import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math'; // +++ NEW: For random number generation
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:lottie/lottie.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'firebase_options.dart';

const String _prefsKeyLastModelName = "last_downloaded_model_name";
// +++ NEW: Key for caching the model list +++
const String _prefsKeyCachedModelList = "cached_models_list";

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return MaterialApp(
      title: 'YOLO Detection App',
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: const Color(0xFF006A6A),
        scaffoldBackgroundColor: const Color(0xFFF0F4F4),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF006A6A),
          brightness: Brightness.light,
          primary: const Color(0xFF006A6A),
          secondary: const Color(0xFF4A6363),
          background: const Color(0xFFF0F4F4),
          error: const Color(0xFFBA1A1A),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(textTheme),
        // --- CORRECTED: Use the proper CardTheme class ---
        cardTheme: const CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
          clipBehavior: Clip.antiAlias,
          margin: EdgeInsets.zero,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF0F4F4),
          foregroundColor: Color(0xFF002020),
          elevation: 0,
          centerTitle: true,
        ),
      ),
      home: const DetectionScreen(),
    );
  }
}


Future<String> getAbsolutePath(String assetPath) async {
  final tempDir = await getTemporaryDirectory();
  final fileName = p.basename(assetPath);
  final tempPath = p.join(tempDir.path, fileName);
  try {
    final byteData = await rootBundle.load(assetPath);
    final buffer = byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes);
    await File(tempPath).writeAsBytes(buffer, flush: true);
    debugPrint("Copied asset $assetPath to $tempPath");
    return tempPath;
  } catch (e) {
    debugPrint("Error copying asset $assetPath: $e");
    throw Exception("Failed to copy asset: $assetPath");
  }
}
// +++ NEW: An enum to represent the different gacha outcomes +++
enum LoaderType { regular, fourStar, standardFiveStar, limitedFiveStar }

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  YOLO? _yoloModel;
  File? _imageFile;
  List<Map<String, dynamic>> _recognitions = [];
  bool _isLoading = false;
  String? _selectedModelName;
  Uint8List? _annotatedImageBytes;
  Uint8List? _maskPngBytes;
  String? _loadingMessage;
  String? _currentModelPath;
  double _originalImageHeight = 0;
  double _originalImageWidth = 0;
  Set<String> _downloadedModelNames = {};

  int? _selectedDetectionIndex;
  bool _showMasks = true;
  double _maskOpacity = 0.5;
  Map<String, Color> _classColorMap = {};

  // --- REFACTORED: Simplified upload state variables ---
  final TextEditingController _modelNameController = TextEditingController();
  File? _selectedTfliteFile;
  String? _selectedTfliteFileName;
  bool _isUploadingModel = false;
  bool _isModelLoadedFromCache = false;

  // +++ NEW: State variable to track internet connectivity +++
  bool _isConnected = true;

  final List<Color> _boxColors = [
  Colors.red, Colors.blue, Colors.green, Colors.yellow.shade700,
  Colors.purple, Colors.orange, Colors.pink, Colors.teal,
  Colors.cyan, Colors.brown, Colors.amber.shade700, Colors.indigo,
  Colors.lime.shade700, Colors.lightGreen.shade700, Colors.deepOrange, Colors.blueGrey
  ];

  List<Map<String, dynamic>> _availableModels = [];
  bool _isFetchingModelList = false;
  late StreamSubscription<List<ConnectivityResult>> _connectivitySubscription;

  // --- MODIFIED: Expanded state for the new gacha system ---
  int _pityCounter5Star = 0;
  int _pityCounter4Star = 0;
  bool _is5050Guaranteed = false;
  LoaderType _loaderType = LoaderType.regular;


  @override
  void initState() {
    super.initState();
    _loadPity(); // +++ NEW: Load pity count on start
    _initializeScreenData();
    _connectivitySubscription = Connectivity().onConnectivityChanged.listen(_updateConnectionStatus);
  }

  // --- MODIFIED: Load and Save methods now handle both pity counters ---
  Future<void> _loadPity() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _pityCounter5Star = prefs.getInt('pity_counter_5_star') ?? 0;
      _pityCounter4Star = prefs.getInt('pity_counter_4_star') ?? 0;
      _is5050Guaranteed = prefs.getBool('is_5050_guaranteed') ?? false;
    });
  }

  Future<void> _savePity() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('pity_counter_5_star', _pityCounter5Star);
    await prefs.setInt('pity_counter_4_star', _pityCounter4Star);
    await prefs.setBool('is_5050_guaranteed', _is5050Guaranteed);
  }

   // --- MODIFIED: The complete HSR gacha simulation logic ---
  void _startLoading(String message) {
    if (mounted) {
      _pityCounter5Star++;
      _pityCounter4Star++;
      debugPrint("5-Star Pity: $_pityCounter5Star | 4-Star Pity: $_pityCounter4Star | Guaranteed: $_is5050Guaranteed");

      bool is5StarPull = false;
      double baseRate5Star = 0.006;
      double softPityIncrease = 0.06;

      // 1. Check for 5-Star (Hard -> Soft -> Base)
      if (_pityCounter5Star >= 90) is5StarPull = true;
      else if (_pityCounter5Star > 73) {
        if (Random().nextDouble() < baseRate5Star + (_pityCounter5Star - 73) * softPityIncrease) is5StarPull = true;
      } else {
        if (Random().nextDouble() < baseRate5Star) is5StarPull = true;
      }
      
      LoaderType currentLoaderType = LoaderType.regular;
      if (is5StarPull) {
        debugPrint("⭐⭐⭐⭐⭐ 5-STAR PULLED AT PITY $_pityCounter5Star! ⭐⭐⭐⭐⭐");
        if (_is5050Guaranteed || Random().nextBool()) {
          debugPrint(">>> You won the 50/50! It's the LIMITED 5-STAR!");
          currentLoaderType = LoaderType.limitedFiveStar;
          _is5050Guaranteed = false;
        } else {
          debugPrint(">>> You lost the 50/50... It's a STANDARD 5-STAR. Next one is guaranteed!");
          currentLoaderType = LoaderType.standardFiveStar;
          _is5050Guaranteed = true;
        }
        _pityCounter5Star = 0;
        _pityCounter4Star = 0; // Pity resets for 4-star too
      } else {
        // 2. If no 5-star, check for 4-Star
        bool is4StarPull = false;
        double baseRate4Star = 0.051; // 5.1% base rate

        if (_pityCounter4Star >= 10) is4StarPull = true;
        else if (Random().nextDouble() < baseRate4Star) is4StarPull = true;

        if (is4StarPull) {
          debugPrint("✨✨✨ 4-STAR PULLED AT PITY $_pityCounter4Star! ✨✨✨");
          currentLoaderType = LoaderType.fourStar;
          _pityCounter4Star = 0; // Reset 4-star pity
        }
      }

      _savePity();

      setState(() {
        _loadingMessage = message;
        _isLoading = true;
        _loaderType = currentLoaderType;
      });
    }
  }

  void _stopLoading() {
    if (mounted) {
      setState(() {
        _isLoading = false;
        _loadingMessage = null;
      });
    }
  }

  // +++ NEW: Function to update connectivity state and show a snackbar +++
  void _updateConnectionStatus(List<ConnectivityResult> results) {
    final bool currentlyConnected = results.contains(ConnectivityResult.mobile) || results.contains(ConnectivityResult.wifi);
    if (mounted && _isConnected != currentlyConnected) {
      setState(() {
        _isConnected = currentlyConnected;
      });
      final message = _isConnected ? "You are back online." : "You've gone offline. Functionality may be limited.";
      final icon = _isConnected ? Icons.wifi_rounded : Icons.wifi_off_rounded;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(
          content: Row(children: [Icon(icon, color: Colors.white), const SizedBox(width: 12), Text(message)]),
          backgroundColor: _isConnected ? Colors.green.shade700 : Colors.orange.shade800,
        ));
    }
  }


  // --- MODIFIED: This function now validates the file type after selection for better compatibility ---
  Future<void> _selectTfliteFile() async {
    // We remove the filter to prevent platform errors on unsupported types.
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any, // Allow any file type
    );

    if (result != null && result.files.single.path != null) {
      final file = result.files.single;

      // Manually check the extension after the file is picked.
      if (file.extension?.toLowerCase() == 'tflite') {
        setState(() {
          _selectedTfliteFile = File(file.path!);
          _selectedTfliteFileName = file.name;
        });
      } else {
        // Show an error message if the wrong file type was selected.
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text("Invalid file type. Please select a .tflite model file."),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } else {
      debugPrint("No file selected.");
    }
  }

  // --- REFACTORED: Upload now only handles the model file ---
  Future<void> _uploadModel() async {
    final modelName = _modelNameController.text.trim();

    if (_selectedTfliteFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please select a .tflite model file.")));
      return;
    }
    if (modelName.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please enter a unique name for the model.")));
      return;
    }
    if (_availableModels.any((m) => m['name'] == modelName)) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("A model named '$modelName' already exists.")));
      return;
    }

    setState(() {
      _isUploadingModel = true;
      _loadingMessage = "Uploading files to Firebase Storage...";
    });

    try {
      final modelFolderPath = 'yoloModels/$modelName';

      // 1. Upload the .tflite file
      final modelRef = FirebaseStorage.instance.ref('$modelFolderPath/model.tflite');
      await modelRef.putFile(_selectedTfliteFile!);
      debugPrint("Model file uploaded to: ${modelRef.fullPath}");

      // 2. Create and upload an empty placeholder labels.txt file to maintain structure
      final labelsRef = FirebaseStorage.instance.ref('$modelFolderPath/labels.txt');
      await labelsRef.putString(''); // Upload an empty string
      debugPrint("Placeholder labels.txt uploaded to: ${labelsRef.fullPath}");

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Model '$modelName' uploaded successfully!"), backgroundColor: Colors.green.shade700),
        );
        setState(() {
          _selectedTfliteFile = null;
          _selectedTfliteFileName = null;
          _modelNameController.clear();
        });
        _handleRefresh();
      }
    } catch (e) {
      debugPrint("Error uploading model: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error uploading model: $e"), backgroundColor: Colors.red.shade700),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isUploadingModel = false;
          _loadingMessage = null;
        });
      }
    }
  }


  // +++ REFACTORED: Check for downloaded models +++
  Future<void> _updateLocalModelStatus() async {
    if (_availableModels.isEmpty) return;
    
    final Set<String> localNames = {};
    for (final model in _availableModels) {
      final modelName = model['name'] as String;
      final modelPath = await _getLocalModelPath(modelName);
      final labelsPath = await _getLocalLabelsPath(modelName);

      if (await File(modelPath).exists() && await File(labelsPath).exists()) {
        localNames.add(modelName);
      }
    }

    if (mounted) {
      setState(() {
        _downloadedModelNames = localNames;
      });
    }
  }

  // +++ REFACTORED: Load initial model based on the new structure +++
  Future<void> _loadInitialModel() async {
    final prefs = await SharedPreferences.getInstance();
    final lastModelName = prefs.getString(_prefsKeyLastModelName);

    if (lastModelName == null) {
      debugPrint("No last-used model found in preferences.");
      if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
      return;
    }

    final modelData = _availableModels.firstWhere(
      (m) => m['name'] == lastModelName,
      orElse: () => <String, dynamic>{},
    );

    if (modelData.isNotEmpty) {
      final String modelPath = await _getLocalModelPath(lastModelName);
      final String labelsPath = await _getLocalLabelsPath(lastModelName);

      if (await File(modelPath).exists() && await File(labelsPath).exists()) {
        debugPrint("Auto-loading last used model '$lastModelName' from local cache.");
        await _prepareAndLoadModel(modelData, isInitialLoad: true);
      } else {
        debugPrint("Files for last used model not found locally. Waiting for user selection.");
        if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
      }
    } else {
      debugPrint("Last used model '$lastModelName' not in the available models list.");
      if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
    }
  }
  
   // --- MODIFIED: Use the new _startLoading and _stopLoading methods ---
  Future<void> _initializeScreenData() async {
    if (!mounted) return;
    _startLoading("Checking for models...");

    final connectivityResult = await Connectivity().checkConnectivity();
    _isConnected = connectivityResult.contains(ConnectivityResult.mobile) || connectivityResult.contains(ConnectivityResult.wifi);

    // ... (rest of the logic is the same)
    List<Map<String, dynamic>> modelsToShow = [];
    if (_isConnected) {
      debugPrint("Online: Attempting to fetch models from Firebase Storage...");
      try {
        modelsToShow = await _fetchModelsFromStorage();
        await _cacheModelList(modelsToShow);
      } catch (e) {
        debugPrint("Firebase fetch failed, falling back to cache: $e");
        modelsToShow = await _loadModelListFromCache();
      }
    } else {
      debugPrint("Offline: Loading models from local cache.");
      modelsToShow = await _loadModelListFromCache();
    }
    
    if (modelsToShow.isEmpty) {
        debugPrint("No models from Firebase or cache. Discovering local files as a fallback.");
        modelsToShow = await _discoverLocalModels();
    }

    if (!mounted) return;
    setState(() {
      _availableModels = modelsToShow;
      _isFetchingModelList = false;
    });

    if (_availableModels.isNotEmpty) {
      await _updateLocalModelStatus();
      await _loadInitialModel();
    } else {
      _stopLoading();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(_isConnected
                ? "No models found in Firebase Storage."
                : "Offline. No models found on device."),
          ),
        );
      }
    }
  }

  // +++ NEW: Caching function to save the model list +++
  Future<void> _cacheModelList(List<Map<String, dynamic>> models) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKeyCachedModelList, jsonEncode(models));
    debugPrint("Model list cached successfully.");
  }

  // +++ NEW: Caching function to load the model list from cache +++
  Future<List<Map<String, dynamic>>> _loadModelListFromCache() async {
    final prefs = await SharedPreferences.getInstance();
    final cachedData = prefs.getString(_prefsKeyCachedModelList);
    if (cachedData != null) {
      final List<dynamic> decoded = jsonDecode(cachedData);
      debugPrint("Loaded ${decoded.length} models from cache.");
      return decoded.cast<Map<String, dynamic>>().toList();
    }
    debugPrint("No cached model list found.");
    return [];
  }
  
  Future<List<Map<String, dynamic>>> _fetchModelsFromStorage() async {
    final List<Map<String, dynamic>> models = [];
    try {
      final listResult = await FirebaseStorage.instance.ref('yoloModels').listAll();
      for (final prefix in listResult.prefixes) {
        models.add({
          'name': prefix.name,
          'storagePath': prefix.fullPath,
        });
      }
      debugPrint("Found ${models.length} models in Firebase Storage.");
      return models;
    } catch (e) {
      debugPrint("Error fetching models from storage: $e");
      return [];
    }
  }


  // --- REFACTORED: Path functions now just use the model display name ---
  Future<String> _getLocalModelPath(String modelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$modelName.tflite");
  }

  Future<String> _getLocalLabelsPath(String modelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$modelName.txt");
  }
  
  // --- REFACTORED: Downloads model and creates dummy labels file ---
  Future<void> _downloadModel(String modelName, String storagePath) async {
    setState(() { _loadingMessage = "Downloading $modelName..."; });
    try {
      final modelRef = FirebaseStorage.instance.ref('$storagePath/model.tflite');
      final localModelFile = File(await _getLocalModelPath(modelName));
      await modelRef.writeToFile(localModelFile);
      debugPrint("Model '$modelName' downloaded to: ${localModelFile.path}");

      // Create a dummy local labels file, as it's required by the package
      final localLabelsFile = File(await _getLocalLabelsPath(modelName));
      if (!await localLabelsFile.exists()) {
        await localLabelsFile.create();
      }
      debugPrint("Created dummy labels file for '$modelName'");

    } catch (e) {
      debugPrint("Error downloading model '$modelName': $e");
      throw Exception("Failed to download model file for '$modelName'.");
    }
  }


  void _clearScreen() {
    setState(() {
      _imageFile = null;
      _recognitions = [];
      _annotatedImageBytes = null;
      _maskPngBytes = null;
    });
  }

  @override
  void dispose() {
    _yoloModel?.dispose();
    _connectivitySubscription.cancel();
    super.dispose();
  }
  
  // --- REFACTORED: Local discovery is now simpler ---
  Future<List<Map<String, dynamic>>> _discoverLocalModels() async {
    debugPrint("Scanning local directory for models...");
    final List<Map<String, dynamic>> foundModels = [];
    final docDir = await getApplicationDocumentsDirectory();
    final files = docDir.listSync();

    final modelFiles = files.where((f) => f.path.endsWith('.tflite')).toList();

    for (final modelFile in modelFiles) {
        final modelName = p.basenameWithoutExtension(modelFile.path);
        final labelsPath = p.join(docDir.path, "$modelName.txt");

        if (await File(labelsPath).exists()) {
            debugPrint("--> Found local model: $modelName");
            foundModels.add({
                'name': modelName,
                // In offline mode, the storage path isn't relevant, but we keep the structure consistent.
                'storagePath': 'yoloModels/$modelName', 
            });
        }
    }
    debugPrint("Discovered ${foundModels.length} models locally.");
    return foundModels;
  }

  // --- REFACTORED: `_prepareAndLoadModel` to provide a valid labels path ---
  Future<void> _prepareAndLoadModel(Map<String, dynamic> modelData, {bool isInitialLoad = false}) async {
    if (!isInitialLoad) _clearScreen();
    setState(() { _isModelLoadedFromCache = false; });

    final String modelName = modelData['name'] as String;
    final String storagePath = modelData['storagePath'] as String;

    if (!mounted) return;
    setState(() {
      _isLoading = true;
      _loadingMessage = "Preparing model: $modelName";
    });

    try {
      final String targetModelPath = await _getLocalModelPath(modelName);
      final modelFile = File(targetModelPath);
      bool loadedFromCache = false;

      if (!await modelFile.exists()) {
        debugPrint("Local file for '$modelName' not found. Downloading...");
        await _downloadModel(modelName, storagePath);
        loadedFromCache = false;
      } else {
        debugPrint("Found local model for '$modelName'. Loading from cache.");
        loadedFromCache = true;
      }

      setState(() { _loadingMessage = "Loading $modelName into memory..."; });

      if (_yoloModel != null) {
        await _yoloModel!.dispose();
      }

      _yoloModel = YOLO(
        modelPath: targetModelPath,
        task: YOLOTask.segment,
      );
      await _yoloModel?.loadModel();

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefsKeyLastModelName, modelName);

      if (!mounted) return;
      setState(() {
        _selectedModelName = modelName;
        _currentModelPath = targetModelPath;
        _isModelLoadedFromCache = loadedFromCache;
      });
      
      final String message = loadedFromCache ? "'$modelName' loaded from cache." : "'$modelName' downloaded successfully.";
      final IconData icon = loadedFromCache ? Icons.storage_rounded : Icons.cloud_download_rounded;
      final Color backgroundColor = loadedFromCache ? Colors.green.shade700 : Colors.blue.shade700;

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Row(children: [Icon(icon, color: Colors.white), const SizedBox(width: 12), Expanded(child: Text(message))]), backgroundColor: backgroundColor),
        );
        await _updateLocalModelStatus();
      }

    } catch (e) {
      debugPrint("Error in _prepareAndLoadModel: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Failed to load model: ${e.toString()}"), backgroundColor: Colors.red.shade700),
        );
        setState(() {
          _yoloModel = null;
          _currentModelPath = null;
          _selectedModelName = null;
        });
      }
    } finally {
      if (mounted) { setState(() { _isLoading = false; _loadingMessage = null; }); }
    }
  }

  // --- REFACTORED: Deletion now uses the model name to find files ---
  Future<void> _deleteLocallyStoredModel() async {
    if (_selectedModelName == null) return;

    final modelNameToDelete = _selectedModelName!;
    
    // Delete the specific files for this model
    final modelFile = File(await _getLocalModelPath(modelNameToDelete));
    final labelsFile = File(await _getLocalLabelsPath(modelNameToDelete));

    if (await modelFile.exists()) await modelFile.delete();
    if (await labelsFile.exists()) await labelsFile.delete();

    // Clear SharedPreferences if the deleted model was the last used one
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(_prefsKeyLastModelName) == modelNameToDelete) {
      await prefs.remove(_prefsKeyLastModelName);
    }

    _clearScreen();
    setState(() {
      _yoloModel = null;
      _currentModelPath = null;
      _selectedModelName = null;
      _isModelLoadedFromCache = false;
      // Remove from the set of downloaded models to update the UI icon
      _downloadedModelNames.remove(modelNameToDelete);
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Local cache for '$modelNameToDelete' has been deleted.")),
    );
  }

  // --- MODIFIED: Model management card with improved styling ---
  Widget _buildModelManagementSection() {
    return Card(
      child: ExpansionTile(
        key: const ValueKey('model-management'),
        leading: Icon(Icons.hub_outlined, color: Theme.of(context).primaryColor),
        title: const Text("Model Management", style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(_selectedModelName ?? "No model selected"),
        initiallyExpanded: _yoloModel == null,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildModelSelector(),
                if (_yoloModel != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 12.0),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: Colors.green.shade700, size: 18),
                        const SizedBox(width: 8),
                        Expanded(child: Text("'$_selectedModelName' is loaded.", style: TextStyle(color: Colors.green.shade800))),
                        if (_isModelLoadedFromCache)
                          IconButton(
                            icon: const Icon(Icons.delete_sweep_outlined, color: Colors.orange),
                            tooltip: "Delete model from local cache",
                            onPressed: () => _showDeleteConfirmation(),
                          ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  // // --- MODIFIED: Upload section with improved styling ---
  // Widget _buildUploadModelSection() {
  //   if (!_isConnected) return const SizedBox.shrink();

  //   return Card(
  //     child: ExpansionTile(
  //       key: const ValueKey('model-upload'),
  //       leading: Icon(Icons.cloud_upload_outlined, color: Theme.of(context).colorScheme.secondary),
  //       title: const Text("Upload New Model", style: TextStyle(fontWeight: FontWeight.bold)),
  //       subtitle: const Text("Add a custom .tflite model"),
  //       children: [
  //         Padding(
  //           padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
  //           child: Column(
  //             crossAxisAlignment: CrossAxisAlignment.start,
  //             children: [
  //               TextField(
  //                 controller: _modelNameController,
  //                 decoration: const InputDecoration(
  //                   labelText: "New Model Unique Name",
  //                   border: OutlineInputBorder(),
  //                 ),
  //               ),
  //               const SizedBox(height: 12),
  //               OutlinedButton.icon(
  //                 icon: const Icon(Icons.attach_file),
  //                 label: Text(_selectedTfliteFileName ?? "Select .tflite File"),
  //                 onPressed: _selectTfliteFile,
  //                 style: OutlinedButton.styleFrom(
  //                   minimumSize: const Size(double.infinity, 50),
  //                   foregroundColor: _selectedTfliteFileName != null ? Colors.green : Theme.of(context).primaryColor,
  //                 ),
  //               ),
  //               const SizedBox(height: 16),
  //               SizedBox(
  //                 width: double.infinity,
  //                 child: ElevatedButton.icon(
  //                   icon: const Icon(Icons.cloud_upload_outlined),
  //                   label: const Text("Upload to Firebase"),
  //                   onPressed: _isUploadingModel ? null : _uploadModel,
  //                 ),
  //               ),
  //             ],
  //           ),
  //         ),
  //       ],
  //     ),
  //   );
  // }

  // Helper method for delete confirmation dialog
  void _showDeleteConfirmation() {
    showDialog(context: context, builder: (ctx) => AlertDialog(
      title: const Text("Confirm Deletion"),
      content: Text("Are you sure you want to delete the local files for '$_selectedModelName'? You can re-download it later if you're online."),
      actions: [
        TextButton(child: const Text("Cancel"), onPressed: () => Navigator.of(ctx).pop()),
        TextButton(child: const Text("Delete"), style: TextButton.styleFrom(foregroundColor: Colors.red), onPressed: () {
          Navigator.of(ctx).pop();
          _deleteLocallyStoredModel();
        }),
      ],
    ));
  }
  
  
  // --- MODIFIED: Model selector dropdown with offline awareness ---
  Widget _buildModelSelector() {
    if (_isFetchingModelList && _availableModels.isEmpty) {
      return const Center(child: Text("Fetching model list..."));
    }
    if (_availableModels.isEmpty && !_isFetchingModelList) {
      return Center(child: Text(
        _isConnected ? "No models available. Pull down to refresh." : "Offline. No cached models found.",
        textAlign: TextAlign.center,
      ));
    }

    return DropdownButtonFormField<String>(
      decoration: const InputDecoration(
        labelText: "Available Models",
        border: OutlineInputBorder(),
      ),
      hint: const Text("Select a Model"),
      value: _selectedModelName,
      isExpanded: true,
      items: _availableModels.map((model) {
        final modelName = model['name'] as String;
        final isDownloaded = _downloadedModelNames.contains(modelName);
        // +++ NEW: An item is only selectable if it's downloaded OR if the user is online +++
        final bool isSelectable = isDownloaded || _isConnected;

        return DropdownMenuItem<String>(
          value: modelName,
          // Disable selection for non-downloaded models when offline
          enabled: isSelectable,
          child: Row(
            children: [
              Expanded(
                child: Text(
                  modelName,
                  style: TextStyle(
                    // Visually grey out non-selectable items
                    color: isSelectable ? null : Colors.grey.shade500,
                  ),
                ),
              ),
              if (isDownloaded)
                const Icon(Icons.download_done, color: Colors.green, size: 20)
              else
                Icon(
                  Icons.cloud_outlined, 
                  color: isSelectable ? Colors.grey : Colors.grey.shade400, 
                  size: 20
                ),
            ],
          ),
        );
      }).toList(),
      onChanged: _isLoading ? null : (String? newValue) {
        if (newValue != null && newValue != _selectedModelName) {
          final selectedModelData = _availableModels.firstWhere((m) => m['name'] == newValue);
          _prepareAndLoadModel(selectedModelData);
        }
      },
    );
  }

  // --- All other methods from this point on can remain largely the same ---
  // --- They handle image picking, segmentation, and UI building for results, ---
  // --- which are not directly affected by the model source change. ---

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      final imageBytes = await image.readAsBytes();
      final decodedImage = await decodeImageFromList(imageBytes);

      setState(() {
        _imageFile = File(image.path);
        _recognitions = [];
        _annotatedImageBytes = null;
        _maskPngBytes = null;
        _originalImageWidth = decodedImage.width.toDouble();
        _originalImageHeight = decodedImage.height.toDouble();
      });

      if (_yoloModel != null && _currentModelPath != null && await File(_currentModelPath!).exists()) {
        _runSegmentation();
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Model not loaded. Please select a model.")),
          );
        }
      }
    }
  }

  void _prettyPrintMap(dynamic data, {String indent = ''}) {
    if (data is Map) {
      data.forEach((key, value) {
        if (value is Map || value is List) {
          debugPrint('$indent$key: (${value.runtimeType})');
          _prettyPrintMap(value, indent: '$indent  ');
        } else if (value is Uint8List) {
          debugPrint('$indent$key: (${value.runtimeType}), Length: ${value.length}');
        }
        else {
          debugPrint('$indent$key: ${value.toString()}');
        }
      });
    } else if (data is List) {
      if (data.isEmpty) {
        debugPrint('$indent- Empty List');
        return;
      }
      debugPrint('$indent[ ... ${data.length} items ... ]');
      debugPrint('$indent  First item:');
      _prettyPrintMap(data.first, indent: '$indent    ');

    } else {
      debugPrint('$indent${data.toString()}');
    }
  }

  Future<void> _runSegmentation() async {
      if (_imageFile == null || _yoloModel == null) return;

      setState(() {
        _isLoading = true;
        _loadingMessage = "Analyzing image...";
        _classColorMap = {}; // Clear previous colors
      });

      try {
        final imageBytes = await _imageFile!.readAsBytes();
        final double confThreshold = 0.5;
        final double nmsThreshold = 0.5;

        debugPrint("Running segmentation with confidence: $confThreshold, IoU: $nmsThreshold");
        debugPrint("--> Using model for prediction: '$_selectedModelName'");

        final detections = await _yoloModel!.predict(
          imageBytes,
          confidenceThreshold: confThreshold,
          iouThreshold: nmsThreshold,
        );

        debugPrint("--- Comprehensive Detections Map Structure ---");
        _prettyPrintMap(detections);
        debugPrint("----------------------------------------------------");

        if (!mounted) return;

        final List<dynamic> boxes = detections['boxes'] ?? [];
        
        final List<Map<String, dynamic>> formattedRecognitions = [];
      int colorIndex = 0;
      final tempColorMap = <String, Color>{};

        for (int i = 0; i < boxes.length; i++) {
          final box = boxes[i];
        final className = box['className'];
          formattedRecognitions.add({
              'x1': box['x1'],
              'y1': box['y1'],
              'x2': box['x2'],
              'y2': box['y2'],
              'className': className,
              'confidence': box['confidence'],
          });

        if (!tempColorMap.containsKey(className)) {
          tempColorMap[className] = _boxColors[colorIndex % _boxColors.length];
          colorIndex++;
        }
        }

        setState(() {
          _recognitions = formattedRecognitions;
        _classColorMap = tempColorMap; 
          _maskPngBytes = detections['maskPng'];
          _annotatedImageBytes = detections['annotatedImage'];
        });

      } catch (e) {
        debugPrint("Error running segmentation: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text("Error during analysis: $e")),
          );
        }
      } finally {
        if (mounted) {
          setState(() {
              _isLoading = false;
              _loadingMessage = null;
          });
        }
      }
    }

  Future<void> _handleRefresh() async {
    debugPrint("Screen refresh triggered.");
    _clearScreen();
    await _initializeScreenData();
  }

  // --- MODIFIED: Main build method for new layout ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Pest Detection", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
        actions: [
          _buildConnectivityIndicator(),
          const SizedBox(width: 16),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _handleRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              _buildModelManagementSection(),
              const SizedBox(height: 16),
              _buildUploadModelSection(),
              const SizedBox(height: 24),
              
              ElevatedButton.icon(
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text("Pick Image from Gallery"),
                onPressed: (_isLoading || _yoloModel == null) ? null : _pickImage,
              ),
              const SizedBox(height: 24),
              
              // --- MODIFIED: Use AnimatedSwitcher for smooth transitions ---
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 500),
                child: _buildContentArea(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // +++ NEW: Persistent connectivity indicator for the AppBar +++
  Widget _buildConnectivityIndicator() {
    final color = _isConnected ? Colors.green.shade700 : Colors.orange.shade800;
    final icon = _isConnected ? Icons.wifi_rounded : Icons.wifi_off_rounded;
    return Tooltip(
      message: _isConnected ? "Online" : "Offline",
      child: Icon(icon, color: color),
    );
  }

  // --- MODIFIED: Upload section with improved styling ---
  Widget _buildUploadModelSection() {
    if (!_isConnected) return const SizedBox.shrink();

    return Card(
      child: ExpansionTile(
        key: const ValueKey('model-upload'),
        leading: Icon(Icons.cloud_upload_outlined, color: Theme.of(context).colorScheme.secondary),
        title: const Text("Upload New Model", style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: const Text("Add a custom .tflite model"),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _modelNameController,
                  decoration: const InputDecoration(
                    labelText: "New Model Unique Name",
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  icon: const Icon(Icons.attach_file),
                  label: Text(_selectedTfliteFileName ?? "Select .tflite File"),
                  onPressed: _selectTfliteFile,
                  style: OutlinedButton.styleFrom(
                    minimumSize: const Size(double.infinity, 50),
                    foregroundColor: _selectedTfliteFileName != null ? Colors.green : Theme.of(context).primaryColor,
                  ),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.cloud_upload_outlined),
                    label: const Text("Upload to Firebase"),
                    onPressed: _isUploadingModel ? null : _uploadModel,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

   // --- MODIFIED: Use the enum and a switch statement to show the correct loader ---
  Widget _buildContentArea() {
    if (_isLoading) {
      Widget loaderWidget;
      switch (_loaderType) {
        case LoaderType.limitedFiveStar:
          loaderWidget = Image.asset('assets/images/loading_limited.gif', width: 120, height: 120);
          break;
        case LoaderType.standardFiveStar:
          loaderWidget = Image.asset('assets/images/loading.gif', width: 200, height: 200);
          break;
        case LoaderType.fourStar:
          loaderWidget = Lottie.asset('assets/animations/loading_4.json', width: 200, height: 200);
          break;
        case LoaderType.regular:
        default:
          loaderWidget = Lottie.asset('assets/animations/loading.json', width: 200, height: 200);
      }

      return Container(
        key: const ValueKey('loading'),
        padding: const EdgeInsets.symmetric(vertical: 50.0),
        child: Column(
          children: [
            loaderWidget,
            if (_loadingMessage != null && _loadingMessage!.isNotEmpty) ...[
              const SizedBox(height: 20),
              Text(
                _loadingMessage!,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade700),
              ),
            ]
          ],
        ),
      );
    }

    // Show results view if an image has been analyzed
    if (_imageFile != null) {
      // Use a LayoutBuilder to create a responsive UI
      return LayoutBuilder(
        builder: (context, constraints) {
          // Use a side-by-side layout for wide screens (tablets, landscape phones)
          if (constraints.maxWidth > 600) {
            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 6, // Image takes 60% of the space
                  child: _buildResultsImage(),
                ),
                const SizedBox(width: 20),
                Expanded(
                  flex: 4, // List and controls take 40%
                  child: _buildResultsList(),
                ),
              ],
            );
          } else {
            // Use a top-and-bottom layout for narrow screens (portrait phones)
            return Column(
              children: [
                _buildResultsImage(),
                const SizedBox(height: 20),
                _buildResultsList(),
              ],
            );
          }
        },
      );
    }
    
     // --- MODIFIED: A much better initial empty state ---
    return Container(
      key: const ValueKey('initial'),
      padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 20),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.science_outlined, size: 80, color: Colors.grey.shade400),
            const SizedBox(height: 20),
            Text(
              "Ready to Analyze",
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Text(
              "Select a model and pick an image",
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInteractiveControls() {
  return Card(
    elevation: 2,
    shadowColor: Colors.black.withOpacity(0.1),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    child: Padding(
      padding: const EdgeInsets.all(12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("Show Masks", style: TextStyle(fontWeight: FontWeight.bold)),
              Switch(
                value: _showMasks,
                onChanged: (value) {
                  setState(() {
                    _showMasks = value;
                  });
                },
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text("Mask Opacity", style: TextStyle(fontWeight: FontWeight.bold)),
          Slider(
            value: _maskOpacity,
            min: 0.1,
            max: 1.0,
            onChanged: (value) {
              setState(() {
                _maskOpacity = value;
              });
            },
          ),
        ],
      ),
    ),
  );
}

  Widget _buildNoDetectionsFound() {
    return Stack(
      alignment: Alignment.center,
      children: [
        Image.file(_imageFile!),
        Container(
          color: Colors.black.withOpacity(0.6),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          child: const Text(
            "No objects detected",
            style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold),
          ),
        ),
      ],
    );
  }
  
  Widget _buildResultsImage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text("Analysis Result", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Card(
          elevation: 4,
          shadowColor: Colors.black.withOpacity(0.2),
          clipBehavior: Clip.antiAlias,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: _recognitions.isEmpty && !_isLoading
              ? _buildNoDetectionsFound() 
              : _buildImageWithDetections(),
        ),
      ],
    );
  }

  Widget _buildResultsList() {
    return Column(
      children: [
        // Only show controls if there are detections
        if (_recognitions.isNotEmpty) ...[
          _buildInteractiveControls(),
          const SizedBox(height: 20),
        ],
        Text("Detected Objects: ${_recognitions.length}", style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 10),
        _buildDetectionList(), // This will now be interactive
        const SizedBox(height: 24),
        if (_annotatedImageBytes != null) ...[
          Text("Library's Raw Output", style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 10),
          Card(
            elevation: 4,
            shadowColor: Colors.black.withOpacity(0.2),
            clipBehavior: Clip.antiAlias,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Image.memory(_annotatedImageBytes!),
          )
        ]
      ],
    );
  }

  Widget _buildImageWithDetections() {
    if (_imageFile == null) return const SizedBox.shrink();

    // Fallback for models that might not produce a mask PNG
    if (_maskPngBytes == null) {
      return _recognitions.isNotEmpty
      ? _buildImageWithBoxesOnly()
      : Image.file(_imageFile!);
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        if (_originalImageWidth == 0 || _originalImageHeight == 0) {
          return const SizedBox.shrink();
        }

        final double scaleRatio = constraints.maxWidth / _originalImageWidth;
        final double responsiveHeight = _originalImageHeight * scaleRatio;

        return FutureBuilder<List<ui.Image>>(
          future: Future.wait([
            _loadImage(_imageFile!),
            _loadImageFromBytes(_maskPngBytes!),
          ]),
          builder: (context, snapshot) {
            if (!snapshot.hasData || snapshot.data!.length < 2) {
              return SizedBox(
                width: constraints.maxWidth,
                height: responsiveHeight,
                child: const Center(child: CircularProgressIndicator())
              );
            }

            final ui.Image originalImage = snapshot.data![0];
            final ui.Image maskImage = snapshot.data![1];

            return CustomPaint(
              size: Size(constraints.maxWidth, responsiveHeight),
              painter: _DetectionPainter(
                originalImage: originalImage,
                maskImage: maskImage,
                recognitions: _recognitions,
                boxColors: _boxColors,
                scaleRatio: scaleRatio,
                // Pass the new interactive state to the painter
                showMasks: _showMasks,
                maskOpacity: _maskOpacity,
                selectedDetectionIndex: _selectedDetectionIndex,
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildImageWithBoxesOnly() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double scaleRatio = constraints.maxWidth / _originalImageWidth;
        final double responsiveHeight = _originalImageHeight * scaleRatio;
        
        return FutureBuilder<ui.Image>(
          future: _loadImage(_imageFile!),
          builder: (context, snapshot) {
            if (!snapshot.hasData) return const SizedBox.shrink();
            
            return CustomPaint(
              size: Size(constraints.maxWidth, responsiveHeight),
              painter: _DetectionPainter(
                originalImage: snapshot.data!,
                recognitions: _recognitions,
                boxColors: _boxColors,
                scaleRatio: scaleRatio,
                showMasks: _showMasks,
                maskOpacity: _maskOpacity,
                selectedDetectionIndex: _selectedDetectionIndex,
              ),
            );
          },
        );
      },
    );
  }

  Future<ui.Image> _loadImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  Future<ui.Image> _loadImageFromBytes(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

   // --- MODIFIED: Results list item with better styling ---
  Widget _buildDetectionList() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _recognitions.length,
      itemBuilder: (context, index) {
      final detection = _recognitions[index];
      final className = detection['className'] ?? 'Unknown';
      final confidence = (detection['confidence'] as num).toDouble();
      final isSelected = _selectedDetectionIndex == index;
      final itemColor = _classColorMap[className] ?? Colors.grey.shade700;

      return Card(
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
          elevation: isSelected ? 8 : 2,
          shadowColor: isSelected ? itemColor.withOpacity(0.5) : Colors.black.withOpacity(0.1),
          shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: isSelected
              ? BorderSide(color: itemColor, width: 2.5)
              : BorderSide(color: Colors.grey.shade300, width: 1),
          ),
          child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () { setState(() { _selectedDetectionIndex = isSelected ? null : index; }); },
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: itemColor.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: itemColor.withOpacity(0.8), width: 1.5)
                    ),
                    child: Text(
                      '${index + 1}',
                      style: TextStyle(
                        color: itemColor,
                        fontSize: 18,
                        fontWeight: FontWeight.bold
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          className,
                          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${(confidence * 100).toStringAsFixed(1)}% Confidence',
                          style: TextStyle(color: Colors.grey.shade700, fontWeight: FontWeight.w500),
                        ),
                      ],
                    ),
                  ),
                  if (isSelected)
                    Icon(Icons.check_circle, color: itemColor, size: 28),
                ],
              ),
            ),
          ),
      );
      },
      );
  }
}

class _DetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final List<Color> boxColors;
  final double scaleRatio;
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;

  _DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.boxColors,
    required this.scaleRatio,
    required this.showMasks,
    required this.maskOpacity,
    this.selectedDetectionIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: originalImage,
      fit: BoxFit.fill,
    );

    if (showMasks && maskImage != null && recognitions.isNotEmpty) {
      final stencilPath = Path();
      // Logic to highlight only the selected mask
      if (selectedDetectionIndex != null) {
          final detection = recognitions[selectedDetectionIndex!];
          final rect = Rect.fromLTRB(
              (detection['x1'] as num) * scaleRatio,
              (detection['y1'] as num) * scaleRatio,
              (detection['x2'] as num) * scaleRatio,
              (detection['y2'] as num) * scaleRatio,
          );
          stencilPath.addRect(rect);
      } else { // Or show all masks
          for (final detection in recognitions) {
              final rect = Rect.fromLTRB(
                  (detection['x1'] as num) * scaleRatio,
                  (detection['y1'] as num) * scaleRatio,
                  (detection['x2'] as num) * scaleRatio,
                  (detection['y2'] as num) * scaleRatio,
              );
              stencilPath.addRect(rect);
          }
      }
      
      canvas.save();
      canvas.clipPath(stencilPath);
      final maskPaint = Paint()..color = Colors.white.withOpacity(maskOpacity);
      canvas.drawImageRect(
          maskImage!,
          Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
          Rect.fromLTWH(0, 0, size.width, size.height),
          maskPaint,
        );
      canvas.restore();
    }

    final Map<String, Color> classColorMap = {};
    int colorIndex = 0;

    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';

      if (!classColorMap.containsKey(className)) {
        classColorMap[className] = boxColors[colorIndex % boxColors.length];
        colorIndex++;
      }
      final color = classColorMap[className]!;
      final isSelected = i == selectedDetectionIndex;
      
      final double x1 = (detection['x1'] as num).toDouble() * scaleRatio;
      final double y1 = (detection['y1'] as num).toDouble() * scaleRatio;
      final double x2 = (detection['x2'] as num).toDouble() * scaleRatio;
      final double y2 = (detection['y2'] as num).toDouble() * scaleRatio;
      final double confidence = (detection['confidence'] as num).toDouble();

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSelected ? 4.0 : 2.5; // Highlight selected box
      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final textPainter = TextPainter(
        text: TextSpan(
          text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            shadows: [Shadow(color: Colors.black, blurRadius: 4)]
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);

      final labelBackgroundPaint = Paint()..color = color.withOpacity(isSelected ? 1.0 : 0.8);
      final labelRect = Rect.fromLTWH(
        x1,
        y1 - (textPainter.height + 4), // Position label above the box
        textPainter.width + 8,
        textPainter.height + 4,
      );
      
      // Ensure label is within view bounds
      double top = y1 - textPainter.height - 4;
      if (top < 0) {
        top = y2 + 2; // If no space above, place it below
      }

      final finalLabelRect = Rect.fromLTWH(x1, top, textPainter.width + 8, textPainter.height + 4);

      canvas.drawRect(finalLabelRect, labelBackgroundPaint);
      
      textPainter.paint(canvas, Offset(x1 + 4, top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return originalImage != oldDelegate.originalImage ||
            maskImage != oldDelegate.maskImage ||
            recognitions != oldDelegate.recognitions ||
            showMasks != oldDelegate.showMasks ||
            maskOpacity != oldDelegate.maskOpacity ||
            selectedDetectionIndex != oldDelegate.selectedDetectionIndex;
  }
}
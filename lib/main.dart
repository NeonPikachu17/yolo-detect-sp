import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui; // Needed for ui.decodeImageFromPixels
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'dart:convert';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_ml_model_downloader/firebase_ml_model_downloader.dart';
import 'firebase_options.dart';

const String _activeModelFileName = "active_custom_yolo_model.tflite";
const String _activeLabelsFileName = "active_custom_yolo_labels.txt";
const String _prefsKeyLastModelName = "last_downloaded_model_name";
const String _prefsKeyLastModelFirebaseName = "last_downloaded_model_firebase_name";

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
    return MaterialApp(
      title: 'YOLO Detection App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
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

  // --- NEW INTERACTIVE STATE VARIABLES ---
  int? _selectedDetectionIndex;
  bool _showMasks = true;
  double _maskOpacity = 0.5;
  // --- END OF NEW VARIABLES ---

  // --- NEW STATE VARIABLE FOR CACHE STATUS ---
  bool _isModelLoadedFromCache = false;


  final List<Color> _boxColors = [
    Colors.red, Colors.blue, Colors.green, Colors.yellow.shade700,
    Colors.purple, Colors.orange, Colors.pink, Colors.teal,
    Colors.cyan, Colors.brown, Colors.amber.shade700, Colors.indigo,
    Colors.lime.shade700, Colors.lightGreen.shade700, Colors.deepOrange, Colors.blueGrey
  ];

  List<Map<String, dynamic>> _availableModels = [];
  bool _isFetchingModelList = false;

  @override
  void initState() {
    super.initState();
    _initializeScreenData();
  }

  Future<void> _initializeScreenData() async {
    if (!mounted) return;
    setState(() {
      _isFetchingModelList = true;
      _isLoading = true;
      _loadingMessage = "Fetching available models...";
    });

    final prefs = await SharedPreferences.getInstance();
    List<Map<String, dynamic>> modelsToShow = [];
    final connectivityResult = await Connectivity().checkConnectivity();
    final bool isConnected = connectivityResult.any((r) =>
        r == ConnectivityResult.wifi || r == ConnectivityResult.mobile || r == ConnectivityResult.ethernet);

    if (isConnected) {
      try {
        debugPrint("Online: Fetching models from Firestore...");
        modelsToShow = await _fetchModelsFromFirestore();
        await prefs.setString('cached_models_list', jsonEncode(modelsToShow));
      } catch (e) {
        debugPrint("Error fetching models from Firestore: $e. Attempting to use cache.");
        modelsToShow = await _loadModelsFromCache(prefs);
      }
    } else { // Offline
      debugPrint("Offline: Loading models from cache or checking local active model.");
      modelsToShow = await _loadModelsFromCache(prefs);
      if (modelsToShow.isEmpty) {
        final Map<String, dynamic>? activeModelData = await _getLocallyActiveModelData(prefs);
        if (activeModelData != null) {
          modelsToShow = [activeModelData];
          debugPrint("Offline: Using only locally available active model: ${activeModelData['name']}");
        }
      }
    }

    if (!mounted) return;
    setState(() {
      _availableModels = modelsToShow;
      _isFetchingModelList = false;
    });

    if (_availableModels.isNotEmpty) {
      await _autoLoadLastUsedModel();
    } else {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _loadingMessage = null;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text(isConnected
                  ? "No models found. Please check Firebase configuration."
                  : "Offline. No models available.")),
        );
      }
    }
  }

  Future<List<Map<String, dynamic>>> _fetchModelsFromFirestore() async {
    QuerySnapshot snapshot = await FirebaseFirestore.instance.collection('yoloModels').get();
    return snapshot.docs.map((doc) {
      final data = doc.data() as Map<String, dynamic>;
      return <String, dynamic>{
        'name': data['name'] as String,
        'firebaseModelName': data['firebaseModelName'] as String,
        'labelsAssetPath': data['labelsAssetPath'] as String,
      };
    }).toList();
  }

  Future<List<Map<String, dynamic>>> _loadModelsFromCache(SharedPreferences prefs) async {
    String? cachedModelsJson = prefs.getString('cached_models_list');
    if (cachedModelsJson != null) {
      try {
        List<dynamic> decodedList = jsonDecode(cachedModelsJson);
        final models = decodedList.map((item) => item as Map<String, dynamic>).toList();
        debugPrint("Loaded ${models.length} models from cache.");
        return models;
      } catch (e) {
        debugPrint("Error decoding cached models: $e");
      }
    }
    return [];
  }

  Future<Map<String, dynamic>?> _getLocallyActiveModelData(SharedPreferences prefs) async {
    final lastModelDisplayName = prefs.getString(_prefsKeyLastModelName);
    final lastFirebaseModelName = prefs.getString(_prefsKeyLastModelFirebaseName);

    if (lastModelDisplayName != null && lastFirebaseModelName != null) {
      // --- FIX: Pass the required 'lastFirebaseModelName' argument ---
      final localModelPath = await _getLocalModelPath(lastFirebaseModelName); 

      if (await File(localModelPath).exists()) {
        return {
          'name': lastModelDisplayName,
          'firebaseModelName': lastFirebaseModelName,
          'labelsAssetPath': 'assets/models/labels.txt', // Assuming a default path
        };
      }
    }
    return null;
  }

  Future<void> _autoLoadLastUsedModel() async {
    final prefs = await SharedPreferences.getInstance();
    final lastModelDisplayName = prefs.getString(_prefsKeyLastModelName);

    if (lastModelDisplayName != null) {
      final modelData = _availableModels.firstWhere(
          (m) => m['name'] == lastModelDisplayName,
          orElse: () => {});
      
      if (modelData.isNotEmpty) {
        // Use the main prepare function for auto-loading
        await _prepareAndLoadModel(modelData, isInitialLoad: true);
      }
    } else {
      if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
    }
  }

  Future<String> _getLocalModelPath(String firebaseModelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$firebaseModelName.tflite");
  }

  Future<String> _getLocalLabelsPath(String firebaseModelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$firebaseModelName.txt");
  }

  Future<void> _downloadModel(String modelName, String firebaseModelName, String labelsAssetPath) async {
    // No longer need to delete old files, as we're saving to a new unique file.
    
    setState(() { _loadingMessage = "Downloading $modelName..."; });
    final FirebaseModelDownloader modelDownloader = FirebaseModelDownloader.instance;
    final FirebaseCustomModel downloadedModel = await modelDownloader.getModel(
      firebaseModelName,
      FirebaseModelDownloadType.latestModel,
      FirebaseModelDownloadConditions(
        iosAllowsCellularAccess: true,
        androidWifiRequired: false,
      ),
    );

    // Get the unique path for this model
    final targetLocalModelPath = await _getLocalModelPath(firebaseModelName);
    await downloadedModel.file.copy(targetLocalModelPath);
    debugPrint("Model '$firebaseModelName' downloaded to: $targetLocalModelPath");
    
    // Get the unique path for the labels
    final targetLocalLabelsPath = await _getLocalLabelsPath(firebaseModelName);
    final assetLabelsTempPath = await getAbsolutePath(labelsAssetPath);
    await File(assetLabelsTempPath).copy(targetLocalLabelsPath);
    debugPrint("Labels for '$firebaseModelName' copied to $targetLocalLabelsPath");
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
    // It's crucial to dispose of the model when the screen is closed.
    _yoloModel?.dispose();
    super.dispose();
  }

  Future<void> _prepareAndLoadModel(Map<String, dynamic> modelData, {bool isInitialLoad = false}) async {
    if (!isInitialLoad) {
      _clearScreen();
    }
    setState(() { _isModelLoadedFromCache = false; });

    final String modelNameDisplay = modelData['name'] as String;
    final String firebaseModelName = modelData['firebaseModelName'] as String;
    final String labelsAssetPath = modelData['labelsAssetPath'] as String;

    if (!mounted) return;
    setState(() {
      _isLoading = true;
      _loadingMessage = "Preparing model: $modelNameDisplay";
    });

    try {
      final String targetModelPath = await _getLocalModelPath(firebaseModelName);
      final modelFile = File(targetModelPath);
      bool loadedFromCache = false;

      if (!await modelFile.exists()) {
        debugPrint("Local file for '$modelNameDisplay' not found. Downloading...");
        await _downloadModel(modelNameDisplay, firebaseModelName, labelsAssetPath);
        loadedFromCache = false;
      } else {
        debugPrint("Found local model for '$modelNameDisplay'. Loading from cache.");
        loadedFromCache = true;
      }

      setState(() { _loadingMessage = "Loading $modelNameDisplay into memory..."; });
      
      // --- NEW DISPOSE AND RE-CREATE LOGIC ---
      // If a model instance already exists, dispose of it first to free up resources.
      if (_yoloModel != null) {
        debugPrint("Disposing previous model instance...");
        await _yoloModel!.dispose();
      }

      // Always create a new instance for the selected model.
      debugPrint("Initializing new YOLO instance for '$modelNameDisplay'...");
      _yoloModel = YOLO(
        modelPath: targetModelPath,
        task: YOLOTask.segment,
      );
      await _yoloModel?.loadModel();
      // --- END NEW LOGIC ---

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefsKeyLastModelName, modelNameDisplay);
      await prefs.setString(_prefsKeyLastModelFirebaseName, firebaseModelName);

      if (!mounted) return;
      setState(() {
        _selectedModelName = modelNameDisplay;
        _currentModelPath = targetModelPath;
        _isModelLoadedFromCache = loadedFromCache;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("$modelNameDisplay ready.")),
      );

    } catch (e) {
      debugPrint("Error in _prepareAndLoadModel: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Failed to load model: ${e.toString()}")),
        );
        setState(() {
          _yoloModel = null;
          _currentModelPath = null;
          _isModelLoadedFromCache = false;
        });
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

  Future<void> _deleteLocallyStoredModel() async {
    if (_yoloModel == null || _selectedModelName == null) return;

    final modelToDelete = _availableModels.firstWhere((m) => m['name'] == _selectedModelName);
    final firebaseModelName = modelToDelete['firebaseModelName'];
    
    setState(() { /* ... loading state ... */ });

    // Delete the specific files for this model
    final modelFile = File(await _getLocalModelPath(firebaseModelName));
    final labelsFile = File(await _getLocalLabelsPath(firebaseModelName));
    if (await modelFile.exists()) await modelFile.delete();
    if (await labelsFile.exists()) await labelsFile.delete();

    // Clear SharedPreferences if the deleted model was the last used one
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(_prefsKeyLastModelFirebaseName) == firebaseModelName) {
      await prefs.remove(_prefsKeyLastModelName);
      await prefs.remove(_prefsKeyLastModelFirebaseName);
    }

    _clearScreen();
    setState(() {
      _yoloModel = null;
      _currentModelPath = null;
      _selectedModelName = null;
      _isModelLoadedFromCache = false;
      _isLoading = false;
      _loadingMessage = null;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Local model '${modelToDelete['name']}' has been deleted.")),
    );
  }

  Widget _buildModelManagementSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                decoration: const InputDecoration(
                  labelText: "Select Model",
                  border: OutlineInputBorder(),
                ),
                hint: const Text("Select a Model to Load"),
                value: _selectedModelName,
                isExpanded: true,
                items: _availableModels.map((model) {
                  return DropdownMenuItem<String>(
                    value: model['name'] as String?,
                    child: Text(model['name'] as String? ?? "Unnamed Model"),
                  );
                }).toList(),
                onChanged: _isLoading ? null : (String? newValue) {
                  if (newValue != null) {
                    final selectedModelData =
                        _availableModels.firstWhere((m) => m['name'] == newValue, orElse: () => {});
                    if (selectedModelData.isNotEmpty) {
                      _prepareAndLoadModel(selectedModelData);
                    }
                  }
                },
              ),
            ),
            // Add the delete button, only visible when a model is loaded
            if (_yoloModel != null && !_isLoading)
              IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.red),
                tooltip: "Delete Local Model Cache",
                onPressed: _deleteLocallyStoredModel,
              ),
          ],
        ),
        // Add the status indicator, only visible when a model is loaded
        if (_yoloModel != null)
          Padding(
            padding: const EdgeInsets.only(top: 8.0),
            child: Row(
              children: [
                Icon(
                  _isModelLoadedFromCache ? Icons.storage_rounded : Icons.cloud_download_rounded,
                  color: Colors.grey.shade600,
                  size: 16,
                ),
                const SizedBox(width: 4),
                Text(
                  _isModelLoadedFromCache ? "Status: Loaded from Local Cache" : "Status: Freshly Downloaded",
                  style: TextStyle(color: Colors.grey.shade600, fontStyle: FontStyle.italic),
                ),
              ],
            ),
          ),
      ],
    );
  }

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

  // Add this helper function inside your _DetectionScreenState class
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
      // Only print details for the first item of a list to avoid flooding the console
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
    });

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final double confThreshold = 0.5;
      final double nmsThreshold = 0.5;

      debugPrint("Running segmentation with confidence: $confThreshold, IoU: $nmsThreshold");

      // --- NEW DEBUG LINE ADDED HERE ---
      debugPrint("--> Using model for prediction: '$_selectedModelName'");

      final detections = await _yoloModel!.predict(
        imageBytes,
        confidenceThreshold: confThreshold,
        iouThreshold: nmsThreshold,
      );

      // ===================================================================
      // ============= COMPREHENSIVE DEBUG OUTPUT STARTS HERE =============
      // ===================================================================
      debugPrint("----------------------------------------------------");
      debugPrint("--- Comprehensive Detections Map Structure ---");
      _prettyPrintMap(detections);
      debugPrint("----------------------------------------------------");
      // ===================================================================
      // ============== COMPREHENSIVE DEBUG OUTPUT ENDS HERE ==============
      // ===================================================================

      if (!mounted) return;

      final List<dynamic> boxes = detections['boxes'] ?? [];
      
      final List<Map<String, dynamic>> formattedRecognitions = [];
      for (int i = 0; i < boxes.length; i++) {
        final box = boxes[i];
        formattedRecognitions.add({
          'x1': box['x1'],
          'y1': box['y1'],
          'x2': box['x2'],
          'y2': box['y2'],
          'className': box['className'],
          'confidence': box['confidence'],
        });
      }

      setState(() {
        _recognitions = formattedRecognitions;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Pest Detection"),
        centerTitle: true,
        elevation: 2,
      ),
      body: RefreshIndicator(
        onRefresh: _handleRefresh,
        child: Builder(
          builder: (context) {
            final screenWidth = MediaQuery.of(context).size.width;
            final horizontalPadding = screenWidth > 600 ? screenWidth * 0.1 : 16.0;

            return SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: EdgeInsets.symmetric(horizontal: horizontalPadding, vertical: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  _buildModelManagementSection(),
                  const SizedBox(height: 20),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text("Pick Image from Gallery"),
                    onPressed: _isLoading ? null : _pickImage,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
                      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                  
                  _buildContentArea(),
                ],
              ),
            );
          }
        ),
      ),
    );
  }

  Widget _buildContentArea() {
    if (_isLoading) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 50.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Ensure you have a 'loading.gif' in an 'assets/images/' directory
              Image.asset( 
                'assets/images/loading.gif',
                width: 120,
                height: 120,
              ),
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
    
    // Default state when no image is picked yet
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 60),
      child: Center(
        child: Text(
          "Select a model and pick an image to begin.",
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(color: Colors.grey.shade600),
        ),
      ),
    );
  }

  // Add this new helper method to build the interactive controls panel
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

  // Add this new helper widget for the "No Detections" case
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
  
  Widget _buildModelSelector() {
    if (_isFetchingModelList && _availableModels.isEmpty) {
      return const Center(child: Text("Fetching model list..."));
    }
    if (_availableModels.isEmpty && !_isFetchingModelList) {
      return const Center(child: Text("No models available to select."));
    }

    return DropdownButtonFormField<String>(
      decoration: const InputDecoration(
        labelText: "Select Model",
        border: OutlineInputBorder(),
      ),
      hint: const Text("Select a Model to Load"),
      value: _selectedModelName,
      isExpanded: true,
      items: _availableModels.map((model) {
        return DropdownMenuItem<String>(
          value: model['name'] as String?,
          child: Text(model['name'] as String? ?? "Unnamed Model"),
        );
      }).toList(),
      onChanged: _isLoading ? null : (String? newValue) {
        if (newValue != null) {
          final selectedModelData =
              _availableModels.firstWhere((m) => m['name'] == newValue, orElse: () => {});
          if (selectedModelData.isNotEmpty) {
            _prepareAndLoadModel(selectedModelData);
          }
        }
      },
    );
  }

  // This method builds the main results image view
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
              ? Center(child: Text("No detections found.")) // Simplified fallback
              : _buildImageWithDetections(),
        ),
      ],
    );
  }

  // This method builds the interactive list and controls
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
                // --- FIX START: Add the missing interactive parameters ---
                showMasks: _showMasks,
                maskOpacity: _maskOpacity,
                selectedDetectionIndex: _selectedDetectionIndex,
                // The `maskImage` is intentionally left null for "boxes only" mode.
                // --- FIX END ---
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

        return Card(
          elevation: isSelected ? 4 : 2,
          color: isSelected ? Theme.of(context).primaryColor.withOpacity(0.1) : null,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
            side: isSelected 
              ? BorderSide(color: Theme.of(context).primaryColor, width: 2) 
              : BorderSide.none,
          ),
          child: ListTile(
            leading: CircleAvatar(child: Text('${index + 1}')),
            title: Text(className),
            subtitle: Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
            onTap: () {
              setState(() {
                // Toggle selection
                _selectedDetectionIndex = isSelected ? null : index;
              });
            },
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
  // New properties for interactivity
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;

  _DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.boxColors,
    required this.scaleRatio,
    // Add new properties to constructor
    required this.showMasks,
    required this.maskOpacity,
    this.selectedDetectionIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Draw the original background image
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: originalImage,
      fit: BoxFit.fill,
    );

    // 2. Selectively draw the mask PNG using clipping
    if (showMasks && maskImage != null && recognitions.isNotEmpty) {
      final stencilPath = Path();
      for (final detection in recognitions) {
        final double x1 = (detection['x1'] as num).toDouble() * scaleRatio;
        final double y1 = (detection['y1'] as num).toDouble() * scaleRatio;
        final double x2 = (detection['x2'] as num).toDouble() * scaleRatio;
        final double y2 = (detection['y2'] as num).toDouble() * scaleRatio;
        stencilPath.addRect(Rect.fromLTRB(x1, y1, x2, y2));
      }
      
      canvas.save();
      canvas.clipPath(stencilPath);
      // Use the Opacity value from the slider
      final maskPaint = Paint()..color = Colors.white.withOpacity(maskOpacity);
      canvas.drawImageRect(
          maskImage!,
          Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
          Rect.fromLTWH(0, 0, size.width, size.height),
          maskPaint,
        );
      canvas.restore();
    }

    // --- 3. DRAW BOUNDING BOXES AND LABELS WITH CLASS-BASED COLORS ---
    
     // 3. Draw the bounding box outlines and labels
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

      // Draw bounding box outline
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      // Draw label
      final textPainter = TextPainter(
        text: TextSpan(
          text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);

      // Draw a padded background for the label
      final labelBackgroundPaint = Paint()..color = color;
      final labelRect = Rect.fromLTWH(
        x1, 
        y1, 
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRect(labelRect, labelBackgroundPaint);
      
      // Draw the text on top of the background
      textPainter.paint(canvas, Offset(x1 + 4, y1 + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    // Update shouldRepaint to include the new interactive properties
    return originalImage != oldDelegate.originalImage ||
           maskImage != oldDelegate.maskImage ||
           recognitions != oldDelegate.recognitions ||
           showMasks != oldDelegate.showMasks ||
           maskOpacity != oldDelegate.maskOpacity ||
           selectedDetectionIndex != oldDelegate.selectedDetectionIndex;
  }
}
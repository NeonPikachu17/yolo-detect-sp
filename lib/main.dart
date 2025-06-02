import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart'; // If using SharedPreferences
import 'package:connectivity_plus/connectivity_plus.dart';

// Add this import statement:
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_ml_model_downloader/firebase_ml_model_downloader.dart';
import 'firebase_options.dart';

const String _activeModelFileName = "active_custom_yolo_model.tflite";
const String _activeLabelsFileName = "active_custom_yolo_labels.txt";
const String _prefsKeyLastModelName = "last_downloaded_model_name";
const String _prefsKeyLastModelFirebaseName = "last_downloaded_model_firebase_name";

Future<void> main() async { // Mark main as async and return Future<void>
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform, // Recommended way after `flutterfire configure`
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
  String? _selectedModelName; // This will reflect the model selected in dropdown
  // String? _currentModelPath; // We'll use the fixed local path
  Uint8List? _annotatedImageBytes;
  String? _loadingMessage;

  String? _currentModelPath;         // To store the path of the actively loaded model
  double _originalImageWidth = 0.0;  // For responsive bounding box scaling
  double _originalImageHeight = 0.0; // For responsive bounding box scaling

  final List<Color> _boxColors = [
    Colors.red, Colors.blue, Colors.green, Colors.yellow.shade700,
    Colors.purple, Colors.orange, Colors.pink, Colors.teal,
    Colors.cyan, Colors.brown, Colors.amber.shade700, Colors.indigo,
    Colors.lime.shade700, Colors.lightGreen.shade700, Colors.deepOrange, Colors.blueGrey
  ];

  final List<Map<String, String>> _availableModels = [
    {
      'name': 'YOLOv8n16',
      'firebaseModelName': 'YOLOv8n16',
      'labelsAssetPath': 'assets/models/labels.txt',
    },
    {
      'name': 'YOLOv8n32',
      'firebaseModelName': 'YOLOv8n32',
      'labelsAssetPath': 'assets/models/labels.txt',
    },
    {
      'name': 'YOLOv11n16',
      'firebaseModelName': 'YOLOv11n16',
      'labelsAssetPath': 'assets/models/labels.txt',
    },
    {
      'name': 'YOLOv11n32',
      'firebaseModelName': 'YOLOv11n32',
      'labelsAssetPath': 'assets/models/labels.txt',
    },
  ];

  @override
  void initState() {
    super.initState();
    _autoLoadLastUsedModel(); // Try to load last used model on init
  }

  Future<void> _autoLoadLastUsedModel() async {
    final prefs = await SharedPreferences.getInstance();
    final lastModelDisplayName = prefs.getString(_prefsKeyLastModelName);
    final lastFirebaseModelName = prefs.getString(_prefsKeyLastModelFirebaseName);

    if (lastModelDisplayName != null && lastFirebaseModelName != null) {
      final localModelPath = await _getLocalModelPath();
      final localLabelsPath = await _getLocalLabelsPath();

      if (await File(localModelPath).exists() && await File(localLabelsPath).exists()) {
        // Find the full model data from _availableModels
        final modelData = _availableModels.firstWhere(
          (m) => m['name'] == lastModelDisplayName && m['firebaseModelName'] == lastFirebaseModelName,
          orElse: () => <String,String>{}, // Return empty map if not found
        );

        if (modelData.isNotEmpty) {
          debugPrint("Last used model '$lastModelDisplayName' found locally. Attempting auto-load.");
          // Set the selected name so dropdown reflects it
          setState(() {
            _selectedModelName = lastModelDisplayName;
          });
          await _prepareAndLoadModel(modelData, isInitialLoad: true, isAutoLoadingPrevious: true);
        } else {
           debugPrint("Could not find model data for $lastModelDisplayName in _availableModels.");
        }
      } else {
        debugPrint("Previously used model files not found locally. Clearing preferences.");
        await prefs.remove(_prefsKeyLastModelName);
        await prefs.remove(_prefsKeyLastModelFirebaseName);
      }
    }
  }

  Future<String> _getLocalModelPath() async {
    final directory = await getApplicationDocumentsDirectory();
    return p.join(directory.path, _activeModelFileName);
  }

  Future<String> _getLocalLabelsPath() async {
    final directory = await getApplicationDocumentsDirectory();
    return p.join(directory.path, _activeLabelsFileName);
  }

  Future<void> _deleteActiveModelFiles() async {
    try {
      final modelPath = await _getLocalModelPath();
      final labelsPath = await _getLocalLabelsPath();
      final modelFile = File(modelPath);
      final labelsFile = File(labelsPath);

      if (await modelFile.exists()) {
        await modelFile.delete();
        debugPrint("Deleted active model file: $modelPath");
      }
      if (await labelsFile.exists()) {
        await labelsFile.delete();
        debugPrint("Deleted active labels file: $labelsPath");
      }
    } catch (e) {
      debugPrint("Error deleting active model files: $e");
    }
  }

  void _clearScreen() {
    setState(() {
      _imageFile = null;
      _recognitions = [];
      _annotatedImageBytes = null;
    });
  }

  // Modified _prepareAndLoadModel
  Future<void> _prepareAndLoadModel(Map<String, String> modelData, {bool isInitialLoad = false, bool isAutoLoadingPrevious = false}) async {
    if (!isInitialLoad) {
      _clearScreen();
    }

    final String modelNameDisplay = modelData['name']!; // Assume 'name' is always present
    final String firebaseModelName = modelData['firebaseModelName']!; // Assume always present

    setState(() {
      _isLoading = true;
      _loadingMessage = "Preparing model: $modelNameDisplay";
    });

    try {
      final prefs = await SharedPreferences.getInstance();
      final String? lastStoredFirebaseModelName = prefs.getString(_prefsKeyLastModelFirebaseName);

      final String targetLocalModelPath = await _getLocalModelPath();
      final String targetLocalLabelsPath = await _getLocalLabelsPath();
      File modelFile = File(targetLocalModelPath);

      bool modelFileExists = await modelFile.exists();
      bool isSwitchingModels = (lastStoredFirebaseModelName != null && lastStoredFirebaseModelName != firebaseModelName);
      bool needsDownloadOrNewCopy = !modelFileExists || isSwitchingModels;

      // If we are auto-loading a previous model, and it exists, we assume labels also exist.
      // If it's a user selection (not auto-load) or model file doesn't exist, we might need to download/recopy.
      if (isAutoLoadingPrevious && modelFileExists) {
          debugPrint("Auto-loading existing model: $modelNameDisplay");
          // Assume labels are also fine if model is fine from previous session
      } else if (needsDownloadOrNewCopy) {
        // Check internet ONLY if we need to download a NEW model from Firebase
        final connectivityResult = await Connectivity().checkConnectivity();
        final bool isConnected = connectivityResult.contains(ConnectivityResult.mobile) ||
                                 connectivityResult.contains(ConnectivityResult.wifi) ||
                                 connectivityResult.contains(ConnectivityResult.ethernet);

        if (!isConnected) {
          throw Exception("Offline. Cannot download '$modelNameDisplay'.");
        }

        // Different model selected or current active model file is missing, and we are online.
        // So, delete whatever was previously active and download the new one.
        await _deleteActiveModelFiles();

        setState(() { _loadingMessage = "Downloading $modelNameDisplay..."; });
        final FirebaseModelDownloader modelDownloader = FirebaseModelDownloader.instance;
        final FirebaseCustomModel downloadedModel = await modelDownloader.getModel(
          firebaseModelName,
          FirebaseModelDownloadType.latestModel,
          FirebaseModelDownloadConditions(
            iosAllowsCellularAccess: true,
            androidWifiRequired: true, // Allow download on cellular
          ),
        );
        await downloadedModel.file.copy(targetLocalModelPath); // Copy to our fixed path
        debugPrint("Model '$firebaseModelName' downloaded to: $targetLocalModelPath");
      } else {
         debugPrint("Model '$modelNameDisplay' is current and exists locally at $targetLocalModelPath.");
      }

      // Always ensure labels are present for the *current* model, copy from assets
      final String? labelsAssetPath = modelData['labelsAssetPath'];
      if (labelsAssetPath != null) {
        setState(() { _loadingMessage = "Preparing labels for $modelNameDisplay..."; });
        final assetLabelsTempPath = await getAbsolutePath(labelsAssetPath);
        await File(assetLabelsTempPath).copy(targetLocalLabelsPath); // Copy to fixed labels path
        debugPrint("Labels copied from assets to $targetLocalLabelsPath");
      } else {
        throw Exception("labelsAssetPath missing for $modelNameDisplay");
      }
      
      if (!await File(targetLocalModelPath).exists() || !await File(targetLocalLabelsPath).exists()) {
        throw Exception("Critical files for '$modelNameDisplay' missing after preparation.");
      }

      setState(() { _loadingMessage = "Loading $modelNameDisplay into memory..."; });
      _yoloModel = YOLO(
        modelPath: targetLocalModelPath,
        task: YOLOTask.detect,
        // !! CRITICAL: How does your plugin use labels? Pass targetLocalLabelsPath if needed !!
        // e.g., labelsPath: targetLocalLabelsPath,
      );
      await _yoloModel?.loadModel();

      await prefs.setString(_prefsKeyLastModelName, modelNameDisplay);
      await prefs.setString(_prefsKeyLastModelFirebaseName, firebaseModelName);

      setState(() {
        _selectedModelName = modelNameDisplay;
        _currentModelPath = targetLocalModelPath; // Update active model path
      });

      if(mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("$modelNameDisplay ready.")),
        );
      }

    } catch (e) {
      debugPrint("Error in _prepareAndLoadModel: $e");
      if(mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Failed to load model: ${e.toString()}")),
        );
        setState(() {
          _yoloModel = null;
          // Don't clear _selectedModelName here, so dropdown shows what failed
          _currentModelPath = null;
        });
      }
    } finally {
      if(mounted){
        setState(() {
          _isLoading = false;
          _loadingMessage = null;
        });
      }
    }
  }

   // In _pickImage(), ensure _currentModelPath is checked or _yoloModel is loaded
   Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      final File imageFile = File(image.path);
      final imageBytesForDimensions = await imageFile.readAsBytes();
      final decodedImage = await decodeImageFromList(imageBytesForDimensions);
      
      setState(() {
        _imageFile = imageFile;
        _originalImageWidth = decodedImage.width.toDouble();
        _originalImageHeight = decodedImage.height.toDouble();
        _recognitions = [];
        _annotatedImageBytes = null;
      });

      // Check if a model is loaded and its file exists
      if (_yoloModel != null && _currentModelPath != null && await File(_currentModelPath!).exists()) {
        _runDetection();
      } else {
         if(mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Model not loaded or file missing. Please select and load a model.")),
            );
         }
      }
    }
  }

  Future<void> _runDetection() async {
    if (_imageFile == null || _yoloModel == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No image selected or model not loaded.")),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    DateTime startTime = DateTime.now(); // Record the start time AFTER _isLoading is true

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final detections = await _yoloModel!.predict(imageBytes);

      setState(() {
        _recognitions = detections['boxes'] ?? [];
        _annotatedImageBytes = detections['annotatedImage'] as Uint8List?;
      });

      if (_recognitions.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("No objects detected.")),
        );
      }
    } 
    catch (e) {
      debugPrint("Error running detection: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error during detection: $e")),
      );
    } 
    finally {
      DateTime endTime = DateTime.now();
      Duration elapsedTime = endTime.difference(startTime);
      // Define your minimum loading display time
      const Duration minLoadingTime = Duration(milliseconds: 700); // Example: 700 milliseconds

      if (elapsedTime < minLoadingTime) {
        // If detection was faster than min display time, wait for the remainder
        await Future.delayed(minLoadingTime - elapsedTime);
      }

      // Always check if the widget is still mounted before calling setState
      // in an async method's finally block, especially after an await (Future.delayed).
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Widget _buildModelSelector() {
    return DropdownButton<String>(
      hint: const Text("Select a Model to Load"),
      value: _selectedModelName,
      isExpanded: true,
      items: _availableModels.map((model) {
        return DropdownMenuItem<String>(
          value: model['name'],
          child: Text(model['name']!),
        );
      }).toList(),
      onChanged: (String? newValue) {
        if (newValue != null) {
          final selectedModelData = _availableModels.firstWhere((m) => m['name'] == newValue);
          _prepareAndLoadModel(selectedModelData);
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Ultralytics YOLO Detection"),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _buildModelSelector(),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.image),
              label: const Text("Pick Image from Gallery"),
              onPressed: _pickImage,
            ),
            const SizedBox(height: 20),
            if (_isLoading)
               Center(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Image.asset(
                        'assets/images/loading.gif', // Your GIF
                        width: 100,
                        height: 100,
                      ),
                      if (_loadingMessage != null && _loadingMessage!.isNotEmpty) ...[
                        const SizedBox(height: 15),
                        Text(_loadingMessage!, textAlign: TextAlign.center),
                      ]
                    ],
                  ),
                ),
              )
            else if (_imageFile != null) ...[
              Text("Original Image", style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 10),
              Image.file(_imageFile!),
              
              // Only show detection results if objects were found
              if (_recognitions.isNotEmpty) ...[
                const SizedBox(height: 20),
                Text("Detection Results", style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                _buildImageWithDetections(_imageFile!, _recognitions),
                const SizedBox(height: 20),
                if (_annotatedImageBytes != null) ...[
                  Text("Annotated Image", style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 10),
                  Image.memory(_annotatedImageBytes!),
                  const SizedBox(height: 20),
                ],
                Text("Detected Objects: ${_recognitions.length}", 
                    style: Theme.of(context).textTheme.titleMedium),
                _buildDetectionList(),
              ] else if (_recognitions.isEmpty && !_isLoading) ...[
                const SizedBox(height: 20),
                const Center(child: Text("No objects detected in the image")),
              ]
            ] else
              const Center(child: Text("Select an image and load a model to begin.")),
          ],
        ),
      ),
    );
  }

  Widget _buildImageWithDetections(File imageFile, List<Map<String, dynamic>> recognitions) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: [
            Image.file(imageFile),
            ...recognitions.asMap().entries.map((entry) {
              final int index = entry.key;
              final Map<String, dynamic> det = entry.value;
              final Color boxColor = _boxColors[index % _boxColors.length];

              final double x1 = (det['x1'] as num?)?.toDouble() ?? 0.0;
              final double y1 = (det['y1'] as num?)?.toDouble() ?? 0.0;
              final double x2 = (det['x2'] as num?)?.toDouble() ?? 0.0;
              final double y2 = (det['y2'] as num?)?.toDouble() ?? 0.0;
              final String className = det['class']?.toString() ?? 'Unknown';
              final double confidence = (det['confidence'] as num?)?.toDouble() ?? 0.0;

              final double boxWidth = x2 - x1;
              final double boxHeight = y2 - y1;

              if (boxWidth <= 0 || boxHeight <= 0) {
                return const SizedBox.shrink();
              }

              return Positioned(
                left: x1,
                top: y1,
                width: boxWidth,
                height: boxHeight,
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: boxColor, width: 2.5),
                  ),
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: Container(
                      color: boxColor.withOpacity(0.6),
                      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                      child: Text(
                        "$className (${(confidence * 100).toStringAsFixed(1)}%)",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ],
        );
      },
    );
  }

  Widget _buildDetectionList() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _recognitions.length,
      itemBuilder: (context, index) {
        final detection = _recognitions[index];
        
        final className = detection['class']?.toString() ?? 'Unknown';
        final confidence = (detection['confidence'] as num?)?.toDouble() ?? 0.0;
        final x1 = (detection['x1'] as num?)?.toDouble() ?? 0.0;
        final y1 = (detection['y1'] as num?)?.toDouble() ?? 0.0;
        final x2 = (detection['x2'] as num?)?.toDouble() ?? 0.0;
        final y2 = (detection['y2'] as num?)?.toDouble() ?? 0.0;

        final boxWidth = x2 - x1;
        final boxHeight = y2 - y1;

        return Card(
          child: ListTile(
            title: Text(className),
            subtitle: Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
            trailing: Text('Box: (${x1.toStringAsFixed(0)},${y1.toStringAsFixed(0)}) W:${boxWidth.toStringAsFixed(0)},H:${boxHeight.toStringAsFixed(0)}'),
          ),
        );
      },
    );
  }
}
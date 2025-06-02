import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

// Add this import statement:
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_ml_model_downloader/firebase_ml_model_downloader.dart';
import 'firebase_options.dart';

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
  String? _selectedModelName;
  String? _currentModelPath;
  Uint8List? _annotatedImageBytes;

  // Add this line to declare the _loadingMessage variable:
  String? _loadingMessage;

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

  void _clearScreen() {
    setState(() {
      _imageFile = null;
      _recognitions = [];
      _annotatedImageBytes = null;
    });
  }

  Future<void> _prepareAndLoadModel(Map<String, String> modelData) async {
    _clearScreen(); // Clear screen before loading new model

    final String modelNameDisplay = modelData['name'] as String;
    final String firebaseModelName = modelData['firebaseModelName'] as String;

    setState(() {
      _isLoading = true;
      _loadingMessage = "Preparing model: $modelNameDisplay";
    });
    
    // if (modelData['modelAssetPath'] == null || modelData['labelsAssetPath'] == null) {
    //   ScaffoldMessenger.of(context).showSnackBar(
    //     const SnackBar(content: Text("Model or labels asset path is missing.")),
    //   );
    //   return;
    // }

        try {
      // 1. Download/Get the TFLite model from Firebase ML
      setState(() { _loadingMessage = "Checking for $modelNameDisplay model updates..."; });

      final FirebaseModelDownloader modelDownloader = FirebaseModelDownloader.instance;
      final FirebaseCustomModel localModel = await modelDownloader.getModel(
        firebaseModelName, // Name given in Firebase ML Console
        FirebaseModelDownloadType.localModelUpdateInBackground, // Or .latestModel, .localModel
        FirebaseModelDownloadConditions(
          iosAllowsCellularAccess: true,
          androidWifiRequired: true,
        ),
      );

      final String localModelPath = localModel.file.path;
      debugPrint("Firebase ML Model '$firebaseModelName' available at: $localModelPath");

      // 2. Prepare the labels file path
      String localLabelsPath;
      final String? labelsAssetPath = modelData['labelsAssetPath'] as String?;
      // final String? labelsStoragePath = modelData['labelsStoragePath'] as String?; // For Option B
      // final String? labelsLocalName = modelData['labelsLocalName'] as String?; // For Option B

      if (labelsAssetPath != null) { // Option A: Labels from app assets
        setState(() { _loadingMessage = "Preparing labels..."; });
        localLabelsPath = await getAbsolutePath(labelsAssetPath); // Your existing helper
      }
      // else if (labelsStoragePath != null && labelsLocalName != null) { // Option B: Download labels
      //   setState(() { _loadingMessage = "Checking labels..."; });
      //   localLabelsPath = await _fileDownloadService.getLocalFilePath(labelsLocalName);
      //   if (!await File(localLabelsPath).exists()) {
      //     bool isConnected = await _connectivityService.isConnected();
      //     if (!isConnected) throw Exception("No internet to download labels.");
      //     setState(() { _loadingMessage = "Downloading labels..."; });
      //     File? downloadedLabelFile = await _fileDownloadService.downloadFile(labelsStoragePath, labelsLocalName);
      //     if (downloadedLabelFile == null) throw Exception("Failed to download labels file.");
      //   }
      // }
      else {
        // This case means labels are neither from assets nor explicitly from remote storage.
        // The YOLO plugin or model must handle labels internally (e.g. embedded metadata).
        // Or, you might need to enforce one of the options.
        // For now, we'll assume if no path, plugin handles it or it's an error later.
        // Consider throwing an exception if labels are strictly required and no path is found.
        debugPrint("Warning: Labels path not specified. Assuming model handles labels or plugin finds them by convention.");
        // If your YOLO plugin REQUIRES a labels path and it's not part of modelData, this will be an issue.
        // For this example, let's assume your plugin might not need it explicitly passed IF it's a common name like labels.txt
        // in the same dir or if your model has metadata. This is highly plugin-dependent.
        // You MIGHT need to ensure labels.txt is copied to the same directory as localModelPath if your plugin expects that.
        // As a fallback if labelsAssetPath wasn't provided for some reason, let's try the default.
        // THIS IS A GUESS - VERIFY YOUR YOLO PLUGIN'S LABEL HANDLING
        localLabelsPath = await getAbsolutePath('assets/models/labels.txt'); // Fallback for example
      }
      
      if (!await File(localLabelsPath).exists()) {
          throw Exception("Labels file could not be found or prepared from: $localLabelsPath");
      }
      debugPrint("Labels file ready at: $localLabelsPath");


      // 3. Load the model using your YOLO plugin
      setState(() { _loadingMessage = "Loading $modelNameDisplay into memory..."; });

      _yoloModel = YOLO(
        modelPath: localModelPath, // Path from FirebaseModelDownloader
        task: YOLOTask.detect,
        // IMPORTANT: How does your specific YOLO plugin handle labels?
        // Does it need `labelsPath: localLabelsPath`?
        // Does it assume `labels.txt` is in the same directory as the model?
        // Does it read labels from model metadata?
        // You MUST verify this for your specific `ultralytics_yolo` plugin.
      );
      await _yoloModel?.loadModel();

      setState(() {
        _selectedModelName = modelNameDisplay;
        // _currentModelPath = localModelPath; // If you need to keep track
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("$modelNameDisplay loaded successfully and ready.")),
      );

    } catch (e) {
      debugPrint("Error in _prepareAndLoadModel: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to load model: ${e.toString()}")),
      );
      setState(() {
        _yoloModel = null;
        _selectedModelName = null;
      });
    } finally {
      setState(() {
        _isLoading = false;
        _loadingMessage = null;
      });
    }

  }

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _recognitions = [];
        _annotatedImageBytes = null;
      });

      if (_yoloModel != null) {
        _runDetection();
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Please select and load a model first.")),
        );
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
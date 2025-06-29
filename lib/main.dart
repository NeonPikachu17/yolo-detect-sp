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

  Future<Map<String, dynamic>?> _getLocallyActiveModelData(
      SharedPreferences prefs) async {
    final lastModelDisplayName = prefs.getString(_prefsKeyLastModelName);
    final lastFirebaseModelName = prefs.getString(_prefsKeyLastModelFirebaseName);

    if (lastModelDisplayName != null && lastFirebaseModelName != null) {
      final localModelPath = await _getLocalModelPath();
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
    final lastFirebaseModelName = prefs.getString(_prefsKeyLastModelFirebaseName);

    if (lastModelDisplayName != null &&
        lastFirebaseModelName != null &&
        _availableModels.isNotEmpty) {
      final localModelPath = await _getLocalModelPath();
      final localLabelsPath = await _getLocalLabelsPath();

      if (await File(localModelPath).exists() &&
          await File(localLabelsPath).exists()) {
        final modelData = _availableModels.firstWhere(
            (m) =>
                m['name'] == lastModelDisplayName &&
                m['firebaseModelName'] == lastFirebaseModelName,
            orElse: () => <String, dynamic>{});
        if (modelData.isNotEmpty) {
          debugPrint("Last used model '$lastModelDisplayName' found locally. Attempting auto-load.");
          if (mounted) {
            setState(() {
              _selectedModelName = lastModelDisplayName;
            });
          }
          await _prepareAndLoadModel(modelData, isInitialLoad: true, isAutoLoadingPrevious: true);
        }
      }
    } else {
      if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
    }
  }

  Future<String> _getLocalModelPath() async => p.join((await getApplicationDocumentsDirectory()).path, _activeModelFileName);
  Future<String> _getLocalLabelsPath() async => p.join((await getApplicationDocumentsDirectory()).path, _activeLabelsFileName);

  Future<void> _downloadModel(String modelName, String firebaseModelName, String labelsAssetPath) async {
    final connectivityResult = await Connectivity().checkConnectivity();
    final bool isConnected = connectivityResult.any((r) => r == ConnectivityResult.wifi || r == ConnectivityResult.mobile || r == ConnectivityResult.ethernet);
    if (!isConnected) {
      throw Exception("Offline. Cannot download or switch to '$modelName'.");
    }

    await _deleteActiveModelFiles();

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

    final targetLocalModelPath = await _getLocalModelPath();
    await downloadedModel.file.copy(targetLocalModelPath);
    debugPrint("Model '$firebaseModelName' downloaded to: $targetLocalModelPath");
    
    final targetLocalLabelsPath = await _getLocalLabelsPath();
    final assetLabelsTempPath = await getAbsolutePath(labelsAssetPath);
    await File(assetLabelsTempPath).copy(targetLocalLabelsPath);
    debugPrint("Labels copied from assets to $targetLocalLabelsPath");
  }

  Future<void> _deleteActiveModelFiles() async {
    try {
      final modelFile = File(await _getLocalModelPath());
      final labelsFile = File(await _getLocalLabelsPath());
      if (await modelFile.exists()) await modelFile.delete();
      if (await labelsFile.exists()) await labelsFile.delete();
    } catch (e) {
      debugPrint("Error deleting active model files: $e");
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

  Future<void> _prepareAndLoadModel(Map<String, dynamic> modelData, {bool isInitialLoad = false, bool isAutoLoadingPrevious = false}) async {
    if (!isInitialLoad) {
      _clearScreen();
    }

    final String modelNameDisplay = modelData['name'] as String;
    final String firebaseModelName = modelData['firebaseModelName'] as String;

    if (!mounted) return;
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

      if (isAutoLoadingPrevious && modelFileExists && !isSwitchingModels) {
        debugPrint("Auto-loading existing model: $modelNameDisplay from $targetLocalModelPath");
      } else if (needsDownloadOrNewCopy) {
        await _downloadModel(modelNameDisplay, firebaseModelName, modelData['labelsAssetPath'] as String);
      } else {
        debugPrint("Model '$modelNameDisplay' is current and already exists locally.");
      }
      
      if (!await File(targetLocalModelPath).exists() || !await File(targetLocalLabelsPath).exists()) {
        throw Exception("Critical files for '$modelNameDisplay' missing after all preparations.");
      }

      setState(() { _loadingMessage = "Loading $modelNameDisplay into memory..."; });
      
      _yoloModel = YOLO(
        modelPath: targetLocalModelPath, 
        task: YOLOTask.segment,
      );
      await _yoloModel?.loadModel();

      await prefs.setString(_prefsKeyLastModelName, modelNameDisplay);
      await prefs.setString(_prefsKeyLastModelFirebaseName, firebaseModelName);

      if (!mounted) return;
      setState(() {
        _selectedModelName = modelNameDisplay;
        _currentModelPath = targetLocalModelPath;
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

  Future<void> _runSegmentation() async {
    if (_imageFile == null || _yoloModel == null) return;

    setState(() {
      _isLoading = true;
      _loadingMessage = "Analyzing image...";
    });

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final double confThreshold = 0.6;
      final double nmsThreshold = 0.5;

      debugPrint("Running segmentation with confidence: $confThreshold, IoU: $nmsThreshold");

      final detections = await _yoloModel!.predict(
        imageBytes,
        confidenceThreshold: confThreshold,
        iouThreshold: nmsThreshold,
      );

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
      ),
      body: RefreshIndicator(
        onRefresh: _handleRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              _buildModelSelector(),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                icon: const Icon(Icons.image_search),
                label: const Text("Pick Image from Gallery"),
                onPressed: _isLoading ? null : _pickImage,
                style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
              ),
              const SizedBox(height: 20),
              
              _buildContentArea(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildContentArea() {
    if (_isLoading) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Image.asset(
                'assets/images/loading.gif',
                width: 100,
                height: 100,
              ),
              if (_loadingMessage != null && _loadingMessage!.isNotEmpty) ...[
                const SizedBox(height: 15),
                Text(_loadingMessage!, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyLarge),
              ]
            ],
          ),
        ),
      );
    }

    if (_imageFile != null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Analysis Result", style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 12),

          if (_recognitions.isNotEmpty)
            _buildImageWithDetections()
          else
            Image.file(_imageFile!),
          
          const SizedBox(height: 20),

          if (_recognitions.isEmpty && !_isLoading)
            const Center(child: Text("No objects detected with high confidence.", style: TextStyle(fontSize: 16))),

          if (_recognitions.isNotEmpty) ...[
            Text("Detected Objects: ${_recognitions.length}", style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 10),
            _buildDetectionList(),
            
            const SizedBox(height: 20),
            if (_annotatedImageBytes != null) ...[
              Text("Library's Annotated Image (for comparison)", style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 10),
              Image.memory(_annotatedImageBytes!),
            ]
          ]
        ],
      );
    }
    
    return const Center(
      child: Text(
        "Select a model and pick an image to begin.",
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 18, color: Colors.grey),
      ),
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
            ),
          );
        },
      );
    });
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
        final x1 = (detection['x1'] as num).toDouble();
        final y1 = (detection['y1'] as num).toDouble();
        final x2 = (detection['x2'] as num).toDouble();
        final y2 = (detection['y2'] as num).toDouble();
        final boxWidth = x2 - x1;
        final boxHeight = y2 - y1;

        return Card(
          child: ListTile(
            leading: CircleAvatar(child: Text('${index + 1}')),
            title: Text(className),
            subtitle: Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
            trailing: Text('W:${boxWidth.toStringAsFixed(0)}, H:${boxHeight.toStringAsFixed(0)}'),
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

  _DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.boxColors,
    required this.scaleRatio,
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
    if (maskImage != null && recognitions.isNotEmpty) {
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
      paintImage(
        canvas: canvas,
        rect: Rect.fromLTWH(0, 0, size.width, size.height),
        image: maskImage!,
        fit: BoxFit.fill,
      );
      canvas.restore();
    }

    // 3. Draw the bounding box outlines and labels on top
    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final color = boxColors[i % boxColors.length];

      final double x1 = (detection['x1'] as num).toDouble() * scaleRatio;
      final double y1 = (detection['y1'] as num).toDouble() * scaleRatio;
      final double x2 = (detection['x2'] as num).toDouble() * scaleRatio;
      final double y2 = (detection['y2'] as num).toDouble() * scaleRatio;
      final String className = detection['className'] ?? 'Unknown';
      final double confidence = (detection['confidence'] as num).toDouble();

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;
      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final textSpan = TextSpan(
        text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: color,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);
      textPainter.paint(canvas, Offset(x1, y1));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return originalImage != oldDelegate.originalImage ||
           maskImage != oldDelegate.maskImage ||
           recognitions != oldDelegate.recognitions;
  }
}
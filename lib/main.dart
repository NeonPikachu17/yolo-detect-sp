import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Enum to manage the selected YOLO task in the app's state.
enum AppYoloTask { segment, detect, classify }

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return MaterialApp(
      title: 'Vision AI (Static)',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: const Color(0xFF455A64),
        scaffoldBackgroundColor: const Color(0xFFECEFF1),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF455A64),
          brightness: Brightness.light,
          primary: const Color(0xFF455A64),
          secondary: const Color(0xFF78909C),
          background: const Color(0xFFECEFF1),
          error: const Color(0xFFD32F2F),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(textTheme).apply(
          bodyColor: const Color(0xFF37474F),
          displayColor: const Color(0xFF263238),
        ),
        cardTheme: const CardThemeData(
          elevation: 1,
          surfaceTintColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
          clipBehavior: Clip.antiAlias,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: const Color(0xFFECEFF1),
          foregroundColor: const Color(0xFF263238),
          elevation: 0,
          centerTitle: true,
          titleTextStyle: GoogleFonts.poppins(
            fontWeight: FontWeight.w600,
            fontSize: 20,
            color: const Color(0xFF263238),
          ),
        ),
      ),
      home: const VisionScreen(),
    );
  }
}

class VisionScreen extends StatefulWidget {
  const VisionScreen({super.key});

  @override
  State<VisionScreen> createState() => _VisionScreenState();
}

class _VisionScreenState extends State<VisionScreen> with SingleTickerProviderStateMixin {

  // --- CONFIGURATION: Map each task to a specific file name ---
  // Ensure these files exist in your assets/ folder!
  final Map<AppYoloTask, String> _modelFiles = {
    AppYoloTask.segment:  "yolo_segment",  // Loads assets/yolo_segment.tflite
    AppYoloTask.detect:   "yolo_detect",   // Loads assets/yolo_detect.tflite
    AppYoloTask.classify: "yolo_classify", // Loads assets/yolo_classify.tflite
  };

  YOLO? _yoloModel;
  File? _imageFile;
  List<Map<String, dynamic>> _recognitions = [];
  bool _isLoading = false;
  String? _selectedModelName;
  String? _loadingMessage;
  
  // Image Dimensions
  double _originalImageHeight = 0;
  double _originalImageWidth = 0;
  double _modelImageHeight = 0;
  double _modelImageWidth = 0;

  int? _selectedDetectionIndex;
  bool _showMasks = true;
  double _maskOpacity = 0.5;
  Map<String, Color> _classColorMap = {};
  Uint8List? _maskPngBytes;

  // Default task
  AppYoloTask _selectedTask = AppYoloTask.segment;

  // Animation variables
  late AnimationController _idleAnimationController;
  late Animation<double> _idleAnimation;

  final List<Color> _boxColors = [
    Colors.deepOrange, Colors.lightBlue, Colors.amber.shade600, Colors.pink,
    Colors.green, Colors.purple, Colors.red, Colors.teal,
    Colors.indigo, Colors.cyan, Colors.brown, Colors.lime.shade800,
  ];

  @override
  void initState() {
    // Initialize the "breathing" idle animation
    _idleAnimationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    _idleAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _idleAnimationController, curve: Curves.easeInOut),
    );

    super.initState();
    _initializeScreenData();
  }
  
  Future<void> _initializeScreenData() async {
    // Load the default task's model on startup
    await _loadStaticModelForTask(_selectedTask);
  }

  /// Switches the task and loads the corresponding model automatically
  Future<void> _switchTask(AppYoloTask newTask) async {
    if (_isLoading || newTask == _selectedTask) return;
    setState(() => _selectedTask = newTask);
    await _loadStaticModelForTask(newTask);
    
    // Re-run inference if an image is already selected
    if (_imageFile != null && _yoloModel != null) {
      _runInference();
    }
  }

  /// Loads the model associated with the specific task from assets
  Future<void> _loadStaticModelForTask(AppYoloTask task) async {
    final modelName = _modelFiles[task];
    if (modelName == null) {
      _showSnackBar("No model defined for ${task.name}", isError: true);
      return;
    }

    _startLoading("Loading ${task.name} model...");

    try {
      // 1. Get the app's private documents directory
      final docDir = await getApplicationDocumentsDirectory();
      final modelPath = p.join(docDir.path, '$modelName.tflite');
      final labelsPath = p.join(docDir.path, '$modelName.txt');
      
      // 2. Check if the model is already copied to the documents directory
      if (!await File(modelPath).exists()) {
        try {
          // Copy model from assets
          final modelData = await rootBundle.load('assets/models/$modelName.tflite');
          await File(modelPath).writeAsBytes(modelData.buffer.asUint8List(
            modelData.offsetInBytes, modelData.lengthInBytes
          ));
          
          // Copy labels file
          try {
            final labelsData = await rootBundle.load('assets/models/$modelName.txt');
             await File(labelsPath).writeAsBytes(labelsData.buffer.asUint8List(
              labelsData.offsetInBytes, labelsData.lengthInBytes
            ));
          } catch (e) {
            await File(labelsPath).create(); // Create empty if missing
          }
        } catch (e) {
          throw Exception("Asset not found: assets/models/$modelName.tflite");
        }
      }

      // 3. Dispose previous model
      if (_yoloModel != null) await _yoloModel!.dispose();
      
      // 4. Map AppYoloTask to the plugin's YOLOTask
      late YOLOTask pluginTask;
      switch (task) {
        case AppYoloTask.segment:
          pluginTask = YOLOTask.segment;
          break;
        case AppYoloTask.detect:
          pluginTask = YOLOTask.detect;
          break;
        case AppYoloTask.classify:
          pluginTask = YOLOTask.classify;
          break;
      }

      // 5. Initialize YOLO
      _yoloModel = YOLO(modelPath: modelPath, task: pluginTask);
      await _yoloModel?.loadModel();

      if (mounted) {
        setState(() { _selectedModelName = modelName; });
        _showSnackBar("'${task.name.toUpperCase()}' model loaded.", isError: false);
      }
    } catch (e) {
      _showSnackBar("Error loading model: ${e.toString()}", isError: true);
      if (mounted) setState(() { _yoloModel = null; _selectedModelName = null; });
    } finally {
      _stopLoading();
    }
  }

  // --- Image Processing Logic ---

  Future<void> _processImage(XFile image) async {
    final imageBytes = await image.readAsBytes();
    final decodedImage = await decodeImageFromList(imageBytes);
    setState(() {
      _imageFile = File(image.path);
      _recognitions = [];
      _maskPngBytes = null;
      _originalImageWidth = decodedImage.width.toDouble();
      _originalImageHeight = decodedImage.height.toDouble();
      _modelImageWidth = 0;
      _modelImageHeight = 0;
      _selectedDetectionIndex = null;
    });
    if (_yoloModel != null) {
      _runInference();
    } else {
      _showSnackBar("Please wait for the model to load.", isError: true);
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) await _processImage(image);
  }

  Future<void> _takePicture() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);
    if (image != null) await _processImage(image);
  }

  Future<void> _runInference() async {
    if (_imageFile == null || _yoloModel == null) return;
    _startLoading("Analyzing image...");

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final detections = await _yoloModel!.predict(imageBytes);
      if (!mounted) return;

      final double modelWidth = (detections['image_width'] as num?)?.toDouble() ?? _originalImageWidth;
      final double modelHeight = (detections['image_height'] as num?)?.toDouble() ?? _originalImageHeight;

      final formattedRecognitions = <Map<String, dynamic>>[];
      final tempColorMap = <String, Color>{};
      Uint8List? newMaskPngBytes;
      int colorIndex = 0;

      switch (_selectedTask) {
        case AppYoloTask.classify:
          final Map<String, dynamic> classificationResult = Map.from((detections as Map?) ?? {});
          final Map? nestedClassificationMap = classificationResult['classification'] as Map?;

          if (nestedClassificationMap != null) {
            final List<dynamic>? top5Classes = nestedClassificationMap['top5Classes'];
            final List<dynamic>? top5Confidences = nestedClassificationMap['top5Confidences'];

            if (top5Classes != null && top5Confidences != null && top5Classes.length == top5Confidences.length) {
              for (int i = 0; i < top5Classes.length; i++) {
                formattedRecognitions.add({
                  'className': top5Classes[i],
                  'confidence': top5Confidences[i],
                });
              }
            }
          }
          break;
        case AppYoloTask.segment:
        case AppYoloTask.detect:
          final Map<String, dynamic> detectionMap = Map.from((detections as Map?) ?? {});
          final List<dynamic> boxes = (detectionMap['boxes'] as List<dynamic>?) ?? [];
          
          for (var box in boxes) {
            double x1 = (box['x1'] as num).toDouble();
            double y1 = (box['y1'] as num).toDouble();
            double x2 = (box['x2'] as num).toDouble();
            double y2 = (box['y2'] as num).toDouble();
            
            // Normalize if needed
            if (x1 <= 1.0 && y1 <= 1.0 && x2 <= 1.0 && y2 <= 1.0) {
              x1 *= modelWidth;
              y1 *= modelHeight;
              x2 *= modelWidth;
              y2 *= modelHeight;
            }
            
            final className = box['className'];
            formattedRecognitions.add({
              'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
              'className': className, 'confidence': box['confidence'],
            });
            if (!tempColorMap.containsKey(className)) {
              tempColorMap[className] = _boxColors[colorIndex % _boxColors.length];
              colorIndex++;
            }
          }
          
          newMaskPngBytes = (_selectedTask == AppYoloTask.segment) ? detectionMap['maskPng'] : null;
          break;
      }

      setState(() {
        _recognitions = formattedRecognitions;
        _classColorMap = tempColorMap;
        _maskPngBytes = newMaskPngBytes;
        _modelImageWidth = modelWidth;
        _modelImageHeight = modelHeight;
      });
    } catch (e) {
      _showSnackBar("Error during analysis: $e", isError: true);
    } finally {
      _stopLoading();
    }
  }
  
  // --- Helpers ---

  void _startLoading(String message) => setState(() { _isLoading = true; _loadingMessage = message; });
  void _stopLoading() => setState(() { _isLoading = false; _loadingMessage = null; });
  void _clearScreen() => setState(() { _imageFile = null; _recognitions = []; _maskPngBytes = null; _selectedDetectionIndex = null; });

  void _showSnackBar(String message, {required bool isError}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)..hideCurrentSnackBar()..showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Theme.of(context).colorScheme.error : Colors.green.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  void dispose() {
    _yoloModel?.dispose();
    _idleAnimationController.dispose();
    super.dispose();
  }

  // --- Build Method ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true, 
      appBar: AppBar(title: const Text("Vision AI")),
      bottomNavigationBar: _buildBottomActionBar(),
      body: Stack(
        children: [
          Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 1200),
              child: ListView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 120), 
                children: <Widget>[
                  _buildModelManagementCard(),
                  const SizedBox(height: 20),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 500),
                    transitionBuilder: (Widget child, Animation<double> animation) {
                      return FadeTransition(
                        opacity: animation,
                        child: ScaleTransition(
                          scale: Tween<double>(begin: 0.95, end: 1.0).animate(animation),
                          child: child,
                        ),
                      );
                    },
                    child: _buildContentArea(),
                  ),
                ],
              ),
            ),
          ),
          if (_isLoading) _buildLoadingModalOverlay(),
        ],
      ),
    );
  }

  // --- Widgets ---

  Widget _buildModelManagementCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("ACTIVE MODEL", style: Theme.of(context).textTheme.labelLarge?.copyWith(color: Theme.of(context).colorScheme.secondary)),
                      Text(
                        _selectedModelName ?? "Loading...",
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: _yoloModel != null ? Theme.of(context).primaryColor : null,
                            ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text("ANALYSIS TASK", style: Theme.of(context).textTheme.labelLarge?.copyWith(color: Theme.of(context).colorScheme.secondary)),
            const SizedBox(height: 8),
            SegmentedButton<AppYoloTask>(
              segments: const [
                ButtonSegment<AppYoloTask>(value: AppYoloTask.segment, label: Text('Segment'), icon: Icon(Icons.grain_rounded)),
                ButtonSegment<AppYoloTask>(value: AppYoloTask.detect, label: Text('Detect'), icon: Icon(Icons.select_all_rounded)),
                ButtonSegment<AppYoloTask>(value: AppYoloTask.classify, label: Text('Classify'), icon: Icon(Icons.label_important_outline)),
              ],
              selected: {_selectedTask},
              onSelectionChanged: (Set<AppYoloTask> newSelection) {
                _switchTask(newSelection.first);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomActionBar() {
    final bool isReadyForAnalysis = _yoloModel != null && !_isLoading;
    
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
        child: Container(
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor.withOpacity(0.85),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text("Gallery"),
                      onPressed: isReadyForAnalysis ? _pickImage : null,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text("Camera"),
                      onPressed: isReadyForAnalysis ? _takePicture : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Theme.of(context).primaryColor,
                        foregroundColor: Theme.of(context).colorScheme.onPrimary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContentArea() {
    if (_imageFile != null) {
      if (_selectedTask == AppYoloTask.classify) {
        return _buildClassificationView();
      } else {
        return _buildDetectionView();
      }
    }
    
    return Container(
      key: const ValueKey('initial'),
      padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 20),
      child: Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          ScaleTransition(
            scale: _idleAnimation,
            child: Icon(Icons.image_search_outlined, size: 100, color: Colors.grey.shade400),
          ),
          const SizedBox(height: 20),
          Text(_yoloModel == null ? "Select a Model" : "Ready to Analyze", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          Text(
            _yoloModel == null 
              ? "Models are loading..."
              : "Use the action bar below to select an image.",
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade600),
          ),
        ]),
      ),
    );
  }

  Widget _buildLoadingModalOverlay() {
    return Stack(
      children: [
        const Opacity(
          opacity: 0.4,
          child: ModalBarrier(dismissible: false, color: Colors.black),
        ),
        Center(
          child: Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 24),
                  if (_loadingMessage != null)
                    Text(
                      _loadingMessage!,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDetectionView() {
    return LayoutBuilder(builder: (context, constraints) {
      if (constraints.maxWidth > 700) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 6, child: _buildResultsImage()),
          const SizedBox(width: 20),
          Expanded(
            flex: 4, 
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 450),
              child: _buildResultsList(),
            )
          ),
        ]);
      } else {
        return Column(children: [
          _buildResultsImage(), const SizedBox(height: 20), _buildResultsList(),
        ]);
      }
    });
  }

  Widget _buildClassificationView() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isWideScreen = constraints.maxWidth > 700;
        if (isWideScreen) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(flex: 5, child: _buildClassificationImageContainer()),
              const SizedBox(width: 20),
              Expanded(flex: 4, child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 450), child: _buildClassificationListContainer())),
            ],
          );
        } else {
          return Column(children: [
            _buildClassificationImageContainer(),
            const SizedBox(height: 20),
            _buildClassificationListContainer(),
          ]);
        }
      },
    );
  }
  
  Widget _buildClassificationImageContainer() {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text("Image", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
            IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
          ],
        ),
        const SizedBox(height: 12),
        _buildClassificationImage(),
      ],
    );
  }

  Widget _buildClassificationListContainer() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.start,
      children: [
        Text("Top 5 Results", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        if (_recognitions.isEmpty)
          const Card(child: Padding(padding: EdgeInsets.all(24.0), child: Text("No results to display.")))
        else
          _buildClassificationList(),
      ],
    );
  }
  
  Widget _buildResultsImage() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text("Analysis Result", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
        IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
      ],
    ),
    const SizedBox(height: 12),
    Card(elevation: 4, shadowColor: Colors.black.withOpacity(0.2), clipBehavior: Clip.antiAlias, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), child: _recognitions.isEmpty && !_isLoading ? _buildNoDetectionsFound() : _buildImageWithDetections())
  ]);

  Widget _buildResultsList() => Column(children: [
    if (_recognitions.isNotEmpty && _selectedTask == AppYoloTask.segment) ...[
      _buildInteractiveControls(), 
      const SizedBox(height: 20)
    ],
    Text("Detected Objects: ${_recognitions.length}", style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
    const SizedBox(height: 10),
    _buildDetectionList(),
  ]);
  
  Widget _buildClassificationImage() {
    if (_imageFile == null) return const SizedBox.shrink();
    return Card(
      elevation: 4,
      shadowColor: Colors.black.withOpacity(0.2),
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Image.file(_imageFile!),
    );
  }
  
  Widget _buildClassificationList() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _recognitions.length,
      itemBuilder: (context, index) {
        final result = _recognitions[index];
        final className = result['className'] ?? 'Unknown';
        final confidence = (result['confidence'] ?? 0.0) as num;

        return Card(
          margin: const EdgeInsets.symmetric(vertical: 5),
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Row(
              children: [
                Container(
                  width: 40,
                  alignment: Alignment.center,
                  child: Text(
                    '#${index + 1}',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(className, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600)),
                      const SizedBox(height: 5),
                      LinearProgressIndicator(
                        value: confidence.toDouble(),
                        backgroundColor: Colors.grey.shade300,
                        valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                      ),
                      const SizedBox(height: 5),
                      Text('${(confidence * 100).toStringAsFixed(1)}% Confidence', style: TextStyle(fontSize: 13, color: Colors.grey.shade700)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildInteractiveControls() => Card(elevation: 2, shadowColor: Colors.black.withOpacity(0.1), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), child: Padding(padding: const EdgeInsets.all(12.0), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [const Text("Show Masks", style: TextStyle(fontWeight: FontWeight.bold)), Switch(value: _showMasks, onChanged: (value) => setState(() => _showMasks = value))]),
    const SizedBox(height: 8),
    const Text("Mask Opacity", style: TextStyle(fontWeight: FontWeight.bold)),
    Slider(value: _maskOpacity, min: 0.1, max: 1.0, divisions: 9, label: _maskOpacity.toStringAsFixed(1), onChanged: (value) => setState(() => _maskOpacity = value))
  ])));

  Widget _buildNoDetectionsFound() => Stack(alignment: Alignment.center, children: [
    if (_imageFile != null) Image.file(_imageFile!),
    Positioned.fill(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 5.0, sigmaY: 5.0),
          child: Container(
            color: Colors.black.withOpacity(0.2),
            alignment: Alignment.center,
            child: const Text("No objects detected", style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold, shadows: [Shadow(color: Colors.black, blurRadius: 4)])),
          ),
        ),
      ),
    )
  ]);

  Widget _buildImageWithDetections() {
    if (_imageFile == null) return const SizedBox.shrink();
    if (_maskPngBytes == null) return _buildImageWithBoxesOnly();

    return LayoutBuilder(builder: (context, constraints) {
      if (_originalImageWidth == 0) return const SizedBox.shrink();
      final scaleRatio = constraints.maxWidth / _originalImageWidth;
      final displayHeight = _originalImageHeight * scaleRatio; 

      return FutureBuilder<List<ui.Image>>(
        future: Future.wait([_loadImage(_imageFile!), _loadImageFromBytes(_maskPngBytes!)]),
        builder: (context, snapshot) {
          if (!snapshot.hasData || snapshot.data!.length < 2) return SizedBox(width: constraints.maxWidth, height: displayHeight, child: const Center(child: CircularProgressIndicator()));
          return CustomPaint(
            size: Size(constraints.maxWidth, displayHeight), 
            painter: _DetectionPainter(
              originalImage: snapshot.data![0],
              maskImage: snapshot.data![1],
              recognitions: _recognitions,
              classColorMap: _classColorMap,
              modelImageWidth: _modelImageWidth,
              modelImageHeight: _modelImageHeight,
              selectedDetectionIndex: _selectedDetectionIndex,
              showMasks: _showMasks,
              maskOpacity: _maskOpacity,
            ),
          );
        },
      );
    });
  }

  Widget _buildImageWithBoxesOnly() => LayoutBuilder(builder: (context, constraints) {
    if (_originalImageWidth == 0) return const SizedBox.shrink();
    final scaleRatio = constraints.maxWidth / _originalImageWidth;
    final displayHeight = _originalImageHeight * scaleRatio;

    return FutureBuilder<ui.Image>(
      future: _loadImage(_imageFile!),
      builder: (context, snapshot) {
        if (!snapshot.hasData) return SizedBox(width: constraints.maxWidth, height: displayHeight, child: const Center(child: CircularProgressIndicator()));
        return CustomPaint(
          size: Size(constraints.maxWidth, displayHeight),
          painter: _DetectionPainter(
            originalImage: snapshot.data!,
            recognitions: _recognitions,
            classColorMap: _classColorMap,
            modelImageWidth: _modelImageWidth,
            modelImageHeight: _modelImageHeight,
            selectedDetectionIndex: _selectedDetectionIndex,
            showMasks: false,
            maskOpacity: 0,
          ),
        );
      }
    );
  });

  Widget _buildDetectionList() => ListView.builder(
    shrinkWrap: true, 
    physics: const NeverScrollableScrollPhysics(), 
    itemCount: _recognitions.length, 
    itemBuilder: (context, index) {
      final detection = _recognitions[index];
      final className = detection['className'] ?? 'Unknown';
      final confidence = (detection['confidence'] as num).toDouble();
      final isSelected = _selectedDetectionIndex == index;
      final itemColor = _classColorMap[className] ?? Colors.grey.shade700;
      
      return GestureDetector(
        onTap: () => setState(() => _selectedDetectionIndex = isSelected ? null : index),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
          margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
          decoration: BoxDecoration(
            color: Theme.of(context).cardTheme.surfaceTintColor ?? Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: isSelected ? itemColor : Colors.grey.shade300, width: isSelected ? 2.5 : 1),
            boxShadow: [BoxShadow(color: isSelected ? itemColor.withOpacity(0.3) : Colors.black.withOpacity(0.05), blurRadius: isSelected ? 8 : 4, offset: const Offset(0, 4))],
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12), 
            child: Row(children: [
              Container(
                width: 44, height: 44, alignment: Alignment.center,
                decoration: BoxDecoration(color: itemColor.withOpacity(0.15), borderRadius: BorderRadius.circular(10), border: Border.all(color: itemColor.withOpacity(0.8), width: 1.5)),
                child: Text('${index + 1}', style: TextStyle(color: itemColor, fontSize: 18, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(className, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                const SizedBox(height: 4),
                Text('${(confidence * 100).toStringAsFixed(1)}% Confidence', style: TextStyle(color: Colors.grey.shade700, fontWeight: FontWeight.w500)),
              ])),
              if (isSelected) Icon(Icons.check_circle, color: itemColor, size: 28),
            ])
          ),
        ),
      );
  });
  
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
}

class _DetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final double modelImageWidth;
  final double modelImageHeight;
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;
  final Map<String, Color> classColorMap;

  _DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.modelImageWidth,
    required this.modelImageHeight,
    required this.classColorMap,
    this.selectedDetectionIndex,
    this.showMasks = false,
    this.maskOpacity = 0.5,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final imageSize = Size(originalImage.width.toDouble(), originalImage.height.toDouble());
    final fittedSizes = applyBoxFit(BoxFit.contain, imageSize, size);
    final sourceRect = Alignment.center.inscribe(fittedSizes.source, Rect.fromLTWH(0, 0, imageSize.width, imageSize.height));
    final destinationRect = Alignment.center.inscribe(fittedSizes.destination, Rect.fromLTWH(0, 0, size.width, size.height));

    canvas.drawImageRect(originalImage, sourceRect, destinationRect, Paint());

    if (modelImageWidth == 0 || modelImageHeight == 0) return;

    final double scale = min(modelImageWidth / imageSize.width, modelImageHeight / imageSize.height);
    final double padX = (modelImageWidth - imageSize.width * scale) / 2.0;
    final double padY = (modelImageHeight - imageSize.height * scale) / 2.0;

    final double scaleToCanvasX = destinationRect.width / imageSize.width;
    final double scaleToCanvasY = destinationRect.height / imageSize.height;

    if (showMasks && maskImage != null) {
      for (int i = 0; i < recognitions.length; i++) {
        if (selectedDetectionIndex != null && i != selectedDetectionIndex) continue;

        final detection = recognitions[i];
        final className = detection['className'] ?? 'Unknown';
        final color = classColorMap[className] ?? Colors.grey;
        final maskPaint = Paint()..colorFilter = ColorFilter.mode(color.withOpacity(maskOpacity), BlendMode.srcIn);
        
        canvas.save();
        canvas.translate(destinationRect.left, destinationRect.top);
        canvas.scale(scaleToCanvasX / scale, scaleToCanvasY / scale);
        canvas.translate(-padX, -padY);
        canvas.drawImageRect(
          maskImage!,
          Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
          Rect.fromLTWH(0, 0, modelImageWidth, modelImageHeight),
          maskPaint,
        );
        canvas.restore();
      }
    }

    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';
      final color = classColorMap[className] ?? Colors.grey;
      final isSelected = i == selectedDetectionIndex;

      final x1 = (detection['x1'] as num).toDouble();
      final y1 = (detection['y1'] as num).toDouble();
      final x2 = (detection['x2'] as num).toDouble();
      final y2 = (detection['y2'] as num).toDouble();

      final originalX1 = (x1 - padX) / scale;
      final originalY1 = (y1 - padY) / scale;
      final originalX2 = (x2 - padX) / scale;
      final originalY2 = (y2 - padY) / scale;

      final canvasLeft = (originalX1 * scaleToCanvasX) + destinationRect.left;
      final canvasTop = (originalY1 * scaleToCanvasY) + destinationRect.top;
      final canvasRight = (originalX2 * scaleToCanvasX) + destinationRect.left;
      final canvasBottom = (originalY2 * scaleToCanvasY) + destinationRect.top;
      
      final boundingBoxRect = Rect.fromLTRB(canvasLeft, canvasTop, canvasRight, canvasBottom);

      final boxPaint = Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = isSelected ? 4.0 : 2.5;
      canvas.drawRect(boundingBoxRect, boxPaint);
      
      final confidence = (detection['confidence'] as num? ?? 0.0);
      final textPainter = TextPainter(
        text: TextSpan(text: '$className (${(confidence * 100).toStringAsFixed(1)}%)', style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold, shadows: [Shadow(color: Colors.black, blurRadius: 4)])),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);

      final labelBackgroundPaint = Paint()..color = color.withOpacity(isSelected ? 1.0 : 0.8);
      double labelTop = canvasTop - textPainter.height - 4;
      if (labelTop < destinationRect.top) labelTop = canvasBottom + 2;
      double labelLeft = canvasLeft;
      if (labelLeft + textPainter.width + 8 > destinationRect.right) {
        labelLeft = destinationRect.right - textPainter.width - 8;
      }
      
      final finalLabelRect = Rect.fromLTWH(labelLeft, labelTop, textPainter.width + 8, textPainter.height + 4);
      canvas.drawRect(finalLabelRect, labelBackgroundPaint);
      textPainter.paint(canvas, Offset(finalLabelRect.left + 4, finalLabelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) =>
    originalImage != oldDelegate.originalImage ||
    maskImage != oldDelegate.maskImage ||
    recognitions != oldDelegate.recognitions ||
    showMasks != oldDelegate.showMasks ||
    maskOpacity != oldDelegate.maskOpacity ||
    selectedDetectionIndex != oldDelegate.selectedDetectionIndex ||
    modelImageWidth != oldDelegate.modelImageWidth ||
    modelImageHeight != oldDelegate.modelImageHeight;
}
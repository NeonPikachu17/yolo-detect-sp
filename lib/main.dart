import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; 
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// NOTE: This is the Static Control app. 
// It mimics a "bundled" app by copying the asset to local storage once on startup.

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // --- THEME COLORS: Maroon + Hat Fusion ---
    const maroonPrimary = Color(0xFF800020);    // Deep Maroon 
    const hatBlueAccent = Color(0xFF6B7BA8);    // Hat Periwinkle
    const hatGoldAccent = Color(0xFFE2B04E);    // Hat Gold
    const hatBackground = Color(0xFFF0F2F5);    // Hat Off-White
    const darkText = Color(0xFF2C3E50);         // Hat Navy

    return MaterialApp(
      title: 'Static Control',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: maroonPrimary,
        scaffoldBackgroundColor: hatBackground,
        colorScheme: ColorScheme.fromSeed(
          seedColor: maroonPrimary,
          brightness: Brightness.light,
          primary: maroonPrimary,
          secondary: hatBlueAccent,
          tertiary: hatGoldAccent,
          background: hatBackground,
          surface: Colors.white,
          error: const Color(0xFFBA1A1A),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(Theme.of(context).textTheme).apply(
          bodyColor: Colors.blueGrey[800],
          displayColor: darkText,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: const BorderRadius.all(Radius.circular(24)),
            side: BorderSide(color: hatBlueAccent.withOpacity(0.15), width: 1),
          ),
          clipBehavior: Clip.antiAlias,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: maroonPrimary,
            foregroundColor: Colors.white,
            elevation: 4,
            shadowColor: maroonPrimary.withOpacity(0.4),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: hatBackground,
          foregroundColor: maroonPrimary,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: GoogleFonts.poppins(
            fontWeight: FontWeight.w800,
            fontSize: 26,
            color: maroonPrimary,
            letterSpacing: -0.5,
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
  YOLO? _yoloModel;
  File? _imageFile;
  List<Map<String, dynamic>> _recognitions = [];
  bool _isLoading = false;
  String? _loadingMessage;
  
  // Image Dimensions
  double _originalImageHeight = 0;
  double _originalImageWidth = 0;
  double _modelImageHeight = 0;
  double _modelImageWidth = 0;

  int? _selectedDetectionIndex;
  bool _showMasks = true;
  double _maskOpacity = 0.5;
  // Static Map for simple visualization
  final Map<String, Color> _classColorMap = {}; 
  Uint8List? _maskPngBytes;

  // Box Colors
  final List<Color> _boxColors = [
    const Color(0xFF800020), // Maroon
    const Color(0xFF6B7BA8), // Hat Blue
    const Color(0xFFE2B04E), // Hat Gold
    const Color(0xFF2C3E50), // Hat Navy
    Colors.teal,
    Colors.orangeAccent,
    Colors.purple,
  ];
  
  // Animation
  late AnimationController _idleAnimationController;
  late Animation<double> _idleAnimation;

  @override
  void initState() {
    super.initState();
     // Setup breathing animation
    _idleAnimationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    _idleAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _idleAnimationController, curve: Curves.easeInOut),
    );

    // STATIC BEHAVIOR: Initialize immediately
    _initializeStaticModel();
  }
  
  @override
  void dispose() {
    _yoloModel?.dispose();
    _idleAnimationController.dispose();
    super.dispose();
  }

  Future<void> _initializeStaticModel() async {
    setState(() { _isLoading = true; _loadingMessage = "Initializing Static Model..."; });
    
    // This delay ensures the UI has time to build before we freeze it with file I/O
    await Future.delayed(const Duration(milliseconds: 500));

    try {
      // 1. Copy asset to local file (Simulating an installed app bundle)
      final modelPath = await _copyAssetToFile("assets/models/static_model.tflite");
      
      // 2. Load the model
      // IMPORTANT: Ensure this matches your model type. 
      // If using yolov8s-seg.tflite, use YOLOTask.segment
      _yoloModel = YOLO(modelPath: modelPath, task: YOLOTask.segment);
      await _yoloModel?.loadModel();
      
      print("Static Model Loaded Successfully at: $modelPath");
      
      if (mounted) {
         ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: const Text("Static Baseline Model Ready"),
          backgroundColor: Theme.of(context).primaryColor,
        ));
      }
    } catch (e) {
      print("Error loading static model: $e");
      if (mounted) {
        showDialog(context: context, builder: (ctx) => AlertDialog(
          title: const Text("Initialization Error"),
          content: Text("Failed to load the static model.\n\nError: $e\n\nPlease restart the app."),
        ));
      }
    } finally {
      if (mounted) setState(() { _isLoading = false; _loadingMessage = null; });
    }
  }

  Future<String> _copyAssetToFile(String assetPath) async {
    try {
      final docDir = await getApplicationDocumentsDirectory();
      final filename = assetPath.split('/').last;
      final file = File('${docDir.path}/$filename');
      
      // Always overwrite to ensure we use the latest asset
      final data = await rootBundle.load(assetPath);
      final bytes = data.buffer.asUint8List();
      await file.writeAsBytes(bytes, flush: true);
      return file.path;
    } catch (e) {
      throw Exception("Could not copy asset $assetPath: $e");
    }
  }

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
    });

    // Run inference immediately after picking
    _runInference();
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

  // --- DATA COLLECTION POINT: Inference Time ---
  Future<void> _runInference() async {
    if (_imageFile == null || _yoloModel == null) return;
    setState(() { _isLoading = true; _loadingMessage = "Analyzing..."; });

    // 1. Start Timer
    final stopwatch = Stopwatch()..start();

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final detections = await _yoloModel!.predict(imageBytes);
      
      // 2. Stop Timer and Log
      stopwatch.stop();
      print("--------------------------------------------------");
      print("[STATIC_BASELINE] Inference Time: ${stopwatch.elapsedMilliseconds} ms");
      print("--------------------------------------------------");
      
      ScaffoldMessenger.of(context).hideCurrentSnackBar();
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text("Inference complete in ${stopwatch.elapsedMilliseconds}ms"),
        backgroundColor: Theme.of(context).primaryColor,
        duration: const Duration(seconds: 1),
      ));

      if (!mounted) return;

      // Parsing Logic (Same as MAMBO to ensure CPU load is comparable)
      final double modelWidth = (detections['image_width'] as num?)?.toDouble() ?? _originalImageWidth;
      final double modelHeight = (detections['image_height'] as num?)?.toDouble() ?? _originalImageHeight;

      final formattedRecognitions = <Map<String, dynamic>>[];
      Uint8List? newMaskPngBytes;
      int colorIndex = 0;

      // Assuming Segment Task for Baseline
      final Map<String, dynamic> detectionMap = Map.from((detections as Map?) ?? {});
      final List<dynamic> boxes = (detectionMap['boxes'] as List<dynamic>?) ?? [];
      
      for (var box in boxes) {
        double x1 = (box['x1'] as num).toDouble();
        double y1 = (box['y1'] as num).toDouble();
        double x2 = (box['x2'] as num).toDouble();
        double y2 = (box['y2'] as num).toDouble();
        
        // Normalize coordinates if they are normalized (0-1)
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
        
        if (!_classColorMap.containsKey(className)) {
          _classColorMap[className] = _boxColors[colorIndex % _boxColors.length];
          colorIndex++;
        }
      }
      
      newMaskPngBytes = detectionMap['maskPng'];

       setState(() {
          _recognitions = formattedRecognitions;
          _maskPngBytes = newMaskPngBytes;
          _modelImageWidth = modelWidth;
          _modelImageHeight = modelHeight;
       });
      
    } catch (e) {
      print("Error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text("Inference Error: $e"),
        backgroundColor: Theme.of(context).colorScheme.error,
      ));
    } finally {
      setState(() { _isLoading = false; _loadingMessage = null; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Static Control App")),
      body: Stack(
        children: [
          Center(
            child: _imageFile != null
                ? Column(
                    children: [
                      // Result Image Card (Expanded to fill space)
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: _buildImageWithDetections(),
                        ),
                      ),
                      // Bottom Controls
                      Container(
                         padding: const EdgeInsets.all(20),
                         decoration: BoxDecoration(
                           color: Colors.white,
                           borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
                           boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, -4))],
                         ),
                         child: Column(
                           mainAxisSize: MainAxisSize.min,
                           children: [
                             Text(
                                "${_recognitions.length} Detections Found",
                                 style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor)
                              ),
                              const SizedBox(height: 16),
                              Row(
                                children: [
                                  Expanded(
                                    child: ElevatedButton.icon(
                                      onPressed: _pickImage, 
                                      icon: const Icon(Icons.photo_library), 
                                      label: const Text("Gallery"),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.white, 
                                        foregroundColor: Theme.of(context).primaryColor,
                                        side: BorderSide(color: Theme.of(context).primaryColor.withOpacity(0.3)),
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 16),
                                  Expanded(
                                    child: ElevatedButton.icon(
                                      onPressed: _takePicture, 
                                      icon: const Icon(Icons.camera_alt), 
                                      label: const Text("Camera")
                                    ),
                                  ),
                                ],
                              )
                           ],
                         ),
                      )
                    ],
                  )
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                       ScaleTransition(
                        scale: _idleAnimation,
                        child: Container(
                          padding: const EdgeInsets.all(32),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: Theme.of(context).primaryColor.withOpacity(0.1),
                                blurRadius: 40,
                                spreadRadius: 10,
                              )
                            ],
                          ),
                          child: Icon(Icons.image_search, size: 80, color: Theme.of(context).primaryColor.withOpacity(0.6)),
                        ),
                      ),
                      const SizedBox(height: 30),
                      Text(
                        "Baseline Testing",
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "Static Model Loaded from Assets.",
                        style: TextStyle(color: Colors.grey.shade600),
                      ),
                      const SizedBox(height: 30),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton.icon(onPressed: _pickImage, icon: const Icon(Icons.photo_library), label: const Text("Gallery")),
                          const SizedBox(width: 16),
                          ElevatedButton.icon(onPressed: _takePicture, icon: const Icon(Icons.camera_alt), label: const Text("Camera")),
                        ],
                      ),
                    ],
                  ),
          ),
           if (_isLoading) _buildLoadingModalOverlay(context),
        ],
      ),
    );
  }

  Widget _buildImageWithDetections() {
    if (_imageFile == null) return const SizedBox.shrink();

    return LayoutBuilder(builder: (context, constraints) {
      if (_originalImageWidth == 0) return const SizedBox.shrink();
      
      // ... (aspect ratio logic remains same) ...

      // FIX STARTS HERE:
      // We create a list of futures that return nullable Images.
      // This allows us to wait for both, even if the mask is null.
      final futures = <Future<ui.Image?>>[
        _loadImage(_imageFile!),
      ];
      
      if (_maskPngBytes != null) {
        futures.add(_loadImageFromBytes(_maskPngBytes!));
      }

      return FutureBuilder<List<ui.Image?>>( // Expect nullable list
        future: Future.wait(futures),
        builder: (context, snapshot) {
          // Check if the first image (original) is loaded
          if (!snapshot.hasData || snapshot.data![0] == null) {
             return const Center(child: CircularProgressIndicator());
          }
          
          return ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: CustomPaint(
              size: Size(constraints.maxWidth, constraints.maxHeight), 
              painter: _StaticDetectionPainter(
                originalImage: snapshot.data![0]!, // Force unwrap original
                // Safely access the second image if it exists
                maskImage: (snapshot.data!.length > 1) ? snapshot.data![1] : null,
                recognitions: _recognitions,
                classColorMap: _classColorMap,
                modelImageWidth: _modelImageWidth,
                modelImageHeight: _modelImageHeight,
                showMasks: _showMasks,
                maskOpacity: _maskOpacity,
              ),
            ),
          );
        },
      );
    });
  }
  
  // Change return type to Future<ui.Image> (non-nullable) since we check for null bytes before calling
  Future<ui.Image> _loadImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  // Only one definition needed. 
  Future<ui.Image> _loadImageFromBytes(Uint8List bytes) async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  Widget _buildLoadingModalOverlay(BuildContext context) {
    return Stack(
      children: [
        BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 4.0, sigmaY: 4.0),
          child: const Opacity(
            opacity: 0.4,
            child: ModalBarrier(dismissible: false, color: Colors.black),
          ),
        ),
        Center(
          child: Card(
            elevation: 10,
            shadowColor: Theme.of(context).primaryColor.withOpacity(0.3),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    height: 40, width: 40,
                    child: CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                      strokeWidth: 3,
                    ),
                  ),
                  const SizedBox(height: 24),
                  if (_loadingMessage != null)
                    Text(
                      _loadingMessage!,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _StaticDetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final double modelImageWidth;
  final double modelImageHeight;
  final bool showMasks;
  final double maskOpacity;
  final Map<String, Color> classColorMap;

  _StaticDetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.modelImageWidth,
    required this.modelImageHeight,
    required this.classColorMap,
    this.showMasks = true,
    this.maskOpacity = 0.5,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Use BoxFit.contain logic to draw image centered
    final imageSize = Size(originalImage.width.toDouble(), originalImage.height.toDouble());
    final fittedSizes = applyBoxFit(BoxFit.contain, imageSize, size);
    final sourceRect = Alignment.center.inscribe(fittedSizes.source, Rect.fromLTWH(0, 0, imageSize.width, imageSize.height));
    final destinationRect = Alignment.center.inscribe(fittedSizes.destination, Rect.fromLTWH(0, 0, size.width, size.height));

    canvas.drawImageRect(originalImage, sourceRect, destinationRect, Paint());

    if (modelImageWidth == 0 || modelImageHeight == 0) return;

    // Calculate scaling factors
    final double scale = min(modelImageWidth / imageSize.width, modelImageHeight / imageSize.height);
    final double padX = (modelImageWidth - imageSize.width * scale) / 2.0;
    final double padY = (modelImageHeight - imageSize.height * scale) / 2.0;

    final double scaleToCanvasX = destinationRect.width / imageSize.width;
    final double scaleToCanvasY = destinationRect.height / imageSize.height;

    // Draw Masks
    if (showMasks && maskImage != null) {
      for (int i = 0; i < recognitions.length; i++) {
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

    // Draw Boxes
    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';
      final color = classColorMap[className] ?? Colors.grey;

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

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRect(boundingBoxRect, boxPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _StaticDetectionPainter oldDelegate) => true; // Always repaint for static
}
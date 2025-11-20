import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// Firebase packages
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'firebase_options.dart'; 

// Keys for storing the user's last session settings
const String _prefsKeyLastModelName = "last_used_model_name";
const String _prefsKeyLastTaskType = "last_used_task_type";

enum AppYoloTask { segment, detect, classify }

void main() async {
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

    // --- THEME COLORS: Maroon + Hat Fusion ---
    const maroonPrimary = Color(0xFF800020);    // Deep Maroon (Primary Action)
    const hatBlueAccent = Color(0xFF6B7BA8);    // Hat Periwinkle (Secondary/Structure)
    const hatGoldAccent = Color(0xFFE2B04E);    // Hat Gold (Highlights)
    const hatBackground = Color(0xFFF0F2F5);    // Hat Off-White (Background)
    const darkText = Color(0xFF2C3E50);         // Hat Navy (Text)

    return MaterialApp(
      title: 'MAMBO',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: maroonPrimary,
        scaffoldBackgroundColor: hatBackground, // Soft hat background
        colorScheme: ColorScheme.fromSeed(
          seedColor: maroonPrimary,
          brightness: Brightness.light,
          primary: maroonPrimary,
          secondary: hatBlueAccent, // Cool blue accent to balance the warm maroon
          tertiary: hatGoldAccent,
          background: hatBackground,
          surface: Colors.white,
          error: const Color(0xFFBA1A1A),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(textTheme).apply(
          bodyColor: Colors.blueGrey[800],
          displayColor: darkText,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: const BorderRadius.all(Radius.circular(24)),
            // Subtle border using the Hat Blue for structure
            side: BorderSide(color: hatBlueAccent.withOpacity(0.15), width: 1),
          ),
          clipBehavior: Clip.antiAlias,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: maroonPrimary, // Maroon buttons
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
          foregroundColor: maroonPrimary, // Maroon Title
          elevation: 0,
          centerTitle: true,
          titleTextStyle: GoogleFonts.poppins(
            fontWeight: FontWeight.w800,
            fontSize: 26,
            color: maroonPrimary,
            letterSpacing: -0.5,
          ),
        ),
        segmentedButtonTheme: SegmentedButtonThemeData(
          style: ButtonStyle(
            backgroundColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) {
                return maroonPrimary.withOpacity(0.1); // Soft maroon tint
              }
              return Colors.transparent;
            }),
            foregroundColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) {
                return maroonPrimary;
              }
              return hatBlueAccent; // Inactive text is Hat Blue
            }),
            iconColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) {
                return maroonPrimary;
              }
              return hatBlueAccent;
            }),
            side: MaterialStateProperty.all(BorderSide(color: hatBlueAccent.withOpacity(0.3))),
          ),
        ),
        sliderTheme: SliderThemeData(
          activeTrackColor: maroonPrimary,
          thumbColor: hatGoldAccent, // Gold thumb for pop
          inactiveTrackColor: maroonPrimary.withOpacity(0.1),
        ),
        switchTheme: SwitchThemeData(
          thumbColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) return maroonPrimary;
            return Colors.blueGrey;
          }),
          trackColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) return maroonPrimary.withOpacity(0.3);
            return Colors.grey.withOpacity(0.2);
          }),
        )
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

  AppYoloTask _selectedTask = AppYoloTask.segment;
  List<Map<String, String>> _availableModels = [];
  
  // Cloud State
  Future<List<String>>? _cloudModelsFuture;

  // Animation
  late AnimationController _idleAnimationController;
  late Animation<double> _idleAnimation;

  // FUSION PALETTE for Boxes: Maroon, Hat Blue, Gold, Navy
  final List<Color> _boxColors = [
    const Color(0xFF800020), // Maroon
    const Color(0xFF6B7BA8), // Hat Blue
    const Color(0xFFE2B04E), // Hat Gold
    const Color(0xFF2C3E50), // Hat Navy
    Colors.teal,
    Colors.orangeAccent,
    Colors.purple,
  ];

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
    
    _initializeScreenData();
  }
  
  @override
  void dispose() {
    _yoloModel?.dispose();
    _idleAnimationController.dispose();
    super.dispose();
  }

  Future<void> _initializeScreenData() async {
    _startLoading("Initializing...");
    _availableModels = await _discoverLocalModels();
    final prefs = await SharedPreferences.getInstance();
    final lastModelName = prefs.getString(_prefsKeyLastModelName);
    var lastTaskIndex = prefs.getInt(_prefsKeyLastTaskType) ?? AppYoloTask.segment.index;

    if (lastTaskIndex >= AppYoloTask.values.length) {
      lastTaskIndex = AppYoloTask.segment.index;
    }

    setState(() {
      _selectedTask = AppYoloTask.values[lastTaskIndex];
    });

    if (lastModelName != null && _availableModels.any((m) => m['name'] == lastModelName)) {
      final modelData = _availableModels.firstWhere((m) => m['name'] == lastModelName);
      await _prepareAndLoadModel(modelData);
    } else {
      _stopLoading();
    }
  }

  // --- File & Image Handling ---

  Future<ui.Image> _resizeAndCropToPortrait(ui.Image originalImage, double targetWidth, double targetHeight) async {
    final double scale = max(targetWidth / originalImage.width, targetHeight / originalImage.height);
    final double newWidth = originalImage.width * scale;
    final double newHeight = originalImage.height * scale;
    final double cropX = (newWidth - targetWidth) / 2;
    final double cropY = (newHeight - targetHeight) / 2;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, targetWidth, targetHeight));

    canvas.drawImageRect(
      originalImage,
      Rect.fromLTWH(cropX / scale, cropY / scale, targetWidth / scale, targetHeight / scale),
      Rect.fromLTWH(0, 0, targetWidth, targetHeight),
      Paint(),
    );

    final picture = recorder.endRecording();
    return await picture.toImage(targetWidth.toInt(), targetHeight.toInt());
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
      _selectedDetectionIndex = null;
    });

    if (_yoloModel != null) {
      _runInference();
    } else {
      _showSnackBar("Please select and load a model first.", isError: true);
    }
  }

  Future<List<Map<String, String>>> _discoverLocalModels() async {
    final docDir = await getApplicationDocumentsDirectory();
    final files = docDir.listSync();
    final modelFiles = files.where((f) => f.path.endsWith('.tflite'));
    return modelFiles.map((modelFile) {
      final modelName = p.basenameWithoutExtension(modelFile.path);
      return {
        'name': modelName,
        'modelPath': modelFile.path,
        'labelsPath': p.join(docDir.path, "$modelName.txt"),
      };
    }).toList();
  }
  
  Future<void> _importModelFromPicker() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any);
    if (result == null || result.files.single.path == null) return;
    final file = result.files.single;

    if (file.extension?.toLowerCase() != 'tflite') {
      _showSnackBar("Invalid file type. Please select a .tflite model.", isError: true);
      return;
    }

    _startLoading("Importing model...");
    try {
      final docDir = await getApplicationDocumentsDirectory();
      final newModelPath = p.join(docDir.path, file.name);

      if (await File(newModelPath).exists()) {
        _showSnackBar("A model with this name already exists.", isError: true);
        _stopLoading();
        return;
      }

      await File(file.path!).copy(newModelPath);
      final newLabelsPath = p.join(docDir.path, "${p.basenameWithoutExtension(file.name)}.txt");
      if (!await File(newLabelsPath).exists()) await File(newLabelsPath).create();
      
      _showSnackBar("'${file.name}' imported successfully!", isError: false);
      await _handleRefresh();
    } catch (e) {
      _showSnackBar("Error importing model: $e", isError: true);
      _stopLoading();
    }
  }

  Future<void> _prepareAndLoadModel(Map<String, String> modelData) async {
    _clearScreen();
    _startLoading("Loading ${modelData['name']}...");

    try {
      final targetModelPath = modelData['modelPath']!;
      if (!await File(targetModelPath).exists()) {
        throw Exception("Model file not found. It may have been deleted.");
      }

      if (_yoloModel != null) await _yoloModel!.dispose();
      
      late YOLOTask yoloTask;
      switch (_selectedTask) {
        case AppYoloTask.segment:
          yoloTask = YOLOTask.segment;
          break;
        case AppYoloTask.detect:
          yoloTask = YOLOTask.detect;
          break;
        case AppYoloTask.classify:
          yoloTask = YOLOTask.classify;
          break;
      }

      _yoloModel = YOLO(modelPath: targetModelPath, task: yoloTask);
      await _yoloModel?.loadModel();

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefsKeyLastModelName, modelData['name']!);
      await prefs.setInt(_prefsKeyLastTaskType, _selectedTask.index);

      if (mounted) {
        setState(() { _selectedModelName = modelData['name']; });
        _showSnackBar("'${modelData['name']}' loaded successfully.", isError: false);
      }
    } catch (e) {
      _showSnackBar("Failed to load model: ${e.toString()}", isError: true);
      if (mounted) setState(() { _yoloModel = null; _selectedModelName = null; });
    } finally {
      _stopLoading();
    }
  }

  Future<void> _deleteLocallyStoredModel(String modelName) async {
    final modelData = _availableModels.firstWhere((m) => m['name'] == modelName);
    final modelFile = File(modelData['modelPath']!);
    final labelsFile = File(modelData['labelsPath']!);

    if (await modelFile.exists()) await modelFile.delete();
    if (await labelsFile.exists()) await labelsFile.delete();

    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(_prefsKeyLastModelName) == modelName) {
      await prefs.remove(_prefsKeyLastModelName);
    }
    
    if (_selectedModelName == modelName) {
      _clearScreen();
      setState(() {
        _yoloModel = null;
        _selectedModelName = null;
      });
    }

    await _handleRefresh();
    _showSnackBar("Deleted '$modelName'.", isError: false);
  }
  
  // --- Firebase Storage Functions ---

  Future<List<String>> _fetchCloudModels() async {
    try {
      final storageRef = FirebaseStorage.instance.ref().child('yoloModels');
      final listResult = await storageRef.listAll().timeout(const Duration(seconds: 15));
      final modelNames = listResult.prefixes.map((prefix) => prefix.name).toList();
      return modelNames;
    } on TimeoutException catch (_) {
      throw Exception("Connection timed out. Check internet.");
    } catch (e) {
      throw Exception("Failed to fetch cloud models: $e");
    }
  }

  Future<void> _downloadModel(String modelName) async {
    _startLoading("Downloading '$modelName'...");
    try {
      final docDir = await getApplicationDocumentsDirectory();
      final localModelPath = p.join(docDir.path, '$modelName.tflite');
      final localLabelsPath = p.join(docDir.path, '$modelName.txt');
      
      if (await File(localModelPath).exists()) {
          _showSnackBar("Model '$modelName' already exists locally.", isError: true);
          _stopLoading();
          return;
      }

      final modelRef = FirebaseStorage.instance.ref('yoloModels/$modelName/model.tflite');
      await modelRef.writeToFile(File(localModelPath));

      try {
        final labelsRef = FirebaseStorage.instance.ref('yoloModels/$modelName/labels.txt');
        await labelsRef.writeToFile(File(localLabelsPath));
      } catch (e) {
        await File(localLabelsPath).create();
      }
      
      _showSnackBar("'$modelName' downloaded successfully.", isError: false);
      await _handleRefresh(); 
    } catch (e) {
      _showSnackBar("Error downloading model: $e", isError: true);
    } finally {
      _stopLoading();
    }
  }
  
  Future<void> _uploadModel(String modelName) async {
    final modelData = _availableModels.firstWhere((m) => m['name'] == modelName);
    final modelFile = File(modelData['modelPath']!);
    final labelsFile = File(modelData['labelsPath']!);

    _startLoading("Uploading '$modelName'...");
    try {
      final metadata = SettableMetadata(
        contentType: 'application/octet-stream',
        customMetadata: {'uploaded_by': 'user_device'},
      );

      final modelRef = FirebaseStorage.instance.ref('yoloModels/$modelName/model.tflite');
      await modelRef.putFile(modelFile, metadata);

      if (await labelsFile.exists()) {
        final labelsRef = FirebaseStorage.instance.ref('yoloModels/$modelName/labels.txt');
        await labelsRef.putFile(labelsFile, SettableMetadata(contentType: 'text/plain'));
      }

      _showSnackBar("'$modelName' uploaded successfully!", isError: false);
      
      // Refresh the cloud list by resetting the future
      setState(() {
        _cloudModelsFuture = _fetchCloudModels();
      });
      
    } on FirebaseException catch (e) {
      String errorMessage = "Upload failed.";
      if (e.code == 'unauthorized') errorMessage = "Permission Denied.";
      else if (e.code == 'retry-limit-exceeded') errorMessage = "Connection Unstable.";
      _showSnackBar(errorMessage, isError: true);
    } catch (e) {
      _showSnackBar("System Error: $e", isError: true);
    } finally {
      _stopLoading();
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
    _startLoading("Analyzing...");

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
  
  // --- Helper Methods ---

  Future<void> _handleRefresh() async {
    _clearScreen();
    await _initializeScreenData();
  }

  void _startLoading(String message) => setState(() { _isLoading = true; _loadingMessage = message; });
  void _stopLoading() => setState(() { _isLoading = false; _loadingMessage = null; });
  void _clearScreen() => setState(() { _imageFile = null; _recognitions = []; _maskPngBytes = null; _selectedDetectionIndex = null; });

  void _showSnackBar(String message, {required bool isError}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)..hideCurrentSnackBar()..showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Theme.of(context).colorScheme.error : Theme.of(context).primaryColor,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  // --- Main Build ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true,
      appBar: AppBar(title: const Text("MAMBO")),
      bottomNavigationBar: _buildBottomActionBar(),
      body: Stack(
        children: [
          Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 1200),
              child: RefreshIndicator(
                color: Theme.of(context).primaryColor,
                onRefresh: _handleRefresh,
                // Use CustomScrollView for responsive "sticky" layout
                child: CustomScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  slivers: [
                    // 1. Top Card
                    SliverPadding(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                      sliver: SliverToBoxAdapter(
                        child: Column(
                          children: [
                            _buildModelManagementCard(),
                            const SizedBox(height: 24),
                          ],
                        ),
                      ),
                    ),
                    // 2. Content Area (Fills remaining space)
                    SliverFillRemaining(
                      hasScrollBody: false,
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 120), // Bottom padding for navbar
                        child: AnimatedSwitcher(
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
                      ),
                    ),
                  ],
                ),
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
    // DYNAMIC SIZING based on screen width
    final screenWidth = MediaQuery.of(context).size.width;
    final isSmallPhone = screenWidth < 380;

    return Card(
      child: Padding(
        padding: EdgeInsets.all(isSmallPhone ? 16.0 : 20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Theme.of(context).primaryColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(Icons.hub_outlined, color: Theme.of(context).primaryColor),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("ACTIVE MODEL", style: Theme.of(context).textTheme.labelMedium?.copyWith(color: Theme.of(context).primaryColor, fontWeight: FontWeight.w700, letterSpacing: 0.5)),
                      const SizedBox(height: 4),
                      Text(
                        _selectedModelName ?? "None Selected",
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: Colors.blueGrey[900],
                              fontSize: isSmallPhone ? 18 : 22, // Scale Text
                            ),
                        overflow: TextOverflow.ellipsis,
                        maxLines: 1,
                      ),
                    ],
                  ),
                ),
                TextButton(
                  onPressed: _showModelSelectionSheet,
                  style: TextButton.styleFrom(foregroundColor: Theme.of(context).primaryColor),
                  child: const Text("Change", style: TextStyle(fontWeight: FontWeight.w700)),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Text("ANALYSIS TASK", style: Theme.of(context).textTheme.labelMedium?.copyWith(color: Theme.of(context).primaryColor, fontWeight: FontWeight.w700, letterSpacing: 0.5)),
            const SizedBox(height: 12),
            SegmentedButton<AppYoloTask>(
              segments: const [
                ButtonSegment<AppYoloTask>(value: AppYoloTask.segment, label: Text('Segment'), icon: Icon(Icons.grain_rounded)),
                ButtonSegment<AppYoloTask>(value: AppYoloTask.detect, label: Text('Detect'), icon: Icon(Icons.select_all_rounded)),
                ButtonSegment<AppYoloTask>(value: AppYoloTask.classify, label: Text('Classify'), icon: Icon(Icons.label_important_outline)),
              ],
              selected: {_selectedTask},
              onSelectionChanged: (newSelection) {
                if (_isLoading) return;
                setState(() => _selectedTask = newSelection.first);
                if (_selectedModelName != null) {
                  final modelData = _availableModels.firstWhere((m) => m['name'] == _selectedModelName);
                  _prepareAndLoadModel(modelData);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showModelSelectionSheet() {
    setState(() {
      _cloudModelsFuture = _fetchCloudModels();
    });

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent, 
      builder: (ctx) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.65,
        maxChildSize: 0.9,
        builder: (_, controller) => Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              // Handle bar
              Center(
                child: Container(
                  margin: const EdgeInsets.symmetric(vertical: 12),
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)),
                ),
              ),
              Expanded(
                child: DefaultTabController(
                  length: 2,
                  child: Column(
                    children: [
                      TabBar(
                        indicatorColor: Theme.of(context).primaryColor,
                        labelColor: Theme.of(context).primaryColor,
                        unselectedLabelColor: Colors.grey,
                        labelStyle: const TextStyle(fontWeight: FontWeight.bold),
                        tabs: const [
                          Tab(icon: Icon(Icons.smartphone_rounded), text: "Local"),
                          Tab(icon: Icon(Icons.cloud_queue_rounded), text: "Cloud"),
                        ],
                      ),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _buildLocalModelsTab(ctx),
                            _buildCloudModelsTab(ctx),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLocalModelsTab(BuildContext modalContext) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_availableModels.isEmpty)
            Expanded(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.folder_off_outlined, size: 48, color: Colors.grey.shade400),
                    const SizedBox(height: 16),
                    Text("No local models found.", style: TextStyle(color: Colors.grey.shade600)),
                  ],
                ),
              ),
            )
          else
            Expanded(
              child: ListView.separated(
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemCount: _availableModels.length,
                itemBuilder: (context, index) {
                  final model = _availableModels[index];
                  final isSelected = _selectedModelName == model['name'];
                  return InkWell(
                    onTap: () {
                      Navigator.of(modalContext).pop(); // Auto-close modal
                      _prepareAndLoadModel(model);
                    },
                    borderRadius: BorderRadius.circular(12),
                    child: Container(
                      decoration: BoxDecoration(
                        color: isSelected ? Theme.of(context).primaryColor.withOpacity(0.08) : Colors.white,
                        border: Border.all(color: isSelected ? Theme.of(context).primaryColor : Colors.grey.shade200),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ListTile(
                        title: Text(model['name']!, style: TextStyle(fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
                        leading: Icon(Icons.extension_outlined, color: isSelected ? Theme.of(context).primaryColor : Theme.of(context).colorScheme.secondary),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              icon: Icon(Icons.cloud_upload_outlined, color: Theme.of(context).primaryColor),
                              tooltip: "Upload to Cloud",
                              onPressed: () {
                                 _uploadModel(model['name']!);
                              },
                            ),
                            IconButton(
                              icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                              onPressed: () {
                                Navigator.of(modalContext).pop(); 
                                _showDeleteConfirmationDialog(model['name']!);
                              },
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            icon: const Icon(Icons.add_rounded),
            label: const Text("Import from Device"),
            onPressed: () {
              Navigator.of(modalContext).pop(); // Auto-close modal
              _importModelFromPicker();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildCloudModelsTab(BuildContext modalContext) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: FutureBuilder<List<String>>(
        future: _cloudModelsFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator(valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor)));
          }
          if (snapshot.hasError) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.cloud_off_rounded, size: 48, color: Colors.grey),
                  const SizedBox(height: 16),
                  Text("Failed to load", style: TextStyle(color: Theme.of(context).colorScheme.error, fontWeight: FontWeight.bold)),
                  Text("Check your internet connection", style: TextStyle(color: Colors.grey.shade600, fontSize: 12)),
                ],
              ),
            );
          }
          if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(child: Text("No models in the cloud."));
          }

          final cloudModels = snapshot.data!;
          return ListView.separated(
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemCount: cloudModels.length,
            itemBuilder: (context, index) {
              final modelName = cloudModels[index];
              final isDownloaded = _availableModels.any((m) => m['name'] == modelName);
              return Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade200),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ListTile(
                  title: Text(modelName, style: const TextStyle(fontWeight: FontWeight.w500)),
                  leading: Icon(Icons.cloud_queue_rounded, color: Theme.of(context).colorScheme.secondary),
                  trailing: isDownloaded
                      ? Icon(Icons.check_circle, color: Colors.green.shade600)
                      : IconButton(
                          icon: Icon(Icons.download_rounded, color: Theme.of(context).primaryColor),
                          tooltip: "Download",
                          onPressed: () {
                            Navigator.of(modalContext).pop(); // Auto-close modal
                            _downloadModel(modelName);
                          },
                        ),
                ),
              );
            },
          );
        },
      ),
    );
  }

  Widget _buildBottomActionBar() {
    final bool isReadyForAnalysis = _yoloModel != null && !_isLoading;
    
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 12.0, sigmaY: 12.0),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.85),
            border: Border(top: BorderSide(color: Theme.of(context).primaryColor.withOpacity(0.1))),
          ),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.photo_library_rounded),
                      label: const Text("Gallery"),
                      onPressed: isReadyForAnalysis ? _pickImage : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: Theme.of(context).primaryColor,
                        elevation: 0,
                        side: BorderSide(color: Theme.of(context).primaryColor.withOpacity(0.3)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.camera_alt_rounded),
                      label: const Text("Camera"),
                      onPressed: isReadyForAnalysis ? _takePicture : null,
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
    
    // Responsive scaling for empty state
    final screenHeight = MediaQuery.of(context).size.height;
    final isSmallScreen = screenHeight < 700;

    final double containerPadding = isSmallScreen ? 24 : 32; 
    final double iconContainerSize = isSmallScreen ? 64 : 80;
    final double verticalSpacing = isSmallScreen ? 16 : 30;
    
    return Container(
      key: const ValueKey('initial'),
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          ScaleTransition(
            scale: _idleAnimation,
            child: Container(
              padding: EdgeInsets.all(containerPadding),
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
              child: Icon(Icons.add_a_photo_rounded, size: iconContainerSize, color: Theme.of(context).primaryColor.withOpacity(0.6)),
            ),
          ),
          SizedBox(height: verticalSpacing),
          Text("Ready to Analyze", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800, color: Theme.of(context).colorScheme.onSurface)),
          const SizedBox(height: 12),
          Text(
            _yoloModel == null 
              ? "Select a model to begin"
              : "Capture or upload an image.",
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.grey.shade600),
          ),
        ]),
      ),
    );
  }

  Widget _buildLoadingModalOverlay() {
    return Stack(
      children: [
        // Backdrop blur
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

  Widget _buildDetectionView() {
    return LayoutBuilder(builder: (context, constraints) {
      if (constraints.maxWidth > 700) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 6, child: _buildResultsImage()),
          const SizedBox(width: 24),
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
          _buildResultsImage(), const SizedBox(height: 24), _buildResultsList(),
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
              const SizedBox(width: 24),
              Expanded(flex: 4, child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 450), child: _buildClassificationListContainer())),
            ],
          );
        } else {
          return Column(children: [
            _buildClassificationImageContainer(),
            const SizedBox(height: 24),
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
            Text("Input Image", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
            IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
          ],
        ),
        const SizedBox(height: 12),
        _buildClassificationImage(),
      ],
    );
  }

  Widget _buildClassificationListContainer() {
    // Responsive Padding for list
    final isSmallScreen = MediaQuery.of(context).size.height < 700;
    
    return Column(
      mainAxisAlignment: MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text("Top Results", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        if (_recognitions.isEmpty)
          const Card(child: Padding(padding: EdgeInsets.all(24.0), child: Center(child: Text("No confident results."))))
        else
          _buildClassificationList(isSmallScreen),
      ],
    );
  }
  
  Widget _buildResultsImage() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text("Analysis Result", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
      ],
    ),
    const SizedBox(height: 12),
    Card(elevation: 0, color: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20), side: BorderSide(color: Colors.grey.shade200)), clipBehavior: Clip.antiAlias, child: _recognitions.isEmpty && !_isLoading ? _buildNoDetectionsFound() : _buildImageWithDetections())
  ]);

  Widget _buildResultsList() {
    final isSmallScreen = MediaQuery.of(context).size.height < 700;
    
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      if (_recognitions.isNotEmpty && _selectedTask == AppYoloTask.segment) ...[
        _buildInteractiveControls(), 
        const SizedBox(height: 24)
      ],
      Text("Detected Objects (${_recognitions.length})", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
      const SizedBox(height: 12),
      _buildDetectionList(isSmallScreen),
    ]);
  }
  
  Widget _buildClassificationImage() {
    if (_imageFile == null) return const SizedBox.shrink();
    return Card(
      elevation: 0,
      color: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20), side: BorderSide(color: Colors.grey.shade200)),
      clipBehavior: Clip.antiAlias,
      child: Image.file(_imageFile!),
    );
  }
  
  Widget _buildClassificationList(bool isSmallScreen) {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _recognitions.length,
      itemBuilder: (context, index) {
        final result = _recognitions[index];
        final className = result['className'] ?? 'Unknown';
        final confidence = (result['confidence'] ?? 0.0) as num;

        return Card(
          margin: EdgeInsets.symmetric(vertical: isSmallScreen ? 4 : 6),
          child: Padding(
            padding: EdgeInsets.all(isSmallScreen ? 12.0 : 16.0),
            child: Row(
              children: [
                Container(
                  width: 48, height: 48,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(color: Theme.of(context).primaryColor.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                  child: Text(
                    '#${index + 1}',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(className, style: TextStyle(fontSize: isSmallScreen ? 15 : 16, fontWeight: FontWeight.w700)),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          minHeight: 6,
                          value: confidence.toDouble(),
                          backgroundColor: Colors.grey.shade200,
                          valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text('${(confidence * 100).toStringAsFixed(1)}% Confidence', style: TextStyle(fontSize: 12, color: Colors.grey.shade600, fontWeight: FontWeight.w500)),
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

  Widget _buildInteractiveControls() => Card(
    child: Padding(
      padding: const EdgeInsets.all(20.0), 
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Row(children: [
            Icon(Icons.layers_outlined, size: 20, color: Theme.of(context).colorScheme.secondary),
            const SizedBox(width: 10),
            const Text("Show Segmentation Masks", style: TextStyle(fontWeight: FontWeight.w600)),
          ]),
          Switch(
            activeColor: Theme.of(context).primaryColor,
            value: _showMasks, 
            onChanged: (value) => setState(() => _showMasks = value)
          )
        ]),
        const SizedBox(height: 16),
        const Text("Mask Opacity", style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Colors.grey)),
        Slider(
          activeColor: Theme.of(context).primaryColor,
          inactiveColor: Theme.of(context).primaryColor.withOpacity(0.1),
          value: _maskOpacity, min: 0.1, max: 1.0, divisions: 9, 
          label: _maskOpacity.toStringAsFixed(1), 
          onChanged: (value) => setState(() => _maskOpacity = value)
        )
      ]),
    ),
  );

  Widget _buildNoDetectionsFound() => Stack(alignment: Alignment.center, children: [
    if (_imageFile != null) Image.file(_imageFile!),
    Positioned.fill(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 8.0, sigmaY: 8.0),
          child: Container(
            color: Colors.black.withOpacity(0.3),
            alignment: Alignment.center,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.search_off, color: Colors.white, size: 48),
                const SizedBox(height: 12),
                const Text("No objects detected", style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.w600, letterSpacing: 0.5)),
              ],
            ),
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

  Widget _buildDetectionList(bool isSmallScreen) => ListView.builder(
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
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeInOut,
          margin: EdgeInsets.symmetric(vertical: isSmallScreen ? 4 : 6),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: isSelected ? itemColor : Colors.transparent,
              width: 2,
            ),
            boxShadow: [
              BoxShadow(
                color: isSelected ? itemColor.withOpacity(0.2) : Colors.grey.shade100,
                blurRadius: isSelected ? 12 : 4,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Padding(
            padding: EdgeInsets.all(isSmallScreen ? 12.0 : 16.0), 
            child: Row(children: [
              Container(
                width: 44, height: 44, alignment: Alignment.center,
                decoration: BoxDecoration(color: itemColor.withOpacity(0.15), borderRadius: BorderRadius.circular(12)),
                child: Text('${index + 1}', style: TextStyle(color: itemColor, fontSize: 18, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(className, style: TextStyle(fontWeight: FontWeight.w700, fontSize: isSmallScreen ? 15 : 16)),
                const SizedBox(height: 4),
                Text('${(confidence * 100).toStringAsFixed(1)}% Confidence', style: TextStyle(color: Colors.grey.shade600, fontSize: 13, fontWeight: FontWeight.w500)),
              ])),
              Icon(isSelected ? Icons.check_circle : Icons.circle_outlined, color: isSelected ? itemColor : Colors.grey.shade300, size: 28),
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
  void _showDeleteConfirmationDialog(String modelName) {
    showDialog(context: context, builder: (ctx) => AlertDialog(
      title: const Text("Confirm Deletion"),
      content: Text("Are you sure you want to delete the local files for '$modelName'? This action cannot be undone."),
      actions: [
        TextButton(child: const Text("Cancel"), onPressed: () => Navigator.of(ctx).pop()),
        TextButton(child: const Text("Delete"), style: TextButton.styleFrom(foregroundColor: Colors.red), onPressed: () {
          Navigator.of(ctx).pop();
          _deleteLocallyStoredModel(modelName);
        }),
      ],
    ));
  }
}

// FIXED: This is the correct painter class that respects aspect ratio
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
    // 1. Calculate the 'fit: BoxFit.contain' rectangle
    final imageSize = Size(originalImage.width.toDouble(), originalImage.height.toDouble());
    final fittedSizes = applyBoxFit(BoxFit.contain, imageSize, size);
    final sourceRect = Alignment.center.inscribe(fittedSizes.source, Rect.fromLTWH(0, 0, imageSize.width, imageSize.height));
    final destinationRect = Alignment.center.inscribe(fittedSizes.destination, Rect.fromLTWH(0, 0, size.width, size.height));

    // 2. Draw the Original Image
    canvas.drawImageRect(
      originalImage,
      sourceRect,
      destinationRect,
      Paint(),
    );

    if (modelImageWidth == 0 || modelImageHeight == 0) return;

    // 3. Calculate Model Padding and Scaling
    final double scale = min(modelImageWidth / imageSize.width, modelImageHeight / imageSize.height);
    final double padX = (modelImageWidth - imageSize.width * scale) / 2.0;
    final double padY = (modelImageHeight - imageSize.height * scale) / 2.0;

    final double scaleToCanvasX = destinationRect.width / imageSize.width;
    final double scaleToCanvasY = destinationRect.height / imageSize.height;

    // 4. Draw Masks
    if (showMasks && maskImage != null) {
      for (int i = 0; i < recognitions.length; i++) {
        if (selectedDetectionIndex != null && i != selectedDetectionIndex) continue;

        final detection = recognitions[i];
        final className = detection['className'] ?? 'Unknown';
        final color = classColorMap[className] ?? Colors.grey;
        final maskPaint = Paint()
          ..colorFilter = ColorFilter.mode(color.withOpacity(maskOpacity), BlendMode.srcIn);
        
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

    // 5. Draw Boxes and Labels
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

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSelected ? 4.0 : 2.5;
      canvas.drawRect(boundingBoxRect, boxPaint);
      
      final confidence = (detection['confidence'] as num? ?? 0.0);
      final textPainter = TextPainter(
        text: TextSpan(
          text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            shadows: [Shadow(color: Colors.black, blurRadius: 4)],
          ),
        ),
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
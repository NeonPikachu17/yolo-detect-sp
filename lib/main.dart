import 'dart:async';
import 'dart:io';
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

// Keys for storing the user's last session settings
const String _prefsKeyLastModelName = "last_used_model_name";
const String _prefsKeyLastTaskType = "last_used_task_type";

/// Enum to manage the selected YOLO task in the app's state.
/// It now includes all three tasks: segment, detect, and classify.
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
      title: 'Vision AI',
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: const Color(0xFF455A64), // Slate Blue
        scaffoldBackgroundColor: const Color(0xFFECEFF1), // Light Grey Background
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF455A64), // Slate Blue Seed
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
          backgroundColor: const Color(0xFFECEFF1), // Match background
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

class _VisionScreenState extends State<VisionScreen> {
  YOLO? _yoloModel;
  File? _imageFile;
  List<Map<String, dynamic>> _recognitions = [];
  bool _isLoading = false;
  String? _selectedModelName;
  String? _loadingMessage;
  double _originalImageHeight = 0;
  double _originalImageWidth = 0;

  int? _selectedDetectionIndex;
  bool _showMasks = true;
  double _maskOpacity = 0.5;
  Map<String, Color> _classColorMap = {};
  Uint8List? _maskPngBytes;

  AppYoloTask _selectedTask = AppYoloTask.segment;
  List<Map<String, String>> _availableModels = [];

  // Vibrant colors for bounding boxes and masks
  final List<Color> _boxColors = [
    Colors.deepOrange, Colors.lightBlue, Colors.amber.shade600, Colors.pink,
    Colors.green, Colors.purple, Colors.red, Colors.teal,
    Colors.indigo, Colors.cyan, Colors.brown, Colors.lime.shade800,
  ];

  @override
  void initState() {
    super.initState();
    _initializeScreenData();
  }
  
  /// Discovers local models and loads the last used model and task.
  Future<void> _initializeScreenData() async {
    _startLoading("Discovering local models...");
    _availableModels = await _discoverLocalModels();
    final prefs = await SharedPreferences.getInstance();
    final lastModelName = prefs.getString(_prefsKeyLastModelName);
    var lastTaskIndex = prefs.getInt(_prefsKeyLastTaskType) ?? AppYoloTask.segment.index;

    // Prevents a range error if the saved task index is no longer valid.
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

  /// Finds all `.tflite` model files in the app's documents directory.
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
  
  /// Imports a new model from the device's storage using a file picker.
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

  /// Loads the selected YOLO model based on the current task type.
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

  /// Deletes the `.tflite` model and associated label file from local storage.
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

  /// Sets the selected image file and original dimensions.
  Future<void> _processImage(XFile image) async {
    final imageBytes = await image.readAsBytes();
    final decodedImage = await decodeImageFromList(imageBytes);

    setState(() {
      _imageFile = File(image.path);
      _recognitions = [];
      _maskPngBytes = null;
      _originalImageWidth = decodedImage.width.toDouble();
      _originalImageHeight = decodedImage.height.toDouble();
      _selectedDetectionIndex = null;
    });

    if (_yoloModel != null) {
      _runInference();
    } else {
      _showSnackBar("Please select and load a model first.", isError: true);
    }
  }
  
  /// Opens the image gallery to pick an image.
  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) await _processImage(image);
  }

  /// Opens the camera to take a picture.
  Future<void> _takePicture() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);
    if (image != null) await _processImage(image);
  }

  /// Runs the YOLO model inference and processes the results based on the task.
  Future<void> _runInference() async {
    if (_imageFile == null || _yoloModel == null) return;
    _startLoading("Analyzing image...");

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final detections = await _yoloModel!.predict(imageBytes);
      if (!mounted) return;

      final double modelImageWidth = (detections['image_width'] as num?)?.toDouble() ?? _originalImageWidth;
      final double modelImageHeight = (detections['image_height'] as num?)?.toDouble() ?? _originalImageHeight;

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
            final className = box['className'];
            formattedRecognitions.add({
              'x1': box['x1'], 'y1': box['y1'], 'x2': box['x2'], 'y2': box['y2'],
              'className': className, 'confidence': box['confidence'],
            });
            if (!tempColorMap.containsKey(className)) {
              tempColorMap[className] = _boxColors[colorIndex % _boxColors.length];
              colorIndex++;
            }
          }
          // Only segmentation tasks will return a mask.
          newMaskPngBytes = (_selectedTask == AppYoloTask.segment) ? detectionMap['maskPng'] : null;
          break;
      }

      setState(() {
        _recognitions = formattedRecognitions;
        _classColorMap = tempColorMap;
        _maskPngBytes = newMaskPngBytes;

        _originalImageWidth = modelImageWidth;
        _originalImageHeight = modelImageHeight;
      });
    } catch (e) {
      _showSnackBar("Error during analysis: $e", isError: true);
    } finally {
      _stopLoading();
    }
  }
  
  // --- Helper Methods for State and UI ---

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
        backgroundColor: isError ? Theme.of(context).colorScheme.error : Colors.green.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  void dispose() {
    _yoloModel?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Vision AI")),
      bottomNavigationBar: _buildBottomActionBar(),
      body: RefreshIndicator(
        onRefresh: _handleRefresh,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          children: <Widget>[
            _buildModelManagementCard(),
            const SizedBox(height: 20),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: _buildContentArea(),
            ),
          ],
        ),
      ),
    );
  }

  // --- UI Widget Builders ---

  /// Builds the card for managing models and selecting the analysis task.
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
                        _selectedModelName ?? "None Selected",
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: _yoloModel != null ? Theme.of(context).primaryColor : null,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                TextButton(
                  onPressed: _showModelSelectionSheet,
                  child: const Text("Change"),
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
              onSelectionChanged: (newSelection) {
                if (_isLoading) return;
                setState(() => _selectedTask = newSelection.first);
                // Reload the model with the new task type if a model is selected
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

  /// Displays a modal sheet to select, import, or delete models.
  void _showModelSelectionSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.6,
        maxChildSize: 0.9,
        builder: (_, controller) => Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text("Available Models", style: Theme.of(context).textTheme.headlineSmall),
              const Divider(height: 24),
              if (_availableModels.isEmpty)
                const Expanded(child: Center(child: Text("No local models found.")))
              else
                Expanded(
                  child: ListView.builder(
                    controller: controller,
                    itemCount: _availableModels.length,
                    itemBuilder: (context, index) {
                      final model = _availableModels[index];
                      return ListTile(
                        title: Text(model['name']!),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        selected: _selectedModelName == model['name'],
                        selectedTileColor: Theme.of(context).primaryColor.withOpacity(0.1),
                        onTap: () {
                          Navigator.of(ctx).pop();
                          _prepareAndLoadModel(model);
                        },
                        trailing: IconButton(
                          icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                          onPressed: () {
                              Navigator.of(ctx).pop();
                            _showDeleteConfirmationDialog(model['name']!);
                          },
                        ),
                      );
                    },
                  ),
                ),
              ElevatedButton.icon(
                icon: const Icon(Icons.note_add_outlined),
                label: const Text("Import New Model"),
                onPressed: () {
                  Navigator.of(ctx).pop();
                  _importModelFromPicker();
                },
                style: ElevatedButton.styleFrom(backgroundColor: Theme.of(context).colorScheme.secondary, foregroundColor: Theme.of(context).colorScheme.onSecondary),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Builds the bottom action bar with Gallery and Camera buttons.
  Widget _buildBottomActionBar() {
    final bool isReadyForAnalysis = _yoloModel != null && !_isLoading;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
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
    );
  }

  /// Determines which content to display: loading, initial placeholder, or results.
  Widget _buildContentArea() {
    if (_isLoading) {
      return Container(
        key: const ValueKey('loading'),
        padding: const EdgeInsets.symmetric(vertical: 50.0),
        child: Center(
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            const CircularProgressIndicator(),
            if (_loadingMessage != null) ...[
              const SizedBox(height: 20),
              Text(_loadingMessage!, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyLarge),
            ]
          ]),
        ),
      );
    }
    if (_imageFile != null) {
      // Show different views based on the selected task.
      if (_selectedTask == AppYoloTask.classify) {
        return _buildClassificationView();
      } else { // Handles both detect and segment tasks
        return _buildDetectionView();
      }
    }
    return Container(
      key: const ValueKey('initial'),
      padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 20),
      child: Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.image_search_outlined, size: 100, color: Colors.grey.shade400),
          const SizedBox(height: 20),
          Text(_yoloModel == null ? "Select a Model" : "Ready to Analyze", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          Text(
            _yoloModel == null 
              ? "Choose a model from the top card to begin."
              : "Use the action bar below to select an image.",
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade600),
          ),
        ]),
      ),
    );
  }

  /// Builds the view for `detect` and `segment` tasks.
  Widget _buildDetectionView() {
    return LayoutBuilder(builder: (context, constraints) {
      // Use a two-column layout on wider screens.
      if (constraints.maxWidth > 700) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 6, child: _buildResultsImage()),
          const SizedBox(width: 20),
          Expanded(flex: 4, child: _buildResultsList()),
        ]);
      } else {
        return Column(children: [
          _buildResultsImage(), const SizedBox(height: 20), _buildResultsList(),
        ]);
      }
    });
  }

  /// Builds the view for the `classify` task.
  Widget _buildClassificationView() {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text("Analysis Result", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
            IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
          ],
        ),
        const SizedBox(height: 12),
        _recognitions.isEmpty
            ? Card(
                elevation: 4,
                shadowColor: Colors.black.withOpacity(0.2),
                clipBehavior: Clip.antiAlias,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: _buildNoDetectionsFound(),
              )
            : Column(
                children: [
                  _buildClassificationImage(),
                  const SizedBox(height: 20),
                  _buildClassificationList(),
                ],
              ),
      ],
    );
  }
  
  /// Displays the processed image with drawn bounding boxes and masks.
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

  /// Displays the list of detected objects.
  Widget _buildResultsList() => Column(children: [
      // Interactive controls are only shown for segmentation tasks.
      if (_recognitions.isNotEmpty && _selectedTask == AppYoloTask.segment) ...[
        _buildInteractiveControls(), 
        const SizedBox(height: 20)
      ],
      Text("Detected Objects: ${_recognitions.length}", style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
      const SizedBox(height: 10),
      _buildDetectionList(),
    ]);
  
  /// Displays the image without overlays for classification.
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
  
  /// Displays the top 5 classification results in a styled list.
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
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Theme.of(context).primaryColor,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        className,
                        style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 5),
                      LinearProgressIndicator(
                        value: confidence.toDouble(),
                        backgroundColor: Colors.grey.shade300,
                        valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                      ),
                      const SizedBox(height: 5),
                      Text(
                        '${(confidence * 100).toStringAsFixed(1)}% Confidence',
                        style: TextStyle(fontSize: 13, color: Colors.grey.shade700),
                      ),
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

  /// Builds controls for mask visibility and opacity.
  Widget _buildInteractiveControls() => Card(elevation: 2, shadowColor: Colors.black.withOpacity(0.1), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), child: Padding(padding: const EdgeInsets.all(12.0), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [const Text("Show Masks", style: TextStyle(fontWeight: FontWeight.bold)), Switch(value: _showMasks, onChanged: (value) => setState(() => _showMasks = value))]),
      const SizedBox(height: 8),
      const Text("Mask Opacity", style: TextStyle(fontWeight: FontWeight.bold)),
      Slider(value: _maskOpacity, min: 0.1, max: 1.0, divisions: 9, label: _maskOpacity.toStringAsFixed(1), onChanged: (value) => setState(() => _maskOpacity = value))
    ])));

  /// Shows an overlay on the image when no objects are detected.
  Widget _buildNoDetectionsFound() => Stack(alignment: Alignment.center, children: [
      if (_imageFile != null) Image.file(_imageFile!),
      Container(
        color: Colors.black.withOpacity(0.6),
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        child: const Text("No objects detected", style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold))
      )
    ]);

  /// Chooses the correct painter based on whether masks are available.
  Widget _buildImageWithDetections() {
    if (_imageFile == null) return const SizedBox.shrink();
    
    // If no mask data, use the simpler painter with only boxes.
    if (_maskPngBytes == null) {
      return _buildImageWithBoxesOnly();
    }

    // Otherwise, build the full painter with masks.
    return LayoutBuilder(builder: (context, constraints) {
      if (_originalImageWidth == 0) return const SizedBox.shrink();
      final scaleRatio = constraints.maxWidth / _originalImageWidth;
      return FutureBuilder<List<ui.Image>>(
        future: Future.wait([_loadImage(_imageFile!), _loadImageFromBytes(_maskPngBytes!)]),
        builder: (context, snapshot) {
          if (!snapshot.hasData || snapshot.data!.length < 2) return SizedBox(width: constraints.maxWidth, height: _originalImageHeight * scaleRatio, child: const Center(child: CircularProgressIndicator()));
          return CustomPaint(size: Size(constraints.maxWidth, _originalImageHeight * scaleRatio), painter: _DetectionPainter(originalImage: snapshot.data![0], maskImage: snapshot.data![1], recognitions: _recognitions, classColorMap: _classColorMap, scaleRatio: scaleRatio, showMasks: _showMasks, maskOpacity: _maskOpacity, selectedDetectionIndex: _selectedDetectionIndex));
        }
      );
    });
  }

  /// Builds the painter for images with only bounding boxes (no masks).
  Widget _buildImageWithBoxesOnly() => LayoutBuilder(builder: (context, constraints) {
    if (_originalImageWidth == 0) return const SizedBox.shrink();
    final scaleRatio = constraints.maxWidth / _originalImageWidth;
    return FutureBuilder<ui.Image>(
      future: _loadImage(_imageFile!),
      builder: (context, snapshot) {
        if (!snapshot.hasData) return const SizedBox.shrink();
        return CustomPaint(size: Size(constraints.maxWidth, _originalImageHeight * scaleRatio), painter: _DetectionPainter(originalImage: snapshot.data!, recognitions: _recognitions, classColorMap: _classColorMap, scaleRatio: scaleRatio, showMasks: false, maskOpacity: 0, selectedDetectionIndex: _selectedDetectionIndex));
      }
    );
  });

  /// Builds the list of detected items, allowing for selection to highlight.
  Widget _buildDetectionList() => ListView.builder(shrinkWrap: true, physics: const NeverScrollableScrollPhysics(), itemCount: _recognitions.length, itemBuilder: (context, index) {
    final detection = _recognitions[index];
    final className = detection['className'] ?? 'Unknown';
    final confidence = (detection['confidence'] as num).toDouble();
    final isSelected = _selectedDetectionIndex == index;
    final itemColor = _classColorMap[className] ?? Colors.grey.shade700;
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
      elevation: isSelected ? 8 : 2,
      shadowColor: isSelected ? itemColor.withOpacity(0.5) : Colors.black.withOpacity(0.1),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12), side: isSelected ? BorderSide(color: itemColor, width: 2.5) : BorderSide(color: Colors.grey.shade300, width: 1)),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () => setState(() => _selectedDetectionIndex = isSelected ? null : index),
        child: Padding(padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12), child: Row(children: [
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
        ])),
      ),
    );
  });
  
  // --- Utility Functions ---

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

/// Custom painter to draw the original image, segmentation masks, and bounding boxes.
class _DetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final double scaleRatio;
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;
  final Map<String, Color> classColorMap;

  _DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.scaleRatio,
    required this.showMasks,
    required this.maskOpacity,
    required this.classColorMap,
    this.selectedDetectionIndex,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    // Draw the original image as the background.
    paintImage(canvas: canvas, rect: Rect.fromLTWH(0, 0, size.width, size.height), image: originalImage, fit: BoxFit.fill);
    
    // Draw segmentation masks if available and enabled.
    if (showMasks && maskImage != null && recognitions.isNotEmpty) {
      canvas.save();
      final stencilPath = Path();
      // If an item is selected, only show its mask. Otherwise, show all.
      if (selectedDetectionIndex != null) {
          final detection = recognitions[selectedDetectionIndex!];
          stencilPath.addRect(Rect.fromLTRB((detection['x1'] ?? 0) * scaleRatio, (detection['y1'] ?? 0) * scaleRatio, (detection['x2'] ?? 0) * scaleRatio, (detection['y2'] ?? 0) * scaleRatio));
      } else {
        for (final detection in recognitions) {
          stencilPath.addRect(Rect.fromLTRB((detection['x1'] ?? 0) * scaleRatio, (detection['y1'] ?? 0) * scaleRatio, (detection['x2'] ?? 0) * scaleRatio, (detection['y2'] ?? 0) * scaleRatio));
        }
      }
      canvas.clipPath(stencilPath);
      final maskPaint = Paint()..color = Colors.white.withOpacity(maskOpacity);
      canvas.drawImageRect(maskImage!, Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()), Rect.fromLTWH(0, 0, size.width, size.height), maskPaint);
      canvas.restore();
    }
    
    // Draw bounding boxes and labels for all detections.
    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';
      final color = classColorMap[className] ?? Colors.grey;
      final isSelected = i == selectedDetectionIndex;
      
      final x1 = (detection['x1'] ?? 0) * scaleRatio;
      final y1 = (detection['y1'] ?? 0) * scaleRatio;
      final x2 = (detection['x2'] ?? 0) * scaleRatio;
      final y2 = (detection['y2'] ?? 0) * scaleRatio;

      final boxPaint = Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = isSelected ? 4.0 : 2.5;
      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final confidence = (detection['confidence'] as num? ?? 0.0);
      final textPainter = TextPainter(text: TextSpan(text: '$className (${(confidence * 100).toStringAsFixed(1)}%)', style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold, shadows: [Shadow(color: Colors.black, blurRadius: 4)])), textDirection: TextDirection.ltr);
      textPainter.layout(minWidth: 0, maxWidth: size.width);
      final labelBackgroundPaint = Paint()..color = color.withOpacity(isSelected ? 1.0 : 0.8);
      double top = y1 - textPainter.height - 4;
      if (top < 0) top = y2 + 2; // Move label to bottom if it doesn't fit on top
      final finalLabelRect = Rect.fromLTWH(x1, top, textPainter.width + 8, textPainter.height + 4);
      canvas.drawRect(finalLabelRect, labelBackgroundPaint);
      textPainter.paint(canvas, Offset(x1 + 4, top + 2));
    }
  }
  
  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) => 
      originalImage != oldDelegate.originalImage || 
      maskImage != oldDelegate.maskImage || 
      recognitions != oldDelegate.recognitions || 
      showMasks != oldDelegate.showMasks || 
      maskOpacity != oldDelegate.maskOpacity || 
      selectedDetectionIndex != oldDelegate.selectedDetectionIndex;
}
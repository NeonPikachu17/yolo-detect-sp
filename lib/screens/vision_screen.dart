import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart' as p;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../core/app_constants.dart';
import '../services/model_service.dart';
import '../painters/detection_painter.dart';

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
  Future<List<String>>? _cloudModelsFuture;

  // Animation
  late AnimationController _idleAnimationController;
  late Animation<double> _idleAnimation;

  @override
  void initState() {
    super.initState();
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

  // --- Core State Logic ---

  Future<void> _initializeScreenData() async {
    _startLoading("Initializing...");
    _availableModels = await ModelService.discoverLocalModels();
    
    final prefs = await SharedPreferences.getInstance();
    final lastModelName = prefs.getString(AppConstants.prefsKeyLastModelName);
    var lastTaskIndex = prefs.getInt(AppConstants.prefsKeyLastTaskType) ?? AppYoloTask.segment.index;

    if (lastTaskIndex >= AppYoloTask.values.length) lastTaskIndex = AppYoloTask.segment.index;

    setState(() => _selectedTask = AppYoloTask.values[lastTaskIndex]);

    if (lastModelName != null && _availableModels.any((m) => m['name'] == lastModelName)) {
      final modelData = _availableModels.firstWhere((m) => m['name'] == lastModelName);
      await _prepareAndLoadModel(modelData);
    } else {
      _stopLoading();
    }
  }

  Future<void> _prepareAndLoadModel(Map<String, String> modelData) async {
    _clearScreen();
    _startLoading("Loading ${modelData['name']}...");

    try {
      final targetModelPath = modelData['modelPath']!;

      if (!await File(targetModelPath).exists()) {
        throw Exception("Model file not found on device.");
      }

      await Future.delayed(const Duration(milliseconds: 100));

      if (_yoloModel != null) await _yoloModel!.dispose();
      
      late YOLOTask yoloTask;
      switch (_selectedTask) {
        case AppYoloTask.segment: yoloTask = YOLOTask.segment; break;
        case AppYoloTask.detect: yoloTask = YOLOTask.detect; break;
        case AppYoloTask.classify: yoloTask = YOLOTask.classify; break;
      }

      // Strip the .tflite extension specifically for the YOLO package
      final String modelPathWithoutExtension = p.withoutExtension(targetModelPath);

      _yoloModel = YOLO(
        modelPath: modelPathWithoutExtension,
        task: yoloTask,
        // Apparently this is needed for new versions to avoid problems
        useMultiInstance: true,
      );
      
      await _yoloModel?.loadModel();

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(AppConstants.prefsKeyLastModelName, modelData['name']!);
      await prefs.setInt(AppConstants.prefsKeyLastTaskType, _selectedTask.index);

      if (mounted) {
        setState(() => _selectedModelName = modelData['name']);
        _showSnackBar("'${modelData['name']}' loaded successfully.", isError: false);
      }
    } catch (e) {
      debugPrint("====== YOLO NATIVE LOAD ERROR ======");
      debugPrint(e.toString()); 
      debugPrint("====================================");
      
      _showSnackBar("Failed to load model: Check debug console.", isError: true);
      if (mounted) setState(() { _yoloModel = null; _selectedModelName = null; });
    } finally {
      _stopLoading();
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
      _selectedDetectionIndex = null;
    });

    if (_yoloModel != null) {
      _runInference();
    } else {
      _showSnackBar("Please select and load a model first.", isError: true);
    }
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
          final Map? nestedMap = (detections as Map?)?['classification'] as Map?;
          if (nestedMap != null) {
            final List<dynamic>? top5Classes = nestedMap['top5Classes'];
            final List<dynamic>? top5Confidences = nestedMap['top5Confidences'];

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
          final List<dynamic> boxes = (detections['boxes'] as List<dynamic>?) ?? [];
          for (var box in boxes) {
            double x1 = (box['x1'] as num).toDouble();
            double y1 = (box['y1'] as num).toDouble();
            double x2 = (box['x2'] as num).toDouble();
            double y2 = (box['y2'] as num).toDouble();
            
            if (x1 <= 1.0 && y1 <= 1.0 && x2 <= 1.0 && y2 <= 1.0) {
              x1 *= modelWidth; y1 *= modelHeight; x2 *= modelWidth; y2 *= modelHeight;
            }
            
            final className = box['className'];
            formattedRecognitions.add({
              'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
              'className': className, 'confidence': box['confidence'],
            });
            if (!tempColorMap.containsKey(className)) {
              tempColorMap[className] = AppConstants.boxColors[colorIndex % AppConstants.boxColors.length];
              colorIndex++;
            }
          }
          newMaskPngBytes = (_selectedTask == AppYoloTask.segment) ? detections['maskPng'] : null;
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

  // --- Storage Delegation Methods ---

  Future<void> _importModel() async {
    _startLoading("Importing model...");
    try {
      final result = await ModelService.pickModelFile();
      if (result == null || result.files.single.path == null) {
         _stopLoading();
         return;
      }
      final file = result.files.single;

      if (file.extension?.toLowerCase() != 'tflite') {
        throw Exception("Invalid file type. Please select a .tflite model.");
      }

      await ModelService.importModelFile(file);
      _showSnackBar("'${file.name}' imported successfully!", isError: false);
      await _handleRefresh();
    } catch (e) {
      _showSnackBar(e.toString().replaceAll("Exception: ", ""), isError: true);
      _stopLoading();
    }
  }

  Future<void> _deleteModel(String modelName) async {
    final modelData = _availableModels.firstWhere((m) => m['name'] == modelName);
    await ModelService.deleteLocallyStoredModel(modelData['modelPath']!, modelData['labelsPath']!);

    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(AppConstants.prefsKeyLastModelName) == modelName) {
      await prefs.remove(AppConstants.prefsKeyLastModelName);
    }
    
    if (_selectedModelName == modelName) {
      _clearScreen();
      setState(() { _yoloModel = null; _selectedModelName = null; });
    }
    await _handleRefresh();
    _showSnackBar("Deleted '$modelName'.", isError: false);
  }

  Future<void> _downloadModel(String modelName) async {
    _startLoading("Downloading '$modelName'...");
    try {
      await ModelService.downloadModel(modelName);
      _showSnackBar("'$modelName' downloaded successfully.", isError: false);
      await _handleRefresh();
    } catch (e) {
      _showSnackBar(e.toString().replaceAll("Exception: ", ""), isError: true);
    } finally {
      _stopLoading();
    }
  }

  Future<void> _uploadModel(String modelName) async {
    final modelData = _availableModels.firstWhere((m) => m['name'] == modelName);
    _startLoading("Uploading '$modelName'...");
    try {
      await ModelService.uploadModel(modelName, modelData['modelPath']!, modelData['labelsPath']!);
      _showSnackBar("'$modelName' uploaded successfully.", isError: false);
      setState(() => _cloudModelsFuture = ModelService.fetchCloudModels());
    } catch (e) {
      _showSnackBar("Error uploading model: $e", isError: true);
    } finally {
      _stopLoading();
    }
  }

  // --- UI Helpers & Image Handlers ---

  Future<void> _pickImage() async {
    final image = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (image != null) await _processImage(image);
  }

  Future<void> _takePicture() async {
    final image = await ImagePicker().pickImage(source: ImageSource.camera);
    if (image != null) await _processImage(image);
  }

  Future<void> _handleRefresh() async {
    _clearScreen();
    await _initializeScreenData();
  }

  void _startLoading(String msg) => setState(() { _isLoading = true; _loadingMessage = msg; });
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
                child: CustomScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  slivers: [
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
                    SliverToBoxAdapter(
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 120),
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
                              fontSize: isSmallPhone ? 18 : 22,
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
    setState(() { _cloudModelsFuture = ModelService.fetchCloudModels(); });

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
                      Navigator.of(modalContext).pop(); 
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
                              onPressed: () => _uploadModel(model['name']!),
                            ),
                            IconButton(
                              icon: Icon(Icons.delete_outline, color: Theme.of(context).colorScheme.error),
                              onPressed: () {
                                Navigator.of(modalContext).pop(); 
                                showDialog(context: context, builder: (ctx) => AlertDialog(
                                  title: const Text("Confirm Deletion"),
                                  content: Text("Delete '${model['name']}'? This cannot be undone."),
                                  actions: [
                                    TextButton(child: const Text("Cancel"), onPressed: () => Navigator.of(ctx).pop()),
                                    TextButton(child: const Text("Delete", style: TextStyle(color: Colors.red)), onPressed: () { Navigator.of(ctx).pop(); _deleteModel(model['name']!); }),
                                  ],
                                ));
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
              Navigator.of(modalContext).pop();
              _importModel();
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
                            Navigator.of(modalContext).pop();
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
    if (_imageFile != null) return _selectedTask == AppYoloTask.classify ? _buildClassificationView() : _buildDetectionView();
    
    final screenHeight = MediaQuery.of(context).size.height;
    final isSmallScreen = screenHeight < 700;
    final double containerPadding = isSmallScreen ? 24 : 32; 
    final double iconContainerSize = isSmallScreen ? 64 : 80;
    final double verticalSpacing = isSmallScreen ? 16 : 30;
    
    return Container(
      key: const ValueKey('initial'),
      height: screenHeight * 0.5, // Added height to keep it centered
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
            _yoloModel == null ? "Select a model to begin" : "Capture or upload an image.",
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
    // Use MediaQuery instead of LayoutBuilder
    final isWideScreen = MediaQuery.of(context).size.width > 700;
    
    if (isWideScreen) {
      return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(flex: 6, child: _buildResultsImage()),
        const SizedBox(width: 24),
        Expanded(flex: 4, child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 450), child: _buildResultsList())),
      ]);
    } else {
      return Column(children: [_buildResultsImage(), const SizedBox(height: 24), _buildResultsList()]);
    }
  }

  Widget _buildClassificationView() {
    // Use MediaQuery instead of LayoutBuilder
    final isWideScreen = MediaQuery.of(context).size.width > 700;
    
    if (isWideScreen) {
      return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(flex: 5, child: _buildClassificationImageContainer()),
        const SizedBox(width: 24),
        Expanded(flex: 4, child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 450), child: _buildClassificationListContainer())),
      ]);
    } else {
      return Column(children: [_buildClassificationImageContainer(), const SizedBox(height: 24), _buildClassificationListContainer()]);
    }
  }
  
  Widget _buildClassificationImageContainer() => Column(children: [
    Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text("Input Image", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        IconButton(onPressed: _clearScreen, icon: const Icon(Icons.close_rounded), tooltip: "Clear Image"),
      ],
    ),
    const SizedBox(height: 12),
    _buildClassificationImage(),
  ]);

  Widget _buildClassificationListContainer() {
    final isSmallScreen = MediaQuery.of(context).size.height < 700;
    return Column(
      mainAxisAlignment: MainAxisAlignment.start, crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text("Top Results", style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        if (_recognitions.isEmpty) const Card(child: Padding(padding: EdgeInsets.all(24.0), child: Center(child: Text("No confident results.")))) else _buildClassificationList(isSmallScreen),
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
    return Card(elevation: 0, color: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20), side: BorderSide(color: Colors.grey.shade200)), clipBehavior: Clip.antiAlias, child: Image.file(_imageFile!));
  }
  
  Widget _buildClassificationList(bool isSmallScreen) {
    return ListView.builder(
      shrinkWrap: true, physics: const NeverScrollableScrollPhysics(), itemCount: _recognitions.length,
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
                  width: 48, height: 48, alignment: Alignment.center,
                  decoration: BoxDecoration(color: Theme.of(context).primaryColor.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                  child: Text('#${index + 1}', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor)),
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
                        child: LinearProgressIndicator(minHeight: 6, value: confidence.toDouble(), backgroundColor: Colors.grey.shade200, valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor)),
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
          Switch(activeColor: Theme.of(context).primaryColor, value: _showMasks, onChanged: (v) => setState(() => _showMasks = v))
        ]),
        const SizedBox(height: 16),
        const Text("Mask Opacity", style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Colors.grey)),
        Slider(activeColor: Theme.of(context).primaryColor, inactiveColor: Theme.of(context).primaryColor.withOpacity(0.1), value: _maskOpacity, min: 0.1, max: 1.0, divisions: 9, label: _maskOpacity.toStringAsFixed(1), onChanged: (v) => setState(() => _maskOpacity = v))
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
            color: Colors.black.withOpacity(0.3), alignment: Alignment.center,
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
            painter: DetectionPainter(
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
          painter: DetectionPainter(
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
    shrinkWrap: true, physics: const NeverScrollableScrollPhysics(), itemCount: _recognitions.length, 
    itemBuilder: (context, index) {
      final detection = _recognitions[index];
      final className = detection['className'] ?? 'Unknown';
      final confidence = (detection['confidence'] as num).toDouble();
      final isSelected = _selectedDetectionIndex == index;
      final itemColor = _classColorMap[className] ?? Colors.grey.shade700;
      
      return GestureDetector(
        onTap: () => setState(() => _selectedDetectionIndex = isSelected ? null : index),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200), curve: Curves.easeInOut,
          margin: EdgeInsets.symmetric(vertical: isSmallScreen ? 4 : 6),
          decoration: BoxDecoration(
            color: Colors.white, borderRadius: BorderRadius.circular(16),
            border: Border.all(color: isSelected ? itemColor : Colors.transparent, width: 2),
            boxShadow: [BoxShadow(color: isSelected ? itemColor.withOpacity(0.2) : Colors.grey.shade100, blurRadius: isSelected ? 12 : 4, offset: const Offset(0, 4))],
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
}
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';


// --- CONSTANTS ---
const String _prefsKeyLastModelName = "last_downloaded_model_name";
const String _prefsKeyCachedModelList = "cached_models_list";

enum LoaderType { regular, fourStar, standardFiveStar, limitedFiveStar }

class DetectionProvider extends ChangeNotifier {
  // --- PRIVATE STATE VARIABLES ---
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
  bool _isModelLoadedFromCache = false;
  bool _isConnected = true;
  List<Map<String, dynamic>> _availableModels = [];
  bool _isFetchingModelList = false;
  late StreamSubscription<List<ConnectivityResult>> _connectivitySubscription;
  int _pityCounter5Star = 0;
  int _pityCounter4Star = 0;
  bool _is5050Guaranteed = false;
  LoaderType _loaderType = LoaderType.regular;

  final List<Color> _boxColors = [
    Colors.red, Colors.blue, Colors.green, Colors.yellow.shade700,
    Colors.purple, Colors.orange, Colors.pink, Colors.teal,
    Colors.cyan, Colors.brown, Colors.amber.shade700, Colors.indigo,
    Colors.lime.shade700, Colors.lightGreen.shade700, Colors.deepOrange, Colors.blueGrey
  ];

  // --- PUBLIC GETTERS ---
  bool get isLoading => _isLoading;
  bool get isConnected => _isConnected;
  String? get loadingMessage => _loadingMessage;
  LoaderType get loaderType => _loaderType;
  YOLO? get yoloModel => _yoloModel;
  List<Map<String, dynamic>> get availableModels => _availableModels;
  String? get selectedModelName => _selectedModelName;
  Set<String> get downloadedModelNames => _downloadedModelNames;
  bool get isModelLoadedFromCache => _isModelLoadedFromCache;
  File? get imageFile => _imageFile;
  List<Map<String, dynamic>> get recognitions => _recognitions;
  Uint8List? get annotatedImageBytes => _annotatedImageBytes;
  Uint8List? get maskPngBytes => _maskPngBytes;
  double get originalImageHeight => _originalImageHeight;
  double get originalImageWidth => _originalImageWidth;
  int? get selectedDetectionIndex => _selectedDetectionIndex;
  bool get showMasks => _showMasks;
  double get maskOpacity => _maskOpacity;
  Map<String, Color> get classColorMap => _classColorMap;
  List<Color> get boxColors => _boxColors;
  
  DetectionProvider() {
    _initialize();
  }
  
  // This is a private initializer to keep the constructor clean
  void _initialize() {
    _loadPity();
    initializeScreenData(); // Public method called on init
    _connectivitySubscription = Connectivity().onConnectivityChanged.listen(_updateConnectionStatus);
  }

  // --- PUBLIC METHODS (The API for your UI) ---
  /// Clears the screen of the current image and detection results.
  void clearScreen() {
    _imageFile = null;
    _recognitions = [];
    _annotatedImageBytes = null;
    _maskPngBytes = null;
    _originalImageHeight = 0;
    _originalImageWidth = 0;
    _selectedDetectionIndex = null;
    _classColorMap = {};
    notifyListeners();
  }

  /// Refreshes the model list from Firebase or cache. Called by pull-to-refresh.
  Future<void> initializeScreenData() async {
    _startLoading("Checking for models...");
    
    final connectivityResult = await Connectivity().checkConnectivity();
    _isConnected = connectivityResult.contains(ConnectivityResult.mobile) || connectivityResult.contains(ConnectivityResult.wifi);

    List<Map<String, dynamic>> modelsToShow = [];
    if (_isConnected) {
      try {
        modelsToShow = await _fetchModelsFromStorage();
        await _cacheModelList(modelsToShow);
      } catch (e) {
        modelsToShow = await _loadModelListFromCache();
      }
    } else {
      modelsToShow = await _loadModelListFromCache();
    }
    
    if (modelsToShow.isEmpty) {
      modelsToShow = await _discoverLocalModels();
    }

    _availableModels = modelsToShow;
    _isFetchingModelList = false;
    notifyListeners();

    if (_availableModels.isNotEmpty) {
      await _updateLocalModelStatus();
      await _loadInitialModel();
    } else {
      _stopLoading();
    }
  }

  // --- REFACTORED: `_prepareAndLoadModel` to provide a valid labels path ---
  /// Loads a selected model into memory. Returns a map with a status message and color for the UI.
  Future<Map<String, dynamic>> prepareAndLoadModel(Map<String, dynamic> modelData, {bool isInitialLoad = false}) async {
    if (!isInitialLoad) clearScreen();
    _isModelLoadedFromCache = false;

    final String modelName = modelData['name'] as String;
    final String storagePath = modelData['storagePath'] as String;

    _isLoading = true;
    _loadingMessage = "Preparing model: $modelName";
    notifyListeners();

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

      _loadingMessage = "Loading $modelName into memory...";
      notifyListeners();

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

      // Update state
      _selectedModelName = modelName;
      _currentModelPath = targetModelPath;
      _isModelLoadedFromCache = loadedFromCache;
      
      // Update the list of downloaded models
      await _updateLocalModelStatus();

      _isLoading = false;
      notifyListeners();

      // Return a success message and color for the UI to use
      return {
        'success': true,
        'message': loadedFromCache ? "'$modelName' loaded from cache." : "'$modelName' downloaded successfully.",
        'color': loadedFromCache ? Colors.green.shade700 : Colors.blue.shade700,
        'icon': loadedFromCache ? Icons.storage_rounded : Icons.cloud_download_rounded,
      };

    } catch (e) {
      debugPrint("Error in _prepareAndLoadModel: $e");
      
      // Reset state on failure
      _yoloModel = null;
      _currentModelPath = null;
      _selectedModelName = null;
      _isLoading = false;
      notifyListeners();

      // Return a failure message and color
      return {
        'success': false,
        'message': "Failed to load model: ${e.toString()}",
        'color': Colors.red.shade700,
        'icon': Icons.error_outline_rounded,
      };
    }
  }

  /// Deletes the currently selected model's files from local storage.
  /// Returns the name of the deleted model on success, or null on failure.
  Future<String?> deleteLocallyStoredModel() async {
    if (_selectedModelName == null) return null;

    final modelNameToDelete = _selectedModelName!;
    
    // --- Business Logic: Delete files and update preferences ---
    final modelFile = File(await _getLocalModelPath(modelNameToDelete));
    final labelsFile = File(await _getLocalLabelsPath(modelNameToDelete));

    if (await modelFile.exists()) await modelFile.delete();
    if (await labelsFile.exists()) await labelsFile.delete();

    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(_prefsKeyLastModelName) == modelNameToDelete) {
      await prefs.remove(_prefsKeyLastModelName);
    }
    
    // --- State Update Logic ---
    clearScreen();
    _yoloModel = null;
    _currentModelPath = null;
    _selectedModelName = null;
    _isModelLoadedFromCache = false;
    _downloadedModelNames.remove(modelNameToDelete);

    // Notify the UI that the state has changed.
    notifyListeners();

    // Return the name so the UI can show a confirmation.
    return modelNameToDelete;
  }

    /// Handles the entire workflow: picking an image, updating state, and running segmentation.
  /// The UI should call this single method.
  Future<void> pickImageAndAnalyze() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image == null) {
      debugPrint("No image selected.");
      return; // User cancelled the picker
    }

    // Update image state first
    final imageBytes = await image.readAsBytes();
    final decodedImage = await decodeImageFromList(imageBytes);
    
    clearScreen(); 

    _imageFile = File(image.path);
    _originalImageWidth = decodedImage.width.toDouble();
    _originalImageHeight = decodedImage.height.toDouble();
    
    // Now, call the private method to do the analysis
    await _runSegmentation();
  }

  /// Updates the state for which detection is currently highlighted or selected.
  /// Toggles selection off if the same index is provided again.
  void selectDetection(int? index) {
    // If the tapped index is already selected, deselect it (set to null).
    // Otherwise, select the new index.
    _selectedDetectionIndex = (_selectedDetectionIndex == index) ? null : index;
    notifyListeners();
  }

  /// Updates the visibility of segmentation masks based on the switch value.
  void setShowMasks(bool value) {
    _showMasks = value;
    notifyListeners();
  }

  /// Updates the opacity of segmentation masks based on the slider value.
  void setMaskOpacity(double value) {
    _maskOpacity = value;
    notifyListeners();
  }


  // --- PRIVATE METHOD ---

  /// Runs the YOLO model prediction. This is now a private helper method.
  /// It is only called internally by `pickImageAndAnalyze`.
  Future<void> _runSegmentation() async {
    if (_imageFile == null || _yoloModel == null) return;

    _startLoading("Analyzing image...");
    // _startLoading already calls notifyListeners()

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final double confThreshold = 0.5;
      final double nmsThreshold = 0.5;

      final detections = await _yoloModel!.predict(
        imageBytes,
        confidenceThreshold: confThreshold,
        iouThreshold: nmsThreshold,
      );

      final List<dynamic> boxes = detections['boxes'] ?? [];
      final List<Map<String, dynamic>> formattedRecognitions = [];
      final tempColorMap = <String, Color>{};
      int colorIndex = 0;

      for (int i = 0; i < boxes.length; i++) {
        final box = boxes[i];
        final className = box['className'];
        formattedRecognitions.add({
          'x1': box['x1'], 'y1': box['y1'],
          'x2': box['x2'], 'y2': box['y2'],
          'className': className,
          'confidence': box['confidence'],
        });

        if (!tempColorMap.containsKey(className)) {
          tempColorMap[className] = _boxColors[colorIndex % _boxColors.length];
          colorIndex++;
        }
      }
      
      // Update the rest of the state with results
      _recognitions = formattedRecognitions;
      _classColorMap = tempColorMap;
      _maskPngBytes = detections['maskPng'];
      _annotatedImageBytes = detections['annotatedImage'];

    } catch (e) {
      debugPrint("Error running segmentation: $e");
      // Optionally, you could set an error message state here for the UI to display
      // _errorMessage = "Error during analysis: $e";
    } finally {
      // This will stop the loading indicator and notify listeners to update the UI with the results
      _stopLoading();
    }
  }

    // --- PRIVATE LOGIC METHODS (Internal Helpers) ---

  void _startLoading(String message) {
    _pityCounter5Star++;
    _pityCounter4Star++;
    debugPrint("5-Star Pity: $_pityCounter5Star | 4-Star Pity: $_pityCounter4Star | Guaranteed: $_is5050Guaranteed");

    bool is5StarPull = false;
    double baseRate5Star = 0.006;
    double softPityIncrease = 0.06;
    LoaderType currentLoaderType;

    // 1. Check for 5-Star
    if (_pityCounter5Star >= 90) is5StarPull = true;
    else if (_pityCounter5Star > 73) {
      if (Random().nextDouble() < baseRate5Star + (_pityCounter5Star - 73) * softPityIncrease) is5StarPull = true;
    } else {
      if (Random().nextDouble() < baseRate5Star) is5StarPull = true;
    }
    
    if (is5StarPull) {
      if (_is5050Guaranteed || Random().nextBool()) {
        currentLoaderType = LoaderType.limitedFiveStar;
        _is5050Guaranteed = false;
      } else {
        currentLoaderType = LoaderType.standardFiveStar;
        _is5050Guaranteed = true;
      }
      _pityCounter5Star = 0;
      _pityCounter4Star = 0;
    } else {
      // 2. Check for 4-Star
      bool is4StarPull = false;
      double baseRate4Star = 0.051;
      if (_pityCounter4Star >= 10) is4StarPull = true;
      else if (Random().nextDouble() < baseRate4Star) is4StarPull = true;

      if (is4StarPull) {
        currentLoaderType = LoaderType.fourStar;
        _pityCounter4Star = 0;
      } else {
        currentLoaderType = LoaderType.regular;
      }
    }

    _savePity();

    // Update state and notify UI
    _loadingMessage = message;
    _isLoading = true;
    _loaderType = currentLoaderType;
    notifyListeners();
  }

  /// Handles hiding the loading indicator.
  void _stopLoading() {
    _isLoading = false;
    _loadingMessage = null;
    notifyListeners();
  }

  /// The callback for the connectivity stream.
  void _updateConnectionStatus(List<ConnectivityResult> results) {
    final bool currentlyConnected = results.contains(ConnectivityResult.mobile) || results.contains(ConnectivityResult.wifi);
    if (_isConnected != currentlyConnected) {
      _isConnected = currentlyConnected;
      // The UI will listen for this change and can display a SnackBar itself.
      notifyListeners();
    }
  }

  /// Loads the gacha pity counters from SharedPreferences.
  Future<void> _loadPity() async {
    final prefs = await SharedPreferences.getInstance();
    _pityCounter5Star = prefs.getInt('pity_counter_5_star') ?? 0;
    _pityCounter4Star = prefs.getInt('pity_counter_4_star') ?? 0;
    _is5050Guaranteed = prefs.getBool('is_5050_guaranteed') ?? false;
    // No need to call notifyListeners() here as this is part of initial setup.
  }

  /// Saves the gacha pity counters to SharedPreferences.
  Future<void> _savePity() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('pity_counter_5_star', _pityCounter5Star);
    await prefs.setInt('pity_counter_4_star', _pityCounter4Star);
    await prefs.setBool('is_5050_guaranteed', _is5050Guaranteed);
  }

  /// Checks local storage to see which models from the available list are downloaded.
  Future<void> _updateLocalModelStatus() async {
    if (_availableModels.isEmpty) return;
    
    final Set<String> localNames = {};
    for (final model in _availableModels) {
      final modelName = model['name'] as String;
      final modelPath = await _getLocalModelPath(modelName);
      // For this check, we only need to see if the main model file exists.
      if (await File(modelPath).exists()) {
        localNames.add(modelName);
      }
    }
    _downloadedModelNames = localNames;
    notifyListeners();
  }

  /// Uploads a new .tflite model file to Firebase Storage.
  /// Returns true on success and false on failure.
  Future<bool> uploadModel(String modelName, File tfliteFile) async {
    // Prevent upload if the name already exists in the available models
    if (_availableModels.any((m) => m['name'] == modelName)) {
      debugPrint("A model named '$modelName' already exists.");
      return false;
    }

    _startLoading("Uploading $modelName...");

    try {
      final modelFolderPath = 'yoloModels/$modelName';

      // Upload the .tflite file
      final modelRef = FirebaseStorage.instance.ref('$modelFolderPath/model.tflite');
      await modelRef.putFile(tfliteFile);

      // Create an empty placeholder labels.txt file
      final labelsRef = FirebaseStorage.instance.ref('$modelFolderPath/labels.txt');
      await labelsRef.putString(''); 
      
      // Refresh the model list to show the newly uploaded model
      await initializeScreenData();
      
      _stopLoading();
      return true; // Success

    } catch (e) {
      debugPrint("Error uploading model: $e");
      _stopLoading();
      return false; // Failure
    }
  }

  /// Helper for initializeScreenData that attempts to load the last used model.
  Future<void> _loadInitialModel() async {
    final prefs = await SharedPreferences.getInstance();
    final lastModelName = prefs.getString(_prefsKeyLastModelName);

    if (lastModelName == null) {
      _stopLoading();
      return;
    }

    final modelData = _availableModels.firstWhere(
      (m) => m['name'] == lastModelName,
      orElse: () => <String, dynamic>{},
    );

    if (modelData.isNotEmpty) {
      final String modelPath = await _getLocalModelPath(lastModelName);
      if (await File(modelPath).exists()) {
        // Call the public method to handle loading
        await prepareAndLoadModel(modelData, isInitialLoad: true);
      } else {
        _stopLoading();
      }
    } else {
      _stopLoading();
    }
  }

  /// Fetches the list of available models from Firebase Storage.
  Future<List<Map<String, dynamic>>> _fetchModelsFromStorage() async {
    final List<Map<String, dynamic>> models = [];
    try {
      final listResult = await FirebaseStorage.instance.ref('yoloModels').listAll();
      for (final prefix in listResult.prefixes) {
        models.add({'name': prefix.name, 'storagePath': prefix.fullPath});
      }
      return models;
    } catch (e) {
      debugPrint("Error fetching models from storage: $e");
      return [];
    }
  }

  /// Saves the fetched model list to local cache (SharedPreferences).
  Future<void> _cacheModelList(List<Map<String, dynamic>> models) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKeyCachedModelList, jsonEncode(models));
  }

  /// Retrieves the model list from local cache when offline.
  Future<List<Map<String, dynamic>>> _loadModelListFromCache() async {
    final prefs = await SharedPreferences.getInstance();
    final cachedData = prefs.getString(_prefsKeyCachedModelList);
    if (cachedData != null) {
      final List<dynamic> decoded = jsonDecode(cachedData);
      return decoded.cast<Map<String, dynamic>>().toList();
    }
    return [];
  }

  /// Scans the local directory for any model files as a fallback.
  Future<List<Map<String, dynamic>>> _discoverLocalModels() async {
    final List<Map<String, dynamic>> foundModels = [];
    final docDir = await getApplicationDocumentsDirectory();
    final files = docDir.listSync();
    final modelFiles = files.where((f) => f.path.endsWith('.tflite')).toList();
    for (final modelFile in modelFiles) {
      final modelName = p.basenameWithoutExtension(modelFile.path);
      foundModels.add({'name': modelName, 'storagePath': 'yoloModels/$modelName'});
    }
    return foundModels;
  }

  /// Downloads the .tflite model file and creates a dummy labels file.
  Future<void> _downloadModel(String modelName, String storagePath) async {
    final modelRef = FirebaseStorage.instance.ref('$storagePath/model.tflite');
    final localModelFile = File(await _getLocalModelPath(modelName));
    await modelRef.writeToFile(localModelFile);

    final localLabelsFile = File(await _getLocalLabelsPath(modelName));
    if (!await localLabelsFile.exists()) {
      await localLabelsFile.create();
    }
  }

  /// Gets the local file path for a given model name.
  Future<String> _getLocalModelPath(String modelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$modelName.tflite");
  }

  /// Gets the local file path for a model's (potentially empty) labels file.
  Future<String> _getLocalLabelsPath(String modelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    return p.join(docDir.path, "$modelName.txt");
  }

  @override
  void dispose() {
    _yoloModel?.dispose();
    _connectivitySubscription.cancel();
    super.dispose();
  }

}
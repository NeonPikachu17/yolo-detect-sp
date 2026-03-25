import 'dart:async';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:firebase_storage/firebase_storage.dart';

class ModelService {
  /// Finds all `.tflite` model files in the app's documents directory.
  static Future<List<Map<String, String>>> discoverLocalModels() async {
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
  static Future<FilePickerResult?> pickModelFile() async {
    return await FilePicker.platform.pickFiles(type: FileType.any);
  }

  /// Copies an imported file into the app's documents directory.
  static Future<void> importModelFile(PlatformFile file) async {
    final docDir = await getApplicationDocumentsDirectory();
    final newModelPath = p.join(docDir.path, file.name);

    if (await File(newModelPath).exists()) {
      throw Exception("A model with this name already exists.");
    }

    await File(file.path!).copy(newModelPath);
    final newLabelsPath = p.join(docDir.path, "${p.basenameWithoutExtension(file.name)}.txt");
    if (!await File(newLabelsPath).exists()) await File(newLabelsPath).create();
  }

  /// Deletes the `.tflite` model and associated label file from local storage.
  static Future<void> deleteLocallyStoredModel(String modelPath, String labelsPath) async {
    final modelFile = File(modelPath);
    final labelsFile = File(labelsPath);

    if (await modelFile.exists()) await modelFile.delete();
    if (await labelsFile.exists()) await labelsFile.delete();
  }

  /// Fetches the list of available models from Firebase Storage.
  static Future<List<String>> fetchCloudModels() async {
    try {
      final storageRef = FirebaseStorage.instance.ref().child('yoloModels');
      // Add a 15-second timeout to the network request.
      final listResult = await storageRef.listAll().timeout(const Duration(seconds: 15));
      return listResult.prefixes.map((prefix) => prefix.name).toList();
    } on TimeoutException catch (_) {
      throw Exception("Failed to connect: The request timed out.");
    } catch (e) {
      throw Exception("Failed to fetch cloud models: $e");
    }
  }

  /// Downloads a model and its label file from Firebase Storage.
  static Future<void> downloadModel(String modelName) async {
    final docDir = await getApplicationDocumentsDirectory();
    final localModelPath = p.join(docDir.path, '$modelName.tflite');
    final localLabelsPath = p.join(docDir.path, '$modelName.txt');

    if (await File(localModelPath).exists()) {
      throw Exception("Model '$modelName' already exists locally.");
    }

    final modelRef = FirebaseStorage.instance.ref('yoloModels/$modelName/model.tflite');
    await modelRef.writeToFile(File(localModelPath));

    try {
      final labelsRef = FirebaseStorage.instance.ref('yoloModels/$modelName/labels.txt');
      await labelsRef.writeToFile(File(localLabelsPath));
    } catch (e) {
      await File(localLabelsPath).create();
    }
  }

  /// Uploads a local model and its label file to Firebase Storage.
  static Future<void> uploadModel(String modelName, String modelPath, String labelsPath) async {
    final modelFile = File(modelPath);
    final labelsFile = File(labelsPath);

    final modelRef = FirebaseStorage.instance.ref('yoloModels/$modelName/model.tflite');
    await modelRef.putFile(modelFile);

    if (await labelsFile.exists()) {
      final labelsRef = FirebaseStorage.instance.ref('yoloModels/$modelName/labels.txt');
      await labelsRef.putFile(labelsFile);
    }
  }
}
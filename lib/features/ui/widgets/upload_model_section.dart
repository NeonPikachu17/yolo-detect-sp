import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';

class UploadModelSection extends StatefulWidget {
  const UploadModelSection({super.key});

  @override
  State<UploadModelSection> createState() => _UploadModelSectionState();
}

class _UploadModelSectionState extends State<UploadModelSection> {
  // UI-specific state stays within the widget
  final _modelNameController = TextEditingController();
  File? _selectedTfliteFile;
  String? _selectedTfliteFileName;

  @override
  void dispose() {
    _modelNameController.dispose();
    super.dispose();
  }

  Future<void> _selectFile() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any);
    if (result != null && result.files.single.path != null) {
      final file = result.files.single;
      if (file.extension?.toLowerCase() == 'tflite') {
        setState(() {
          _selectedTfliteFile = File(file.path!);
          _selectedTfliteFileName = file.name;
        });
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text("Invalid file type. Please select a .tflite model file."),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  Future<void> _handleUpload() async {
    final provider = context.read<DetectionProvider>();
    final modelName = _modelNameController.text.trim();

    if (_selectedTfliteFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please select a model file.")));
      return;
    }
    if (modelName.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please enter a model name.")));
      return;
    }

    final success = await provider.uploadModel(modelName, _selectedTfliteFile!);
    
    if (mounted) {
      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("'$modelName' uploaded successfully!")));
        setState(() {
          _modelNameController.clear();
          _selectedTfliteFile = null;
          _selectedTfliteFileName = null;
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Upload failed. Model name might already exist.")));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Only build if connected
    if (!context.watch<DetectionProvider>().isConnected) {
      return const SizedBox.shrink();
    }

    return Card(
      child: ExpansionTile(
        key: const ValueKey('model-upload'),
        leading: Icon(Icons.cloud_upload_outlined, color: Theme.of(context).colorScheme.secondary),
        title: const Text("Upload New Model", style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: const Text("Add a custom .tflite model"),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _modelNameController,
                  decoration: const InputDecoration(
                    labelText: "New Model Unique Name",
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        icon: const Icon(Icons.attach_file),
                        label: Text(_selectedTfliteFileName ?? "Select .tflite File"),
                        onPressed: _selectFile,
                        style: OutlinedButton.styleFrom(
                          minimumSize: const Size(double.infinity, 50),
                          foregroundColor: _selectedTfliteFileName != null
                              ? Colors.green
                              : Theme.of(context).primaryColor,
                        ),
                      ),
                    ),
                    // This button only appears when a file is selected
                    if (_selectedTfliteFile != null)
                      IconButton(
                        icon: const Icon(Icons.close),
                        tooltip: "Clear selection",
                        onPressed: () {
                          setState(() {
                            _selectedTfliteFile = null;
                            _selectedTfliteFileName = null;
                          });
                        },
                      )
                  ],
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.cloud_upload_outlined),
                    label: const Text("Upload to Firebase"),
                    onPressed: context.watch<DetectionProvider>().isLoading ? null : _handleUpload,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
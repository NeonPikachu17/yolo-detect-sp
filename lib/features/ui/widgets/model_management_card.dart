import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';

class ModelManagementCard extends StatelessWidget {
  const ModelManagementCard({super.key});

  @override
  Widget build(BuildContext context) {
    // Use context.watch to rebuild this widget when the provider's state changes.
    final provider = context.watch<DetectionProvider>();
    final theme = Theme.of(context);

    return Card(
      child: ExpansionTile(
        key: const ValueKey('model-management'),
        leading: Icon(Icons.hub_outlined, color: theme.primaryColor),
        title: const Text("Model Management", style: TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(provider.selectedModelName ?? "No model selected"),
        initiallyExpanded: provider.yoloModel == null,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildModelSelector(context), // Helper for the dropdown
                if (provider.yoloModel != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 12.0),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: Colors.green.shade700, size: 18),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            "'${provider.selectedModelName}' is loaded.",
                            style: TextStyle(color: Colors.green.shade800),
                          ),
                        ),
                        if (provider.isModelLoadedFromCache)
                          IconButton(
                            icon: const Icon(Icons.delete_sweep_outlined, color: Colors.orange),
                            tooltip: "Delete model from local cache",
                            onPressed: () => _showDeleteConfirmation(context),
                          ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // This helper builds the dropdown using the provider's data.
  Widget _buildModelSelector(BuildContext context) {
    // Here we use context.read inside a callback, as it doesn't need to listen.
    final provider = context.watch<DetectionProvider>();

    return DropdownButtonFormField<String>(
      decoration: const InputDecoration(
        labelText: "Available Models",
        border: OutlineInputBorder(),
      ),
      hint: const Text("Select a Model"),
      value: provider.selectedModelName,
      isExpanded: true,
      items: provider.availableModels.map((model) {
        final modelName = model['name'] as String;
        final isDownloaded = provider.downloadedModelNames.contains(modelName);
        final bool isSelectable = isDownloaded || provider.isConnected;

        return DropdownMenuItem<String>(
          value: modelName,
          enabled: isSelectable,
          child: Row(
            children: [
              Expanded(child: Text(modelName)),
              if (isDownloaded)
                const Icon(Icons.download_done, color: Colors.green, size: 20)
              else
                Icon(Icons.cloud_outlined, color: Colors.grey, size: 20),
            ],
          ),
        );
      }).toList(),
      onChanged: provider.isLoading ? null : (String? newValue) {
        if (newValue != null && newValue != provider.selectedModelName) {
          final selectedModelData = provider.availableModels.firstWhere((m) => m['name'] == newValue);
          // Call the provider method to handle the logic
          context.read<DetectionProvider>().prepareAndLoadModel(selectedModelData);
        }
      },
    );
  }

  // This UI-specific logic (showing a dialog) belongs with the widget.
  void _showDeleteConfirmation(BuildContext context) {
    final provider = context.read<DetectionProvider>();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Confirm Deletion"),
        content: Text("Are you sure you want to delete the local files for '${provider.selectedModelName}'?"),
        actions: [
          TextButton(child: const Text("Cancel"), onPressed: () => Navigator.of(ctx).pop()),
          TextButton(
            child: const Text("Delete", style: TextStyle(color: Colors.red)),
            onPressed: () {
              // Call the provider's logic method, then close the dialog.
              provider.deleteLocallyStoredModel();
              Navigator.of(ctx).pop();
            },
          ),
        ],
      ),
    );
  }
}
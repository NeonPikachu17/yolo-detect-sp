import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';
import 'package:yolo_detect/features/ui/widgets/upload_model_section.dart';
import 'widgets/model_management_card.dart'; // We will create this
import 'widgets/results_display_area.dart'; // And this

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  @override
  void initState() {
    super.initState();
    // Trigger the initial data load. `listen: false` is important here.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<DetectionProvider>(context, listen: false).initializeScreenData();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Pest Detection"),
        actions: [
          // You can even make this a tiny widget!
          Consumer<DetectionProvider>(
              builder: (context, provider, _) =>
                  Icon(provider.isConnected ? Icons.wifi : Icons.wifi_off)),
          const SizedBox(width: 16),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () => context.read<DetectionProvider>().initializeScreenData(),
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              // High-level, readable layout
              const ModelManagementCard(),
              const SizedBox(height: 16),
              const UploadModelSection(), // This would be the stateful widget
              const SizedBox(height: 24),
              
              Consumer<DetectionProvider>(
                builder: (context, provider, child) => ElevatedButton.icon(
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text("Pick Image & Analyze"),
                  onPressed: (provider.isLoading || provider.yoloModel == null)
                      ? null
                      : provider.pickImageAndAnalyze,
                ),
              ),
              const SizedBox(height: 24),
              
              // The main content area handles all the switching logic internally.
              const ResultsDisplayArea(),
            ],
          ),
        ),
      ),
    );
  }
}
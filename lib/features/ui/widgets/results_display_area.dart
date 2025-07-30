import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';
import 'loading_indicator.dart';
import 'detection_results_view.dart';

class ResultsDisplayArea extends StatelessWidget {
  const ResultsDisplayArea({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<DetectionProvider>();

    Widget content;

    if (provider.isLoading) {
      content = LoadingIndicator(key: const ValueKey('loading'));
    } else if (provider.imageFile != null) {
      content = DetectionResultsView(key: const ValueKey('results'));
    } else {
      content = Container(
        key: const ValueKey('initial'),
        padding: const EdgeInsets.symmetric(vertical: 60, horizontal: 20),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.science_outlined, size: 80, color: Colors.grey.shade400),
              const SizedBox(height: 20),
              Text(
                "Ready to Analyze",
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 10),
              Text(
                "Step 1: Select a model from the dropdown above", // Changed
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade600),
              ),
            ],
          ),
        ),
      );
    }
    
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 500),
      child: content,
    );
  }
}
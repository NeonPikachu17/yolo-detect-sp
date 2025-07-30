import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';
import 'package:yolo_detect/features/ui/widgets/interactive_results.dart';
import 'detection_painter.dart';

class DetectionResultsView extends StatelessWidget {
  const DetectionResultsView({super.key});

  @override
  Widget build(BuildContext context) {
    // Use a LayoutBuilder for a responsive UI
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth > 600) {
          // Side-by-side layout for wide screens
          return const Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(flex: 6, child: _ResultsImage()),
              SizedBox(width: 20),
              Expanded(flex: 4, child: _ResultsList()),
            ],
          );
        } else {
          // Stacked layout for narrow screens
          return const Column(
            children: [
              _ResultsImage(),
              SizedBox(height: 20),
              _ResultsList(),
            ],
          );
        }
      },
    );
  }
}


class _ResultsImage extends StatelessWidget {
  const _ResultsImage();

  Future<ui.Image> _loadImage(dynamic imageSource) async {
    final Completer<ui.Image> completer = Completer();
    final bytes = (imageSource is File) ? await imageSource.readAsBytes() : imageSource as Uint8List;
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<DetectionProvider>();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text("Analysis Result", style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Card(
          elevation: 4,
          shadowColor: Colors.black.withOpacity(0.2),
          clipBehavior: Clip.antiAlias,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: _buildImageWithDetections(context, provider),
        ),
      ],
    );
  }

  Widget _buildImageWithDetections(BuildContext context, DetectionProvider provider) {
    if (provider.imageFile == null) return const SizedBox.shrink();

    if (provider.recognitions.isEmpty) {
      return Stack(
        alignment: Alignment.center,
        children: [
          Image.file(provider.imageFile!),
          Container(
            color: Colors.black.withOpacity(0.6),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            child: const Text(
              "No objects detected",
              style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      );
    }
    
    return LayoutBuilder(
      builder: (context, constraints) {
        if (provider.originalImageWidth == 0) return const SizedBox.shrink();
        final scaleRatio = constraints.maxWidth / provider.originalImageWidth;
        final responsiveHeight = provider.originalImageHeight * scaleRatio;

        return FutureBuilder<List<ui.Image?>>(
          future: Future.wait([
            _loadImage(provider.imageFile!),
            if (provider.maskPngBytes != null) _loadImage(provider.maskPngBytes!),
          ]),
          builder: (context, snapshot) {
            if (!snapshot.hasData || snapshot.data?[0] == null) {
              return SizedBox(
                width: constraints.maxWidth,
                height: responsiveHeight,
                child: const Center(child: CircularProgressIndicator()),
              );
            }
            return CustomPaint(
              size: Size(constraints.maxWidth, responsiveHeight),
              painter: DetectionPainter(
                originalImage: snapshot.data![0]!,
                maskImage: snapshot.data!.length > 1 ? snapshot.data![1] : null,
                recognitions: provider.recognitions,
                boxColors: provider.boxColors,
                scaleRatio: scaleRatio,
                showMasks: provider.showMasks,
                maskOpacity: provider.maskOpacity,
                selectedDetectionIndex: provider.selectedDetectionIndex,
              ),
            );
          },
        );
      },
    );
  }
}

class _ResultsList extends StatelessWidget {
  const _ResultsList();

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<DetectionProvider>();
    final recognitions = provider.recognitions;

    return Column(
      children: [
        if (recognitions.isNotEmpty) ...[
          const InteractiveControls(),
          const SizedBox(height: 20),
        ],
        Text("Detected Objects: ${recognitions.length}", style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 10),
        ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: recognitions.length,
          itemBuilder: (context, index) {
            final detection = recognitions[index];
            final className = detection['className'] ?? 'Unknown';
            final confidence = (detection['confidence'] as num).toDouble();
            final isSelected = provider.selectedDetectionIndex == index;
            final itemColor = provider.classColorMap[className] ?? Colors.grey.shade700;
            
            return Card(
              margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
              elevation: isSelected ? 8 : 2,
              child: InkWell(
                borderRadius: BorderRadius.circular(12),
                onTap: () => context.read<DetectionProvider>().selectDetection(index),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Row(
                    children: [
                      CircleAvatar(
                        backgroundColor: itemColor,
                        child: Text(
                          '${index + 1}',
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(className, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                            Text('${(confidence * 100).toStringAsFixed(1)}% Confidence'),
                          ],
                        ),
                      ),
                      if (isSelected)
                        Icon(Icons.check_circle, color: itemColor),
                    ],
                  ),
                ),
                // THE CODE SHOULD NOT BE HERE
              ),
            );
          },
        ),

        // --- CORRECT LOCATION: After the ListView.builder ---
        const SizedBox(height: 24),
        if (provider.annotatedImageBytes != null) ...[
          Text("Library's Raw Output", style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 10),
          Card(
            elevation: 4,
            shadowColor: Colors.black.withOpacity(0.2),
            clipBehavior: Clip.antiAlias,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Image.memory(provider.annotatedImageBytes!),
          )
        ]
        // --- END OF CORRECT LOCATION ---
      ],
    );
  }
}
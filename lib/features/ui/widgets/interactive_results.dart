import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';

class InteractiveControls extends StatelessWidget {
  const InteractiveControls({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<DetectionProvider>();

    return Card(
      elevation: 2,
      shadowColor: Colors.black.withOpacity(0.1),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text("Show Masks", style: TextStyle(fontWeight: FontWeight.bold)),
                Switch(
                  value: provider.showMasks,
                  onChanged: (value) => context.read<DetectionProvider>().setShowMasks(value),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text("Mask Opacity", style: TextStyle(fontWeight: FontWeight.bold)),
            Slider(
              value: provider.maskOpacity,
              min: 0.1,
              max: 1.0,
              onChanged: (value) => context.read<DetectionProvider>().setMaskOpacity(value),
            ),
          ],
        ),
      ),
    );
  }
}
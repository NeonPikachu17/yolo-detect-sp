import 'dart:ui' as ui;
import 'package:flutter/material.dart';


class DetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final List<Color> boxColors;
  final double scaleRatio;
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;

  DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.boxColors,
    required this.scaleRatio,
    required this.showMasks,
    required this.maskOpacity,
    this.selectedDetectionIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: originalImage,
      fit: BoxFit.fill,
    );

    if (showMasks && maskImage != null && recognitions.isNotEmpty) {
      final stencilPath = Path();
      // Logic to highlight only the selected mask
      if (selectedDetectionIndex != null) {
          final detection = recognitions[selectedDetectionIndex!];
          final rect = Rect.fromLTRB(
              (detection['x1'] as num) * scaleRatio,
              (detection['y1'] as num) * scaleRatio,
              (detection['x2'] as num) * scaleRatio,
              (detection['y2'] as num) * scaleRatio,
          );
          stencilPath.addRect(rect);
      } else { // Or show all masks
          for (final detection in recognitions) {
              final rect = Rect.fromLTRB(
                  (detection['x1'] as num) * scaleRatio,
                  (detection['y1'] as num) * scaleRatio,
                  (detection['x2'] as num) * scaleRatio,
                  (detection['y2'] as num) * scaleRatio,
              );
              stencilPath.addRect(rect);
          }
      }
      
      canvas.save();
      canvas.clipPath(stencilPath);
      final maskPaint = Paint()..color = Colors.white.withOpacity(maskOpacity);
      canvas.drawImageRect(
          maskImage!,
          Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
          Rect.fromLTWH(0, 0, size.width, size.height),
          maskPaint,
        );
      canvas.restore();
    }

    final Map<String, Color> classColorMap = {};
    int colorIndex = 0;

    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';

      if (!classColorMap.containsKey(className)) {
        classColorMap[className] = boxColors[colorIndex % boxColors.length];
        colorIndex++;
      }
      final color = classColorMap[className]!;
      final isSelected = i == selectedDetectionIndex;
      
      final double x1 = (detection['x1'] as num).toDouble() * scaleRatio;
      final double y1 = (detection['y1'] as num).toDouble() * scaleRatio;
      final double x2 = (detection['x2'] as num).toDouble() * scaleRatio;
      final double y2 = (detection['y2'] as num).toDouble() * scaleRatio;
      final double confidence = (detection['confidence'] as num).toDouble();

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSelected ? 4.0 : 2.5; // Highlight selected box
      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final textPainter = TextPainter(
        text: TextSpan(
          text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            shadows: [Shadow(color: Colors.black, blurRadius: 4)]
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);

      final labelBackgroundPaint = Paint()..color = color.withOpacity(isSelected ? 1.0 : 0.8);
      // final labelRect = Rect.fromLTWH(
      //   x1,
      //   y1 - (textPainter.height + 4), // Position label above the box
      //   textPainter.width + 8,
      //   textPainter.height + 4,
      // );
      
      // Ensure label is within view bounds
      double top = y1 - textPainter.height - 4;
      if (top < 0) {
        top = y2 + 2; // If no space above, place it below
      }

      final finalLabelRect = Rect.fromLTWH(x1, top, textPainter.width + 8, textPainter.height + 4);

      canvas.drawRect(finalLabelRect, labelBackgroundPaint);
      
      textPainter.paint(canvas, Offset(x1 + 4, top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return originalImage != oldDelegate.originalImage ||
            maskImage != oldDelegate.maskImage ||
            recognitions != oldDelegate.recognitions ||
            showMasks != oldDelegate.showMasks ||
            maskOpacity != oldDelegate.maskOpacity ||
            selectedDetectionIndex != oldDelegate.selectedDetectionIndex;
  }
}
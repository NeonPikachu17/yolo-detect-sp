import 'dart:math';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class DetectionPainter extends CustomPainter {
  final ui.Image originalImage;
  final ui.Image? maskImage;
  final List<Map<String, dynamic>> recognitions;
  final double modelImageWidth;
  final double modelImageHeight;
  final bool showMasks;
  final double maskOpacity;
  final int? selectedDetectionIndex;
  final Map<String, Color> classColorMap;

  DetectionPainter({
    required this.originalImage,
    this.maskImage,
    required this.recognitions,
    required this.modelImageWidth,
    required this.modelImageHeight,
    required this.classColorMap,
    this.selectedDetectionIndex,
    this.showMasks = false,
    this.maskOpacity = 0.5,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // --- 1. Calculate the 'fit: BoxFit.contain' rectangle ---
    final imageSize = Size(originalImage.width.toDouble(), originalImage.height.toDouble());
    final fittedSizes = applyBoxFit(BoxFit.contain, imageSize, size);
    final sourceRect = Alignment.center.inscribe(fittedSizes.source, Rect.fromLTWH(0, 0, imageSize.width, imageSize.height));
    final destinationRect = Alignment.center.inscribe(fittedSizes.destination, Rect.fromLTWH(0, 0, size.width, size.height));

    // --- 2. Draw the Original Image ---
    canvas.drawImageRect(
      originalImage,
      sourceRect,
      destinationRect,
      Paint(),
    );

    if (modelImageWidth == 0 || modelImageHeight == 0) return;

    // --- 3. Calculate Model Padding and Scaling ---
    final double scale = min(modelImageWidth / imageSize.width, modelImageHeight / imageSize.height);
    final double padX = (modelImageWidth - imageSize.width * scale) / 2.0;
    final double padY = (modelImageHeight - imageSize.height * scale) / 2.0;

    final double scaleToCanvasX = destinationRect.width / imageSize.width;
    final double scaleToCanvasY = destinationRect.height / imageSize.height;

    // --- 4. Draw Masks ---
    if (showMasks && maskImage != null) {
      for (int i = 0; i < recognitions.length; i++) {
        if (selectedDetectionIndex != null && i != selectedDetectionIndex) continue;

        final detection = recognitions[i];
        final className = detection['className'] ?? 'Unknown';
        final color = classColorMap[className] ?? Colors.grey;
        final maskPaint = Paint()
          ..colorFilter = ColorFilter.mode(color.withOpacity(maskOpacity), BlendMode.srcIn);
        
        canvas.save();
        
        canvas.translate(destinationRect.left, destinationRect.top);
        canvas.scale(scaleToCanvasX / scale, scaleToCanvasY / scale);
        canvas.translate(-padX, -padY);
        
        canvas.drawImageRect(
          maskImage!,
          Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble()),
          Rect.fromLTWH(0, 0, modelImageWidth, modelImageHeight),
          maskPaint,
        );
        
        canvas.restore();
      }
    }

    // --- 5. Draw Boxes and Labels ---
    for (int i = 0; i < recognitions.length; i++) {
      final detection = recognitions[i];
      final className = detection['className'] ?? 'Unknown';
      final color = classColorMap[className] ?? Colors.grey;
      final isSelected = i == selectedDetectionIndex;

      final x1 = (detection['x1'] as num).toDouble();
      final y1 = (detection['y1'] as num).toDouble();
      final x2 = (detection['x2'] as num).toDouble();
      final y2 = (detection['y2'] as num).toDouble();

      // --- Convert to Original Image Space ---
      final originalX1 = (x1 - padX) / scale;
      final originalY1 = (y1 - padY) / scale;
      final originalX2 = (x2 - padX) / scale;
      final originalY2 = (y2 - padY) / scale;

      // --- Convert to Canvas Space ---
      final canvasLeft = (originalX1 * scaleToCanvasX) + destinationRect.left;
      final canvasTop = (originalY1 * scaleToCanvasY) + destinationRect.top;
      final canvasRight = (originalX2 * scaleToCanvasX) + destinationRect.left;
      final canvasBottom = (originalY2 * scaleToCanvasY) + destinationRect.top;
      
      final boundingBoxRect = Rect.fromLTRB(canvasLeft, canvasTop, canvasRight, canvasBottom);

      // --- Draw Bounding Box ---
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSelected ? 4.0 : 2.5;
      canvas.drawRect(boundingBoxRect, boxPaint);
      
      // --- Draw Label ---
      final confidence = (detection['confidence'] as num? ?? 0.0);
      final textPainter = TextPainter(
        text: TextSpan(
          text: '$className (${(confidence * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            shadows: [Shadow(color: Colors.black, blurRadius: 4)],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);

      final labelBackgroundPaint = Paint()..color = color.withOpacity(isSelected ? 1.0 : 0.8);
      double labelTop = canvasTop - textPainter.height - 4;
      if (labelTop < destinationRect.top) labelTop = canvasBottom + 2;
      double labelLeft = canvasLeft;
      if (labelLeft + textPainter.width + 8 > destinationRect.right) {
        labelLeft = destinationRect.right - textPainter.width - 8;
      }
      
      final finalLabelRect = Rect.fromLTWH(labelLeft, labelTop, textPainter.width + 8, textPainter.height + 4);
      canvas.drawRect(finalLabelRect, labelBackgroundPaint);
      textPainter.paint(canvas, Offset(finalLabelRect.left + 4, finalLabelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) =>
    originalImage != oldDelegate.originalImage ||
    maskImage != oldDelegate.maskImage ||
    recognitions != oldDelegate.recognitions ||
    showMasks != oldDelegate.showMasks ||
    maskOpacity != oldDelegate.maskOpacity ||
    selectedDetectionIndex != oldDelegate.selectedDetectionIndex ||
    modelImageWidth != oldDelegate.modelImageWidth ||
    modelImageHeight != oldDelegate.modelImageHeight;
}
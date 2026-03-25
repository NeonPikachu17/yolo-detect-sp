import 'package:flutter/material.dart';

/// Enum to manage the selected YOLO task in the app's state.
enum AppYoloTask { segment, detect, classify }

class AppConstants {
  static const String prefsKeyLastModelName = "last_used_model_name";
  static const String prefsKeyLastTaskType = "last_used_task_type";

  // --- THEME COLORS: Maroon + Hat Fusion ---
  static const Color maroonPrimary = Color(0xFF800020);   // Deep Maroon (Primary Action)
  static const Color hatBlueAccent = Color(0xFF6B7BA8);   // Hat Periwinkle (Secondary/Structure)
  static const Color hatGoldAccent = Color(0xFFE2B04E);   // Hat Gold (Highlights)
  static const Color hatBackground = Color(0xFFF0F2F5);   // Hat Off-White (Background)
  static const Color darkText = Color(0xFF2C3E50);        // Hat Navy (Text)

  // FUSION PALETTE for Boxes
  static final List<Color> boxColors = [
    maroonPrimary,
    hatBlueAccent,
    hatGoldAccent,
    darkText,
    Colors.teal,
    Colors.orangeAccent,
    Colors.purple,
  ];
}
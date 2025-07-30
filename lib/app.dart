import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:yolo_detect/features/ui/detection_screen.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return MaterialApp(
      title: 'YOLO Detection App',
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: const Color(0xFF006A6A),
        scaffoldBackgroundColor: const Color(0xFFF0F4F4),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF006A6A),
          brightness: Brightness.light,
          primary: const Color(0xFF006A6A),
          secondary: const Color(0xFF4A6363),
          background: const Color(0xFFF0F4F4),
          error: const Color(0xFFBA1A1A),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(textTheme),
        // --- CORRECTED: Use the proper CardTheme class ---
        cardTheme: const CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
          clipBehavior: Clip.antiAlias,
          margin: EdgeInsets.zero,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF0F4F4),
          foregroundColor: Color(0xFF002020),
          elevation: 0,
          centerTitle: true,
        ),
      ),
      home: const DetectionScreen(),
    );
  }
}
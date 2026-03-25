import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart'; 

import 'core/app_constants.dart';
import 'screens/vision_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return MaterialApp(
      title: 'MAMBO',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: AppConstants.maroonPrimary,
        scaffoldBackgroundColor: AppConstants.hatBackground,
        colorScheme: ColorScheme.fromSeed(
          seedColor: AppConstants.maroonPrimary,
          brightness: Brightness.light,
          primary: AppConstants.maroonPrimary,
          secondary: AppConstants.hatBlueAccent,
          tertiary: AppConstants.hatGoldAccent,
          background: AppConstants.hatBackground,
          surface: Colors.white,
          error: const Color(0xFFBA1A1A),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(textTheme).apply(
          bodyColor: Colors.blueGrey[800],
          displayColor: AppConstants.darkText,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: const BorderRadius.all(Radius.circular(24)),
            side: BorderSide(color: AppConstants.hatBlueAccent.withOpacity(0.15), width: 1),
          ),
          clipBehavior: Clip.antiAlias,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppConstants.maroonPrimary,
            foregroundColor: Colors.white,
            elevation: 4,
            shadowColor: AppConstants.maroonPrimary.withOpacity(0.4),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: AppConstants.hatBackground,
          foregroundColor: AppConstants.maroonPrimary,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: GoogleFonts.poppins(
            fontWeight: FontWeight.w800,
            fontSize: 26,
            color: AppConstants.maroonPrimary,
            letterSpacing: -0.5,
          ),
        ),
        segmentedButtonTheme: SegmentedButtonThemeData(
          style: ButtonStyle(
            backgroundColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) return AppConstants.maroonPrimary.withOpacity(0.1);
              return Colors.transparent;
            }),
            foregroundColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) return AppConstants.maroonPrimary;
              return AppConstants.hatBlueAccent;
            }),
            iconColor: MaterialStateProperty.resolveWith<Color>((states) {
              if (states.contains(MaterialState.selected)) return AppConstants.maroonPrimary;
              return AppConstants.hatBlueAccent;
            }),
            side: MaterialStateProperty.all(BorderSide(color: AppConstants.hatBlueAccent.withOpacity(0.3))),
          ),
        ),
        sliderTheme: SliderThemeData(
          activeTrackColor: AppConstants.maroonPrimary,
          thumbColor: AppConstants.hatGoldAccent,
          inactiveTrackColor: AppConstants.maroonPrimary.withOpacity(0.1),
        ),
        switchTheme: SwitchThemeData(
          thumbColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) return AppConstants.maroonPrimary;
            return Colors.blueGrey;
          }),
          trackColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) return AppConstants.maroonPrimary.withOpacity(0.3);
            return Colors.grey.withOpacity(0.2);
          }),
        )
      ),
      home: const VisionScreen(),
    );
  }
}
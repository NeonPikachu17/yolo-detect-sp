import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:provider/provider.dart'; // <-- The required import

import 'app.dart'; 
import 'features/detection/logic/detection_provider.dart';
import 'firebase_options.dart';
import 'package:statsfl/statsfl.dart'; // +++ NEW: Import the package

Future<void> main() async {
  // Ensure Flutter is ready
  WidgetsFlutterBinding.ensureInitialized();
  
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(
    // 1. The Provider is at the highest level to provide state.
    ChangeNotifierProvider(
      create: (_) => DetectionProvider(),
      // 2. The StatsFl widget wraps your app to show the overlay.
      child: StatsFl(
        align: Alignment.bottomRight, // Position the overlay
        // 3. MyApp is the root of your application UI.
        child: const MyApp(),
      ),
    ),
  );
}
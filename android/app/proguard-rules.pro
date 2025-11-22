# Keep Flutter Plugin Registrant (Auto-generated)
-keep class io.flutter.plugins.GeneratedPluginRegistrant { *; }

# Keep Ultralytics Plugin (The Critical Part)
-keep class com.ultralytics.** { *; }
-keep class com.ultralytics.ultralytics_yolo.** { *; }

# Keep Native Methods
-keepclasseswithmembernames class * {
    native <methods>;
}
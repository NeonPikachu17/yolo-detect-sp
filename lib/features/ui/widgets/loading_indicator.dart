import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:provider/provider.dart';
import 'package:yolo_detect/features/detection/logic/detection_provider.dart';

class LoadingIndicator extends StatelessWidget {
  const LoadingIndicator({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<DetectionProvider>();
    final loaderType = provider.loaderType;
    final loadingMessage = provider.loadingMessage;
    
    Widget loaderWidget;
    switch (loaderType) {
      case LoaderType.limitedFiveStar:
        loaderWidget = Image.asset('assets/images/loading_limited.gif', width: 120, height: 120);
        break;
      case LoaderType.standardFiveStar:
        loaderWidget = Image.asset('assets/images/loading.gif', width: 200, height: 200);
        break;
      case LoaderType.fourStar:
        loaderWidget = Lottie.asset('assets/animations/loading_4.json', width: 200, height: 200);
        break;
      case LoaderType.regular:
      default:
        loaderWidget = Lottie.asset('assets/animations/loading.json', width: 200, height: 200);
    }

    return Container(
      padding: const EdgeInsets.symmetric(vertical: 50.0),
      child: Column(
        children: [
          loaderWidget,
          if (loadingMessage != null && loadingMessage.isNotEmpty) ...[
            const SizedBox(height: 20),
            Text(
              loadingMessage,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade700),
            ),
          ]
        ],
      ),
    );
  }
}
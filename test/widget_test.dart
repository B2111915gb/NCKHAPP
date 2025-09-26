import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:banana/main.dart';

void main() {
  testWidgets('Camera screen UI test', (WidgetTester tester) async {
    // Build CameraApp
    await tester.pumpWidget(CameraApp());

    // Kiểm tra có AppBar với tiêu đề "Chụp ảnh"
    expect(find.text('Chụp ảnh'), findsOneWidget);

    // Kiểm tra có FloatingActionButton với icon camera
    expect(find.byIcon(Icons.camera_alt), findsOneWidget);
  });

  testWidgets('Tap camera button', (WidgetTester tester) async {
    await tester.pumpWidget(CameraApp());

    // Tìm nút camera
    final cameraButton = find.byIcon(Icons.camera_alt);
    expect(cameraButton, findsOneWidget);

    // Bấm vào nút camera
    await tester.tap(cameraButton);

    // Trigger rebuild
    await tester.pump();

    // Ở đây ta không thể test ảnh thật,
    // nhưng có thể xác nhận không crash sau khi bấm.
    expect(find.byIcon(Icons.camera_alt), findsOneWidget);
  });
}

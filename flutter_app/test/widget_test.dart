import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';

import 'package:hit_song_app/main.dart';

void main() {
  testWidgets('Home screen shows input form', (tester) async {
    tester.view.physicalSize = const Size(1400, 1000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const HitSongApp());
    expect(find.text('Hit Song Predictor'), findsOneWidget);
    expect(find.text('Upload audio'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);
  });
}

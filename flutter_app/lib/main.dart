import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'theme.dart';

void main() => runApp(const HitSongApp());

class HitSongApp extends StatelessWidget {
  const HitSongApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hit Song Predictor',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(),
      home: const HomeScreen(),
    );
  }
}

import 'package:flutter/material.dart';

class AppColors {
  static const background = Color(0xFFF6F7FB);
  static const surface = Colors.white;
  static const border = Color(0xFFE3E6EE);
  static const textPrimary = Color(0xFF1A1D29);
  static const textSecondary = Color(0xFF5F6577);
  static const accent = Color(0xFF5B4BDB);
  static const negative = Color(0xFFD64545);

  /// Color scale for 0–100 scores.
  static Color score(double value) {
    if (value >= 70) return const Color(0xFF1E9E6A);
    if (value >= 50) return const Color(0xFF2F7FD8);
    if (value >= 30) return const Color(0xFFD98E04);
    return negative;
  }
}

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: AppColors.accent,
    brightness: Brightness.light,
    surface: AppColors.surface,
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: AppColors.background,
    dividerTheme: const DividerThemeData(color: AppColors.border),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.surface,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: AppColors.border),
      ),
    ),
  );
}

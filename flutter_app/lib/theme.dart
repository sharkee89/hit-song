import 'package:flutter/material.dart';

class AppColors {
  static const background = Color(0xFF0A0E19);
  static const surface = Color(0xFF121829);
  static const surfaceHigh = Color(0xFF1A2236);
  static const border = Color(0xFF242E46);
  static const textPrimary = Color(0xFFE8ECF6);
  static const textSecondary = Color(0xFF8C97B2);
  static const textMuted = Color(0xFF5D6782);

  static const violet = Color(0xFF7C5CFF);
  static const pink = Color(0xFFFF4D8D);
  static const positive = Color(0xFF2DD4BF);
  static const negative = Color(0xFFFB7185);
  static const warning = Color(0xFFF5B544);

  static const brandGradient = LinearGradient(
    colors: [violet, pink],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  /// Same three groups ("agents") used across the research code.
  static Color group(String group) => switch (group) {
    'Production agent' => warning,
    'Talent agent' => const Color(0xFF9D8CFF),
    _ => positive,
  };

  /// Color scale for 0–100 scores.
  static Color score(double value) {
    if (value >= 70) return positive;
    if (value >= 50) return const Color(0xFF7DD3FC);
    if (value >= 30) return warning;
    return negative;
  }
}

ThemeData buildTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppColors.violet,
      brightness: Brightness.dark,
      surface: AppColors.surface,
      primary: AppColors.violet,
      secondary: AppColors.pink,
    ),
    scaffoldBackgroundColor: AppColors.background,
  );

  return base.copyWith(
    textTheme: base.textTheme.apply(
      bodyColor: AppColors.textPrimary,
      displayColor: AppColors.textPrimary,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.surfaceHigh,
      hintStyle: const TextStyle(color: AppColors.textMuted),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.violet, width: 1.5),
      ),
    ),
    tooltipTheme: TooltipThemeData(
      decoration: BoxDecoration(
        color: AppColors.surfaceHigh,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.border),
      ),
      textStyle: const TextStyle(color: AppColors.textPrimary, fontSize: 12),
    ),
    dividerColor: AppColors.border,
  );
}

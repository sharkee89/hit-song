import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

class SectionCard extends StatelessWidget {
  final String? title;
  final String? subtitle;
  final IconData? icon;
  final Widget? trailing;
  final Widget child;
  final EdgeInsetsGeometry padding;

  const SectionCard({
    super.key,
    this.title,
    this.subtitle,
    this.icon,
    this.trailing,
    required this.child,
    this.padding = const EdgeInsets.all(22),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (title != null) ...[
            Row(
              children: [
                if (icon != null) ...[
                  Icon(icon, size: 18, color: AppColors.textSecondary),
                  const SizedBox(width: 10),
                ],
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title!,
                        style: const TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                        ),
                      ),
                      if (subtitle != null) ...[
                        const SizedBox(height: 2),
                        Text(
                          subtitle!,
                          style: const TextStyle(
                            fontSize: 12,
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                if (trailing != null) trailing!,
              ],
            ),
            const SizedBox(height: 18),
          ],
          child,
        ],
      ),
    );
  }
}

class Pill extends StatelessWidget {
  final String label;
  final Color color;
  final IconData? icon;

  const Pill(
    this.label, {
    super.key,
    this.color = AppColors.textSecondary,
    this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 13, color: color),
            const SizedBox(width: 5),
          ],
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

/// Large animated semicircular gauge for the 0–100 popularity score.
class ScoreGauge extends StatelessWidget {
  final double value;
  final double size;
  final String caption;

  const ScoreGauge({
    super.key,
    required this.value,
    this.size = 220,
    this.caption = '/ 100',
  });

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: value.clamp(0, 100)),
      duration: const Duration(milliseconds: 1400),
      curve: Curves.easeOutCubic,
      builder: (context, v, _) => SizedBox(
        width: size,
        height: size * 0.82,
        child: CustomPaint(
          painter: _GaugePainter(v),
          child: Padding(
            padding: EdgeInsets.only(top: size * 0.2),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  v.toStringAsFixed(1),
                  style: TextStyle(
                    fontSize: size * 0.24,
                    fontWeight: FontWeight.w700,
                    height: 1.0,
                    color: AppColors.textPrimary,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  caption,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double value;
  _GaugePainter(this.value);

  static const _start = math.pi * 0.75;
  static const _sweep = math.pi * 1.5;

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = size.width * 0.075;
    final rect = Rect.fromCircle(
      center: Offset(size.width / 2, size.width / 2),
      radius: size.width / 2 - stroke,
    );

    canvas.drawArc(
      rect,
      _start,
      _sweep,
      false,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke
        ..strokeCap = StrokeCap.round
        ..color = AppColors.surfaceHigh,
    );

    // Tick marks every 10 points
    final tickPaint = Paint()
      ..color = AppColors.textMuted.withValues(alpha: 0.6)
      ..strokeWidth = 1.5;
    final c = rect.center;
    final r = rect.width / 2;
    for (var i = 0; i <= 10; i++) {
      final a = _start + _sweep * i / 10;
      final inner = r - stroke * 0.9 - 4;
      final outer = r - stroke * 0.9 - (i % 5 == 0 ? 12 : 8);
      canvas.drawLine(
        c + Offset(math.cos(a) * inner, math.sin(a) * inner),
        c + Offset(math.cos(a) * outer, math.sin(a) * outer),
        tickPaint,
      );
    }

    if (value <= 0) return;
    canvas.drawArc(
      rect,
      _start,
      _sweep * value / 100,
      false,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke
        ..strokeCap = StrokeCap.round
        ..shader = const SweepGradient(
          startAngle: _start,
          endAngle: _start + _sweep,
          colors: [AppColors.violet, AppColors.pink, Color(0xFFFFB86B)],
          transform: GradientRotation(0),
        ).createShader(rect),
    );
  }

  @override
  bool shouldRepaint(_GaugePainter old) => old.value != value;
}

/// Small circular score indicator used for LLM metrics.
class ScoreRing extends StatelessWidget {
  final double? value;
  final double size;

  const ScoreRing({super.key, required this.value, this.size = 52});

  @override
  Widget build(BuildContext context) {
    final v = value;
    final color = v == null ? AppColors.textMuted : AppColors.score(v);
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: (v ?? 0).clamp(0, 100)),
      duration: const Duration(milliseconds: 1100),
      curve: Curves.easeOutCubic,
      builder: (context, a, _) => SizedBox(
        width: size,
        height: size,
        child: Stack(
          alignment: Alignment.center,
          children: [
            SizedBox.expand(
              child: CircularProgressIndicator(
                value: a / 100,
                strokeWidth: size * 0.09,
                strokeCap: StrokeCap.round,
                backgroundColor: AppColors.surfaceHigh,
                valueColor: AlwaysStoppedAnimation(color),
              ),
            ),
            Text(
              v == null ? '–' : a.round().toString(),
              style: TextStyle(
                fontSize: size * 0.3,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Horizontal labelled bar for a value inside [0, max].
class MeterBar extends StatelessWidget {
  final String label;
  final String? hint;
  final double value;
  final double max;
  final Color color;
  final String? valueText;

  const MeterBar({
    super.key,
    required this.label,
    required this.value,
    required this.max,
    this.hint,
    this.color = AppColors.violet,
    this.valueText,
  });

  @override
  Widget build(BuildContext context) {
    final ratio = (value / max).clamp(0.0, 1.0);
    return Tooltip(
      message: hint ?? '',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
              Text(
                valueText ?? value.toStringAsFixed(2),
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 7),
          ClipRRect(
            borderRadius: BorderRadius.circular(99),
            child: Stack(
              children: [
                Container(height: 8, color: AppColors.surfaceHigh),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0, end: ratio),
                  duration: const Duration(milliseconds: 900),
                  curve: Curves.easeOutCubic,
                  builder: (context, r, _) => FractionallySizedBox(
                    widthFactor: r,
                    child: Container(
                      height: 8,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [color.withValues(alpha: 0.65), color],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

String formatCompact(num n) {
  if (n >= 1e9) return '${(n / 1e9).toStringAsFixed(1)}B';
  if (n >= 1e6) return '${(n / 1e6).toStringAsFixed(1)}M';
  if (n >= 1e3) return '${(n / 1e3).toStringAsFixed(1)}K';
  return n.toStringAsFixed(0);
}

String formatTime(double seconds) {
  final s = seconds.round();
  return '${s ~/ 60}:${(s % 60).toString().padLeft(2, '0')}';
}

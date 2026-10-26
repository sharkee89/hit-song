import 'package:flutter/material.dart';

import '../api/models.dart';
import '../theme.dart';
import 'common.dart';

class ScoreCard extends StatelessWidget {
  final PredictionResult result;
  const ScoreCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final delta = result.popularity - result.baseValue;
    final tierColor = AppColors.score(result.popularity);
    return SectionCard(
      title: 'Predicted popularity',
      subtitle: 'XGBoost · Spotify popularity scale',
      icon: Icons.insights_rounded,
      trailing: Pill(
        result.tier,
        color: tierColor,
        icon: Icons.local_fire_department_rounded,
      ),
      child: Column(
        children: [
          ScoreGauge(value: result.popularity),
          const SizedBox(height: 4),
          Text(
            result.fileName,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 13,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 18),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppColors.surfaceHigh,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                _Figure('Model baseline', result.baseValue.toStringAsFixed(1)),
                const _Op('+'),
                _Figure(
                  'This song',
                  '${delta >= 0 ? '+' : '−'}${delta.abs().toStringAsFixed(1)}',
                  color: delta >= 0 ? AppColors.positive : AppColors.negative,
                ),
                const _Op('='),
                _Figure(
                  'Prediction',
                  result.popularity.toStringAsFixed(1),
                  color: tierColor,
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            "Baseline is the model's expected output (SHAP base value); "
            "the song's offset is the sum of its exact TreeSHAP feature contributions.",
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 11.5, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }
}

class _Figure extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _Figure(this.label, this.value, {this.color = AppColors.textPrimary});

  @override
  Widget build(BuildContext context) => Expanded(
    child: Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
        ),
      ],
    ),
  );
}

class _Op extends StatelessWidget {
  final String op;
  const _Op(this.op);

  @override
  Widget build(BuildContext context) => Text(
    op,
    style: const TextStyle(
      fontSize: 16,
      color: AppColors.textMuted,
      fontWeight: FontWeight.w300,
    ),
  );
}

class ArtistCard extends StatelessWidget {
  final ArtistInfo artist;
  final AnalysisWindow window;
  const ArtistCard({super.key, required this.artist, required this.window});

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'Artist context',
      subtitle: 'Talent agent',
      icon: Icons.person_search_rounded,
      trailing: artist.foundInDataset
          ? const Pill(
              'In dataset',
              color: AppColors.positive,
              icon: Icons.verified_rounded,
            )
          : const Pill(
              'Not in dataset',
              color: AppColors.warning,
              icon: Icons.info_outline_rounded,
            ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 48,
                height: 48,
                alignment: Alignment.center,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: AppColors.brandGradient,
                ),
                child: Text(
                  artist.name.isEmpty
                      ? '?'
                      : artist.name.characters.first.toUpperCase(),
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      artist.name,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    if (artist.followers != null)
                      Text(
                        '${formatCompact(artist.followers!)} followers',
                        style: const TextStyle(
                          fontSize: 12.5,
                          color: AppColors.textSecondary,
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          if (artist.foundInDataset && artist.popularity != null)
            MeterBar(
              label: 'Artist popularity',
              value: artist.popularity!,
              max: 100,
              valueText: artist.popularity!.toStringAsFixed(0),
              color: AppColors.score(artist.popularity!),
            )
          else
            const Text(
              'Artist not found in the Spotify dataset — the model used neutral '
              'defaults (popularity 10, 10K followers), so the score relies on audio alone.',
              style: TextStyle(
                fontSize: 12.5,
                color: AppColors.textSecondary,
                height: 1.4,
              ),
            ),
          if (artist.genres.isNotEmpty) ...[
            const SizedBox(height: 16),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final g in artist.genres)
                  Pill(g, color: const Color(0xFF9D8CFF)),
              ],
            ),
          ],
          const SizedBox(height: 18),
          const Divider(height: 1),
          const SizedBox(height: 14),
          Row(
            children: [
              const Icon(
                Icons.content_cut_rounded,
                size: 15,
                color: AppColors.textSecondary,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Analysed window ${formatTime(window.startSec)} – ${formatTime(window.endSec)} '
                  'of ${formatTime(window.totalSec)}',
                  style: const TextStyle(
                    fontSize: 12.5,
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _WindowBar(window: window),
        ],
      ),
    );
  }
}

class _WindowBar extends StatelessWidget {
  final AnalysisWindow window;
  const _WindowBar({required this.window});

  @override
  Widget build(BuildContext context) {
    final total = window.totalSec <= 0 ? 1.0 : window.totalSec;
    return LayoutBuilder(
      builder: (context, c) {
        final left = c.maxWidth * window.startSec / total;
        final width = (c.maxWidth * (window.endSec - window.startSec) / total)
            .clamp(4.0, c.maxWidth);
        return SizedBox(
          height: 10,
          child: Stack(
            children: [
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surfaceHigh,
                  borderRadius: BorderRadius.circular(99),
                ),
              ),
              Positioned(
                left: left,
                width: width,
                top: 0,
                bottom: 0,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: AppColors.brandGradient,
                    borderRadius: BorderRadius.circular(99),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class ExplainabilityCard extends StatelessWidget {
  final PredictionResult result;
  const ExplainabilityCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final items = result.contributions;
    final maxAbs = items.fold<double>(
      0.01,
      (m, c) => c.contribution.abs() > m ? c.contribution.abs() : m,
    );
    const groups = ['Production agent', 'Talent agent', 'Audio agent'];

    return SectionCard(
      title: 'Why this score?',
      subtitle: 'Top feature contributions (SHAP) in popularity points',
      icon: Icons.account_tree_rounded,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              for (final g in groups)
                Expanded(
                  child: _GroupTotal(
                    group: g,
                    total: result.groupContributions[g] ?? 0,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 20),
          for (final c in items) _ContributionRow(c, maxAbs),
          const SizedBox(height: 10),
          const Row(
            children: [
              _Legend(AppColors.positive, 'Raises popularity'),
              SizedBox(width: 16),
              _Legend(AppColors.negative, 'Lowers popularity'),
            ],
          ),
        ],
      ),
    );
  }
}

class _GroupTotal extends StatelessWidget {
  final String group;
  final double total;
  const _GroupTotal({required this.group, required this.total});

  @override
  Widget build(BuildContext context) {
    final color = AppColors.group(group);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 4),
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          Text(
            '${total >= 0 ? '+' : '−'}${total.abs().toStringAsFixed(1)}',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            group,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 11.5,
              color: AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}

class _ContributionRow extends StatelessWidget {
  final Contribution c;
  final double maxAbs;
  const _ContributionRow(this.c, this.maxAbs);

  @override
  Widget build(BuildContext context) {
    final positive = c.contribution >= 0;
    final color = positive ? AppColors.positive : AppColors.negative;
    final ratio = (c.contribution.abs() / maxAbs).clamp(0.0, 1.0);
    return Tooltip(
      message: '${c.group} · input value ${_formatValue(c)}',
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: AppColors.group(c.group),
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 140,
              child: Text(
                c.label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 12.5,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
            Expanded(
              child: LayoutBuilder(
                builder: (context, box) {
                  final half = box.maxWidth / 2;
                  return SizedBox(
                    height: 18,
                    child: Stack(
                      children: [
                        Positioned(
                          left: half - 0.5,
                          top: 0,
                          bottom: 0,
                          child: Container(width: 1, color: AppColors.border),
                        ),
                        TweenAnimationBuilder<double>(
                          tween: Tween(begin: 0, end: ratio),
                          duration: const Duration(milliseconds: 900),
                          curve: Curves.easeOutCubic,
                          builder: (context, r, _) => Positioned(
                            left: positive ? half : half - half * r,
                            width: half * r,
                            top: 3,
                            bottom: 3,
                            child: Container(
                              decoration: BoxDecoration(
                                color: color.withValues(alpha: 0.85),
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
            SizedBox(
              width: 56,
              child: Text(
                '${positive ? '+' : '−'}${c.contribution.abs().toStringAsFixed(2)}',
                textAlign: TextAlign.right,
                style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatValue(Contribution c) {
    if (c.feature == 'followers') return formatCompact(c.value);
    if (c.value.abs() >= 100) return c.value.toStringAsFixed(0);
    return c.value.toStringAsFixed(3);
  }
}

class _Legend extends StatelessWidget {
  final Color color;
  final String label;
  const _Legend(this.color, this.label);

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(3),
        ),
      ),
      const SizedBox(width: 6),
      Text(
        label,
        style: const TextStyle(fontSize: 11.5, color: AppColors.textSecondary),
      ),
    ],
  );
}

class AudioProfileCard extends StatelessWidget {
  final PredictionResult result;
  const AudioProfileCard({super.key, required this.result});

  static const _audioboxLabels = {
    'PQ': (
      'Production Quality',
      'Technical quality: clarity, fidelity, dynamics',
    ),
    'PC': ('Production Complexity', 'Number and richness of audio components'),
    'CE': (
      'Content Enjoyment',
      'Predicted listener enjoyment / artistic appeal',
    ),
    'CU': ('Content Usefulness', 'Usefulness as material for content creation'),
  };
  static const _pitches = [
    'C',
    'C#',
    'D',
    'D#',
    'E',
    'F',
    'F#',
    'G',
    'G#',
    'A',
    'A#',
    'B',
  ];

  @override
  Widget build(BuildContext context) {
    final f = result.audioFeatures;
    return SectionCard(
      title: 'Audio profile',
      subtitle:
          'Production agent (Meta Audiobox Aesthetics) + Audio agent (Librosa)',
      icon: Icons.graphic_eq_rounded,
      child: LayoutBuilder(
        builder: (context, c) {
          final wide = c.maxWidth > 640;
          final audiobox = Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const _SubHeading('Aesthetics scores (1–10)'),
              for (final e in _audioboxLabels.entries)
                if (result.audiobox[e.key] != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 14),
                    child: MeterBar(
                      label: '${e.value.$1} (${e.key})',
                      hint: e.value.$2,
                      value: result.audiobox[e.key]!,
                      max: 10,
                      color: AppColors.warning,
                    ),
                  ),
            ],
          );
          final features = Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const _SubHeading('Signal descriptors'),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  _Stat('Tempo', '${(f['Tempo'] ?? 0).toStringAsFixed(0)} BPM'),
                  _Stat(
                    'Energy (RMS)',
                    (f['Energy (RMS)'] ?? 0).toStringAsFixed(3),
                  ),
                  _Stat(
                    'Spectral centroid',
                    '${(f['Spectral centroid'] ?? 0).toStringAsFixed(0)} Hz',
                  ),
                  _Stat(
                    'Spectral rolloff',
                    '${(f['Spectral rolloff'] ?? 0).toStringAsFixed(0)} Hz',
                  ),
                  _Stat(
                    'Bandwidth',
                    '${(f['Spectral bandwidth'] ?? 0).toStringAsFixed(0)} Hz',
                  ),
                  _Stat(
                    'Flatness',
                    (f['Spectral flatness'] ?? 0).toStringAsFixed(4),
                  ),
                  _Stat(
                    'Zero-crossing',
                    (f['Zero-crossing rate'] ?? 0).toStringAsFixed(4),
                  ),
                ],
              ),
              const SizedBox(height: 18),
              const _SubHeading('Chroma / pitch-class profile'),
              _ChromaChart(values: result.chroma, labels: _pitches),
            ],
          );
          if (!wide) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [audiobox, const SizedBox(height: 10), features],
            );
          }
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(flex: 5, child: audiobox),
              const SizedBox(width: 28),
              Expanded(flex: 6, child: features),
            ],
          );
        },
      ),
    );
  }
}

class _SubHeading extends StatelessWidget {
  final String text;
  const _SubHeading(this.text);

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: Text(
      text.toUpperCase(),
      style: const TextStyle(
        fontSize: 11,
        letterSpacing: 1.1,
        fontWeight: FontWeight.w600,
        color: AppColors.textMuted,
      ),
    ),
  );
}

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  const _Stat(this.label, this.value);

  @override
  Widget build(BuildContext context) => Container(
    width: 132,
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    decoration: BoxDecoration(
      color: AppColors.surfaceHigh,
      borderRadius: BorderRadius.circular(10),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: AppColors.textSecondary),
        ),
        const SizedBox(height: 3),
        Text(
          value,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
        ),
      ],
    ),
  );
}

class _ChromaChart extends StatelessWidget {
  final List<double> values;
  final List<String> labels;
  const _ChromaChart({required this.values, required this.labels});

  @override
  Widget build(BuildContext context) {
    final maxV = values.fold<double>(0.01, (m, v) => v > m ? v : m);
    return SizedBox(
      height: 96,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          for (var i = 0; i < values.length && i < labels.length; i++)
            Expanded(
              child: Tooltip(
                message: '${labels[i]}: ${values[i].toStringAsFixed(3)}',
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0, end: values[i] / maxV),
                      duration: Duration(milliseconds: 700 + i * 40),
                      curve: Curves.easeOutCubic,
                      builder: (context, r, _) => Container(
                        height: 70 * r,
                        margin: const EdgeInsets.symmetric(horizontal: 3),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(4),
                          gradient: LinearGradient(
                            begin: Alignment.bottomCenter,
                            end: Alignment.topCenter,
                            colors: [
                              AppColors.positive.withValues(alpha: 0.35),
                              values[i] == maxV
                                  ? AppColors.pink
                                  : AppColors.positive,
                            ],
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      labels[i],
                      style: const TextStyle(
                        fontSize: 10.5,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

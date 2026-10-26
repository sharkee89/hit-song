import 'package:flutter/material.dart';

import '../api/models.dart';
import '../theme.dart';
import 'common.dart';

/// Display order of LLM metrics (keys produced by api/pipeline.py).
const _metricOrder = [
  'prediction_grade',
  'production_quality',
  'audio_quality',
  'genre_placement',
  'possible_improvements',
  'prediction_detail_analysis',
  'analysis',
];

int _orderOf(String key) {
  final i = _metricOrder.indexOf(key);
  return i == -1 ? _metricOrder.length : i;
}

class LlmLoadingCard extends StatelessWidget {
  final int elapsedSeconds;
  const LlmLoadingCard({super.key, required this.elapsedSeconds});

  @override
  Widget build(BuildContext context) {
    final stage = elapsedSeconds < 20
        ? 'Gemini and GPT are listening to the full track…'
        : 'DeepSeek is reconciling both expert opinions…';
    return SectionCard(
      title: 'AI expert panel',
      subtitle: 'Gemini · GPT audio · DeepSeek consensus',
      icon: Icons.psychology_alt_rounded,
      trailing: Text(
        '${elapsedSeconds}s',
        style: const TextStyle(color: AppColors.textSecondary, fontSize: 13),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(99),
            child: const LinearProgressIndicator(
              minHeight: 6,
              backgroundColor: AppColors.surfaceHigh,
              color: AppColors.violet,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              const Icon(
                Icons.headphones_rounded,
                size: 18,
                color: AppColors.violet,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 300),
                  child: Text(
                    stage,
                    key: ValueKey(stage),
                    style: const TextStyle(color: AppColors.textSecondary),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'This usually takes 20–60 seconds. The ML prediction above is already final.',
            style: TextStyle(fontSize: 12, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }
}

class LlmErrorCard extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const LlmErrorCard({super.key, required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'AI expert panel',
      subtitle: 'The LLM analysis could not be completed',
      icon: Icons.psychology_alt_rounded,
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: AppColors.negative),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(
                color: AppColors.textSecondary,
                fontSize: 13,
              ),
            ),
          ),
          const SizedBox(width: 12),
          OutlinedButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh_rounded, size: 18),
            label: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class ConsensusCard extends StatelessWidget {
  final LlmAnalysis analysis;
  const ConsensusCard({super.key, required this.analysis});

  @override
  Widget build(BuildContext context) {
    final metrics = [...analysis.consensus.metrics]
      ..sort((a, b) => _orderOf(a.key).compareTo(_orderOf(b.key)));
    return SectionCard(
      title: 'Consensus verdict',
      subtitle: 'DeepSeek meta-analysis of Gemini and GPT',
      icon: Icons.hub_rounded,
      trailing: Pill(
        '${analysis.elapsedSec.toStringAsFixed(0)} s',
        icon: Icons.timer_outlined,
      ),
      child: metrics.isEmpty
          ? _RawText(analysis.consensus.raw ?? 'No consensus returned.')
          : LayoutBuilder(
              builder: (context, c) {
                final columns = c.maxWidth > 900
                    ? 3
                    : (c.maxWidth > 560 ? 2 : 1);
                final width = (c.maxWidth - (columns - 1) * 14) / columns;
                return Wrap(
                  spacing: 14,
                  runSpacing: 14,
                  children: [
                    for (final m in metrics)
                      SizedBox(
                        width: width,
                        child: _MetricTile(metric: m),
                      ),
                  ],
                );
              },
            ),
    );
  }
}

class _MetricTile extends StatefulWidget {
  final LlmMetric metric;
  const _MetricTile({required this.metric});

  @override
  State<_MetricTile> createState() => _MetricTileState();
}

class _MetricTileState extends State<_MetricTile> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final m = widget.metric;
    return Material(
      color: AppColors.surfaceHigh,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () => setState(() => _expanded = !_expanded),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  ScoreRing(value: m.score),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          m.label,
                          style: const TextStyle(
                            fontSize: 14.5,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        if (m.key == 'possible_improvements')
                          const Text(
                            'higher = fewer changes needed',
                            style: TextStyle(
                              fontSize: 11,
                              color: AppColors.textMuted,
                            ),
                          ),
                      ],
                    ),
                  ),
                  Icon(
                    _expanded
                        ? Icons.expand_less_rounded
                        : Icons.expand_more_rounded,
                    color: AppColors.textMuted,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              AnimatedSize(
                duration: const Duration(milliseconds: 220),
                alignment: Alignment.topCenter,
                child: Text(
                  m.analysis,
                  maxLines: _expanded ? null : 4,
                  overflow: _expanded
                      ? TextOverflow.visible
                      : TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.5,
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class AgreementCard extends StatelessWidget {
  final LlmAnalysis analysis;
  final double xgboostPopularity;
  const AgreementCard({
    super.key,
    required this.analysis,
    required this.xgboostPopularity,
  });

  @override
  Widget build(BuildContext context) {
    final keys = <String>{
      for (final r in [analysis.gemini, analysis.gpt, analysis.consensus])
        for (final m in r.metrics) m.key,
    }.toList()..sort((a, b) => _orderOf(a).compareTo(_orderOf(b)));
    String labelFor(String key) =>
        [analysis.consensus, analysis.gemini, analysis.gpt]
            .map((r) => r.metric(key)?.label)
            .firstWhere((l) => l != null, orElse: () => key)!;

    final llmGrade = analysis.consensus.metric('prediction_grade')?.score;

    return SectionCard(
      title: 'Model agreement',
      subtitle:
          'Scores per model · spread shows how much Gemini and GPT disagree',
      icon: Icons.compare_arrows_rounded,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _CrossCheck(xgboost: xgboostPopularity, llmGrade: llmGrade),
          const SizedBox(height: 18),
          const _MatrixRow(
            cells: ['Metric', 'Gemini', 'GPT', 'Consensus', 'Spread'],
            header: true,
          ),
          const Divider(height: 1),
          for (final k in keys)
            _MatrixRow(
              cells: [labelFor(k)],
              scores: [
                analysis.gemini.metric(k)?.score,
                analysis.gpt.metric(k)?.score,
                analysis.consensus.metric(k)?.score,
              ],
            ),
        ],
      ),
    );
  }
}

class _CrossCheck extends StatelessWidget {
  final double xgboost;
  final double? llmGrade;
  const _CrossCheck({required this.xgboost, required this.llmGrade});

  @override
  Widget build(BuildContext context) {
    final gap = llmGrade == null ? null : (llmGrade! - xgboost);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        gradient: LinearGradient(
          colors: [
            AppColors.violet.withValues(alpha: 0.14),
            AppColors.pink.withValues(alpha: 0.08),
          ],
        ),
        border: Border.all(color: AppColors.violet.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          _CrossFigure('XGBoost popularity', xgboost, 'data-driven'),
          const Icon(Icons.sync_alt_rounded, color: AppColors.textMuted),
          _CrossFigure('LLM hit potential', llmGrade, 'expert listening'),
          Container(width: 1, height: 44, color: AppColors.border),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(left: 16),
              child: Text(
                gap == null
                    ? 'No consensus hit-potential score available.'
                    : gap.abs() < 10
                    ? 'Both approaches agree closely (Δ ${gap.abs().toStringAsFixed(0)}).'
                    : gap > 0
                    ? 'LLM experts rate the track ${gap.toStringAsFixed(0)} pts higher '
                          'than the data-driven popularity forecast.'
                    : 'The data-driven forecast is ${(-gap).toStringAsFixed(0)} pts higher '
                          "than the LLM experts' rating.",
                style: const TextStyle(
                  fontSize: 12.5,
                  height: 1.45,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CrossFigure extends StatelessWidget {
  final String label;
  final double? value;
  final String caption;
  const _CrossFigure(this.label, this.value, this.caption);

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 14),
    child: Column(
      children: [
        Text(
          value == null ? '–' : value!.toStringAsFixed(0),
          style: TextStyle(
            fontSize: 26,
            fontWeight: FontWeight.w700,
            color: value == null
                ? AppColors.textMuted
                : AppColors.score(value!),
          ),
        ),
        Text(label, style: const TextStyle(fontSize: 11.5)),
        Text(
          caption,
          style: const TextStyle(fontSize: 10.5, color: AppColors.textMuted),
        ),
      ],
    ),
  );
}

class _MatrixRow extends StatelessWidget {
  final List<String> cells;
  final List<double?>? scores;
  final bool header;
  const _MatrixRow({required this.cells, this.scores, this.header = false});

  @override
  Widget build(BuildContext context) {
    final headerStyle = const TextStyle(
      fontSize: 11,
      letterSpacing: 0.8,
      fontWeight: FontWeight.w600,
      color: AppColors.textMuted,
    );
    Widget scoreCell(double? v) => Expanded(
      flex: 2,
      child: Text(
        v == null ? '—' : v.toStringAsFixed(0),
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: v == null ? AppColors.textMuted : AppColors.score(v),
        ),
      ),
    );

    final List<Widget> children;
    if (header) {
      children = [
        Expanded(
          flex: 4,
          child: Text(cells[0].toUpperCase(), style: headerStyle),
        ),
        for (final c in cells.skip(1))
          Expanded(
            flex: 2,
            child: Text(
              c.toUpperCase(),
              textAlign: TextAlign.center,
              style: headerStyle,
            ),
          ),
      ];
    } else {
      final s = scores!;
      final spread = (s[0] != null && s[1] != null)
          ? (s[0]! - s[1]!).abs()
          : null;
      children = [
        Expanded(
          flex: 4,
          child: Text(cells[0], style: const TextStyle(fontSize: 13.5)),
        ),
        scoreCell(s[0]),
        scoreCell(s[1]),
        scoreCell(s[2]),
        Expanded(
          flex: 2,
          child: Center(
            child: spread == null
                ? const Text('—', style: TextStyle(color: AppColors.textMuted))
                : Pill(
                    '±${spread.toStringAsFixed(0)}',
                    color: spread <= 5
                        ? AppColors.positive
                        : spread <= 12
                        ? AppColors.warning
                        : AppColors.negative,
                  ),
          ),
        ),
      ];
    }
    return Padding(
      padding: EdgeInsets.symmetric(vertical: header ? 8 : 10),
      child: Row(children: children),
    );
  }
}

class ModelReportsCard extends StatefulWidget {
  final LlmAnalysis analysis;
  const ModelReportsCard({super.key, required this.analysis});

  @override
  State<ModelReportsCard> createState() => _ModelReportsCardState();
}

class _ModelReportsCardState extends State<ModelReportsCard> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final reports = [
      ('Gemini', widget.analysis.gemini),
      ('GPT audio', widget.analysis.gpt),
      ('DeepSeek', widget.analysis.consensus),
    ];
    final report = reports[_tab].$2;
    final metrics = [...report.metrics]
      ..sort((a, b) => _orderOf(a.key).compareTo(_orderOf(b.key)));

    return SectionCard(
      title: 'Individual expert reports',
      subtitle: 'Full reasoning of each language model',
      icon: Icons.article_outlined,
      trailing: SegmentedButton<int>(
        showSelectedIcon: false,
        style: const ButtonStyle(visualDensity: VisualDensity.compact),
        segments: [
          for (var i = 0; i < reports.length; i++)
            ButtonSegment(value: i, label: Text(reports[i].$1)),
        ],
        selected: {_tab},
        onSelectionChanged: (s) => setState(() => _tab = s.first),
      ),
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 250),
        child: KeyedSubtree(
          key: ValueKey(_tab),
          child: report.isEmpty
              ? const Text(
                  'This model did not return a response.',
                  style: TextStyle(color: AppColors.textSecondary),
                )
              : metrics.isEmpty
              ? _RawText(report.raw ?? '')
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    for (final m in metrics)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 18),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            ScoreRing(value: m.score, size: 42),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    m.label,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  SelectableText(
                                    m.analysis,
                                    style: const TextStyle(
                                      fontSize: 13,
                                      height: 1.55,
                                      color: AppColors.textSecondary,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
        ),
      ),
    );
  }
}

class _RawText extends StatelessWidget {
  final String text;
  const _RawText(this.text);

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: AppColors.surfaceHigh,
      borderRadius: BorderRadius.circular(12),
    ),
    child: SelectableText(
      text,
      style: const TextStyle(
        fontSize: 12.5,
        height: 1.5,
        color: AppColors.textSecondary,
      ),
    ),
  );
}

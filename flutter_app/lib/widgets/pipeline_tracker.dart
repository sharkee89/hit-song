import 'package:flutter/material.dart';

import '../theme.dart';
import 'common.dart';

enum StepStatus { pending, running, done, error }

class PipelineStep {
  final String title;
  final String detail;
  final IconData icon;
  const PipelineStep(this.title, this.detail, this.icon);
}

/// Mirrors inference/inference_hit.py: phase 1 (ML) and phase 2 (LLM ensemble).
const mlSteps = [
  PipelineStep(
    'Peak 30 s window',
    'Highest-energy segment, matching Spotify previews',
    Icons.content_cut_rounded,
  ),
  PipelineStep(
    'Audio features',
    'Librosa (31) + Meta Audiobox Aesthetics (4)',
    Icons.graphic_eq_rounded,
  ),
  PipelineStep(
    'Talent agent',
    'Artist popularity & followers from Spotify dataset',
    Icons.person_search_rounded,
  ),
  PipelineStep(
    'XGBoost regression',
    'Popularity 0–100 with SHAP explanation',
    Icons.account_tree_rounded,
  ),
];

const llmSteps = [
  PipelineStep(
    'Gemini + GPT listen',
    'Both models receive the audio and features',
    Icons.headphones_rounded,
  ),
  PipelineStep(
    'DeepSeek consensus',
    'Meta-analysis of agreement & disagreement',
    Icons.hub_rounded,
  ),
];

class PipelineTracker extends StatelessWidget {
  final StepStatus mlStatus;
  final StepStatus llmStatus;
  final double? mlSeconds;
  final double? llmSeconds;

  const PipelineTracker({
    super.key,
    required this.mlStatus,
    required this.llmStatus,
    this.mlSeconds,
    this.llmSeconds,
  });

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'Pipeline',
      subtitle: 'Hybrid machine learning + LLM ensemble',
      icon: Icons.route_rounded,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PhaseHeader('Phase 1 · Machine learning', mlStatus, mlSeconds),
          for (var i = 0; i < mlSteps.length; i++)
            _StepRow(mlSteps[i], mlStatus, isLast: i == mlSteps.length - 1),
          const SizedBox(height: 14),
          _PhaseHeader('Phase 2 · LLM expert panel', llmStatus, llmSeconds),
          for (var i = 0; i < llmSteps.length; i++)
            _StepRow(llmSteps[i], llmStatus, isLast: i == llmSteps.length - 1),
        ],
      ),
    );
  }
}

class _PhaseHeader extends StatelessWidget {
  final String label;
  final StepStatus status;
  final double? seconds;
  const _PhaseHeader(this.label, this.status, this.seconds);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label.toUpperCase(),
              style: const TextStyle(
                fontSize: 11,
                letterSpacing: 1.1,
                fontWeight: FontWeight.w600,
                color: AppColors.textMuted,
              ),
            ),
          ),
          if (status == StepStatus.done && seconds != null)
            Text(
              '${seconds!.toStringAsFixed(1)} s',
              style: const TextStyle(fontSize: 12, color: AppColors.positive),
            ),
          if (status == StepStatus.error)
            const Text(
              'failed',
              style: TextStyle(fontSize: 12, color: AppColors.negative),
            ),
        ],
      ),
    );
  }
}

class _StepRow extends StatelessWidget {
  final PipelineStep step;
  final StepStatus status;
  final bool isLast;
  const _StepRow(this.step, this.status, {required this.isLast});

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      StepStatus.done => AppColors.positive,
      StepStatus.running => AppColors.violet,
      StepStatus.error => AppColors.negative,
      StepStatus.pending => AppColors.textMuted,
    };
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            width: 30,
            child: Column(
              children: [
                SizedBox(
                  width: 26,
                  height: 26,
                  child: _StatusIcon(status, step.icon, color),
                ),
                if (!isLast)
                  Expanded(
                    child: Container(
                      width: 1.5,
                      margin: const EdgeInsets.symmetric(vertical: 3),
                      color: status == StepStatus.done
                          ? AppColors.positive.withValues(alpha: 0.4)
                          : AppColors.border,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Padding(
              padding: EdgeInsets.only(bottom: isLast ? 0 : 12, top: 2),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    step.title,
                    style: TextStyle(
                      fontSize: 13.5,
                      fontWeight: FontWeight.w600,
                      color: status == StepStatus.pending
                          ? AppColors.textSecondary
                          : AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    step.detail,
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textMuted,
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

class _StatusIcon extends StatelessWidget {
  final StepStatus status;
  final IconData icon;
  final Color color;
  const _StatusIcon(this.status, this.icon, this.color);

  @override
  Widget build(BuildContext context) {
    final inner = switch (status) {
      StepStatus.done => const Icon(
        Icons.check_rounded,
        size: 15,
        color: AppColors.positive,
      ),
      StepStatus.error => const Icon(
        Icons.close_rounded,
        size: 15,
        color: AppColors.negative,
      ),
      _ => Icon(icon, size: 14, color: color),
    };
    return Stack(
      alignment: Alignment.center,
      children: [
        Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color.withValues(alpha: 0.12),
            border: Border.all(color: color.withValues(alpha: 0.5)),
          ),
        ),
        if (status == StepStatus.running)
          const SizedBox.expand(
            child: CircularProgressIndicator(
              strokeWidth: 1.8,
              color: AppColors.violet,
            ),
          ),
        inner,
      ],
    );
  }
}

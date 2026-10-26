import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/hit_song_api.dart';
import '../api/models.dart';
import '../theme.dart';
import '../widgets/common.dart';
import '../widgets/input_panel.dart';
import '../widgets/llm_section.dart';
import '../widgets/pipeline_tracker.dart';
import '../widgets/prediction_section.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = HitSongApi();
  final _artistController = TextEditingController();

  ModelInfo? _modelInfo;
  bool _backendOnline = true;

  PlatformFile? _file;
  Uint8List? _fileBytes;

  StepStatus _mlStatus = StepStatus.pending;
  StepStatus _llmStatus = StepStatus.pending;
  PredictionResult? _prediction;
  LlmAnalysis? _llm;
  String? _mlError;
  String? _llmError;

  Timer? _ticker;
  int _llmElapsed = 0;

  bool get _busy => _mlStatus == StepStatus.running;

  @override
  void initState() {
    super.initState();
    _checkBackend();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _artistController.dispose();
    super.dispose();
  }

  Future<void> _checkBackend() async {
    try {
      final info = await _api.health();
      if (mounted) setState(() => (_modelInfo = info, _backendOnline = true));
    } catch (_) {
      if (mounted) setState(() => _backendOnline = false);
    }
  }

  Future<void> _onFilePicked(PlatformFile file) async {
    final bytes = await file.readAsBytes();
    setState(() {
      _file = file;
      _fileBytes = bytes;
    });
  }

  Future<void> _runPrediction() async {
    final file = _file;
    final bytes = _fileBytes;
    if (file == null || bytes == null) return;

    _ticker?.cancel();
    setState(() {
      _mlStatus = StepStatus.running;
      _llmStatus = StepStatus.pending;
      _prediction = null;
      _llm = null;
      _mlError = null;
      _llmError = null;
    });

    try {
      final result = await _api.predict(
        artistName: _artistController.text.trim(),
        fileName: file.name,
        bytes: bytes,
      );
      if (!mounted) return;
      setState(() {
        _prediction = result;
        _mlStatus = StepStatus.done;
        _backendOnline = true;
      });
      _runLlm();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _mlStatus = StepStatus.error;
        _mlError = _describe(e);
      });
    }
  }

  Future<void> _runLlm() async {
    final jobId = _prediction?.jobId;
    if (jobId == null) return;

    _ticker?.cancel();
    setState(() {
      _llmStatus = StepStatus.running;
      _llmError = null;
      _llmElapsed = 0;
    });
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _llmElapsed++);
    });

    try {
      final result = await _api.analyze(jobId);
      if (!mounted || _prediction?.jobId != jobId) return;
      final failed = result.error != null && result.consensus.isEmpty;
      setState(() {
        _llm = result;
        _llmStatus = failed ? StepStatus.error : StepStatus.done;
        _llmError = failed ? result.error : null;
      });
    } catch (e) {
      if (!mounted || _prediction?.jobId != jobId) return;
      setState(() {
        _llmStatus = StepStatus.error;
        _llmError = _describe(e);
      });
    } finally {
      _ticker?.cancel();
    }
  }

  String _describe(Object e) {
    if (e is ApiException) return e.message;
    _backendOnline = false;
    return 'Cannot reach the prediction server at $apiBaseUrl. '
        'Start it with run_api.ps1 in the project root.';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          const _Backdrop(),
          SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final wide = constraints.maxWidth >= 1100;
                final hPad = constraints.maxWidth < 600 ? 16.0 : 32.0;
                final sidebar = _buildSidebar();
                final results = _buildResults();

                return SingleChildScrollView(
                  padding: EdgeInsets.fromLTRB(hPad, 24, hPad, 40),
                  child: Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 1440),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          _Header(
                            info: _modelInfo,
                            online: _backendOnline,
                            onRetry: _checkBackend,
                          ),
                          const SizedBox(height: 28),
                          if (wide)
                            Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                SizedBox(width: 380, child: sidebar),
                                const SizedBox(width: 24),
                                Expanded(child: results),
                              ],
                            )
                          else ...[
                            sidebar,
                            const SizedBox(height: 24),
                            results,
                          ],
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidebar() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InputPanel(
          api: _api,
          artistController: _artistController,
          file: _file,
          fileSize: _fileBytes?.length,
          busy: _busy,
          onFilePicked: _onFilePicked,
          onSubmit: _runPrediction,
        ),
        const SizedBox(height: 20),
        PipelineTracker(
          mlStatus: _mlStatus,
          llmStatus: _llmStatus,
          mlSeconds: _prediction?.elapsedSec,
          llmSeconds: _llm?.elapsedSec,
        ),
      ],
    );
  }

  Widget _buildResults() {
    final prediction = _prediction;

    if (_mlStatus == StepStatus.error) {
      return _MessageCard(
        icon: Icons.error_outline_rounded,
        color: AppColors.negative,
        title: 'Prediction failed',
        message: _mlError ?? 'Unknown error',
      );
    }
    if (_mlStatus == StepStatus.running) {
      return const _MessageCard(
        icon: Icons.graphic_eq_rounded,
        color: AppColors.violet,
        title: 'Extracting features and running XGBoost…',
        message:
            'Finding the peak 30-second window, computing Librosa and Audiobox '
            'features and looking up the artist.',
        loading: true,
      );
    }
    if (prediction == null) return const _EmptyState();

    const gap = SizedBox(height: 20);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        LayoutBuilder(
          builder: (context, c) {
            final score = ScoreCard(result: prediction);
            final artist = ArtistCard(
              artist: prediction.artist,
              window: prediction.window,
            );
            if (c.maxWidth < 760) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [score, gap, artist],
              );
            }
            return IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: score),
                  const SizedBox(width: 20),
                  Expanded(child: artist),
                ],
              ),
            );
          },
        ),
        gap,
        ExplainabilityCard(result: prediction),
        gap,
        AudioProfileCard(result: prediction),
        const SizedBox(height: 36),
        const _SectionDivider('LLM expert analysis'),
        const SizedBox(height: 20),
        ..._buildLlmCards(prediction),
      ],
    );
  }

  List<Widget> _buildLlmCards(PredictionResult prediction) {
    const gap = SizedBox(height: 20);
    switch (_llmStatus) {
      case StepStatus.pending:
      case StepStatus.running:
        return [LlmLoadingCard(elapsedSeconds: _llmElapsed)];
      case StepStatus.error:
        return [
          LlmErrorCard(message: _llmError ?? 'Unknown error', onRetry: _runLlm),
        ];
      case StepStatus.done:
        final llm = _llm!;
        return [
          if (llm.error != null) ...[
            _MessageCard(
              icon: Icons.warning_amber_rounded,
              color: AppColors.warning,
              title: 'Partial result',
              message: llm.error!,
            ),
            gap,
          ],
          ConsensusCard(analysis: llm),
          gap,
          AgreementCard(
            analysis: llm,
            xgboostPopularity: prediction.popularity,
          ),
          gap,
          ModelReportsCard(analysis: llm),
        ];
    }
  }
}

class _Header extends StatelessWidget {
  final ModelInfo? info;
  final bool online;
  final VoidCallback onRetry;
  const _Header({
    required this.info,
    required this.online,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final status = online && info != null
        ? Pill(
            '${info!.model} · ${info!.nFeatures} features · online',
            color: AppColors.positive,
            icon: Icons.circle,
          )
        : online
        ? const Pill('Connecting…', icon: Icons.circle)
        : InkWell(
            onTap: onRetry,
            borderRadius: BorderRadius.circular(99),
            child: const Pill(
              'Server offline · tap to retry',
              color: AppColors.negative,
              icon: Icons.circle,
            ),
          );

    final title = Row(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            gradient: AppColors.brandGradient,
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              BoxShadow(
                color: AppColors.violet.withValues(alpha: 0.4),
                blurRadius: 20,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: const Icon(
            Icons.equalizer_rounded,
            color: Colors.white,
            size: 28,
          ),
        ),
        const SizedBox(width: 14),
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Hit Song Predictor',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.3,
                ),
              ),
              SizedBox(height: 2),
              Text(
                'Song popularity forecasting with XGBoost and a multi-LLM expert panel',
                style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
              ),
            ],
          ),
        ),
      ],
    );

    return LayoutBuilder(
      builder: (context, c) => c.maxWidth < 760
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [title, const SizedBox(height: 14), status],
            )
          : Row(
              children: [
                Expanded(child: title),
                const SizedBox(width: 16),
                status,
              ],
            ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    const features = [
      (
        Icons.account_tree_rounded,
        'XGBoost regression',
        '38 features: Audiobox aesthetics, artist reach and Librosa descriptors.',
      ),
      (
        Icons.insights_rounded,
        'Explainable output',
        'TreeSHAP shows exactly which features raise or lower the score.',
      ),
      (
        Icons.headphones_rounded,
        'LLMs that listen',
        'Gemini and GPT-audio review the actual recording, not only the numbers.',
      ),
      (
        Icons.hub_rounded,
        'Consensus arbitration',
        'DeepSeek reconciles both experts into one final report.',
      ),
    ];
    return SectionCard(
      padding: const EdgeInsets.all(36),
      child: Column(
        children: [
          Container(
            width: 84,
            height: 84,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.violet.withValues(alpha: 0.12),
              border: Border.all(
                color: AppColors.violet.withValues(alpha: 0.35),
              ),
            ),
            child: const Icon(
              Icons.library_music_rounded,
              size: 38,
              color: AppColors.violet,
            ),
          ),
          const SizedBox(height: 20),
          const Text(
            'Predict the popularity of a song',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          const Text(
            'Enter the artist, upload an MP3 or WAV file and run the hybrid pipeline.',
            textAlign: TextAlign.center,
            style: TextStyle(color: AppColors.textSecondary),
          ),
          const SizedBox(height: 32),
          LayoutBuilder(
            builder: (context, c) {
              final columns = c.maxWidth > 700 ? 2 : 1;
              final width = (c.maxWidth - (columns - 1) * 16) / columns;
              return Wrap(
                spacing: 16,
                runSpacing: 16,
                children: [
                  for (final f in features)
                    SizedBox(
                      width: width,
                      child: Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: AppColors.surfaceHigh,
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(f.$1, color: AppColors.pink, size: 22),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    f.$2,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    f.$3,
                                    style: const TextStyle(
                                      fontSize: 12.5,
                                      height: 1.45,
                                      color: AppColors.textSecondary,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

class _MessageCard extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title;
  final String message;
  final bool loading;

  const _MessageCard({
    required this.icon,
    required this.color,
    required this.title,
    required this.message,
    this.loading = false,
  });

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      child: Row(
        children: [
          SizedBox(
            width: 44,
            height: 44,
            child: Stack(
              alignment: Alignment.center,
              children: [
                if (loading)
                  SizedBox.expand(
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      color: color,
                    ),
                  ),
                Icon(icon, color: color),
              ],
            ),
          ),
          const SizedBox(width: 18),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  message,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.45,
                    color: AppColors.textSecondary,
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

class _SectionDivider extends StatelessWidget {
  final String label;
  const _SectionDivider(this.label);

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Text(
        label.toUpperCase(),
        style: const TextStyle(
          fontSize: 12,
          letterSpacing: 1.4,
          fontWeight: FontWeight.w600,
          color: AppColors.textSecondary,
        ),
      ),
      const SizedBox(width: 14),
      const Expanded(child: Divider(height: 1)),
    ],
  );
}

class _Backdrop extends StatelessWidget {
  const _Backdrop();

  @override
  Widget build(BuildContext context) => Positioned.fill(
    child: IgnorePointer(
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: RadialGradient(
            center: const Alignment(-0.9, -1.1),
            radius: 1.4,
            colors: [
              AppColors.violet.withValues(alpha: 0.16),
              AppColors.background,
            ],
          ),
        ),
      ),
    ),
  );
}

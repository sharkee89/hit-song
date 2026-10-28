import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/hit_song_api.dart';
import '../api/models.dart';
import '../theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = HitSongApi();
  final _artistController = TextEditingController();

  String? _fileName;
  Uint8List? _fileBytes;

  bool _loading = false;
  String? _error;
  PredictionResult? _prediction;
  LlmAnalysis? _llm;
  String? _llmError;

  bool get _canSubmit =>
      !_loading &&
      _fileBytes != null &&
      _artistController.text.trim().isNotEmpty;

  @override
  void dispose() {
    _artistController.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    final file = await FilePicker.pickFile(
      type: FileType.custom,
      allowedExtensions: const ['mp3', 'wav'],
    );
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _fileName = file.name;
      _fileBytes = bytes;
    });
  }

  Future<void> _run() async {
    final bytes = _fileBytes;
    if (bytes == null) return;

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
      _llm = null;
      _llmError = null;
    });

    try {
      final prediction = await _api.predict(
        artistName: _artistController.text.trim(),
        fileName: _fileName!,
        bytes: bytes,
      );
      LlmAnalysis? llm;
      String? llmError;
      try {
        llm = await _api.analyze(prediction.jobId);
      } catch (e) {
        llmError = _describe(e);
      }
      if (!mounted) return;
      setState(() {
        _prediction = prediction;
        _llm = llm;
        _llmError = llmError;
      });
    } catch (e) {
      if (mounted) setState(() => _error = _describe(e));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _describe(Object e) => e is ApiException
      ? e.message
      : 'Cannot reach the prediction server at $apiBaseUrl.';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 32),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 760),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Hit Song Predictor',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 24),
                  _buildForm(),
                  const SizedBox(height: 28),
                  ..._buildResults(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildForm() {
    return _Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _artistController,
            enabled: !_loading,
            decoration: const InputDecoration(labelText: 'Artist name'),
            onChanged: (_) => setState(() {}),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              OutlinedButton.icon(
                onPressed: _loading ? null : _pickFile,
                icon: const Icon(Icons.upload_file_rounded),
                label: const Text('Upload audio'),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  _fileName ?? 'No file selected (MP3 or WAV)',
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: AppColors.textSecondary),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          FilledButton(
            onPressed: _canSubmit ? _run : null,
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            child: const Text('Predict'),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildResults() {
    if (_loading) {
      return const [
        Padding(
          padding: EdgeInsets.symmetric(vertical: 40),
          child: Center(child: CircularProgressIndicator()),
        ),
      ];
    }
    if (_error != null) {
      return [_Message(_error!, color: AppColors.negative)];
    }
    final prediction = _prediction;
    if (prediction == null) return const [];

    final reports = _llm?.reports ?? const <LlmReport>[];
    return [
      _ScoreCard(prediction),
      const SizedBox(height: 20),
      if (reports.isEmpty)
        _Message(
          _llmError ?? 'LLM explanations are not available for this track.',
        ),
      for (final report in reports) ...[
        _ReportCard(report),
        const SizedBox(height: 20),
      ],
    ];
  }
}

class _Card extends StatelessWidget {
  final Widget child;
  const _Card({required this.child});

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(24),
    decoration: BoxDecoration(
      color: AppColors.surface,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: AppColors.border),
    ),
    child: child,
  );
}

class _Message extends StatelessWidget {
  final String text;
  final Color color;
  const _Message(this.text, {this.color = AppColors.textSecondary});

  @override
  Widget build(BuildContext context) => _Card(
    child: Text(text, style: TextStyle(color: color, height: 1.45)),
  );
}

class _ScoreCard extends StatelessWidget {
  final PredictionResult result;
  const _ScoreCard(this.result);

  @override
  Widget build(BuildContext context) {
    final color = AppColors.score(result.popularity);
    return _Card(
      child: Column(
        children: [
          const Text(
            'Predicted popularity (XGBoost)',
            style: TextStyle(color: AppColors.textSecondary),
          ),
          const SizedBox(height: 8),
          Text.rich(
            TextSpan(
              children: [
                TextSpan(
                  text: result.popularity.toStringAsFixed(1),
                  style: TextStyle(
                    fontSize: 56,
                    fontWeight: FontWeight.w700,
                    color: color,
                  ),
                ),
                const TextSpan(
                  text: ' / 100',
                  style: TextStyle(
                    fontSize: 20,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text(
            result.tier,
            style: TextStyle(fontWeight: FontWeight.w600, color: color),
          ),
          const Divider(height: 32),
          Row(
            children: [
              Expanded(
                child: _Stat(
                  value: '${(result.hitProbability * 100).toStringAsFixed(0)}%',
                  label:
                      'Hit probability\n(popularity ≥ ${result.hitThreshold})',
                ),
              ),
              Expanded(
                child: _Stat(
                  value: '${result.percentile.toStringAsFixed(0)}%',
                  label: 'Scores higher than\nthis share of songs',
                ),
              ),
            ],
          ),
          if (!result.artistFound) ...[
            const SizedBox(height: 20),
            const Text(
              'Artist not found in the dataset. A typical artist profile was '
              'assumed, so the prediction is based mostly on the audio.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 13,
                height: 1.45,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String value;
  final String label;
  const _Stat({required this.value, required this.label});

  @override
  Widget build(BuildContext context) => Column(
    children: [
      Text(
        value,
        style: const TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w700,
          color: AppColors.textPrimary,
        ),
      ),
      const SizedBox(height: 4),
      Text(
        label,
        textAlign: TextAlign.center,
        style: const TextStyle(
          fontSize: 12.5,
          height: 1.4,
          color: AppColors.textSecondary,
        ),
      ),
    ],
  );
}

class _ReportCard extends StatelessWidget {
  final LlmReport report;
  const _ReportCard(this.report);

  @override
  Widget build(BuildContext context) {
    return _Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            report.name,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          if (report.metrics.isEmpty)
            SelectableText(
              report.raw ?? '',
              style: const TextStyle(height: 1.5),
            ),
          for (final (i, metric) in report.metrics.indexed) ...[
            if (i > 0) const Divider(height: 28),
            Row(
              children: [
                Expanded(
                  child: Text(
                    metric.label,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
                if (metric.score != null)
                  Text(
                    metric.score!.toStringAsFixed(0),
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: AppColors.score(metric.score!),
                    ),
                  ),
              ],
            ),
            if (metric.analysis.isNotEmpty) ...[
              const SizedBox(height: 6),
              SelectableText(
                metric.analysis,
                style: const TextStyle(
                  height: 1.5,
                  color: AppColors.textSecondary,
                ),
              ),
            ],
          ],
        ],
      ),
    );
  }
}

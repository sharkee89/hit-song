double _d(dynamic v) => v == null ? 0.0 : (v as num).toDouble();

class PredictionResult {
  final String jobId;
  final double popularity;
  final String tier;
  final double hitProbability;
  final int hitThreshold;
  final double percentile;
  final bool artistFound;

  PredictionResult.fromJson(Map<String, dynamic> j)
    : jobId = j['job_id'] as String,
      popularity = _d(j['prediction']['popularity']),
      tier = j['prediction']['tier'] as String,
      hitProbability = _d(j['prediction']['hit_probability']),
      hitThreshold = (j['prediction']['hit_threshold'] as num?)?.toInt() ?? 60,
      percentile = _d(j['prediction']['percentile']),
      artistFound = j['prediction']['artist_found'] as bool? ?? true;
}

class LlmMetric {
  final String label;
  final double? score;
  final String analysis;

  LlmMetric.fromJson(Map<String, dynamic> j)
    : label = j['label'] as String,
      score = (j['score'] as num?)?.toDouble(),
      analysis = j['analysis'] as String? ?? '';
}

class LlmReport {
  final String name;
  final bool ok;
  final List<LlmMetric> metrics;
  final String? raw;

  LlmReport.fromJson(this.name, Map<String, dynamic> j)
    : ok = j['ok'] as bool? ?? false,
      metrics = (j['metrics'] as List? ?? const [])
          .map((e) => LlmMetric.fromJson(e as Map<String, dynamic>))
          .toList(),
      raw = j['raw'] as String?;
}

class LlmAnalysis {
  /// Only the models that returned a usable answer, consensus first.
  final List<LlmReport> reports;

  LlmAnalysis.fromJson(Map<String, dynamic> j)
    : reports = [
        for (final (key, name) in const [
          ('deepseek', 'DeepSeek consensus'),
          ('gemini', 'Gemini'),
          ('gpt', 'GPT'),
        ])
          if (j['models']?[key] case final Map<String, dynamic> m)
            LlmReport.fromJson(name, m),
      ].where((r) => r.ok).toList();
}

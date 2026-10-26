double _d(dynamic v) => v == null ? 0.0 : (v as num).toDouble();

class ArtistSuggestion {
  final String name;
  final double popularity;
  final int followers;

  ArtistSuggestion.fromJson(Map<String, dynamic> j)
    : name = j['name'] as String,
      popularity = _d(j['artist_popularity']),
      followers = (j['followers'] as num?)?.toInt() ?? 0;
}

class ModelInfo {
  final String model;
  final int nFeatures;
  final int? nEstimators;
  final int? maxDepth;
  final List<String> llms;

  ModelInfo.fromJson(Map<String, dynamic> j)
    : model = j['model'] as String? ?? 'XGBoost',
      nFeatures = (j['n_features'] as num?)?.toInt() ?? 0,
      nEstimators = (j['n_estimators'] as num?)?.toInt(),
      maxDepth = (j['max_depth'] as num?)?.toInt(),
      llms = List<String>.from(j['llms'] as List? ?? const []);
}

class ArtistInfo {
  final String name;
  final bool foundInDataset;
  final double? popularity;
  final int? followers;
  final List<String> genres;
  final String? country;

  ArtistInfo.fromJson(Map<String, dynamic> j)
    : name = j['name'] as String? ?? '',
      foundInDataset = j['found_in_dataset'] as bool? ?? false,
      popularity = (j['artist_popularity'] as num?)?.toDouble(),
      followers = (j['followers'] as num?)?.toInt(),
      genres = List<String>.from(j['genres'] as List? ?? const []),
      country = j['country'] as String?;
}

class Contribution {
  final String feature;
  final String label;
  final String group;
  final double value;
  final double contribution;

  Contribution.fromJson(Map<String, dynamic> j)
    : feature = j['feature'] as String,
      label = j['label'] as String,
      group = j['group'] as String,
      value = _d(j['value']),
      contribution = _d(j['contribution']);
}

class AnalysisWindow {
  final double startSec;
  final double endSec;
  final double totalSec;

  AnalysisWindow.fromJson(Map<String, dynamic> j)
    : startSec = _d(j['start_sec']),
      endSec = _d(j['end_sec']),
      totalSec = _d(j['total_duration_sec']);
}

class PredictionResult {
  final String jobId;
  final String fileName;
  final double elapsedSec;
  final double popularity;
  final String tier;
  final double baseValue;
  final AnalysisWindow window;
  final ArtistInfo artist;
  final Map<String, double> audiobox;
  final Map<String, double> audioFeatures;
  final List<double> chroma;
  final List<Contribution> contributions;
  final Map<String, double> groupContributions;

  PredictionResult.fromJson(Map<String, dynamic> j)
    : jobId = j['job_id'] as String,
      fileName = j['file_name'] as String? ?? '',
      elapsedSec = _d(j['elapsed_sec']),
      popularity = _d(j['prediction']['popularity']),
      tier = j['prediction']['tier'] as String,
      baseValue = _d(j['prediction']['base_value']),
      window = AnalysisWindow.fromJson(j['window'] as Map<String, dynamic>),
      artist = ArtistInfo.fromJson(j['artist'] as Map<String, dynamic>),
      audiobox = (j['audiobox'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, _d(v)),
      ),
      audioFeatures = (j['audio_features'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, _d(v)),
      ),
      chroma = (j['chroma'] as List).map(_d).toList(),
      contributions = (j['contributions'] as List)
          .map((e) => Contribution.fromJson(e as Map<String, dynamic>))
          .toList(),
      groupContributions = (j['group_contributions'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, _d(v)));
}

class LlmMetric {
  final String key;
  final String label;
  final double? score;
  final String analysis;

  LlmMetric.fromJson(Map<String, dynamic> j)
    : key = j['key'] as String,
      label = j['label'] as String,
      score = (j['score'] as num?)?.toDouble(),
      analysis = j['analysis'] as String? ?? '';
}

class LlmReport {
  final List<LlmMetric> metrics;
  final String? raw;
  final bool parsed;

  LlmReport.fromJson(Map<String, dynamic> j)
    : metrics = (j['metrics'] as List? ?? const [])
          .map((e) => LlmMetric.fromJson(e as Map<String, dynamic>))
          .toList(),
      raw = j['raw'] as String?,
      parsed = j['parsed'] as bool? ?? false;

  LlmMetric? metric(String key) {
    for (final m in metrics) {
      if (m.key == key) return m;
    }
    return null;
  }

  bool get isEmpty => metrics.isEmpty && (raw == null || raw!.isEmpty);
}

class LlmAnalysis {
  final String? error;
  final double elapsedSec;
  final LlmReport consensus;
  final LlmReport gemini;
  final LlmReport gpt;

  LlmAnalysis.fromJson(Map<String, dynamic> j)
    : error = j['error'] as String?,
      elapsedSec = _d(j['elapsed_sec']),
      consensus = LlmReport.fromJson(j['consensus'] as Map<String, dynamic>),
      gemini = LlmReport.fromJson(
        j['models']['gemini'] as Map<String, dynamic>,
      ),
      gpt = LlmReport.fromJson(j['models']['gpt'] as Map<String, dynamic>);
}

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

/// Python backend (api/server.py). Override with:
///   flutter run -d chrome --dart-define=API_URL=http://HOST:8000
const String apiBaseUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://localhost:8000',
);

class ApiException implements Exception {
  final String message;
  ApiException(this.message);

  @override
  String toString() => message;
}

class HitSongApi {
  final String baseUrl;
  final http.Client _client = http.Client();

  HitSongApi({this.baseUrl = apiBaseUrl});

  Uri _uri(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$path').replace(queryParameters: query);

  Future<ModelInfo> health() async {
    final res = await _client
        .get(_uri('/api/health'))
        .timeout(const Duration(seconds: 5));
    return ModelInfo.fromJson(_decode(res) as Map<String, dynamic>);
  }

  Future<List<ArtistSuggestion>> searchArtists(String query) async {
    if (query.trim().length < 2) return const [];
    final res = await _client
        .get(_uri('/api/artists', {'q': query}))
        .timeout(const Duration(seconds: 5));
    return (_decode(res) as List)
        .map((e) => ArtistSuggestion.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// Phase 1: peak 30 s window -> Librosa + Audiobox -> TalentAgent -> XGBoost.
  Future<PredictionResult> predict({
    required String artistName,
    required String fileName,
    required Uint8List bytes,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/api/predict'))
      ..fields['artist_name'] = artistName
      ..files.add(
        http.MultipartFile.fromBytes('audio', bytes, filename: fileName),
      );
    final streamed = await _client
        .send(request)
        .timeout(const Duration(minutes: 3));
    final res = await http.Response.fromStream(streamed);
    return PredictionResult.fromJson(_decode(res) as Map<String, dynamic>);
  }

  /// Phase 2: Gemini + GPT listen to the track, DeepSeek builds the consensus.
  Future<LlmAnalysis> analyze(String jobId) async {
    final res = await _client
        .post(_uri('/api/analysis/$jobId'))
        .timeout(const Duration(minutes: 10));
    return LlmAnalysis.fromJson(_decode(res) as Map<String, dynamic>);
  }

  dynamic _decode(http.Response res) {
    final body = utf8.decode(res.bodyBytes);
    if (res.statusCode >= 200 && res.statusCode < 300) return jsonDecode(body);
    try {
      final detail = (jsonDecode(body) as Map<String, dynamic>)['detail'];
      throw ApiException(
        detail?.toString() ?? 'Request failed (${res.statusCode}).',
      );
    } on FormatException {
      throw ApiException('Request failed (${res.statusCode}).');
    }
  }
}

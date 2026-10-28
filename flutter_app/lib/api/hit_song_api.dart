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

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  /// XGBoost popularity prediction.
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

  /// LLM explanations (Gemini, GPT, DeepSeek consensus) for the same upload.
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

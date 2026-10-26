import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/hit_song_api.dart';
import '../api/models.dart';
import '../theme.dart';
import 'common.dart';

class InputPanel extends StatefulWidget {
  final HitSongApi api;
  final TextEditingController artistController;
  final PlatformFile? file;
  final int? fileSize;
  final bool busy;
  final ValueChanged<PlatformFile> onFilePicked;
  final VoidCallback onSubmit;

  const InputPanel({
    super.key,
    required this.api,
    required this.artistController,
    required this.file,
    required this.fileSize,
    required this.busy,
    required this.onFilePicked,
    required this.onSubmit,
  });

  @override
  State<InputPanel> createState() => _InputPanelState();
}

class _InputPanelState extends State<InputPanel> {
  final FocusNode _artistFocus = FocusNode();
  Timer? _debounce;
  Completer<List<ArtistSuggestion>>? _pending;

  @override
  void initState() {
    super.initState();
    widget.artistController.addListener(_rebuild);
  }

  @override
  void dispose() {
    widget.artistController.removeListener(_rebuild);
    _debounce?.cancel();
    _artistFocus.dispose();
    super.dispose();
  }

  void _rebuild() => setState(() {});

  Future<List<ArtistSuggestion>> _suggest(String query) {
    _debounce?.cancel();
    if (_pending != null && !_pending!.isCompleted)
      _pending!.complete(const []);
    final completer = _pending = Completer<List<ArtistSuggestion>>();
    _debounce = Timer(const Duration(milliseconds: 250), () async {
      try {
        final results = await widget.api.searchArtists(query);
        if (!completer.isCompleted) completer.complete(results);
      } catch (_) {
        if (!completer.isCompleted) completer.complete(const []);
      }
    });
    return completer.future;
  }

  Future<void> _pickFile() async {
    final file = await FilePicker.pickFile(
      type: FileType.custom,
      allowedExtensions: const ['mp3', 'wav'],
    );
    if (file != null) widget.onFilePicked(file);
  }

  bool get _canSubmit =>
      !widget.busy &&
      widget.file != null &&
      widget.artistController.text.trim().isNotEmpty;

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'New analysis',
      subtitle: 'Artist + audio file → ML prediction and LLM expert review',
      icon: Icons.tune_rounded,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _FieldLabel('Artist name'),
          LayoutBuilder(
            builder: (context, constraints) =>
                RawAutocomplete<ArtistSuggestion>(
                  textEditingController: widget.artistController,
                  focusNode: _artistFocus,
                  displayStringForOption: (a) => a.name,
                  optionsBuilder: (value) => _suggest(value.text),
                  fieldViewBuilder:
                      (context, controller, focusNode, onSubmitted) =>
                          TextField(
                            controller: controller,
                            focusNode: focusNode,
                            enabled: !widget.busy,
                            textInputAction: TextInputAction.done,
                            onSubmitted: (_) =>
                                _canSubmit ? widget.onSubmit() : null,
                            decoration: const InputDecoration(
                              hintText: 'e.g. Charlie Puth',
                              prefixIcon: Icon(
                                Icons.person_outline_rounded,
                                color: AppColors.textSecondary,
                              ),
                            ),
                          ),
                  optionsViewBuilder: (context, onSelected, options) => Align(
                    alignment: Alignment.topLeft,
                    child: Material(
                      color: AppColors.surfaceHigh,
                      elevation: 8,
                      borderRadius: BorderRadius.circular(12),
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          maxWidth: constraints.maxWidth,
                          maxHeight: 320,
                        ),
                        child: ListView(
                          padding: const EdgeInsets.symmetric(vertical: 6),
                          shrinkWrap: true,
                          children: [
                            for (final a in options)
                              ListTile(
                                dense: true,
                                onTap: () => onSelected(a),
                                leading: const Icon(
                                  Icons.mic_external_on_rounded,
                                  size: 18,
                                  color: AppColors.textSecondary,
                                ),
                                title: Text(a.name),
                                subtitle: Text(
                                  '${formatCompact(a.followers)} followers',
                                  style: const TextStyle(
                                    color: AppColors.textMuted,
                                    fontSize: 12,
                                  ),
                                ),
                                trailing: Pill(
                                  'pop ${a.popularity.round()}',
                                  color: AppColors.score(a.popularity),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Suggestions come from the Spotify artist dataset used by the TalentAgent.',
            style: TextStyle(fontSize: 11.5, color: AppColors.textMuted),
          ),
          const SizedBox(height: 20),
          const _FieldLabel('Audio file'),
          _UploadZone(
            file: widget.file,
            fileSize: widget.fileSize,
            enabled: !widget.busy,
            onTap: _pickFile,
          ),
          const SizedBox(height: 24),
          _GradientButton(
            label: widget.busy ? 'Analyzing…' : 'Predict popularity',
            icon: widget.busy ? null : Icons.auto_graph_rounded,
            busy: widget.busy,
            onPressed: _canSubmit ? widget.onSubmit : null,
          ),
        ],
      ),
    );
  }
}

class _FieldLabel extends StatelessWidget {
  final String text;
  const _FieldLabel(this.text);

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Text(
      text,
      style: const TextStyle(
        fontSize: 12.5,
        color: AppColors.textSecondary,
        fontWeight: FontWeight.w500,
      ),
    ),
  );
}

class _UploadZone extends StatefulWidget {
  final PlatformFile? file;
  final int? fileSize;
  final bool enabled;
  final VoidCallback onTap;

  const _UploadZone({
    required this.file,
    required this.fileSize,
    required this.enabled,
    required this.onTap,
  });

  @override
  State<_UploadZone> createState() => _UploadZoneState();
}

class _UploadZoneState extends State<_UploadZone> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    final file = widget.file;
    final active = _hover && widget.enabled;
    return MouseRegion(
      cursor: widget.enabled
          ? SystemMouseCursors.click
          : SystemMouseCursors.basic,
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: GestureDetector(
        onTap: widget.enabled ? widget.onTap : null,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 22),
          decoration: BoxDecoration(
            color: active
                ? AppColors.violet.withValues(alpha: 0.08)
                : AppColors.surfaceHigh,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: file != null
                  ? AppColors.positive.withValues(alpha: 0.6)
                  : active
                  ? AppColors.violet
                  : AppColors.border,
              width: 1.3,
            ),
          ),
          child: file == null
              ? const Column(
                  children: [
                    Icon(
                      Icons.cloud_upload_outlined,
                      size: 34,
                      color: AppColors.violet,
                    ),
                    SizedBox(height: 10),
                    Text(
                      'Choose an audio file',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 14,
                      ),
                    ),
                    SizedBox(height: 4),
                    Text(
                      'MP3 or WAV · up to 50 MB',
                      style: TextStyle(
                        fontSize: 12,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                )
              : Row(
                  children: [
                    Container(
                      width: 42,
                      height: 42,
                      decoration: BoxDecoration(
                        gradient: AppColors.brandGradient,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.music_note_rounded,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            file.name,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            widget.fileSize == null
                                ? (file.extension ?? '').toUpperCase()
                                : '${(file.extension ?? '').toUpperCase()} · '
                                      '${(widget.fileSize! / 1048576).toStringAsFixed(1)} MB',
                            style: const TextStyle(
                              fontSize: 12,
                              color: AppColors.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Text(
                      'Change',
                      style: TextStyle(
                        color: AppColors.violet,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

class _GradientButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final bool busy;
  final VoidCallback? onPressed;

  const _GradientButton({
    required this.label,
    required this.icon,
    required this.busy,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 200),
      opacity: enabled || busy ? 1 : 0.45,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: AppColors.brandGradient,
          borderRadius: BorderRadius.circular(14),
          boxShadow: enabled
              ? [
                  BoxShadow(
                    color: AppColors.pink.withValues(alpha: 0.28),
                    blurRadius: 24,
                    offset: const Offset(0, 8),
                  ),
                ]
              : null,
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(14),
            onTap: onPressed,
            child: SizedBox(
              height: 54,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (busy)
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2.2,
                        color: Colors.white,
                      ),
                    )
                  else if (icon != null)
                    Icon(icon, color: Colors.white, size: 20),
                  const SizedBox(width: 10),
                  Text(
                    label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/chat_message.dart';
import '../providers/chat_provider.dart';
import '../providers/garden_provider.dart';
import '../providers/notifications_provider.dart';
import '../providers/settings_provider.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  final _scroll = ScrollController();
  int _lastMessageCount = 0;
  String _selectedModel = 'vgg16';

  static const _inputBarHeight = 48.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ChatProvider>().loadHistory();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 240),
        curve: Curves.easeOutCubic,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatProvider>();
    final scheme = Theme.of(context).colorScheme;

    if (chat.messages.length != _lastMessageCount) {
      _lastMessageCount = chat.messages.length;
      _scrollToEnd();
    }

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              'AI',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    height: 1.0,
                  ),
            ),
            const SizedBox(width: 8),
            PopupMenuButton<String>(
              tooltip: 'Chọn model AI',
              padding: EdgeInsets.zero,
              initialValue: _selectedModel,
              onSelected: chat.sending
                  ? null
                  : (value) {
                      setState(() => _selectedModel = value);
                    },
              itemBuilder: (context) => const [
                PopupMenuItem(
                  value: 'vgg16',
                  child: Text('VGG16'),
                ),
                PopupMenuItem(
                  value: 'resnet',
                  child: Text('ResNet'),
                ),
              ],
              child: SizedBox(
                height: 30,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: scheme.outline.withValues(alpha: 0.45)),
                    color: scheme.surfaceContainerHighest,
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Text(
                        _selectedModel == 'vgg16' ? 'VGG16' : 'ResNet',
                        style: Theme.of(context).textTheme.labelMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                              height: 1.0,
                            ),
                      ),
                      const SizedBox(width: 2),
                      const Icon(Icons.expand_more_rounded, size: 18),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    scheme.surface,
                    scheme.surfaceContainerLow,
                  ],
                ),
              ),
              child: chat.loadingHistory
                  ? const Center(child: CircularProgressIndicator())
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.fromLTRB(18, 16, 18, 12),
                      itemCount: chat.messages.length,
                      itemBuilder: (context, i) {
                        return _Bubble(message: chat.messages[i]);
                      },
                    ),
            ),
          ),
          if (chat.lastError != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  chat.lastError!,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: scheme.error,
                        fontWeight: FontWeight.w600,
                      ),
                ),
              ),
            ),
          SizedBox(
            height: 44,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              itemCount: ChatProvider.suggestions.length,
              separatorBuilder: (_, __) => const SizedBox(width: 8),
              itemBuilder: (context, i) {
                final s = ChatProvider.suggestions[i];
                return OutlinedButton(
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                  onPressed: chat.sending
                      ? null
                      : () async {
                          if (s == 'Kiểm tra sức khỏe cây') {
                            final garden = context.read<GardenProvider>();
                            final settings = context.read<SettingsProvider>();
                            final preferredImage = (garden.latestImageUrl?.trim().isNotEmpty ?? false)
                                ? garden.latestImageUrl!.trim()
                                : settings.cameraUrl.trim();
                            final reply = await context.read<ChatProvider>().analyzeCurrentCameraImage(
                                  model: _selectedModel,
                                  preferredImageUrl:
                                      preferredImage.isEmpty ? null : preferredImage,
                                );
                            if (!context.mounted) return;
                            if (reply != null && reply.trim().isNotEmpty) {
                              context.read<GardenProvider>().setAiAnalysisFromServer(reply.trim());
                              await context.read<NotificationsProvider>().add(
                                    title: 'Phân tích AI',
                                    body: reply.trim(),
                                  );
                            }
                            return;
                          }
                          final garden = context.read<GardenProvider>();
                          final reply = await context.read<ChatProvider>().runSmartSuggestion(
                                intent: s,
                                garden: garden,
                              );
                          if (!context.mounted) return;
                          if (reply != null && reply.trim().isNotEmpty) {
                            await context.read<NotificationsProvider>().add(
                                  title: 'Phân tích AI',
                                  body: reply.trim(),
                                );
                          }
                        },
                  child: Text(
                    s,
                    style: Theme.of(context).textTheme.labelMedium,
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 6),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: scheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(22),
                border: Border.all(
                  color: scheme.outline.withValues(alpha: 0.38),
                ),
                boxShadow: [
                  BoxShadow(
                    color: scheme.primary.withValues(alpha: 0.06),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(4, 6, 8, 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(
                      height: _inputBarHeight,
                      child: TextButton(
                        style: TextButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          minimumSize: Size.zero,
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          alignment: Alignment.center,
                        ),
                        onPressed: chat.sending
                            ? null
                            : () async {
                                final reply = await context
                                    .read<ChatProvider>()
                                    .pickImageAndPredict(
                                      model: _selectedModel,
                                    );
                                if (!context.mounted) return;
                                if (reply != null && reply.trim().isNotEmpty) {
                                  context
                                      .read<GardenProvider>()
                                      .setAiAnalysisFromServer(reply.trim());
                                  await context
                                      .read<NotificationsProvider>()
                                      .add(
                                        title: 'Phân tích AI',
                                        body: reply.trim(),
                                      );
                                }
                              },
                        child: const Text('Ảnh'),
                      ),
                    ),
                    Expanded(
                      child: Align(
                        alignment: Alignment.center,
                        child: TextField(
                          controller: _controller,
                          minLines: 1,
                          maxLines: 4,
                          textAlignVertical: TextAlignVertical.center,
                          textInputAction: TextInputAction.send,
                          decoration: InputDecoration(
                            hintText: 'Nhập tin nhắn',
                            isDense: true,
                            filled: true,
                            fillColor: Colors.white,
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: BorderSide.none,
                            ),
                            enabledBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: BorderSide.none,
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: BorderSide(
                                color: scheme.primary.withValues(alpha: 0.45),
                                width: 1.2,
                              ),
                            ),
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 14,
                              vertical: 14,
                            ),
                          ),
                          onSubmitted: (_) => _send(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                    SizedBox(
                      height: _inputBarHeight,
                      child: FilledButton(
                        style: FilledButton.styleFrom(
                          minimumSize: const Size(52, _inputBarHeight),
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          alignment: Alignment.center,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                        onPressed: chat.sending ? null : _send,
                        child: chat.sending
                            ? SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2.2,
                                  color: scheme.onPrimary,
                                ),
                              )
                            : const Text('Gửi'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _send() async {
    final t = _controller.text;
    _controller.clear();
    await context.read<ChatProvider>().sendUserText(t);
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isUser = message.senderType == SenderType.user;
    final time =
        '${message.createdAt.hour.toString().padLeft(2, '0')}:${message.createdAt.minute.toString().padLeft(2, '0')}';

    final radius = BorderRadius.only(
      topLeft: const Radius.circular(20),
      topRight: const Radius.circular(20),
      bottomLeft: Radius.circular(isUser ? 20 : 5),
      bottomRight: Radius.circular(isUser ? 5 : 20),
    );

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.sizeOf(context).width * 0.84,
        ),
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: isUser
                ? scheme.primary.withValues(alpha: 0.14)
                : scheme.surfaceContainerHighest,
            borderRadius: radius,
            border: Border.all(
              color: scheme.outline.withValues(alpha: isUser ? 0.22 : 0.4),
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.04),
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isUser ? 'Bạn' : 'Hệ thống',
                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      letterSpacing: 0.6,
                      fontWeight: FontWeight.w800,
                      color: scheme.onSurface.withValues(alpha: 0.42),
                    ),
              ),
              const SizedBox(height: 8),
              if (message.localImagePath != null &&
                  message.localImagePath!.trim().isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Image.file(
                      File(message.localImagePath!),
                      height: 140,
                      width: double.infinity,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        height: 72,
                        alignment: Alignment.center,
                        color: scheme.surfaceContainerLow,
                        child: Text(
                          'Không tải được ảnh đính kèm',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ),
                    ),
                  ),
                ),
              Text(
                message.text,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      height: 1.45,
                      fontWeight: FontWeight.w500,
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                time,
                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: scheme.onSurface.withValues(alpha: 0.38),
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

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:home_widget/home_widget.dart';
import 'package:quick_actions/quick_actions.dart';

/// Points d'entrée depuis l'écran d'accueil du téléphone :
///
///  * le widget SOS (Android AppWidget / iOS WidgetKit) ;
///  * le raccourci obtenu par appui long sur l'icône de l'app.
///
/// Les deux ouvrent directement l'écran d'alerte, sans passer par l'accueil.
class HomeEntryService {
  HomeEntryService();

  /// Doit correspondre à l'App Group configuré dans Xcode (cible principale
  /// + extension widget) — voir README, section iOS.
  static const String iosAppGroupId = 'group.com.astrotarget.asthmalerte';

  static const String androidWidgetName = 'SosWidgetProvider';
  static const String iosWidgetName = 'SosWidget';

  static const String quickActionSos = 'action_sos';

  final _sosRequests = StreamController<String>.broadcast();
  final QuickActions _quickActions = const QuickActions();
  StreamSubscription<Uri?>? _widgetSubscription;

  /// Émet à chaque demande d'alerte venue de l'extérieur de l'app.
  Stream<String> get sosRequests => _sosRequests.stream;

  Future<void> initialize() async {
    await _initWidget();
    await _initQuickActions();
  }

  Future<void> _initWidget() async {
    try {
      await HomeWidget.setAppGroupId(iosAppGroupId);
      await _widgetSubscription?.cancel();
      _widgetSubscription = HomeWidget.widgetClicked.listen(_onWidgetUri);
      await refreshWidget();
    } catch (e) {
      debugPrint('Widget indisponible : $e');
    }
  }

  Future<void> _initQuickActions() async {
    try {
      _quickActions.initialize((type) {
        if (type == quickActionSos) _sosRequests.add('shortcut');
      });
      await _quickActions.setShortcutItems(const <ShortcutItem>[
        ShortcutItem(
          type: quickActionSos,
          localizedTitle: 'Alerte SOS',
          icon: 'ic_shortcut_sos',
        ),
      ]);
    } catch (e) {
      debugPrint('Raccourcis indisponibles : $e');
    }
  }

  /// L'app a-t-elle été lancée par un appui sur le widget ?
  Future<bool> launchedFromWidget() async {
    try {
      final uri = await HomeWidget.initiallyLaunchedFromHomeWidget();
      return _isSosUri(uri);
    } catch (_) {
      return false;
    }
  }

  /// Met à jour le libellé affiché dans le widget (nombre de proches prêts).
  Future<void> refreshWidget({int contactCount = 0}) async {
    try {
      await HomeWidget.saveWidgetData<String>(
        'sos_subtitle',
        contactCount > 0
            ? '$contactCount proche${contactCount > 1 ? 's' : ''} prévenu${contactCount > 1 ? 's' : ''}'
            : 'Ajouter un proche',
      );
      await HomeWidget.updateWidget(
        androidName: androidWidgetName,
        iOSName: iosWidgetName,
      );
    } catch (e) {
      debugPrint('Rafraîchissement du widget impossible : $e');
    }
  }

  void _onWidgetUri(Uri? uri) {
    if (_isSosUri(uri)) _sosRequests.add('widget');
  }

  bool _isSosUri(Uri? uri) =>
      uri != null && (uri.host == 'sos' || uri.path.contains('sos'));

  Future<void> dispose() async {
    await _widgetSubscription?.cancel();
    await _sosRequests.close();
  }
}

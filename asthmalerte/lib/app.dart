import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'screens/home_screen.dart';
import 'screens/onboarding_screen.dart';
import 'screens/sos_screen.dart';
import 'services/alert_service.dart';
import 'services/home_entry_service.dart';
import 'state/app_state.dart';
import 'theme.dart';

/// Accès à l'état et aux services depuis n'importe quel écran.
class AppScope extends InheritedNotifier<AppState> {
  const AppScope({
    super.key,
    required AppState appState,
    required this.alertService,
    required this.homeEntry,
    required super.child,
  }) : super(notifier: appState);

  final AlertService alertService;
  final HomeEntryService homeEntry;

  static AppScope of(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<AppScope>();
    assert(scope != null, 'AppScope manquant au-dessus de ce widget');
    return scope!;
  }

  AppState get state => notifier!;
}

class AsthmAlerteApp extends StatefulWidget {
  const AsthmAlerteApp({
    super.key,
    required this.appState,
    required this.homeEntry,
    this.openSosOnStart = false,
  });

  final AppState appState;
  final HomeEntryService homeEntry;

  /// L'app a été ouverte via le widget d'écran d'accueil ou le raccourci.
  final bool openSosOnStart;

  @override
  State<AsthmAlerteApp> createState() => _AsthmAlerteAppState();
}

class _AsthmAlerteAppState extends State<AsthmAlerteApp> {
  final GlobalKey<NavigatorState> _navigatorKey = GlobalKey<NavigatorState>();
  late final AlertService _alertService =
      AlertService(appState: widget.appState);
  StreamSubscription<String>? _sosSubscription;

  @override
  void initState() {
    super.initState();
    _sosSubscription = widget.homeEntry.sosRequests.listen((_) => _openSos());
    widget.appState.addListener(_syncWidget);
    if (widget.openSosOnStart) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _openSos());
    }
  }

  @override
  void dispose() {
    widget.appState.removeListener(_syncWidget);
    _sosSubscription?.cancel();
    super.dispose();
  }

  void _syncWidget() {
    widget.homeEntry
        .refreshWidget(contactCount: widget.appState.smsContacts.length);
  }

  void _openSos() {
    final navigator = _navigatorKey.currentState;
    if (navigator == null) return;
    // Une seule alerte à la fois : on ne réempile pas l'écran.
    navigator.popUntil((route) => route.isFirst);
    navigator.push(MaterialPageRoute<void>(
      builder: (_) => const SosScreen(autoStart: true),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return AppScope(
      appState: widget.appState,
      alertService: _alertService,
      homeEntry: widget.homeEntry,
      child: MaterialApp(
        navigatorKey: _navigatorKey,
        title: 'AsthmAlerte',
        debugShowCheckedModeBanner: false,
        locale: const Locale('fr', 'FR'),
        supportedLocales: const [Locale('fr', 'FR'), Locale('en')],
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        theme: buildAppTheme(Brightness.light),
        darkTheme: buildAppTheme(Brightness.dark),
        home: AnimatedBuilder(
          animation: widget.appState,
          builder: (context, _) {
            if (!widget.appState.isReady) {
              return const Scaffold(
                body: Center(child: CircularProgressIndicator()),
              );
            }
            return widget.appState.onboarded
                ? const HomeScreen()
                : const OnboardingScreen();
          },
        ),
      ),
    );
  }
}

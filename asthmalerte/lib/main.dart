import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/date_symbol_data_local.dart';

import 'app.dart';
import 'services/home_entry_service.dart';
import 'state/app_state.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Dates en français dans l'historique.
  await initializeDateFormatting('fr_FR');

  // L'app est pensée en portrait, bouton au centre.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  final appState = AppState();
  await appState.load();

  final homeEntry = HomeEntryService();
  await homeEntry.initialize();
  await homeEntry.refreshWidget(contactCount: appState.smsContacts.length);

  final launchedFromWidget = await homeEntry.launchedFromWidget();

  runApp(AsthmAlerteApp(
    appState: appState,
    homeEntry: homeEntry,
    openSosOnStart: launchedFromWidget && appState.onboarded,
  ));
}

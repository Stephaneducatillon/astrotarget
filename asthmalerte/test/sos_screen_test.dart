import 'package:asthmalerte/app.dart';
import 'package:asthmalerte/models/app_settings.dart';
import 'package:asthmalerte/models/emergency_contact.dart';
import 'package:asthmalerte/models/medical_profile.dart';
import 'package:asthmalerte/screens/sos_screen.dart';
import 'package:asthmalerte/services/alert_message.dart';
import 'package:asthmalerte/services/alert_service.dart';
import 'package:asthmalerte/services/home_entry_service.dart';
import 'package:asthmalerte/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Service d'alerte simulé : on teste l'écran, pas le GPS ni l'app SMS.
class FakeAlertService extends AlertService {
  FakeAlertService({required super.appState});

  int triggerCount = 0;
  bool allClearSent = false;

  @override
  Future<AlertOutcome> trigger({void Function(AlertStep)? onStep}) async {
    triggerCount++;
    onStep?.call(AlertStep.locating);
    onStep?.call(AlertStep.sending);
    onStep?.call(AlertStep.done);
    return AlertOutcome(
      message: 'Message de test',
      recipients: appState.smsContacts,
      position: const ResolvedPosition(
        latitude: 48.85837,
        longitude: 2.29448,
        address: '5 avenue Anatole France, 75007 Paris',
      ),
      smsOpened: true,
      callPlaced: true,
    );
  }

  @override
  Future<bool> sendAllClear() async {
    allClearSent = true;
    return true;
  }
}

void main() {
  late AppState state;
  late FakeAlertService alertService;

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    state = AppState();
    await state.load();
    await state.saveProfile(const MedicalProfile(
      fullName: 'Camille Roux',
      treatment: 'Ventoline 2 bouffées',
    ));
    await state.upsertContact(const EmergencyContact(
      id: '1',
      name: 'Sacha',
      phone: '+33612345678',
    ));
    alertService = FakeAlertService(appState: state);
  });

  Widget wrap() {
    return AppScope(
      appState: state,
      alertService: alertService,
      homeEntry: HomeEntryService(),
      child: const MaterialApp(home: SosScreen()),
    );
  }

  testWidgets('le décompte s\'affiche puis l\'alerte part', (tester) async {
    await state.saveSettings(const AppSettings(countdownSeconds: 3));
    await tester.pumpWidget(wrap());
    await tester.pump();

    expect(find.text('ALERTE DANS'), findsOneWidget);
    expect(find.text('3'), findsOneWidget);
    expect(alertService.triggerCount, 0);

    await tester.pump(const Duration(seconds: 1));
    expect(find.text('2'), findsOneWidget);

    await tester.pump(const Duration(seconds: 1));
    await tester.pump(const Duration(seconds: 1));
    await tester.pumpAndSettle();

    expect(alertService.triggerCount, 1);
    expect(find.text('Alerte envoyée'), findsOneWidget);
    expect(find.textContaining('Sacha'), findsWidgets);
    expect(find.textContaining('Anatole France'), findsOneWidget);
  });

  testWidgets('« Annuler » stoppe le décompte sans rien envoyer',
      (tester) async {
    await state.saveSettings(const AppSettings(countdownSeconds: 10));
    await tester.pumpWidget(wrap());
    await tester.pump();

    await tester.tap(find.text('ANNULER'));
    await tester.pumpAndSettle();
    // Bien au-delà du décompte : rien ne doit partir après l'annulation.
    await tester.pump(const Duration(seconds: 15));

    expect(alertService.triggerCount, 0);
  });

  testWidgets('« Envoyer tout de suite » court-circuite le décompte',
      (tester) async {
    await state.saveSettings(const AppSettings(countdownSeconds: 10));
    await tester.pumpWidget(wrap());
    await tester.pump();

    await tester.tap(find.text('Envoyer tout de suite'));
    await tester.pumpAndSettle();

    expect(alertService.triggerCount, 1);
    expect(find.text('Alerte envoyée'), findsOneWidget);
  });

  testWidgets('un délai nul envoie immédiatement', (tester) async {
    await state.saveSettings(const AppSettings(countdownSeconds: 0));
    await tester.pumpWidget(wrap());
    await tester.pumpAndSettle();

    expect(find.text('ALERTE DANS'), findsNothing);
    expect(alertService.triggerCount, 1);
  });

  testWidgets('l\'écran de résultat rappelle la fiche médicale et rassure',
      (tester) async {
    await state.saveSettings(const AppSettings(countdownSeconds: 0));
    await tester.pumpWidget(wrap());
    await tester.pumpAndSettle();

    expect(find.textContaining('Ventoline'), findsWidgets);
    expect(find.text('Appeler les secours (112)'), findsOneWidget);

    final allClear = find.text('Je vais mieux — rassurer mes proches');
    await tester.scrollUntilVisible(allClear, 200,
        scrollable: find.byType(Scrollable).first);
    await tester.tap(allClear);
    await tester.pumpAndSettle();

    expect(alertService.allClearSent, isTrue);
  });
}

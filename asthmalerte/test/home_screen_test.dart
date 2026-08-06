import 'package:asthmalerte/app.dart';
import 'package:asthmalerte/models/emergency_contact.dart';
import 'package:asthmalerte/screens/home_screen.dart';
import 'package:asthmalerte/services/alert_service.dart';
import 'package:asthmalerte/services/home_entry_service.dart';
import 'package:asthmalerte/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Vérifie que l'écran d'accueil se construit vraiment et réagit à l'ajout
/// d'un proche — le chemin que l'utilisateur voit en premier.
void main() {
  late AppState state;

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    state = AppState();
    await state.load();
  });

  Widget wrap() {
    return AppScope(
      appState: state,
      alertService: AlertService(appState: state),
      homeEntry: HomeEntryService(),
      child: const MaterialApp(home: HomeScreen()),
    );
  }

  testWidgets('sans proche : bouton désactivé et bandeau d\'avertissement',
      (tester) async {
    await tester.pumpWidget(wrap());

    expect(find.text('SOS'), findsOneWidget);
    expect(find.text('Ajoutez un proche'), findsOneWidget);
    expect(
      find.textContaining('l\'alerte ne partira nulle part'),
      findsOneWidget,
    );
  });

  testWidgets('avec un proche : le bouton annonce le destinataire',
      (tester) async {
    await tester.pumpWidget(wrap());

    await state.upsertContact(const EmergencyContact(
      id: '1',
      name: 'Camille',
      phone: '+33612345678',
    ));
    await tester.pump();

    expect(find.text('Prévenir 1 proche'), findsOneWidget);
    expect(find.text('Appeler Camille'), findsOneWidget);
    expect(find.textContaining('ne partira nulle part'), findsNothing);
  });

  testWidgets('deux proches : le pluriel est correct', (tester) async {
    await tester.pumpWidget(wrap());

    await state.upsertContact(const EmergencyContact(
        id: '1', name: 'Camille', phone: '+33612345678'));
    await state.upsertContact(const EmergencyContact(
        id: '2', name: 'Sacha', phone: '+33698765432'));
    await tester.pump();

    expect(find.text('Prévenir 2 proches'), findsOneWidget);
    // Le premier de la liste reste celui qu'on appelle.
    expect(find.text('Appeler Camille'), findsOneWidget);
  });
}

import 'package:asthmalerte/app.dart';
import 'package:asthmalerte/models/emergency_contact.dart';
import 'package:asthmalerte/models/medical_profile.dart';
import 'package:asthmalerte/screens/contact_form_screen.dart';
import 'package:asthmalerte/screens/contacts_screen.dart';
import 'package:asthmalerte/screens/history_screen.dart';
import 'package:asthmalerte/screens/onboarding_screen.dart';
import 'package:asthmalerte/screens/profile_screen.dart';
import 'package:asthmalerte/screens/settings_screen.dart';
import 'package:asthmalerte/services/alert_service.dart';
import 'package:asthmalerte/services/home_entry_service.dart';
import 'package:asthmalerte/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Chaque écran doit se construire sans exception — un plantage ici serait un
/// plantage sur le téléphone.
void main() {
  late AppState state;

  setUpAll(() async {
    await initializeDateFormatting('fr_FR');
  });

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
      relation: 'Conjoint',
    ));
  });

  Widget wrap(Widget screen) {
    return AppScope(
      appState: state,
      alertService: AlertService(appState: state),
      homeEntry: HomeEntryService(),
      child: MaterialApp(home: screen),
    );
  }

  testWidgets('liste des proches', (tester) async {
    await tester.pumpWidget(wrap(const ContactsScreen()));
    expect(find.text('Sacha'), findsOneWidget);
    expect(find.text('Appelé en 1er'), findsOneWidget);
  });

  testWidgets('formulaire de proche pré-rempli', (tester) async {
    await tester.pumpWidget(wrap(ContactFormScreen(
      contact: state.contacts.first,
    )));
    expect(find.text('Sacha'), findsOneWidget);
    expect(find.text('+33612345678'), findsOneWidget);
  });

  testWidgets('fiche médicale pré-remplie', (tester) async {
    await tester.pumpWidget(wrap(const ProfileScreen()));
    expect(find.text('Camille Roux'), findsOneWidget);
    expect(find.text('Ventoline 2 bouffées'), findsOneWidget);
  });

  testWidgets('réglages : aperçu du message affiché', (tester) async {
    await tester.pumpWidget(wrap(const SettingsScreen()));
    expect(find.text('Délai avant envoi'), findsOneWidget);
    expect(find.text('Partager ma position'), findsOneWidget);

    // L'aperçu du SMS est plus bas dans la liste.
    await tester.scrollUntilVisible(
      find.textContaining('ALERTE ASTHME'),
      300,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.textContaining('ALERTE ASTHME'), findsOneWidget);
    expect(find.textContaining('Ventoline'), findsOneWidget);
  });

  testWidgets('historique vide', (tester) async {
    await tester.pumpWidget(wrap(const HistoryScreen()));
    expect(find.text('Aucune alerte enregistrée.'), findsOneWidget);
  });

  testWidgets('onboarding : le bouton attend un contact valide',
      (tester) async {
    await tester.pumpWidget(wrap(const OnboardingScreen()));
    expect(find.text('Bienvenue'), findsOneWidget);

    final continuer = find.widgetWithText(FilledButton, 'Continuer');
    expect(tester.widget<FilledButton>(continuer).onPressed, isNotNull);

    await tester.tap(continuer);
    await tester.pumpAndSettle();

    // Étape 2 : pas de proche saisi, on ne peut pas continuer.
    expect(find.text('Un proche à prévenir'), findsOneWidget);
    expect(tester.widget<FilledButton>(continuer).onPressed, isNull);

    await tester.enterText(find.byType(TextField).first, 'Marie');
    await tester.enterText(find.byType(TextField).last, '0612345678');
    await tester.pump();

    expect(tester.widget<FilledButton>(continuer).onPressed, isNotNull);
  });
}

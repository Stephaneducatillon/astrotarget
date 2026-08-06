import 'package:asthmalerte/models/app_settings.dart';
import 'package:asthmalerte/models/medical_profile.dart';
import 'package:asthmalerte/services/alert_message.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final now = DateTime(2026, 3, 14, 9, 5);

  const profile = MedicalProfile(
    fullName: 'Camille Roux',
    treatment: 'Ventoline 2 bouffées',
    allergies: 'Pollen',
  );

  const position = ResolvedPosition(
    latitude: 48.85837,
    longitude: 2.29448,
    address: '5 avenue Anatole France, 75007 Paris',
  );

  group('buildAlertMessage', () {
    test('contient le nom, l\'heure, l\'adresse et le lien carte', () {
      final message = buildAlertMessage(
        settings: const AppSettings(),
        profile: profile,
        now: now,
        position: position,
      );

      expect(message, contains('Camille Roux'));
      expect(message, contains('09h05'));
      expect(message, contains('5 avenue Anatole France'));
      expect(message, contains('https://maps.google.com/?q=48.85837,2.29448'));
      expect(message, contains('Ventoline'));
    });

    test('sans position, aucune coordonnée ni lien vide ne traîne', () {
      final message = buildAlertMessage(
        settings: const AppSettings(),
        profile: profile,
        now: now,
      );

      expect(message, contains('non disponible'));
      expect(message, isNot(contains('maps.google.com')));
      // Pas de ligne vide laissée par {lien}.
      expect(message.split('\n').any((l) => l.trim().isEmpty), isFalse);
    });

    test('respecte la désactivation du partage de position', () {
      final message = buildAlertMessage(
        settings: const AppSettings(shareLocation: false),
        profile: profile,
        now: now,
        position: position,
      );

      expect(message, isNot(contains('maps.google.com')));
    });

    test('respecte la désactivation des infos médicales', () {
      final message = buildAlertMessage(
        settings: const AppSettings(includeMedicalInfo: false),
        profile: profile,
        now: now,
        position: position,
      );

      expect(message, isNot(contains('Ventoline')));
      expect(message, isNot(contains('Pollen')));
    });

    test('retombe sur un libellé générique si le nom est vide', () {
      final message = buildAlertMessage(
        settings: const AppSettings(),
        profile: const MedicalProfile(),
        now: now,
        position: position,
      );

      expect(message, contains('Une personne asthmatique'));
    });

    test('utilise les coordonnées quand l\'adresse est introuvable', () {
      final message = buildAlertMessage(
        settings: const AppSettings(),
        profile: profile,
        now: now,
        position: const ResolvedPosition(latitude: 48.1, longitude: 2.2),
      );

      expect(message, contains('48.10000, 2.20000'));
    });

    test('un modèle personnalisé est bien substitué', () {
      final message = buildAlertMessage(
        settings: const AppSettings(messageTemplate: 'SOS {nom} -> {lien}'),
        profile: profile,
        now: now,
        position: position,
      );

      expect(message, 'SOS Camille Roux -> ${position.mapsLink}');
    });
  });

  test('buildAllClearMessage rassure avec le nom et l\'heure', () {
    final message = buildAllClearMessage(profile: profile, now: now);
    expect(message, contains('Camille Roux'));
    expect(message, contains('09h05'));
  });
}

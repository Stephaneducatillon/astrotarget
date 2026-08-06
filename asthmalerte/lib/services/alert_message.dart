import '../models/app_settings.dart';
import '../models/medical_profile.dart';

/// Position résolue au moment de l'alerte.
class ResolvedPosition {
  const ResolvedPosition({
    required this.latitude,
    required this.longitude,
    this.address = '',
    this.accuracyMeters,
  });

  final double latitude;
  final double longitude;
  final String address;
  final double? accuracyMeters;

  String get mapsLink => 'https://maps.google.com/?q=$latitude,$longitude';

  String get coordinates =>
      '${latitude.toStringAsFixed(5)}, ${longitude.toStringAsFixed(5)}';
}

/// Construit le texte du SMS d'alerte.
///
/// Fonction pure (aucun plugin) pour rester testable : voir
/// `test/alert_message_test.dart`.
String buildAlertMessage({
  required AppSettings settings,
  required MedicalProfile profile,
  required DateTime now,
  ResolvedPosition? position,
  String? nameOverride,
}) {
  final name = (nameOverride ?? profile.fullName).trim().isEmpty
      ? 'Une personne asthmatique'
      : (nameOverride ?? profile.fullName).trim();

  final heure = '${_two(now.hour)}h${_two(now.minute)}';

  final bool withPosition = settings.shareLocation && position != null;
  final adresse = withPosition
      ? (position.address.trim().isNotEmpty
          ? position.address.trim()
          : position.coordinates)
      : 'non disponible';
  final lien = withPosition ? position.mapsLink : '';

  final medical =
      settings.includeMedicalInfo ? profile.smsSummary.trim() : '';

  final raw = settings.messageTemplate
      .replaceAll('{nom}', name)
      .replaceAll('{heure}', heure)
      .replaceAll('{adresse}', adresse)
      .replaceAll('{lien}', lien)
      .replaceAll('{medical}', medical);

  return _tidy(raw);
}

/// Message « je vais mieux » envoyé aux mêmes proches après coup.
String buildAllClearMessage({
  required MedicalProfile profile,
  required DateTime now,
  String? nameOverride,
}) {
  final name = (nameOverride ?? profile.fullName).trim().isEmpty
      ? 'La personne alertée'
      : (nameOverride ?? profile.fullName).trim();
  return _tidy(AppSettings.defaultAllClearTemplate
      .replaceAll('{nom}', name)
      .replaceAll('{heure}', '${_two(now.hour)}h${_two(now.minute)}'));
}

String _two(int value) => value.toString().padLeft(2, '0');

/// Supprime les lignes vides laissées par les variables non renseignées.
String _tidy(String text) {
  final lines = text
      .split('\n')
      .map((line) => line.trimRight())
      .where((line) => line.trim().isNotEmpty)
      .toList();
  return lines.join('\n').trim();
}

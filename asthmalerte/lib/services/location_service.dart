import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';

import 'alert_message.dart';

class LocationFailure implements Exception {
  LocationFailure(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Récupération de la position + adresse lisible.
///
/// En urgence, mieux vaut une position approximative tout de suite qu'une
/// position parfaite trop tard : on borne l'attente et on retombe sur la
/// dernière position connue si nécessaire.
class LocationService {
  static const Duration fixTimeout = Duration(seconds: 12);
  static const Duration geocodeTimeout = Duration(seconds: 5);

  /// Demande l'autorisation. Retourne `true` si on peut localiser.
  Future<bool> ensurePermission() async {
    if (!await Geolocator.isLocationServiceEnabled()) return false;

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    return permission == LocationPermission.always ||
        permission == LocationPermission.whileInUse;
  }

  /// Position + adresse. Lève [LocationFailure] si rien n'est disponible.
  Future<ResolvedPosition> resolve() async {
    if (!await ensurePermission()) {
      throw LocationFailure(
        'Localisation indisponible (service désactivé ou autorisation refusée).',
      );
    }

    Position? position;
    try {
      position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: fixTimeout,
        ),
      );
    } catch (_) {
      position = await Geolocator.getLastKnownPosition();
    }

    if (position == null) {
      throw LocationFailure('Position introuvable dans le délai imparti.');
    }

    final address = await _address(position.latitude, position.longitude);

    return ResolvedPosition(
      latitude: double.parse(position.latitude.toStringAsFixed(6)),
      longitude: double.parse(position.longitude.toStringAsFixed(6)),
      address: address,
      accuracyMeters: position.accuracy,
    );
  }

  /// Adresse postale. Best effort : une adresse vide n'empêche pas l'alerte.
  Future<String> _address(double lat, double lon) async {
    try {
      final marks =
          await placemarkFromCoordinates(lat, lon).timeout(geocodeTimeout);
      if (marks.isEmpty) return '';
      final m = marks.first;
      final parts = <String>[
        if ((m.street ?? '').trim().isNotEmpty) m.street!.trim(),
        if ((m.postalCode ?? '').trim().isNotEmpty) m.postalCode!.trim(),
        if ((m.locality ?? '').trim().isNotEmpty) m.locality!.trim(),
      ];
      return parts.join(', ');
    } catch (_) {
      return '';
    }
  }
}

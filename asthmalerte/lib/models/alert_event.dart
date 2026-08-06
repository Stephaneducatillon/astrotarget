/// Trace d'une alerte déclenchée, conservée dans l'historique local.
class AlertEvent {
  const AlertEvent({
    required this.id,
    required this.date,
    required this.recipients,
    this.latitude,
    this.longitude,
    this.address = '',
    this.message = '',
    this.cancelled = false,
    this.called = false,
  });

  final String id;
  final DateTime date;

  /// Noms des proches prévenus.
  final List<String> recipients;

  final double? latitude;
  final double? longitude;
  final String address;
  final String message;

  /// L'utilisateur a annulé pendant le décompte.
  final bool cancelled;

  /// Un appel a été lancé dans la foulée.
  final bool called;

  bool get hasPosition => latitude != null && longitude != null;

  String? get mapsLink => hasPosition
      ? 'https://maps.google.com/?q=$latitude,$longitude'
      : null;

  Map<String, dynamic> toJson() => {
        'id': id,
        'date': date.toIso8601String(),
        'recipients': recipients,
        'latitude': latitude,
        'longitude': longitude,
        'address': address,
        'message': message,
        'cancelled': cancelled,
        'called': called,
      };

  factory AlertEvent.fromJson(Map<String, dynamic> json) {
    return AlertEvent(
      id: json['id'] as String,
      date: DateTime.tryParse(json['date'] as String? ?? '') ?? DateTime.now(),
      recipients:
          (json['recipients'] as List<dynamic>? ?? []).map((e) => '$e').toList(),
      latitude: (json['latitude'] as num?)?.toDouble(),
      longitude: (json['longitude'] as num?)?.toDouble(),
      address: json['address'] as String? ?? '',
      message: json['message'] as String? ?? '',
      cancelled: json['cancelled'] as bool? ?? false,
      called: json['called'] as bool? ?? false,
    );
  }
}

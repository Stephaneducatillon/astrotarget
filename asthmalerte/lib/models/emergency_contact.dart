/// Un proche à prévenir en cas de crise.
class EmergencyContact {
  const EmergencyContact({
    required this.id,
    required this.name,
    required this.phone,
    this.relation = '',
    this.sendSms = true,
    this.callable = true,
  });

  final String id;
  final String name;

  /// Numéro tel que saisi. Normalisé via [dialablePhone] avant usage.
  final String phone;

  /// « Conjoint », « Mère », « Médecin »… Affiché sur la fiche.
  final String relation;

  /// Ce contact reçoit-il le SMS d'alerte ?
  final bool sendSms;

  /// Peut-on l'appeler directement depuis l'écran d'alerte ?
  final bool callable;

  /// Numéro nettoyé pour les URI `sms:` / `tel:` (espaces, points, tirets…).
  String get dialablePhone {
    final cleaned = phone.replaceAll(RegExp(r'[^0-9+]'), '');
    // Un « + » n'a de sens qu'en tête de numéro.
    if (cleaned.isEmpty) return cleaned;
    final plus = cleaned.startsWith('+') ? '+' : '';
    return plus + cleaned.replaceAll('+', '');
  }

  bool get isValid => name.trim().isNotEmpty && dialablePhone.length >= 6;

  EmergencyContact copyWith({
    String? name,
    String? phone,
    String? relation,
    bool? sendSms,
    bool? callable,
  }) {
    return EmergencyContact(
      id: id,
      name: name ?? this.name,
      phone: phone ?? this.phone,
      relation: relation ?? this.relation,
      sendSms: sendSms ?? this.sendSms,
      callable: callable ?? this.callable,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'phone': phone,
        'relation': relation,
        'sendSms': sendSms,
        'callable': callable,
      };

  factory EmergencyContact.fromJson(Map<String, dynamic> json) {
    return EmergencyContact(
      id: json['id'] as String,
      name: json['name'] as String? ?? '',
      phone: json['phone'] as String? ?? '',
      relation: json['relation'] as String? ?? '',
      sendSms: json['sendSms'] as bool? ?? true,
      callable: json['callable'] as bool? ?? true,
    );
  }
}

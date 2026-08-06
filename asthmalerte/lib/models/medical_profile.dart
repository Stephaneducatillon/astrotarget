/// Fiche médicale affichée à l'écran pendant une crise (pour les secours
/// ou un passant) et, en option, jointe au SMS d'alerte.
class MedicalProfile {
  const MedicalProfile({
    this.fullName = '',
    this.birthDate = '',
    this.treatment = '',
    this.allergies = '',
    this.doctorName = '',
    this.doctorPhone = '',
    this.notes = '',
  });

  final String fullName;
  final String birthDate;

  /// Traitement de crise : « Ventoline 2 bouffées », etc.
  final String treatment;
  final String allergies;
  final String doctorName;
  final String doctorPhone;
  final String notes;

  bool get isEmpty =>
      fullName.trim().isEmpty &&
      treatment.trim().isEmpty &&
      allergies.trim().isEmpty;

  /// Résumé court, ajouté au SMS si l'option est activée.
  String get smsSummary {
    final parts = <String>[];
    if (treatment.trim().isNotEmpty) parts.add('Traitement : ${treatment.trim()}');
    if (allergies.trim().isNotEmpty) parts.add('Allergies : ${allergies.trim()}');
    return parts.join(' — ');
  }

  MedicalProfile copyWith({
    String? fullName,
    String? birthDate,
    String? treatment,
    String? allergies,
    String? doctorName,
    String? doctorPhone,
    String? notes,
  }) {
    return MedicalProfile(
      fullName: fullName ?? this.fullName,
      birthDate: birthDate ?? this.birthDate,
      treatment: treatment ?? this.treatment,
      allergies: allergies ?? this.allergies,
      doctorName: doctorName ?? this.doctorName,
      doctorPhone: doctorPhone ?? this.doctorPhone,
      notes: notes ?? this.notes,
    );
  }

  Map<String, dynamic> toJson() => {
        'fullName': fullName,
        'birthDate': birthDate,
        'treatment': treatment,
        'allergies': allergies,
        'doctorName': doctorName,
        'doctorPhone': doctorPhone,
        'notes': notes,
      };

  factory MedicalProfile.fromJson(Map<String, dynamic> json) {
    return MedicalProfile(
      fullName: json['fullName'] as String? ?? '',
      birthDate: json['birthDate'] as String? ?? '',
      treatment: json['treatment'] as String? ?? '',
      allergies: json['allergies'] as String? ?? '',
      doctorName: json['doctorName'] as String? ?? '',
      doctorPhone: json['doctorPhone'] as String? ?? '',
      notes: json['notes'] as String? ?? '',
    );
  }
}

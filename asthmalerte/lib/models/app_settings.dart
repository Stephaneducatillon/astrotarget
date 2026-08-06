/// Comment l'alerte est envoyée quand on appuie sur le bouton SOS.
class AppSettings {
  const AppSettings({
    this.countdownSeconds = 5,
    this.shareLocation = true,
    this.includeMedicalInfo = true,
    this.autoCallAfterSms = true,
    this.oneSmsPerContact = false,
    this.vibrate = true,
    this.messageTemplate = defaultTemplate,
  });

  /// Délai d'annulation avant l'envoi. 0 = envoi immédiat.
  final int countdownSeconds;

  final bool shareLocation;
  final bool includeMedicalInfo;

  /// Ouvre l'appel vers le 1er contact juste après le SMS.
  final bool autoCallAfterSms;

  /// Certains téléphones Android gèrent mal les SMS groupés : cette option
  /// prépare un SMS par contact, l'un après l'autre.
  final bool oneSmsPerContact;

  final bool vibrate;

  /// Variables disponibles : {nom} {lien} {adresse} {heure} {medical}
  final String messageTemplate;

  static const String defaultTemplate =
      '🚨 ALERTE ASTHME 🚨\n{nom} fait une crise et a besoin d\'aide.\n'
      'Position : {adresse}\n{lien}\n{medical}\nEnvoyé à {heure} depuis AsthmAlerte.';

  static const String defaultAllClearTemplate =
      '✅ Fausse alerte / je vais mieux. {nom} — {heure}.';

  AppSettings copyWith({
    int? countdownSeconds,
    bool? shareLocation,
    bool? includeMedicalInfo,
    bool? autoCallAfterSms,
    bool? oneSmsPerContact,
    bool? vibrate,
    String? messageTemplate,
  }) {
    return AppSettings(
      countdownSeconds: countdownSeconds ?? this.countdownSeconds,
      shareLocation: shareLocation ?? this.shareLocation,
      includeMedicalInfo: includeMedicalInfo ?? this.includeMedicalInfo,
      autoCallAfterSms: autoCallAfterSms ?? this.autoCallAfterSms,
      oneSmsPerContact: oneSmsPerContact ?? this.oneSmsPerContact,
      vibrate: vibrate ?? this.vibrate,
      messageTemplate: messageTemplate ?? this.messageTemplate,
    );
  }

  Map<String, dynamic> toJson() => {
        'countdownSeconds': countdownSeconds,
        'shareLocation': shareLocation,
        'includeMedicalInfo': includeMedicalInfo,
        'autoCallAfterSms': autoCallAfterSms,
        'oneSmsPerContact': oneSmsPerContact,
        'vibrate': vibrate,
        'messageTemplate': messageTemplate,
      };

  factory AppSettings.fromJson(Map<String, dynamic> json) {
    return AppSettings(
      countdownSeconds: json['countdownSeconds'] as int? ?? 5,
      shareLocation: json['shareLocation'] as bool? ?? true,
      includeMedicalInfo: json['includeMedicalInfo'] as bool? ?? true,
      autoCallAfterSms: json['autoCallAfterSms'] as bool? ?? true,
      oneSmsPerContact: json['oneSmsPerContact'] as bool? ?? false,
      vibrate: json['vibrate'] as bool? ?? true,
      messageTemplate:
          json['messageTemplate'] as String? ?? AppSettings.defaultTemplate,
    );
  }
}

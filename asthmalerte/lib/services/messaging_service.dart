import 'dart:io';

import 'package:url_launcher/url_launcher.dart';

/// Ouverture de l'application SMS pré-remplie et lancement d'appels.
///
/// On passe par les applications système : c'est ce qui fonctionne sur les
/// deux plateformes sans permission sensible (`SEND_SMS`), et ce qui est
/// accepté sur l'App Store. Il reste un appui sur « Envoyer ».
/// Pour un envoi 100 % automatique, voir la section « Aller plus loin » du
/// README (passerelle SMS côté serveur).
class MessagingService {
  /// Compose un SMS vers [recipients] avec [body] déjà rempli.
  Future<bool> sendSms(List<String> recipients, String body) async {
    final numbers = recipients
        .map((n) => n.trim())
        .where((n) => n.isNotEmpty)
        .toList();
    if (numbers.isEmpty) return false;

    return _launch(smsUri(numbers, body));
  }

  /// Un SMS par contact, séquentiellement (téléphones qui gèrent mal le
  /// multi-destinataires). Retourne le nombre d'envois ouverts.
  Future<int> sendSmsIndividually(
    List<String> recipients,
    String body, {
    Duration between = const Duration(milliseconds: 900),
  }) async {
    var opened = 0;
    for (final number in recipients) {
      if (await sendSms([number], body)) opened++;
      await Future<void>.delayed(between);
    }
    return opened;
  }

  Future<bool> call(String phone) async {
    if (phone.trim().isEmpty) return false;
    return _launch(Uri(scheme: 'tel', path: phone.trim()));
  }

  Future<bool> openMaps(double latitude, double longitude) {
    return _launch(Uri.parse('https://maps.google.com/?q=$latitude,$longitude'));
  }

  Future<bool> _launch(Uri uri) async {
    try {
      return await launchUrl(uri, mode: LaunchMode.externalApplication);
    } catch (_) {
      return false;
    }
  }

  /// URI SMS adaptée à la plateforme.
  ///
  /// iOS attend `sms:num1,num2&body=…`, Android `smsto:num1;num2?body=…`.
  static Uri smsUri(List<String> numbers, String body) {
    final encodedBody = Uri.encodeComponent(body);
    if (Platform.isIOS) {
      return Uri.parse('sms:${numbers.join(',')}&body=$encodedBody');
    }
    return Uri.parse('smsto:${numbers.join(';')}?body=$encodedBody');
  }
}

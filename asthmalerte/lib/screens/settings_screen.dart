import 'package:flutter/material.dart';

import '../app.dart';
import '../models/app_settings.dart';
import '../services/alert_message.dart';
import '../widgets/section_card.dart';

/// Réglages de l'alerte + aperçu en direct du SMS envoyé.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final scope = AppScope.of(context);
    final state = scope.state;
    final settings = state.settings;

    Future<void> update(AppSettings next) => state.saveSettings(next);

    final preview = buildAlertMessage(
      settings: settings,
      profile: state.profile,
      now: DateTime.now(),
      position: const ResolvedPosition(
        latitude: 48.85837,
        longitude: 2.29448,
        address: '5 avenue Anatole France, 75007 Paris',
      ),
    );

    return Scaffold(
      appBar: AppBar(title: const Text('Réglages')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Délai avant envoi',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text(
                  settings.countdownSeconds == 0
                      ? 'Envoi immédiat, sans possibilité d\'annuler.'
                      : 'Vous avez ${settings.countdownSeconds} s pour annuler '
                          'en cas d\'appui involontaire.',
                  style: const TextStyle(fontSize: 15),
                ),
                Slider(
                  value: settings.countdownSeconds.toDouble(),
                  min: 0,
                  max: 15,
                  divisions: 15,
                  label: '${settings.countdownSeconds} s',
                  onChanged: (v) =>
                      update(settings.copyWith(countdownSeconds: v.round())),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SectionCard(
            padding: EdgeInsets.zero,
            child: Column(
              children: [
                SwitchListTile(
                  value: settings.shareLocation,
                  onChanged: (v) =>
                      update(settings.copyWith(shareLocation: v)),
                  title: const Text('Partager ma position'),
                  subtitle: const Text('Adresse + lien vers la carte'),
                ),
                SwitchListTile(
                  value: settings.includeMedicalInfo,
                  onChanged: (v) =>
                      update(settings.copyWith(includeMedicalInfo: v)),
                  title: const Text('Joindre mes infos médicales'),
                  subtitle: const Text('Traitement et allergies dans le SMS'),
                ),
                SwitchListTile(
                  value: settings.autoCallAfterSms,
                  onChanged: (v) =>
                      update(settings.copyWith(autoCallAfterSms: v)),
                  title: const Text('Appeler après le SMS'),
                  subtitle: const Text('Appel du 1er proche de la liste'),
                ),
                SwitchListTile(
                  value: settings.oneSmsPerContact,
                  onChanged: (v) =>
                      update(settings.copyWith(oneSmsPerContact: v)),
                  title: const Text('Un SMS par proche'),
                  subtitle: const Text(
                      'À activer si les SMS groupés ne partent pas'),
                ),
                SwitchListTile(
                  value: settings.vibrate,
                  onChanged: (v) => update(settings.copyWith(vibrate: v)),
                  title: const Text('Vibrer pendant le décompte'),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Message envoyé',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                Text(preview,
                    style: const TextStyle(fontSize: 15, height: 1.4)),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  icon: const Icon(Icons.edit),
                  label: const Text('Personnaliser le message'),
                  onPressed: () => _editTemplate(context, settings, update),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Accès rapide',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                const Text(
                  'Ajoutez le widget « SOS » sur votre écran d\'accueil : '
                  'appui long sur l\'écran d\'accueil → Widgets → AsthmAlerte. '
                  'Un appui long sur l\'icône de l\'app donne aussi le '
                  'raccourci « Alerte SOS ».',
                  style: TextStyle(fontSize: 15),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  icon: const Icon(Icons.widgets),
                  label: const Text('Rafraîchir le widget'),
                  onPressed: () => scope.homeEntry
                      .refreshWidget(contactCount: state.smsContacts.length),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'AsthmAlerte ne remplace pas un appel aux secours (15 ou 112).',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Future<void> _editTemplate(
    BuildContext context,
    AppSettings settings,
    Future<void> Function(AppSettings) update,
  ) async {
    final controller = TextEditingController(text: settings.messageTemplate);
    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Message d\'alerte'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: controller, maxLines: 8),
            const SizedBox(height: 8),
            const Text(
              'Variables : {nom} {adresse} {lien} {heure} {medical}',
              style: TextStyle(fontSize: 13),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () =>
                Navigator.of(context).pop(AppSettings.defaultTemplate),
            child: const Text('Par défaut'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Annuler'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(controller.text),
            child: const Text('Enregistrer'),
          ),
        ],
      ),
    );
    if (result != null && result.trim().isNotEmpty) {
      await update(settings.copyWith(messageTemplate: result));
    }
  }
}

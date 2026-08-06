import 'package:flutter/material.dart';

import '../app.dart';
import '../models/emergency_contact.dart';
import '../theme.dart';
import '../widgets/sos_button.dart';
import 'contacts_screen.dart';
import 'history_screen.dart';
import 'profile_screen.dart';
import 'settings_screen.dart';
import 'sos_screen.dart';

/// Écran d'accueil : le bouton SOS et rien qui puisse distraire.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final scope = AppScope.of(context);
    final state = scope.state;
    final ready = state.canAlert;
    final callTarget = state.primaryCallContact;

    return Scaffold(
      appBar: AppBar(
        title: const Text('AsthmAlerte'),
        actions: [
          IconButton(
            tooltip: 'Historique',
            iconSize: 28,
            icon: const Icon(Icons.history),
            onPressed: () => _push(context, const HistoryScreen()),
          ),
          IconButton(
            tooltip: 'Réglages',
            iconSize: 28,
            icon: const Icon(Icons.settings),
            onPressed: () => _push(context, const SettingsScreen()),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            children: [
              if (!ready) const _NoContactBanner(),
              Expanded(
                child: SosButton(
                  enabled: ready,
                  sublabel: ready
                      ? 'Prévenir ${state.smsContacts.length} proche${state.smsContacts.length > 1 ? 's' : ''}'
                      : 'Ajoutez un proche',
                  onPressed: () => _push(
                    context,
                    const SosScreen(autoStart: true),
                  ),
                ),
              ),
              if (callTarget != null) _QuickCall(contact: callTarget),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _NavTile(
                      icon: Icons.group,
                      label: 'Mes proches',
                      badge: '${state.contacts.length}',
                      onTap: () => _push(context, const ContactsScreen()),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _NavTile(
                      icon: Icons.medical_information,
                      label: 'Fiche médicale',
                      badge: state.profile.isEmpty ? '!' : null,
                      onTap: () => _push(context, const ProfileScreen()),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  void _push(BuildContext context, Widget screen) {
    Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => screen));
  }
}

class _NoContactBanner extends StatelessWidget {
  const _NoContactBanner();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.warning.fade(0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.warning.fade(0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: AppColors.warning, size: 28),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'Aucun proche enregistré : l\'alerte ne partira nulle part.',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(builder: (_) => const ContactsScreen()),
            ),
            child: const Text('Ajouter'),
          ),
        ],
      ),
    );
  }
}

/// Appel direct du proche prioritaire, sans passer par l'alerte complète.
class _QuickCall extends StatelessWidget {
  const _QuickCall({required this.contact});

  final EmergencyContact contact;

  @override
  Widget build(BuildContext context) {
    final scope = AppScope.of(context);
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.safe,
          foregroundColor: Colors.white,
        ),
        icon: const Icon(Icons.phone, size: 28),
        label: Text('Appeler ${contact.name}'),
        onPressed: () => scope.alertService.call(contact),
      ),
    );
  }
}

class _NavTile extends StatelessWidget {
  const _NavTile({
    required this.icon,
    required this.label,
    required this.onTap,
    this.badge,
  });

  final IconData icon;
  final String label;
  final String? badge;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 12),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1A1D23) : Colors.white,
          borderRadius: BorderRadius.circular(18),
          border:
              Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        ),
        child: Column(
          children: [
            Icon(icon, size: 30),
            const SizedBox(height: 8),
            Text(
              label,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            if (badge != null) ...[
              const SizedBox(height: 4),
              Text(
                badge!,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

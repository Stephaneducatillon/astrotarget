import 'package:flutter/material.dart';

import '../app.dart';
import '../models/emergency_contact.dart';
import '../theme.dart';
import 'contact_form_screen.dart';

/// Liste des proches à prévenir. L'ordre compte : le premier est appelé.
class ContactsScreen extends StatelessWidget {
  const ContactsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final scope = AppScope.of(context);
    final contacts = scope.state.contacts;

    return Scaffold(
      appBar: AppBar(title: const Text('Mes proches')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _openForm(context, null),
        icon: const Icon(Icons.person_add),
        label: const Text('Ajouter'),
      ),
      body: contacts.isEmpty
          ? const _EmptyState()
          : Column(
              children: [
                const Padding(
                  padding: EdgeInsets.fromLTRB(20, 8, 20, 12),
                  child: Text(
                    'Maintenez un proche appuyé pour le déplacer. '
                    'Le premier de la liste est appelé en priorité.',
                    style: TextStyle(fontSize: 15),
                  ),
                ),
                Expanded(
                  child: ReorderableListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 96),
                    itemCount: contacts.length,
                    // `onReorder` est déprécié depuis Flutter 3.41 au profit
                    // de `onReorderItem`, qui n'existe pas sur les versions
                    // antérieures : on garde l'API compatible partout.
                    // ignore: deprecated_member_use
                    onReorder: scope.state.reorderContacts,
                    itemBuilder: (context, index) {
                      final contact = contacts[index];
                      return _ContactTile(
                        key: ValueKey(contact.id),
                        contact: contact,
                        isPrimary: index == 0,
                        onTap: () => _openForm(context, contact),
                        onCall: () => scope.alertService.call(contact),
                      );
                    },
                  ),
                ),
              ],
            ),
    );
  }

  void _openForm(BuildContext context, EmergencyContact? contact) {
    Navigator.of(context).push(MaterialPageRoute<void>(
      builder: (_) => ContactFormScreen(contact: contact),
    ));
  }
}

class _ContactTile extends StatelessWidget {
  const _ContactTile({
    super.key,
    required this.contact,
    required this.isPrimary,
    required this.onTap,
    required this.onCall,
  });

  final EmergencyContact contact;
  final bool isPrimary;
  final VoidCallback onTap;
  final VoidCallback onCall;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Material(
        color: isDark ? const Color(0xFF1A1D23) : Colors.white,
        borderRadius: BorderRadius.circular(18),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 26,
                  backgroundColor:
                      isPrimary ? AppColors.sos : Colors.blueGrey,
                  child: Text(
                    contact.name.isNotEmpty
                        ? contact.name[0].toUpperCase()
                        : '?',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        contact.name,
                        style: const TextStyle(
                            fontSize: 19, fontWeight: FontWeight.w700),
                      ),
                      Text(
                        [
                          if (contact.relation.isNotEmpty) contact.relation,
                          contact.phone,
                        ].join(' · '),
                        style: const TextStyle(fontSize: 15),
                      ),
                      const SizedBox(height: 4),
                      Wrap(
                        spacing: 6,
                        children: [
                          if (isPrimary) const _Chip('Appelé en 1er'),
                          if (contact.sendSms) const _Chip('SMS'),
                          if (!contact.isValid)
                            const _Chip('Numéro incomplet', warning: true),
                        ],
                      ),
                    ],
                  ),
                ),
                IconButton(
                  iconSize: 30,
                  color: AppColors.safe,
                  icon: const Icon(Icons.phone),
                  tooltip: 'Appeler',
                  onPressed: onCall,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip(this.label, {this.warning = false});

  final String label;
  final bool warning;

  @override
  Widget build(BuildContext context) {
    final color = warning ? AppColors.warning : Theme.of(context).colorScheme.primary;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.fade(0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(
            fontSize: 12, fontWeight: FontWeight.w700, color: color),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Padding(
        padding: EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.group_add, size: 72),
            SizedBox(height: 16),
            Text(
              'Aucun proche enregistré',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 8),
            Text(
              'Ajoutez au moins une personne à prévenir : conjoint, parent, '
              'voisin, médecin…',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

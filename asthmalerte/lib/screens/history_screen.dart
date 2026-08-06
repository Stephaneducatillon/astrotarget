import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../app.dart';
import '../widgets/section_card.dart';

/// Historique des alertes : utile pour en parler au pneumologue.
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final scope = AppScope.of(context);
    final history = scope.state.history;
    final format = DateFormat('EEEE d MMMM, HH:mm', 'fr_FR');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Historique'),
        actions: [
          if (history.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_sweep),
              tooltip: 'Tout effacer',
              onPressed: () async {
                final ok = await showDialog<bool>(
                  context: context,
                  builder: (context) => AlertDialog(
                    title: const Text('Effacer l\'historique ?'),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.of(context).pop(false),
                        child: const Text('Annuler'),
                      ),
                      TextButton(
                        onPressed: () => Navigator.of(context).pop(true),
                        child: const Text('Effacer'),
                      ),
                    ],
                  ),
                );
                if (ok == true) await scope.state.clearHistory();
              },
            ),
        ],
      ),
      body: history.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(32),
                child: Text(
                  'Aucune alerte enregistrée.',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(20),
              itemCount: history.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final event = history[index];
                return SectionCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _capitalize(format.format(event.date)),
                        style: const TextStyle(
                            fontSize: 17, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        event.recipients.isEmpty
                            ? 'Aucun proche prévenu'
                            : 'Prévenus : ${event.recipients.join(', ')}',
                        style: const TextStyle(fontSize: 15),
                      ),
                      if (event.address.isNotEmpty)
                        Text(event.address,
                            style: const TextStyle(fontSize: 15)),
                      if (event.called)
                        const Text('Appel lancé',
                            style: TextStyle(fontSize: 15)),
                      if (event.hasPosition) ...[
                        const SizedBox(height: 8),
                        OutlinedButton.icon(
                          icon: const Icon(Icons.map),
                          label: const Text('Voir sur la carte'),
                          onPressed: () => scope.alertService.openMaps(
                            event.latitude!,
                            event.longitude!,
                          ),
                        ),
                      ],
                    ],
                  ),
                );
              },
            ),
    );
  }

  String _capitalize(String value) =>
      value.isEmpty ? value : value[0].toUpperCase() + value.substring(1);
}

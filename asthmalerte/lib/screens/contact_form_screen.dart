import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app.dart';
import '../models/emergency_contact.dart';

/// Ajout / modification d'un proche.
class ContactFormScreen extends StatefulWidget {
  const ContactFormScreen({super.key, this.contact});

  final EmergencyContact? contact;

  @override
  State<ContactFormScreen> createState() => _ContactFormScreenState();
}

class _ContactFormScreenState extends State<ContactFormScreen> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _name =
      TextEditingController(text: widget.contact?.name ?? '');
  late final TextEditingController _phone =
      TextEditingController(text: widget.contact?.phone ?? '');
  late final TextEditingController _relation =
      TextEditingController(text: widget.contact?.relation ?? '');
  late bool _sendSms = widget.contact?.sendSms ?? true;
  late bool _callable = widget.contact?.callable ?? true;

  @override
  void dispose() {
    _name.dispose();
    _phone.dispose();
    _relation.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!(_formKey.currentState?.validate() ?? false)) return;
    final scope = AppScope.of(context);
    final contact = EmergencyContact(
      id: widget.contact?.id ??
          DateTime.now().microsecondsSinceEpoch.toString(),
      name: _name.text.trim(),
      phone: _phone.text.trim(),
      relation: _relation.text.trim(),
      sendSms: _sendSms,
      callable: _callable,
    );
    await scope.state.upsertContact(contact);
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _delete() async {
    final contact = widget.contact;
    if (contact == null) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Supprimer ce proche ?'),
        content: Text('${contact.name} ne sera plus prévenu en cas de crise.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Annuler'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Supprimer'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    if (!mounted) return;
    await AppScope.of(context).state.removeContact(contact.id);
    if (mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.contact == null ? 'Nouveau proche' : 'Modifier'),
        actions: [
          if (widget.contact != null)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Supprimer',
              onPressed: _delete,
            ),
        ],
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            TextFormField(
              controller: _name,
              textCapitalization: TextCapitalization.words,
              decoration: const InputDecoration(labelText: 'Nom'),
              validator: (value) => (value == null || value.trim().isEmpty)
                  ? 'Indiquez un nom'
                  : null,
            ),
            const SizedBox(height: 16),
            TextFormField(
              controller: _phone,
              keyboardType: TextInputType.phone,
              inputFormatters: [
                FilteringTextInputFormatter.allow(RegExp(r'[0-9+ .\-()]')),
              ],
              decoration: const InputDecoration(
                labelText: 'Téléphone mobile',
                hintText: '+33 6 12 34 56 78',
              ),
              validator: (value) {
                final digits =
                    (value ?? '').replaceAll(RegExp(r'[^0-9]'), '');
                if (digits.length < 6) return 'Numéro incomplet';
                return null;
              },
            ),
            const SizedBox(height: 16),
            TextFormField(
              controller: _relation,
              textCapitalization: TextCapitalization.sentences,
              decoration: const InputDecoration(
                labelText: 'Lien (facultatif)',
                hintText: 'Conjoint, mère, voisin, médecin…',
              ),
            ),
            const SizedBox(height: 8),
            SwitchListTile(
              value: _sendSms,
              onChanged: (v) => setState(() => _sendSms = v),
              title: const Text('Recevoir le SMS d\'alerte'),
              subtitle: const Text('Avec la position et le lien carte'),
            ),
            SwitchListTile(
              value: _callable,
              onChanged: (v) => setState(() => _callable = v),
              title: const Text('Peut être appelé'),
              subtitle: const Text('Proposé en un appui après l\'alerte'),
            ),
            const SizedBox(height: 24),
            FilledButton(onPressed: _save, child: const Text('Enregistrer')),
          ],
        ),
      ),
    );
  }
}

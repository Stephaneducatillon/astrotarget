import 'package:flutter/material.dart';

import '../app.dart';
import '../models/medical_profile.dart';

/// Fiche médicale : affichée pendant une crise, résumée dans le SMS.
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late MedicalProfile _initial;
  late final TextEditingController _fullName;
  late final TextEditingController _birthDate;
  late final TextEditingController _treatment;
  late final TextEditingController _allergies;
  late final TextEditingController _doctorName;
  late final TextEditingController _doctorPhone;
  late final TextEditingController _notes;
  bool _initialised = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_initialised) return;
    _initial = AppScope.of(context).state.profile;
    _fullName = TextEditingController(text: _initial.fullName);
    _birthDate = TextEditingController(text: _initial.birthDate);
    _treatment = TextEditingController(text: _initial.treatment);
    _allergies = TextEditingController(text: _initial.allergies);
    _doctorName = TextEditingController(text: _initial.doctorName);
    _doctorPhone = TextEditingController(text: _initial.doctorPhone);
    _notes = TextEditingController(text: _initial.notes);
    _initialised = true;
  }

  @override
  void dispose() {
    _fullName.dispose();
    _birthDate.dispose();
    _treatment.dispose();
    _allergies.dispose();
    _doctorName.dispose();
    _doctorPhone.dispose();
    _notes.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    await AppScope.of(context).state.saveProfile(MedicalProfile(
          fullName: _fullName.text.trim(),
          birthDate: _birthDate.text.trim(),
          treatment: _treatment.text.trim(),
          allergies: _allergies.text.trim(),
          doctorName: _doctorName.text.trim(),
          doctorPhone: _doctorPhone.text.trim(),
          notes: _notes.text.trim(),
        ));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Fiche médicale enregistrée')),
    );
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Fiche médicale')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Ces informations restent sur votre téléphone. Elles sont '
            'affichées pendant une alerte et peuvent être jointes au SMS.',
            style: TextStyle(fontSize: 15),
          ),
          const SizedBox(height: 20),
          _field(_fullName, 'Nom et prénom',
              capitalization: TextCapitalization.words),
          _field(_birthDate, 'Date de naissance', hint: '12/04/1985'),
          _field(_treatment, 'Traitement de crise',
              hint: 'Ventoline — 2 bouffées', lines: 2),
          _field(_allergies, 'Allergies', hint: 'Pollen, arachide…', lines: 2),
          _field(_doctorName, 'Médecin traitant',
              capitalization: TextCapitalization.words),
          _field(_doctorPhone, 'Téléphone du médecin',
              keyboard: TextInputType.phone),
          _field(_notes, 'Autres informations utiles', lines: 3),
          const SizedBox(height: 12),
          FilledButton(onPressed: _save, child: const Text('Enregistrer')),
        ],
      ),
    );
  }

  Widget _field(
    TextEditingController controller,
    String label, {
    String? hint,
    int lines = 1,
    TextInputType? keyboard,
    TextCapitalization capitalization = TextCapitalization.sentences,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: TextField(
        controller: controller,
        maxLines: lines,
        keyboardType: keyboard,
        textCapitalization: capitalization,
        decoration: InputDecoration(labelText: label, hintText: hint),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../app.dart';
import '../models/emergency_contact.dart';
import '../models/medical_profile.dart';
import '../services/location_service.dart';
import '../theme.dart';

/// Première ouverture : le strict minimum pour que le bouton serve à
/// quelque chose — un proche, le traitement, la localisation.
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _pageController = PageController();
  final _name = TextEditingController();
  final _treatment = TextEditingController();
  final _contactName = TextEditingController();
  final _contactPhone = TextEditingController();
  bool _locationGranted = false;
  int _page = 0;

  @override
  void dispose() {
    _pageController.dispose();
    _name.dispose();
    _treatment.dispose();
    _contactName.dispose();
    _contactPhone.dispose();
    super.dispose();
  }

  bool get _canContinue {
    if (_page == 1) {
      return _contactName.text.trim().isNotEmpty &&
          _contactPhone.text.replaceAll(RegExp(r'[^0-9]'), '').length >= 6;
    }
    return true;
  }

  Future<void> _next() async {
    if (_page < 2) {
      setState(() => _page++);
      await _pageController.animateToPage(
        _page,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
      return;
    }
    await _finish();
  }

  Future<void> _finish() async {
    final state = AppScope.of(context).state;
    await state.saveProfile(MedicalProfile(
      fullName: _name.text.trim(),
      treatment: _treatment.text.trim(),
    ));
    await state.upsertContact(EmergencyContact(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      name: _contactName.text.trim(),
      phone: _contactPhone.text.trim(),
    ));
    await state.completeOnboarding();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView(
                controller: _pageController,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  _step(
                    title: 'Bienvenue',
                    subtitle:
                        'En cas de crise, un seul appui prévient vos proches '
                        'avec votre position et lance l\'appel.',
                    icon: Icons.favorite,
                    children: [
                      TextField(
                        controller: _name,
                        textCapitalization: TextCapitalization.words,
                        decoration:
                            const InputDecoration(labelText: 'Votre prénom'),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: _treatment,
                        decoration: const InputDecoration(
                          labelText: 'Traitement de crise',
                          hintText: 'Ventoline — 2 bouffées',
                        ),
                      ),
                    ],
                  ),
                  _step(
                    title: 'Un proche à prévenir',
                    subtitle:
                        'Il recevra le SMS d\'alerte et sera appelé en premier. '
                        'Vous pourrez en ajouter d\'autres ensuite.',
                    icon: Icons.group,
                    children: [
                      TextField(
                        controller: _contactName,
                        textCapitalization: TextCapitalization.words,
                        onChanged: (_) => setState(() {}),
                        decoration: const InputDecoration(labelText: 'Nom'),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: _contactPhone,
                        keyboardType: TextInputType.phone,
                        onChanged: (_) => setState(() {}),
                        decoration: const InputDecoration(
                          labelText: 'Téléphone mobile',
                          hintText: '+33 6 12 34 56 78',
                        ),
                      ),
                    ],
                  ),
                  _step(
                    title: 'Localisation',
                    subtitle:
                        'Sans position, vos proches savent que vous allez mal '
                        'mais pas où vous êtes. Autorisez la localisation.',
                    icon: Icons.my_location,
                    children: [
                      FilledButton.icon(
                        icon: Icon(_locationGranted
                            ? Icons.check
                            : Icons.location_on),
                        label: Text(_locationGranted
                            ? 'Localisation autorisée'
                            : 'Autoriser la localisation'),
                        style: FilledButton.styleFrom(
                          backgroundColor:
                              _locationGranted ? AppColors.safe : null,
                        ),
                        onPressed: () async {
                          final granted =
                              await LocationService().ensurePermission();
                          if (mounted) {
                            setState(() => _locationGranted = granted);
                          }
                        },
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Pensez aussi à poser le widget « SOS » sur votre '
                        'écran d\'accueil : appui long sur l\'écran d\'accueil '
                        '→ Widgets → AsthmAlerte.',
                        style: TextStyle(fontSize: 15),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: FilledButton(
                onPressed: _canContinue ? _next : null,
                child: Text(_page < 2 ? 'Continuer' : 'Terminer'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _step({
    required String title,
    required String subtitle,
    required IconData icon,
    required List<Widget> children,
  }) {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        const SizedBox(height: 24),
        Icon(icon, size: 64, color: AppColors.sos),
        const SizedBox(height: 24),
        Text(title,
            style: const TextStyle(fontSize: 30, fontWeight: FontWeight.w800)),
        const SizedBox(height: 12),
        Text(subtitle, style: const TextStyle(fontSize: 17, height: 1.4)),
        const SizedBox(height: 32),
        ...children,
      ],
    );
  }
}

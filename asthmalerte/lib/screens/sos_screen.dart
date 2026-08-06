import 'dart:async';

import 'package:flutter/material.dart';
import 'package:vibration/vibration.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../app.dart';
import '../services/alert_service.dart';
import '../theme.dart';

enum _Phase { countdown, sending, result }

/// Écran d'alerte : décompte annulable, envoi, puis récapitulatif.
class SosScreen extends StatefulWidget {
  const SosScreen({super.key, this.autoStart = true});

  final bool autoStart;

  @override
  State<SosScreen> createState() => _SosScreenState();
}

class _SosScreenState extends State<SosScreen> {
  _Phase _phase = _Phase.countdown;
  int _remaining = 0;
  Timer? _timer;
  AlertStep _step = AlertStep.locating;
  AlertOutcome? _outcome;
  String? _error;

  @override
  void initState() {
    super.initState();
    WakelockPlus.enable();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (widget.autoStart) _startCountdown();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    WakelockPlus.disable();
    super.dispose();
  }

  void _startCountdown() {
    final settings = AppScope.of(context).state.settings;
    _remaining = settings.countdownSeconds;
    if (_remaining <= 0) {
      _send();
      return;
    }
    setState(() => _phase = _Phase.countdown);
    _buzz(120);
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) return;
      setState(() => _remaining--);
      _buzz(80);
      if (_remaining <= 0) {
        timer.cancel();
        _send();
      }
    });
  }

  Future<void> _buzz(int ms) async {
    if (!mounted) return;
    if (!AppScope.of(context).state.settings.vibrate) return;
    try {
      // Le type de retour a changé entre les versions du plugin : la
      // comparaison explicite fonctionne pour `bool` comme pour `bool?`.
      final hasVibrator = await Vibration.hasVibrator();
      if (hasVibrator == true) {
        await Vibration.vibrate(duration: ms);
      }
    } catch (_) {
      // Le retour haptique est un confort, jamais un bloquant.
    }
  }

  Future<void> _send() async {
    if (!mounted) return;
    setState(() {
      _phase = _Phase.sending;
      _error = null;
    });
    _buzz(400);

    final scope = AppScope.of(context);
    try {
      final outcome = await scope.alertService.trigger(
        onStep: (step) {
          if (mounted) setState(() => _step = step);
        },
      );
      if (!mounted) return;
      setState(() {
        _outcome = outcome;
        _phase = _Phase.result;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _phase = _Phase.result;
      });
    }
  }

  void _cancel() {
    _timer?.cancel();
    Navigator.of(context).maybePop();
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      // Pendant le décompte, seul le gros bouton « Annuler » sort de l'écran :
      // on évite l'annulation par un geste involontaire.
      canPop: _phase != _Phase.countdown,
      child: Scaffold(
        backgroundColor:
            _phase == _Phase.result ? null : AppColors.sosDark,
        body: SafeArea(
          child: switch (_phase) {
            _Phase.countdown => _buildCountdown(),
            _Phase.sending => _buildSending(),
            _Phase.result => _buildResult(),
          },
        ),
      ),
    );
  }

  Widget _buildCountdown() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const SizedBox(height: 24),
          const Text(
            'ALERTE DANS',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.w700,
              letterSpacing: 3,
            ),
          ),
          Expanded(
            child: Center(
              child: Text(
                '$_remaining',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 180,
                  height: 1,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
          ),
          const Text(
            'Vos proches vont recevoir votre position.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white, fontSize: 18),
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 96,
            width: double.infinity,
            child: FilledButton(
              style: FilledButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: AppColors.sosDark,
                textStyle: const TextStyle(
                    fontSize: 26, fontWeight: FontWeight.w800),
              ),
              onPressed: _cancel,
              child: const Text('ANNULER'),
            ),
          ),
          const SizedBox(height: 12),
          TextButton(
            onPressed: () {
              _timer?.cancel();
              _send();
            },
            child: const Text(
              'Envoyer tout de suite',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSending() {
    final label = switch (_step) {
      AlertStep.locating => 'Localisation en cours…',
      AlertStep.composing => 'Préparation du message…',
      AlertStep.sending => 'Envoi aux proches…',
      AlertStep.calling => 'Appel en cours…',
      AlertStep.done => 'Terminé',
    };
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 72,
            height: 72,
            child: CircularProgressIndicator(
                color: Colors.white, strokeWidth: 6),
          ),
          const SizedBox(height: 32),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResult() {
    final scope = AppScope.of(context);
    final outcome = _outcome;
    final position = outcome?.position;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Row(
          children: [
            Icon(
              _error == null ? Icons.check_circle : Icons.error,
              color: _error == null ? AppColors.safe : AppColors.sos,
              size: 40,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _error == null ? 'Alerte envoyée' : 'Échec de l\'alerte',
                style: const TextStyle(
                    fontSize: 26, fontWeight: FontWeight.w800),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        if (_error != null)
          Text(_error!, style: const TextStyle(fontSize: 16))
        else if (outcome != null) ...[
          Text(
            'Prévenus : ${outcome.recipients.map((c) => c.name).join(', ')}',
            style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          if (position != null)
            Text('Position : ${position.address.isNotEmpty ? position.address : position.coordinates}',
                style: const TextStyle(fontSize: 16))
          else
            Text(
              outcome.locationError ?? 'Position non partagée.',
              style: const TextStyle(fontSize: 16, color: AppColors.warning),
            ),
          if (!outcome.smsOpened) ...[
            const SizedBox(height: 8),
            const Text(
              'L\'application SMS ne s\'est pas ouverte. Utilisez le bouton ci-dessous.',
              style: TextStyle(fontSize: 16, color: AppColors.warning),
            ),
          ],
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Text(outcome.message,
                style: const TextStyle(fontSize: 15, height: 1.4)),
          ),
        ],
        const SizedBox(height: 24),
        FilledButton.icon(
          style: FilledButton.styleFrom(
            backgroundColor: AppColors.sos,
            foregroundColor: Colors.white,
          ),
          icon: const Icon(Icons.local_hospital, size: 26),
          label: const Text('Appeler les secours (112)'),
          onPressed: () => scope.alertService.callEmergencyServices(),
        ),
        const SizedBox(height: 12),
        if (scope.state.primaryCallContact != null)
          OutlinedButton.icon(
            icon: const Icon(Icons.phone),
            label: Text('Appeler ${scope.state.primaryCallContact!.name}'),
            onPressed: () =>
                scope.alertService.call(scope.state.primaryCallContact!),
          ),
        const SizedBox(height: 12),
        if (position != null)
          OutlinedButton.icon(
            icon: const Icon(Icons.map),
            label: const Text('Voir ma position sur la carte'),
            onPressed: () => scope.alertService
                .openMaps(position.latitude, position.longitude),
          ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          icon: const Icon(Icons.refresh),
          label: const Text('Renvoyer l\'alerte'),
          onPressed: _send,
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          style: OutlinedButton.styleFrom(foregroundColor: AppColors.safe),
          icon: const Icon(Icons.sentiment_satisfied_alt),
          label: const Text('Je vais mieux — rassurer mes proches'),
          onPressed: () async {
            final sent = await scope.alertService.sendAllClear();
            if (!mounted) return;
            if (!sent) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Aucun proche à rassurer.')),
              );
            }
          },
        ),
        const SizedBox(height: 24),
        if (!scope.state.profile.isEmpty) const _MedicalRecap(),
        const SizedBox(height: 24),
        TextButton(
          onPressed: () => Navigator.of(context).maybePop(),
          child: const Text('Fermer', style: TextStyle(fontSize: 18)),
        ),
      ],
    );
  }
}

/// Rappel du traitement, lisible par un proche ou un secouriste.
class _MedicalRecap extends StatelessWidget {
  const _MedicalRecap();

  @override
  Widget build(BuildContext context) {
    final profile = AppScope.of(context).state.profile;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.safe.fade(0.10),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.safe.fade(0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Fiche médicale',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
          const SizedBox(height: 8),
          if (profile.fullName.isNotEmpty) Text('Nom : ${profile.fullName}'),
          if (profile.treatment.isNotEmpty)
            Text('Traitement : ${profile.treatment}',
                style: const TextStyle(
                    fontSize: 17, fontWeight: FontWeight.w700)),
          if (profile.allergies.isNotEmpty)
            Text('Allergies : ${profile.allergies}'),
          if (profile.doctorPhone.isNotEmpty)
            Text('Médecin : ${profile.doctorName} — ${profile.doctorPhone}'),
        ],
      ),
    );
  }
}

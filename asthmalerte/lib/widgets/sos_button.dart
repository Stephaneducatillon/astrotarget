import 'package:flutter/material.dart';

import '../theme.dart';

/// Le bouton d'alerte : rond, rouge, énorme, pulsant.
///
/// Il occupe volontairement la majeure partie de l'écran d'accueil : en crise,
/// on ne cherche pas un bouton, on appuie n'importe où au centre.
class SosButton extends StatefulWidget {
  const SosButton({
    super.key,
    required this.onPressed,
    this.label = 'SOS',
    this.sublabel = 'Prévenir mes proches',
    this.enabled = true,
  });

  final VoidCallback onPressed;
  final String label;
  final String sublabel;
  final bool enabled;

  @override
  State<SosButton> createState() => _SosButtonState();
}

class _SosButtonState extends State<SosButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1600),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.biggest.shortestSide.clamp(180.0, 360.0);
        return Center(
          child: Semantics(
            button: true,
            label: 'Bouton d\'alerte SOS. ${widget.sublabel}',
            child: AnimatedBuilder(
              animation: _pulse,
              builder: (context, child) {
                final t = widget.enabled ? _pulse.value : 0.0;
                return Container(
                  width: size,
                  height: size,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.sos.fade(0.28 * (1 - t) + 0.10),
                        blurRadius: 24 + 36 * t,
                        spreadRadius: 4 + 18 * t,
                      ),
                    ],
                  ),
                  child: child,
                );
              },
              child: Material(
                color: widget.enabled ? AppColors.sos : Colors.grey.shade500,
                shape: const CircleBorder(),
                child: InkWell(
                  customBorder: const CircleBorder(),
                  onTap: widget.enabled ? widget.onPressed : null,
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.warning_amber_rounded,
                            size: 56, color: Colors.white),
                        const SizedBox(height: 8),
                        Text(
                          widget.label,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 64,
                            height: 1,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 4,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          widget.sublabel,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

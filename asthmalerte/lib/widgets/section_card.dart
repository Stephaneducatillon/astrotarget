import 'package:flutter/material.dart';

/// Carte arrondie utilisée dans les écrans de réglages / fiche médicale.
///
/// C'est un [Material] et non un [Container] décoré : les `ListTile` et
/// `SwitchListTile` qu'elle contient peignent leur fond et leurs ondes de
/// contact sur le `Material` le plus proche, qui doit donc être la carte
/// elle-même — sinon l'effet passe sous la couleur de fond.
class SectionCard extends StatelessWidget {
  const SectionCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
  });

  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Material(
      color: isDark ? const Color(0xFF1A1D23) : Colors.white,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: padding,
        child: SizedBox(width: double.infinity, child: child),
      ),
    );
  }
}

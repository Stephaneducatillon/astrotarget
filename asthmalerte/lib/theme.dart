import 'package:flutter/material.dart';

/// Palette volontairement contrastée : l'écran doit rester lisible en pleine
/// crise, à bout de bras, éventuellement en plein soleil.
class AppColors {
  static const Color sos = Color(0xFFD32029);
  static const Color sosDark = Color(0xFF8E0F16);
  static const Color safe = Color(0xFF1B7F4B);
  static const Color warning = Color(0xFFB26A00);
  static const Color ink = Color(0xFF14171C);
  static const Color surface = Color(0xFFF6F7F9);
}

extension AppColorFade on Color {
  /// Équivalent de `withOpacity`, qui est déprécié sur les Flutter récents.
  Color fade(double opacity) =>
      withAlpha((opacity.clamp(0.0, 1.0) * 255).round());
}

ThemeData buildAppTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final scheme = ColorScheme.fromSeed(
    seedColor: AppColors.sos,
    brightness: brightness,
  ).copyWith(
    primary: AppColors.sos,
    error: AppColors.sos,
  );

  final base = ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: isDark ? const Color(0xFF101216) : AppColors.surface,
  );

  return base.copyWith(
    // Cibles tactiles généreuses : on vise le bouton en tremblant.
    materialTapTargetSize: MaterialTapTargetSize.padded,
    appBarTheme: AppBarTheme(
      centerTitle: false,
      elevation: 0,
      backgroundColor: base.scaffoldBackgroundColor,
      foregroundColor: isDark ? Colors.white : AppColors.ink,
      titleTextStyle: TextStyle(
        fontSize: 24,
        fontWeight: FontWeight.w700,
        color: isDark ? Colors.white : AppColors.ink,
      ),
    ),
    textTheme: base.textTheme.apply(fontSizeFactor: 1.05),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(60),
        textStyle: const TextStyle(fontSize: 19, fontWeight: FontWeight.w700),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(56),
        textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: isDark ? const Color(0xFF1A1D23) : Colors.white,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(14),
        borderSide: BorderSide(color: scheme.outlineVariant),
      ),
      contentPadding:
          const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
    ),
    listTileTheme: const ListTileThemeData(
      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
    ),
  );
}

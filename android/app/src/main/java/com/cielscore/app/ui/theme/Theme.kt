package com.cielscore.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** Palette nocturne : l'application est faite pour etre ouverte dehors, la nuit. */
private val NightScheme = darkColorScheme(
    primary = Color(0xFF7FB3FF),
    onPrimary = Color(0xFF06132B),
    primaryContainer = Color(0xFF16305C),
    onPrimaryContainer = Color(0xFFD3E4FF),
    secondary = Color(0xFFF2C14E),
    onSecondary = Color(0xFF241A00),
    secondaryContainer = Color(0xFF4A3800),
    onSecondaryContainer = Color(0xFFFFE08A),
    background = Color(0xFF0B1020),
    onBackground = Color(0xFFE3E6F0),
    surface = Color(0xFF121A2E),
    onSurface = Color(0xFFE3E6F0),
    surfaceVariant = Color(0xFF1C2740),
    onSurfaceVariant = Color(0xFFB6BFD4),
    outline = Color(0xFF44506B),
    error = Color(0xFFFF8A80),
)

/** Mode nuit rouge, pour preserver la vision nocturne sur le terrain. */
private val RedScheme = darkColorScheme(
    primary = Color(0xFFFF6B5A),
    onPrimary = Color(0xFF1A0301),
    primaryContainer = Color(0xFF4A0F08),
    onPrimaryContainer = Color(0xFFFFCDC6),
    secondary = Color(0xFFFF8A70),
    onSecondary = Color(0xFF1A0301),
    background = Color(0xFF0A0201),
    onBackground = Color(0xFFFFB4A6),
    surface = Color(0xFF160604),
    onSurface = Color(0xFFFFB4A6),
    surfaceVariant = Color(0xFF25100C),
    onSurfaceVariant = Color(0xFFD98A7E),
    outline = Color(0xFF5C2A22),
    error = Color(0xFFFFB4A6),
)

private val AppTypography = Typography(
    titleLarge = TextStyle(fontSize = 21.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
    bodyMedium = TextStyle(fontSize = 14.sp),
    bodySmall = TextStyle(fontSize = 12.sp),
    labelSmall = TextStyle(fontSize = 11.sp, fontWeight = FontWeight.Medium),
)

@Composable
fun CielScoreTheme(
    nightMode: Boolean = false,
    content: @Composable () -> Unit,
) {
    // L'application reste sombre en permanence : elle s'utilise dehors, la nuit.
    val scheme = if (nightMode) RedScheme else NightScheme
    MaterialTheme(colorScheme = scheme, typography = AppTypography, content = content)
}

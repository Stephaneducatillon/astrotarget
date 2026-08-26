package com.cielscore.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.cielscore.app.scoring.Formulas

/** Carte de section, avec un titre et un contenu. */
@Composable
fun SectionCard(
    title: String,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            if (subtitle != null) {
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Column(Modifier.padding(top = 10.dp)) { content() }
        }
    }
}

/** Ligne « libelle / valeur ». */
@Composable
fun LabeledValue(label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
    }
}

/**
 * Pastille de score, coloree selon la lecture du score de la section 6.5.
 */
@Composable
fun ScoreBadge(score: Double, modifier: Modifier = Modifier, size: Int = 44) {
    val color = scoreColor(score)
    Box(
        modifier = modifier
            .size(size.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(color.copy(alpha = 0.18f)),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            "%.0f".format(score),
            style = MaterialTheme.typography.titleMedium,
            color = color,
            fontWeight = FontWeight.Bold,
        )
    }
}

/** Couleurs de la lecture du score (section 6.5). */
fun scoreColor(score: Double): Color = when {
    score >= 85 -> Color(0xFF4CAF50)
    score >= 70 -> Color(0xFF8BC34A)
    score >= 50 -> Color(0xFFF2C14E)
    score >= 25 -> Color(0xFFFF9800)
    score >= 1 -> Color(0xFFFF5722)
    else -> Color(0xFFB0BEC5)
}

/** Libelle d'interpretation du score (section 6.5). */
@Composable
fun ScoreInterpretation(score: Double, modifier: Modifier = Modifier) {
    val (label, advice) = Formulas.scoreInterpretation(score)
    Column(modifier) {
        Text(label, style = MaterialTheme.typography.titleMedium, color = scoreColor(score))
        Text(
            advice,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** Petite etiquette d'information. */
@Composable
fun Chip(text: String, color: Color = MaterialTheme.colorScheme.surfaceVariant) {
    Box(
        Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(color)
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Text(text, style = MaterialTheme.typography.labelSmall)
    }
}

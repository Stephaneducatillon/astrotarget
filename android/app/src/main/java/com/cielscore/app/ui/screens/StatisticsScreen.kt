package com.cielscore.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.cielscore.app.data.db.ObservationRepository
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Statistiques — section 2.6 : compteurs, progression Messier et
 * Caldwell, heatmap d'activite sur douze mois, favoris, types et sites.
 */
@Composable
fun StatisticsScreen(viewModel: AppViewModel, state: AppUiState) {
    val user = state.user ?: return
    val repository = viewModel.observations

    val stats by remember(user.username) { repository.stats(user.username) }
        .collectAsState(initial = ObservationRepository.Stats(0, 0, 0, 0.0))
    val favourites by remember(user.username) { repository.favourites(user.username) }
        .collectAsState(initial = emptyList())
    val byType by remember(user.username) { repository.byType(user.username) }
        .collectAsState(initial = emptyList())
    val bySite by remember(user.username) { repository.bySite(user.username) }
        .collectAsState(initial = emptyList())
    val activity by remember(user.username) { repository.activity(user.username) }
        .collectAsState(initial = emptyMap())

    var progress by remember(user.username) { mutableStateOf(0 to 0) }
    LaunchedEffect(user.username, stats.observations) {
        progress = repository.progress(user.username)
    }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item {
            SectionCard(title = "Compteurs") {
                LabeledValue("Soirees", "${stats.sessions}")
                LabeledValue("Observations totales", "${stats.observations}")
                LabeledValue("Objets uniques", "${stats.distinctObjects}")
                LabeledValue("Score moyen", "%.0f / 100".format(stats.averageScore))
            }
        }

        item {
            SectionCard(title = "Progression") {
                ProgressRow("Messier", progress.first, 110)
                ProgressRow("Caldwell", progress.second, 109)
            }
        }

        item {
            SectionCard(title = "Activite", subtitle = "12 derniers mois") {
                ActivityHeatmap(activity)
            }
        }

        if (favourites.isNotEmpty()) {
            item {
                SectionCard(title = "Objets favoris") {
                    favourites.forEach {
                        LabeledValue(it.label, "${it.total} passage(s)")
                    }
                }
            }
        }

        if (byType.isNotEmpty()) {
            item {
                SectionCard(title = "Repartition par type") {
                    byType.forEach { LabeledValue(it.label, "${it.total}") }
                }
            }
        }

        if (bySite.isNotEmpty()) {
            item {
                SectionCard(title = "Repartition par site") {
                    bySite.forEach { LabeledValue(it.label, "${it.total}") }
                }
            }
        }
    }
}

@Composable
private fun ProgressRow(label: String, done: Int, total: Int) {
    Column(Modifier.padding(vertical = 4.dp)) {
        LabeledValue(label, "$done / $total")
        LinearProgressIndicator(
            progress = { if (total == 0) 0f else done.toFloat() / total },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

/** Heatmap type GitHub : un carre par jour sur les douze derniers mois. */
@Composable
private fun ActivityHeatmap(activity: Map<String, Int>) {
    val formatter = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.FRANCE)
        .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
    val today = java.util.Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"))
    val days = 364
    val maxCount = (activity.values.maxOrNull() ?: 1).coerceAtLeast(1)

    // 52 colonnes de 7 jours, la plus recente a droite.
    Column {
        for (weekday in 0 until 7) {
            Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                for (week in 0 until 52) {
                    val offset = days - (week * 7 + (6 - weekday))
                    val c = today.clone() as java.util.Calendar
                    c.add(java.util.Calendar.DAY_OF_YEAR, -offset)
                    val count = activity[formatter.format(c.time)] ?: 0
                    val alpha = if (count == 0) 0.08f
                    else 0.30f + 0.70f * (count.toFloat() / maxCount)
                    Box(
                        Modifier
                            .padding(vertical = 1.dp)
                            .size(5.dp)
                            .clip(RoundedCornerShape(1.dp))
                            .background(Color(0xFF4CAF50).copy(alpha = alpha))
                    )
                }
            }
        }
        Text(
            "Un carre par jour ; plus la couleur est vive, plus il y a d'observations.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp),
        )
    }
}

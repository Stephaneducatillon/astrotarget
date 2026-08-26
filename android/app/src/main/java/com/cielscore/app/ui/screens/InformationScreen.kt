package com.cielscore.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.cielscore.app.astro.SkyCalendar
import com.cielscore.app.astro.Twilight
import com.cielscore.app.scoring.Formulas
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.Chip
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Informations — section 2.1 : tableau de bord astronomique du jour,
 * accessible sans connexion.
 */
@Composable
fun InformationScreen(viewModel: AppViewModel, state: AppUiState) {
    LaunchedEffect(Unit) { viewModel.loadInformationTab() }

    val calendar = remember(state.params.epochMillis) {
        SkyCalendar.upcoming(state.params.epochMillis, 60)
    }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item { WelcomeBanner(state) }
        item { ApodCard(state) }
        item { SunAndMoonCard(state) }
        item { AstronomicalNightCard(state) }
        item { AuroraCard(state) }
        item { CalendarCard(calendar) }
        item { LaunchesCard(state) }
    }
}

/**
 * Bandeau d'accueil : score du meilleur objet, Bortle, seeing.
 *
 * RG-INFO-01 — sans commune selectionnee, une estimation France en Bortle 5 est
 * affichee et l'utilisateur est invite a choisir sa commune.
 */
@Composable
private fun WelcomeBanner(state: AppUiState) {
    val best = state.topTargets.firstOrNull()
    val noCommune = state.params.site.name == com.cielscore.app.model.ObservingSite.DEFAULT.name

    SectionCard(
        title = "Ce soir",
        subtitle = state.params.site.name +
            if (state.params.site.department.isNotBlank()) " (${state.params.site.department})" else "",
    ) {
        if (best != null) {
            LabeledValue("Meilleure cible", best.target.displayName)
            LabeledValue("Score", "%.0f / 100".format(best.score))
        } else {
            Text(
                "Lancez une session depuis le Dashboard pour obtenir le classement des cibles.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        LabeledValue(
            "Indice de Bortle",
            "${state.params.site.bortle} — ${Formulas.bortleDescription(state.params.site.bortle)}",
        )
        LabeledValue(
            "Seeing",
            "${state.conditions.seeingIndex} — ${state.conditions.seeingLabel}" +
                if (!state.conditions.ok) " (valeur de repli)" else "",
        )
        if (noCommune) {
            Text(
                "Estimation France, Bortle 5. Choisissez votre commune depuis le Dashboard " +
                    "pour un calcul adapte a votre ciel.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

/** Image du jour de la NASA. */
@Composable
private fun ApodCard(state: AppUiState) {
    SectionCard(title = "Image du jour", subtitle = "NASA APOD") {
        val apod = state.apod
        when {
            apod != null -> {
                if (apod.mediaType == "image" && apod.imageUrl != null) {
                    AsyncImage(
                        model = apod.imageUrl,
                        contentDescription = apod.title,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .clip(RoundedCornerShape(10.dp)),
                    )
                }
                Text(
                    apod.title,
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(top = 8.dp),
                )
                Text(
                    apod.explanation,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
                apod.copyright?.let {
                    Text(
                        "© $it",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                }
            }
            else -> Text(
                state.apodError ?: "Chargement…",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SunAndMoonCard(state: AppUiState) {
    SectionCard(title = "Soleil & Lune") {
        LabeledValue("Lever du Soleil", formatTime(state.sunRise))
        LabeledValue("Coucher du Soleil", formatTime(state.sunSet))
        LabeledValue("Hauteur du Soleil", "%.1f°".format(state.sunAltitudeDeg))
        state.moon?.let { moon ->
            LabeledValue("Phase lunaire", "${moon.phaseName} — %.0f %%".format(moon.phasePercent))
            LabeledValue("Altitude de la Lune", "%.1f°".format(moon.altitudeDeg))
        }
        LabeledValue("Lever de la Lune", formatTime(state.moonRise))
        LabeledValue("Coucher de la Lune", formatTime(state.moonSet))
    }
}

/** Nuit astronomique : debut, fin, duree (sections 2.1 et 7.5). */
@Composable
private fun AstronomicalNightCard(state: AppUiState) {
    val night = state.night ?: return
    val phase = night.currentPhase

    SectionCard(title = "Nuit astronomique", subtitle = phase.message) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Chip(phase.label, Color(phase.colorArgb).copy(alpha = 0.28f))
            Chip("Score nuit %.0f".format(Twilight.nightScore(night.sunAltitudeDeg)))
        }
        LabeledValue("Crepuscule civil", formatWindow(night.civil))
        LabeledValue("Crepuscule nautique", formatWindow(night.nautical))
        LabeledValue("Crepuscule astronomique", formatWindow(night.astronomical))
        if (!night.astronomical.exists) {
            Text(
                "Aucune nuit astronomique a cette date et a cette latitude : le Soleil ne " +
                    "descend pas sous -18°. C'est le cas au-dela d'environ 49° N de la mi-mai " +
                    "a la fin juillet.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary,
                modifier = Modifier.padding(top = 6.dp),
            )
        } else {
            LabeledValue(
                "Duree de ciel noir",
                "%.1f h".format(night.astronomical.durationMinutes / 60.0),
            )
        }
    }
}

/** Aurores boreales : indice Kp et historique 24 h. */
@Composable
private fun AuroraCard(state: AppUiState) {
    SectionCard(title = "Aurores boreales", subtitle = state.kp?.source ?: "GFZ Potsdam / NOAA") {
        val kp = state.kp
        if (kp == null) {
            Text(
                "Indice Kp indisponible.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            LabeledValue("Indice Kp", "%.1f".format(kp.current))
            LabeledValue("Interpretation", kp.label)
            LabeledValue("Mesures sur 24 h", "${kp.history.size}")
        }
    }
}

/** Calendrier sur 60 jours : phases, solstices, meteores. */
@Composable
private fun CalendarCard(events: List<SkyCalendar.Event>) {
    SectionCard(title = "Calendrier", subtitle = "60 prochains jours") {
        if (events.isEmpty()) {
            Text("Aucun evenement.", style = MaterialTheme.typography.bodySmall)
        }
        events.forEach { event ->
            Row(
                Modifier.fillMaxWidth().padding(vertical = 3.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(event.title, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        event.detail,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(formatDate(event.epochMillis), style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/** Cinq prochains tirs spatiaux. */
@Composable
private fun LaunchesCard(state: AppUiState) {
    SectionCard(title = "Prochains lancements", subtitle = "The Space Devs") {
        if (state.launches.isEmpty()) {
            Text(
                "Aucun lancement recupere.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        state.launches.forEach { launch ->
            Column(Modifier.padding(vertical = 4.dp)) {
                Text(launch.name, style = MaterialTheme.typography.bodyMedium)
                Text(
                    listOfNotNull(
                        launch.provider.takeIf { it.isNotBlank() },
                        launch.padName.takeIf { it.isNotBlank() },
                        launch.netMillis?.let { formatDateTime(it) },
                    ).joinToString(" — "),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

internal fun formatTime(epochMillis: Long?): String =
    epochMillis?.let {
        java.text.SimpleDateFormat("HH'h'mm", java.util.Locale.FRANCE).format(java.util.Date(it))
    } ?: "—"

internal fun formatDate(epochMillis: Long): String =
    java.text.SimpleDateFormat("d MMM", java.util.Locale.FRANCE)
        .format(java.util.Date(epochMillis))

internal fun formatDateTime(epochMillis: Long): String =
    java.text.SimpleDateFormat("d MMM 'a' HH'h'mm", java.util.Locale.FRANCE)
        .format(java.util.Date(epochMillis))

private fun formatWindow(window: Twilight.Window): String =
    if (!window.exists) "Inexistante"
    else "${formatTime(window.startMillis)} — ${formatTime(window.endMillis)}"

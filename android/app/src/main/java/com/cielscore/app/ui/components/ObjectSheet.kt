package com.cielscore.app.ui.components

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import coil.compose.AsyncImage
import com.cielscore.app.catalog.Constellation
import com.cielscore.app.catalog.Star
import com.cielscore.app.data.net.SkyImagery
import com.cielscore.app.scoring.Formulas
import com.cielscore.app.scoring.ScoringEngine
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import kotlinx.coroutines.launch

/**
 * Fiche objet — section 2.2 : score, image reelle, carte du ciel, guide IA,
 * altitude et magnitude.
 *
 * Le panneau lateral offre les trois vues de la section 3.2 :
 *   1. Carte du ciel   vue hemispherique depuis le lieu d'observation
 *   2. Aladin DSS2     image reelle avec cercle du champ oculaire
 *   3. Lien externe    ouverture de Stellarium Web
 */
@Composable
fun ObjectSheet(
    scored: ScoringEngine.Scored,
    state: AppUiState,
    viewModel: AppViewModel,
    stars: List<Star>,
    starsById: Map<String, Star>,
    constellations: List<Constellation>,
    modifier: Modifier = Modifier,
) {
    val target = scored.target
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var view by remember(target.id) { mutableIntStateOf(0) }
    var survey by remember { mutableStateOf(SkyImagery.Survey.DSS2_COLOR) }
    var guide by remember(target.id) { mutableStateOf<String?>(null) }
    var guideLoading by remember(target.id) { mutableStateOf(false) }
    var guideError by remember(target.id) { mutableStateOf<String?>(null) }

    Column(modifier.verticalScroll(rememberScrollState())) {

        SectionCard(title = target.displayName, subtitle = subtitleFor(scored)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                ScoreBadge(scored.score, size = 56)
                ScoreInterpretation(scored.score, Modifier.padding(start = 12.dp))
            }

            Column(Modifier.padding(top = 10.dp)) {
                LabeledValue("Altitude", "%.1f°".format(scored.altitudeDeg))
                LabeledValue("Azimut", "%.0f° (${cardinal(scored.azimuthDeg)})".format(scored.azimuthDeg))
                LabeledValue("Magnitude", target.magnitude?.let { "%.1f".format(it) } ?: "—")
                LabeledValue("Taille apparente", target.sizeLabel)
                scored.surfaceBrightness?.let {
                    LabeledValue(
                        "Brillance de surface",
                        "%.2f mag/arcsec² (limite %.1f)".format(
                            it, Formulas.surfaceBrightnessLimit(state.params.site.bortle)
                        ),
                    )
                }
                LabeledValue("Fenetre au-dessus de 30°", "%.0f min".format(scored.windowMinutes))
                LabeledValue("Distance a la Lune", "%.0f°".format(scored.moonDistanceDeg))
                if (target.constellation.isNotBlank()) {
                    LabeledValue("Constellation", target.constellation)
                }
            }
        }

        // Detail des criteres ponderes.
        SectionCard(
            title = "Detail du score",
            subtitle = if (target.isSolarSystem) "Formule planetes (section 6.3)"
            else if (state.params.isSmartMode) "Formule smart telescope (section 6.4)"
            else "Huit criteres ponderes (section 6.1)",
            modifier = Modifier.padding(top = 10.dp),
        ) {
            val b = scored.breakdown
            if (target.isSolarSystem) {
                CriterionRow("Altitude", 40, b.altitude)
                CriterionRow("Magnitude", 30, b.magnitude)
                CriterionRow("Transparence", 20, b.transparency)
                CriterionRow("Lune", 10, b.moon)
            } else if (state.params.isSmartMode) {
                CriterionRow("Altitude", 25, b.altitude)
                CriterionRow("Transparence", 20, b.transparency)
                CriterionRow("Seeing", 15, b.seeing)
                CriterionRow("Bortle", 15, b.bortle)
                CriterionRow("Lune", 15, b.moon)
                CriterionRow("F/D", 5, b.focalRatio)
                CriterionRow("Champ", 5, b.fieldMatch)
            } else {
                CriterionRow("Altitude", 25, b.altitude)
                CriterionRow("Fenetre", 15, b.window)
                CriterionRow("Seeing", 11, b.seeing)
                CriterionRow("Transparence", 13, b.transparency)
                CriterionRow("Bortle", 8, b.bortle)
                CriterionRow("Lune", 6, b.moon)
                CriterionRow("Brillance de surface", 15, b.surfaceBrightness)
                CriterionRow("Nuit astronomique", 7, b.night)
            }
        }

        // Courbe d'altitude sur 10 heures.
        if (state.altitudeCurve.isNotEmpty()) {
            SectionCard(
                title = "Courbe d'altitude",
                subtitle = "10 heures, seuil optimal a 30°",
                modifier = Modifier.padding(top = 10.dp),
            ) {
                AltitudeCurve(state.altitudeCurve)
            }
        }

        // Les trois vues de la section 3.2.
        SectionCard(title = "Ou pointer", modifier = Modifier.padding(top = 10.dp)) {
            TabRow(selectedTabIndex = view, containerColor = MaterialTheme.colorScheme.surface) {
                listOf("Carte du ciel", "Aladin", "Stellarium").forEachIndexed { index, label ->
                    Tab(
                        selected = view == index,
                        onClick = { view = index },
                        text = { Text(label, style = MaterialTheme.typography.labelSmall) },
                    )
                }
            }

            when (view) {
                0 -> SkyMapView(
                    stars = stars,
                    starsById = starsById,
                    constellations = constellations,
                    latitude = state.params.site.latitude,
                    longitude = state.params.site.longitude,
                    epochMillis = state.params.epochMillis,
                    target = target,
                    nightMode = state.nightMode,
                    modifier = Modifier.padding(top = 10.dp),
                )

                1 -> Column(Modifier.padding(top = 10.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        SkyImagery.Survey.entries.forEach { s ->
                            FilterChip(
                                selected = survey == s,
                                onClick = { survey = s },
                                label = {
                                    Text(s.label, style = MaterialTheme.typography.labelSmall)
                                },
                            )
                        }
                    }
                    AladinWebView(
                        target = target,
                        survey = survey,
                        fieldDeg = SkyImagery.suggestedFieldDeg(target),
                        fieldCircleArcmin = eyepieceFieldArcmin(state),
                        nightMode = state.nightMode,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(320.dp)
                            .padding(top = 8.dp)
                            .clip(RoundedCornerShape(10.dp)),
                    )
                    Text(
                        "Le cercle dore represente le champ reel de votre oculaire.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }

                else -> Column(Modifier.padding(top = 10.dp)) {
                    Text(
                        "Ouvre Stellarium Web sur la position, la date et l'objet courants.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    TextButton(onClick = {
                        val url = SkyImagery.stellariumWebUrl(
                            target,
                            state.params.site.latitude,
                            state.params.site.longitude,
                            state.params.epochMillis,
                        )
                        context.startActivity(Intent(Intent.ACTION_VIEW, url.toUri()))
                    }) { Text("Ouvrir Stellarium Web") }
                }
            }
        }

        // Image reelle hips2fits.
        if (!target.isSolarSystem) {
            SectionCard(
                title = "Image reelle",
                subtitle = "CDS hips2fits — ${survey.label}",
                modifier = Modifier.padding(top = 10.dp),
            ) {
                AsyncImage(
                    model = SkyImagery.thumbnailUrl(
                        target.raDeg,
                        target.decDeg,
                        SkyImagery.suggestedFieldDeg(target),
                        survey,
                    ),
                    contentDescription = target.displayName,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(260.dp)
                        .clip(RoundedCornerShape(10.dp)),
                )
            }
        }

        // Guide IA.
        SectionCard(
            title = "Guide d'observation",
            subtitle = "Genere par Mistral",
            modifier = Modifier.padding(top = 10.dp, bottom = 16.dp),
        ) {
            when {
                guideLoading -> CircularProgressIndicator(Modifier.padding(8.dp))
                guide != null -> Text(guide!!, style = MaterialTheme.typography.bodyMedium)
                guideError != null -> Text(
                    guideError!!,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
                else -> AssistChip(
                    onClick = {
                        guideLoading = true
                        guideError = null
                        scope.launch {
                            viewModel.objectGuide(target)
                                .onSuccess { guide = it; guideLoading = false }
                                .onFailure { guideError = it.message; guideLoading = false }
                        }
                    },
                    label = { Text("Demander le guide de " + target.id) },
                )
            }
        }
    }
}

@Composable
private fun CriterionRow(label: String, weightPercent: Int, score: Double) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            "$label ($weightPercent %)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            "%.0f → %.1f pt".format(score, score * weightPercent / 100.0),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private fun subtitleFor(scored: ScoringEngine.Scored): String =
    listOfNotNull(
        scored.target.type.label,
        scored.target.catalog.label,
        scored.target.designation.takeIf { it != scored.target.id && it.isNotBlank() },
    ).joinToString(" — ")

/** Champ reel de l'oculaire courant, en arcminutes, pour le cercle Aladin. */
private fun eyepieceFieldArcmin(state: AppUiState): Double? {
    val params = state.params
    if (params.isSmartMode) return params.smartTelescope?.fieldWidthArcmin
    if (params.instrument == com.cielscore.app.model.InstrumentType.NAKED_EYE) return null
    // Oculaire de reference : 25 mm de focale, 52 degres de champ apparent.
    val magnification = Formulas.magnification(params.effectiveFocalMm, 25.0)
    return Formulas.trueFieldDeg(52.0, magnification) * 60.0
}

/** Point cardinal correspondant a un azimut compte depuis le Nord. */
fun cardinal(azimuthDeg: Double): String {
    val labels = listOf("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO")
    val index = ((azimuthDeg % 360.0 + 360.0) % 360.0 / 22.5).toInt() % 16
    return labels[index]
}

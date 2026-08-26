package com.cielscore.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cielscore.app.model.SmartTelescope
import com.cielscore.app.scoring.Formulas
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Equipement — section 2.5 : oculaires, astrophotographie et
 * smart telescopes.
 */
@Composable
fun EquipmentScreen(state: AppUiState) {
    var tab by remember { mutableIntStateOf(0) }

    Column(Modifier.fillMaxWidth()) {
        TabRow(selectedTabIndex = tab, containerColor = MaterialTheme.colorScheme.surface) {
            listOf("Oculaires", "Astrophoto", "Smart telescopes").forEachIndexed { index, label ->
                Tab(
                    selected = tab == index,
                    onClick = { tab = index },
                    text = { Text(label, style = MaterialTheme.typography.labelSmall) },
                )
            }
        }
        when (tab) {
            0 -> EyepieceTab(state)
            1 -> AstrophotoTab(state)
            else -> SmartTelescopeTab(state)
        }
    }
}

/** Oculaires : grossissement, champ reel, pupille de sortie, diagnostic. */
@Composable
private fun EyepieceTab(state: AppUiState) {
    val diameter = state.params.effectiveDiameterMm
    val focal = state.params.effectiveFocalMm
    val ratio = Formulas.focalRatio(focal, diameter)
    val focalLengths = listOf(4.0, 6.0, 9.0, 12.5, 17.0, 25.0, 32.0, 40.0)

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item {
            SectionCard(
                title = "Instrument",
                subtitle = "%.0f mm f/%.1f — focale %.0f mm".format(diameter, ratio, focal),
            ) {
                LabeledValue(
                    "Grossissement minimal",
                    "x%.0f (pupille %.1f mm)".format(
                        Formulas.minMagnification(diameter, state.params.eyePupilMm),
                        state.params.eyePupilMm,
                    ),
                )
                LabeledValue(
                    "Grossissement ideal",
                    "x%.0f".format(Formulas.idealMagnification(diameter)),
                )
                LabeledValue(
                    "Grossissement maximal",
                    "x%.0f".format(Formulas.maxMagnification(diameter)),
                )
                LabeledValue(
                    "Magnitude limite visuelle",
                    "%.1f".format(Formulas.instrumentLimitingMagnitude(diameter)),
                )
            }
        }

        item {
            SectionCard(title = "Par focale d'oculaire", subtitle = "Champ apparent 52°") {
                Row(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
                    HeaderCell("Focale", 1f)
                    HeaderCell("Gross.", 1f)
                    HeaderCell("Champ", 1f)
                    HeaderCell("Pupille", 1f)
                }
                focalLengths.forEach { eyepiece ->
                    val magnification = Formulas.magnification(focal, eyepiece)
                    val field = Formulas.trueFieldDeg(52.0, magnification)
                    val exitPupil = Formulas.exitPupilMm(eyepiece, ratio)
                    Column(Modifier.padding(vertical = 4.dp)) {
                        Row(Modifier.fillMaxWidth()) {
                            BodyCell("%.1f mm".format(eyepiece), 1f)
                            BodyCell("x%.0f".format(magnification), 1f)
                            BodyCell("%.2f°".format(field), 1f)
                            BodyCell("%.1f mm".format(exitPupil), 1f)
                        }
                        Text(
                            Formulas.eyepieceDiagnosis(
                                magnification, diameter, state.params.eyePupilMm
                            ),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

/** Astrophotographie : F/D effectif, bornes de Shannon, echantillonnage, champ. */
@Composable
private fun AstrophotoTab(state: AppUiState) {
    var pixelSize by remember { mutableFloatStateOf(3.76f) }
    var barlow by remember { mutableFloatStateOf(1f) }
    var sensorWidth by remember { mutableFloatStateOf(23.5f) }
    var sensorHeight by remember { mutableFloatStateOf(15.7f) }

    val diameter = state.params.effectiveDiameterMm
    val focal = state.params.effectiveFocalMm
    val effectiveFocal = focal * barlow
    val effectiveRatio = Formulas.effectiveFocalRatio(focal, barlow.toDouble(), diameter)
    val sampling = Formulas.samplingArcsecPerPixel(pixelSize.toDouble(), effectiveFocal)

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item {
            SectionCard(title = "Capteur") {
                SliderRow("Taille de pixel", pixelSize, 1f..10f, "µm", 2) { pixelSize = it }
                SliderRow("Barlow / reducteur", barlow, 0.5f..3f, "x", 2) { barlow = it }
                SliderRow("Largeur du capteur", sensorWidth, 4f..40f, "mm", 1) { sensorWidth = it }
                SliderRow("Hauteur du capteur", sensorHeight, 3f..30f, "mm", 1) { sensorHeight = it }
            }
        }
        item {
            SectionCard(title = "Resultats") {
                LabeledValue("F/D effectif", "f/%.1f".format(effectiveRatio))
                LabeledValue("Focale effective", "%.0f mm".format(effectiveFocal))
                LabeledValue(
                    "F/D recommande (Shannon)",
                    "min f/%.1f — ideal f/%.1f — max f/%.1f".format(
                        Formulas.minRecommendedFocalRatio(pixelSize.toDouble()),
                        Formulas.idealFocalRatio(pixelSize.toDouble()),
                        Formulas.maxRecommendedFocalRatio(pixelSize.toDouble()),
                    ),
                )
                LabeledValue("Echantillonnage", "%.2f \"/px".format(sampling))
                Text(
                    Formulas.samplingDiagnosis(sampling),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.padding(top = 4.dp),
                )
                LabeledValue(
                    "Champ du capteur",
                    "%.2f° x %.2f°".format(
                        Formulas.sensorFieldDeg(sensorWidth.toDouble(), effectiveFocal),
                        Formulas.sensorFieldDeg(sensorHeight.toDouble(), effectiveFocal),
                    ),
                )
            }
        }
    }
}

/** Smart telescopes : les modeles integres et leur magnitude limite. */
@Composable
private fun SmartTelescopeTab(state: AppUiState) {
    var exposure by remember { mutableFloatStateOf(60f) }
    val bortle = state.params.site.bortle

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item {
            SectionCard(
                title = "Duree de pose cumulee",
                subtitle = "Magnitude limite calculee pour un ciel Bortle $bortle",
            ) {
                SliderRow("Pose", exposure, 5f..240f, "min", 0) { exposure = it }
            }
        }
        items(SmartTelescope.CATALOG.size) { index ->
            val scope = SmartTelescope.CATALOG[index]
            SectionCard(
                title = scope.name,
                subtitle = "%.0f mm f/%.1f".format(scope.diameterMm, scope.focalRatio),
            ) {
                LabeledValue("Ouverture", "%.0f mm".format(scope.diameterMm))
                LabeledValue("Focale", "%.0f mm".format(scope.focalMm))
                LabeledValue(
                    "Capteur",
                    "%.1f x %.1f mm, pixels %.1f µm".format(
                        scope.sensorWidthMm, scope.sensorHeightMm, scope.pixelSizeUm
                    ),
                )
                LabeledValue(
                    "Champ",
                    "%.0f' x %.0f'".format(scope.fieldWidthArcmin, scope.fieldHeightArcmin),
                )
                LabeledValue(
                    "Magnitude limite",
                    "%.1f".format(scope.limitingMagnitude(exposure.toDouble(), bortle)),
                )
                LabeledValue(
                    "Echantillonnage",
                    "%.2f \"/px".format(
                        Formulas.samplingArcsecPerPixel(scope.pixelSizeUm, scope.focalMm)
                    ),
                )
            }
        }
    }
}

@Composable
private fun SliderRow(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    unit: String,
    decimals: Int,
    onChange: (Float) -> Unit,
) {
    Column(Modifier.padding(vertical = 4.dp)) {
        Text(
            "$label : %.${decimals}f $unit".format(value),
            style = MaterialTheme.typography.bodySmall,
        )
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}

@Composable
private fun androidx.compose.foundation.layout.RowScope.HeaderCell(text: String, weight: Float) {
    Text(
        text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.weight(weight),
    )
}

@Composable
private fun androidx.compose.foundation.layout.RowScope.BodyCell(text: String, weight: Float) {
    Text(text, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(weight))
}

package com.cielscore.app.ui.screens

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.cielscore.app.catalog.Catalog
import com.cielscore.app.model.InstrumentType
import com.cielscore.app.model.SmartTelescope
import com.cielscore.app.scoring.Formulas
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.Chip
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.ObjectSheet
import com.cielscore.app.ui.components.ScoreBadge
import com.cielscore.app.ui.components.SectionCard
import com.google.android.gms.location.LocationServices

/**
 * Onglet Dashboard — section 2.2 : coeur de l'application, produit la liste
 * classee des cibles.
 */
@Composable
fun DashboardScreen(viewModel: AppViewModel, state: AppUiState) {
    var showSheet by remember { mutableStateOf(false) }

    val stars by produceStars(viewModel)

    if (showSheet && state.selected != null) {
        Column(Modifier.fillMaxWidth()) {
            TextButton(onClick = { showSheet = false }) { Text("← Retour au Top des cibles") }
            ObjectSheet(
                scored = state.selected,
                state = state,
                viewModel = viewModel,
                stars = stars.first,
                starsById = stars.second,
                constellations = stars.third,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
            )
        }
        return
    }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        item { SiteCard(viewModel, state) }
        item { InstrumentCard(viewModel, state) }
        item { CatalogCard(viewModel, state) }
        item { ComputeCard(viewModel, state) }
        if (state.topTargets.isNotEmpty()) {
            item { ConditionsBanner(state) }
            item {
                Text(
                    "Top des cibles",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
            items(state.topTargets, key = { it.target.id + it.target.designation }) { scored ->
                TargetRow(scored) {
                    // RG-DASH-01 : ouverture de la fiche sans recalculer la session.
                    viewModel.selectTarget(scored)
                    showSheet = true
                }
            }
        }
    }
}

/** Recherche de commune, position GPS et indice de Bortle. */
@SuppressLint("MissingPermission")
@Composable
private fun SiteCard(viewModel: AppViewModel, state: AppUiState) {
    val context = LocalContext.current
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) requestLocation(context, viewModel)
    }

    SectionCard(title = "Lieu d'observation") {
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = state.communeQuery,
                onValueChange = viewModel::setCommuneQuery,
                label = { Text("Rechercher une commune") },
                singleLine = true,
                modifier = Modifier.weight(1f),
            )
            IconButton(onClick = {
                val granted = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.ACCESS_COARSE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED
                if (granted) requestLocation(context, viewModel)
                else permissionLauncher.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
            }) {
                Icon(Icons.Filled.MyLocation, contentDescription = "Utiliser ma position")
            }
        }

        state.communeResults.forEach { site ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .clickable { viewModel.selectCommune(site) }
                    .padding(vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    "${site.name} (${site.department})",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text("Bortle ~${site.bortle}", style = MaterialTheme.typography.labelSmall)
            }
        }

        Column(Modifier.padding(top = 8.dp)) {
            LabeledValue("Commune", state.params.site.name)
            LabeledValue(
                "Coordonnees",
                "%.4f°, %.4f°".format(state.params.site.latitude, state.params.site.longitude),
            )
            Text(
                "Indice de Bortle : ${state.params.site.bortle} — " +
                    Formulas.bortleDescription(state.params.site.bortle),
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(top = 6.dp),
            )
            Slider(
                value = state.params.site.bortle.toFloat(),
                onValueChange = { viewModel.setBortle(it.toInt()) },
                valueRange = 1f..9f,
                steps = 7,
            )
            Text(
                "SB limite %.1f mag/arcsec² — %s".format(
                    Formulas.surfaceBrightnessLimit(state.params.site.bortle),
                    Formulas.bortlePlaceExample(state.params.site.bortle),
                ),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (state.params.site.bortleEstimated) {
                Text(
                    "Indice estime d'apres la population de la commune. Ajustez-le si vous " +
                        "connaissez la qualite reelle de votre ciel.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}

/** Instrument, diametre, focale, pupille — valeurs par defaut de la section 2.2. */
@Composable
private fun InstrumentCard(viewModel: AppViewModel, state: AppUiState) {
    SectionCard(title = "Instrument") {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            InstrumentType.entries.forEach { type ->
                FilterChip(
                    selected = state.params.instrument == type && !state.params.isSmartMode ||
                        (type == InstrumentType.SMART_TELESCOPE && state.params.isSmartMode),
                    onClick = {
                        if (type == InstrumentType.SMART_TELESCOPE) {
                            viewModel.updateParams {
                                it.copy(
                                    instrument = type,
                                    smartTelescope = it.smartTelescope
                                        ?: SmartTelescope.CATALOG[1],
                                )
                            }
                        } else {
                            viewModel.updateParams {
                                it.copy(instrument = type, smartTelescope = null)
                            }
                        }
                    },
                    label = { Text(type.label, style = MaterialTheme.typography.labelSmall) },
                )
            }
        }

        if (state.params.isSmartMode) {
            Column(Modifier.padding(top = 10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    SmartTelescope.CATALOG.take(4).forEach { scope ->
                        FilterChip(
                            selected = state.params.smartTelescope?.name == scope.name,
                            onClick = {
                                viewModel.updateParams { it.copy(smartTelescope = scope) }
                            },
                            label = {
                                Text(scope.model, style = MaterialTheme.typography.labelSmall)
                            },
                        )
                    }
                }
                Row(
                    Modifier.padding(top = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    SmartTelescope.CATALOG.drop(4).forEach { scope ->
                        FilterChip(
                            selected = state.params.smartTelescope?.name == scope.name,
                            onClick = {
                                viewModel.updateParams { it.copy(smartTelescope = scope) }
                            },
                            label = {
                                Text(scope.model, style = MaterialTheme.typography.labelSmall)
                            },
                        )
                    }
                }
                Text(
                    "Duree de pose cumulee : %.0f min".format(state.params.smartExposureMinutes),
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 8.dp),
                )
                Slider(
                    value = state.params.smartExposureMinutes.toFloat(),
                    onValueChange = {
                        viewModel.updateParams { p -> p.copy(smartExposureMinutes = it.toDouble()) }
                    },
                    valueRange = 5f..240f,
                )
                Text(
                    "RG-P-05 : les planetes sont desactivees en mode smart telescope.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
        } else if (state.params.instrument != InstrumentType.NAKED_EYE) {
            ParamSlider(
                label = "Diametre",
                value = state.params.diameterMm,
                range = 50f..600f,
                unit = "mm",
            ) { viewModel.updateParams { p -> p.copy(diameterMm = it) } }

            ParamSlider(
                label = "Focale",
                value = state.params.focalMm,
                range = 300f..3000f,
                unit = "mm",
            ) { viewModel.updateParams { p -> p.copy(focalMm = it) } }
        }

        ParamSlider(
            label = "Pupille de l'oeil",
            value = state.params.eyePupilMm,
            range = 4f..8f,
            unit = "mm",
            decimals = 1,
        ) { viewModel.updateParams { p -> p.copy(eyePupilMm = it) } }

        Column(Modifier.padding(top = 8.dp)) {
            LabeledValue("Magnitude limite", "%.1f".format(state.magnitudeLimit))
            LabeledValue("Rapport F/D", "f/%.1f".format(state.params.focalRatio))
        }
    }
}

@Composable
private fun ParamSlider(
    label: String,
    value: Double,
    range: ClosedFloatingPointRange<Float>,
    unit: String,
    decimals: Int = 0,
    onChange: (Double) -> Unit,
) {
    Column(Modifier.padding(top = 8.dp)) {
        Text(
            "$label : %.${decimals}f $unit".format(value),
            style = MaterialTheme.typography.bodySmall,
        )
        Slider(
            value = value.toFloat().coerceIn(range.start, range.endInclusive),
            onValueChange = { onChange(it.toDouble()) },
            valueRange = range,
        )
    }
}

@Composable
private fun CatalogCard(viewModel: AppViewModel, state: AppUiState) {
    SectionCard(title = "Catalogues") {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Catalog.entries.forEach { catalog ->
                val enabled = catalog != Catalog.PLANETS || !state.params.isSmartMode
                FilterChip(
                    selected = catalog in state.params.catalogs && enabled,
                    enabled = enabled,
                    onClick = {
                        viewModel.updateParams { p ->
                            val updated = if (catalog in p.catalogs) p.catalogs - catalog
                            else p.catalogs + catalog
                            p.copy(catalogs = updated)
                        }
                    },
                    label = { Text(catalog.label, style = MaterialTheme.typography.labelSmall) },
                )
            }
        }
    }
}

@Composable
private fun ComputeCard(viewModel: AppViewModel, state: AppUiState) {
    SectionCard(title = "Session", subtitle = formatDateTime(state.params.epochMillis)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TextButton(onClick = {
                viewModel.updateParams { it.copy(epochMillis = System.currentTimeMillis()) }
            }) { Text("Maintenant") }
            TextButton(onClick = {
                viewModel.updateParams { it.copy(epochMillis = it.epochMillis - 3_600_000L) }
            }) { Text("−1 h") }
            TextButton(onClick = {
                viewModel.updateParams { it.copy(epochMillis = it.epochMillis + 3_600_000L) }
            }) { Text("+1 h") }
        }
        Button(
            onClick = viewModel::computeSession,
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            enabled = !state.calculating,
        ) {
            if (state.calculating) {
                CircularProgressIndicator(
                    Modifier.padding(end = 8.dp),
                    strokeWidth = 2.dp,
                )
            }
            Text(if (state.calculating) "Calcul en cours…" else "Calculer le Top des cibles")
        }
    }
}

/** Bandeau conditions — section 2.2. */
@Composable
private fun ConditionsBanner(state: AppUiState) {
    SectionCard(title = "Conditions") {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Chip(state.params.site.name)
            Chip("Bortle ${state.params.site.bortle}")
            Chip("mag %.1f".format(state.magnitudeLimit))
        }
        Column(Modifier.padding(top = 8.dp)) {
            LabeledValue(
                "Instrument",
                state.params.smartTelescope?.name
                    ?: "${state.params.instrument.label} ${state.params.effectiveDiameterMm.toInt()} mm",
            )
            state.moon?.let {
                LabeledValue("Lune", "${it.phaseName} — %.0f %%".format(it.phasePercent))
            }
            LabeledValue(
                "Nuages",
                "%.0f %%%s".format(
                    state.conditions.cloudCoverPercent,
                    if (state.conditions.ok) "" else " (repli, Open-Meteo indisponible)",
                ),
            )
            LabeledValue(
                "Seeing",
                "${state.conditions.seeingIndex} — ${state.conditions.seeingLabel}",
            )
            LabeledValue("Hauteur du Soleil", "%.1f°".format(state.sunAltitudeDeg))
        }
    }
}

@Composable
private fun TargetRow(
    scored: com.cielscore.app.scoring.ScoringEngine.Scored,
    onClick: () -> Unit,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        ScoreBadge(scored.score)
        Column(Modifier.weight(1f).padding(start = 12.dp)) {
            Text(scored.target.displayName, style = MaterialTheme.typography.bodyMedium)
            Text(
                "%s — alt %.0f° — mag %s".format(
                    scored.target.type.label,
                    scored.altitudeDeg,
                    scored.target.magnitude?.let { "%.1f".format(it) } ?: "—",
                ),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@SuppressLint("MissingPermission")
private fun requestLocation(context: android.content.Context, viewModel: AppViewModel) {
    val client = LocationServices.getFusedLocationProviderClient(context)
    client.lastLocation.addOnSuccessListener { location ->
        if (location != null) viewModel.useDeviceLocation(location.latitude, location.longitude)
    }
}

/** Chargement paresseux du fond d'etoiles pour la carte du ciel. */
@Composable
internal fun produceStars(
    viewModel: AppViewModel,
): androidx.compose.runtime.State<Triple<
    List<com.cielscore.app.catalog.Star>,
    Map<String, com.cielscore.app.catalog.Star>,
    List<com.cielscore.app.catalog.Constellation>,
    >> {
    val empty = Triple(
        emptyList<com.cielscore.app.catalog.Star>(),
        emptyMap<String, com.cielscore.app.catalog.Star>(),
        emptyList<com.cielscore.app.catalog.Constellation>(),
    )
    val state = remember { androidx.compose.runtime.mutableStateOf(empty) }
    LaunchedEffect(Unit) {
        state.value = Triple(
            viewModel.starCatalog.stars(),
            viewModel.starCatalog.starsById(),
            viewModel.starCatalog.constellations(),
        )
    }
    return state
}

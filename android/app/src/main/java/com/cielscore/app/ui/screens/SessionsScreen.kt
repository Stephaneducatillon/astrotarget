package com.cielscore.app.ui.screens

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.cielscore.app.data.db.ObservationEntity
import com.cielscore.app.export.PdfExporter
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.SectionCard
import kotlinx.coroutines.launch

/**
 * Onglet Sessions — section 2.4 : plan de soiree IA, export PDF, enregistrement
 * d'observation et carnet.
 */
@Composable
fun SessionsScreen(viewModel: AppViewModel, state: AppUiState) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val user = state.user ?: return

    val observations by remember(user.username) {
        viewModel.observations.recent(user.username)
    }.collectAsState(initial = emptyList())

    var draft by remember { mutableStateOf<ObservationEntity?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        // --- Plan de soiree IA (RG-SESS-01) ---
        item {
            SectionCard(
                title = "Plan de soiree",
                subtitle = "Genere par Mistral a partir du dernier calcul Dashboard",
            ) {
                if (state.topTargets.isEmpty()) {
                    Text(
                        "Lancez d'abord une session depuis le Dashboard : le plan de soiree " +
                            "s'appuie sur le dernier calcul.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
                val plan = state.eveningPlan
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = viewModel::generateEveningPlan,
                        enabled = !state.planLoading && state.topTargets.isNotEmpty(),
                    ) {
                        Text(if (state.planLoading) "Generation…" else "Generer le plan")
                    }
                    if (plan != null) {
                        TextButton(onClick = {
                            scope.launch {
                                val uri = PdfExporter.exportEveningPlan(
                                    context,
                                    plan,
                                    state.topTargets,
                                    state.params,
                                    state.conditions,
                                )
                                val share = Intent(Intent.ACTION_SEND).apply {
                                    type = "application/pdf"
                                    putExtra(Intent.EXTRA_STREAM, uri)
                                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                }
                                context.startActivity(
                                    Intent.createChooser(share, "Exporter le plan en PDF")
                                )
                            }
                        }) { Text("Export PDF") }
                    }
                }
                if (state.planLoading) CircularProgressIndicator(Modifier.padding(top = 8.dp))
                plan?.let {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                }
            }
        }

        // --- Enregistrer une observation ---
        item {
            SectionCard(title = "Enregistrer une observation") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = { draft = viewModel.observationDraft() }) {
                        Text("Reprendre le dernier calcul")
                    }
                    if (draft != null) {
                        TextButton(onClick = { draft = null }) { Text("Annuler") }
                    }
                }
                if (draft == null) {
                    Text(
                        "Pre-remplit le formulaire depuis la session Dashboard : objet, date, " +
                            "site, instrument, Bortle, lune, nuages, seeing et score.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    ObservationForm(
                        draft = draft!!,
                        onChange = { draft = it },
                        onSave = {
                            viewModel.saveObservation(it)
                            draft = null
                        },
                    )
                }
            }
        }

        // --- Mon carnet ---
        item {
            Text(
                "Mon carnet — ${observations.size} observation(s)",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
        items(observations, key = { it.id }) { observation ->
            SectionCard(
                title = observation.objectName,
                subtitle = formatDateTime(observation.observationDate),
            ) {
                LabeledValue("Site", "${observation.site} — Bortle ${observation.bortle}")
                LabeledValue("Instrument", observation.instrument)
                LabeledValue(
                    "Conditions",
                    "Lune %.0f %% — nuages %.0f %% — seeing %d".format(
                        observation.moonPhasePercent,
                        observation.cloudCoverPercent,
                        observation.seeingIndex,
                    ),
                )
                LabeledValue("Score", "%.0f / 100".format(observation.score))
                if (observation.notes.isNotBlank()) {
                    Text(
                        observation.notes,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
                TextButton(onClick = { viewModel.deleteObservation(observation.id) }) {
                    Text("Supprimer")
                }
            }
        }
    }
}

@Composable
private fun ObservationForm(
    draft: ObservationEntity,
    onChange: (ObservationEntity) -> Unit,
    onSave: (ObservationEntity) -> Unit,
) {
    Column(Modifier.padding(top = 8.dp)) {
        OutlinedTextField(
            value = draft.objectName,
            onValueChange = { onChange(draft.copy(objectName = it)) },
            label = { Text("Objet") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = draft.site,
            onValueChange = { onChange(draft.copy(site = it)) },
            label = { Text("Site") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        OutlinedTextField(
            value = draft.instrument,
            onValueChange = { onChange(draft.copy(instrument = it)) },
            label = { Text("Instrument") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        Text(
            "Indice de Bortle : ${draft.bortle}",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(top = 8.dp),
        )
        Slider(
            value = draft.bortle.toFloat(),
            onValueChange = { onChange(draft.copy(bortle = it.toInt())) },
            valueRange = 1f..9f,
            steps = 7,
        )
        Text("Seeing : ${draft.seeingIndex}", style = MaterialTheme.typography.bodySmall)
        Slider(
            value = draft.seeingIndex.toFloat(),
            onValueChange = { onChange(draft.copy(seeingIndex = it.toInt())) },
            valueRange = 1f..5f,
            steps = 3,
        )
        OutlinedTextField(
            value = draft.notes,
            onValueChange = { onChange(draft.copy(notes = it)) },
            label = { Text("Notes") },
            minLines = 3,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        Button(
            onClick = { onSave(draft) },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
        ) { Text("Enregistrer dans le carnet") }
    }
}

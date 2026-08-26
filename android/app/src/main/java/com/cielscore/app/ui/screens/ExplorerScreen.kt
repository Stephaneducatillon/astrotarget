package com.cielscore.app.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cielscore.app.catalog.Catalog
import com.cielscore.app.catalog.ObjectType
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.ObjectSheet
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Explorer — section 2.3 : recherche libre dans les catalogues,
 * independamment des conditions du soir.
 *
 * Les resultats sont tries par magnitude ; la fiche detaillee reste accessible.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ExplorerScreen(viewModel: AppViewModel, state: AppUiState) {
    var showSheet by remember { mutableStateOf(false) }
    val stars by produceStars(viewModel)

    if (showSheet && state.selected != null) {
        Column(Modifier.fillMaxWidth()) {
            TextButton(onClick = { showSheet = false }) { Text("← Retour a la recherche") }
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
        item {
            SectionCard(title = "Recherche") {
                OutlinedTextField(
                    value = state.explorerQuery,
                    onValueChange = viewModel::setExplorerQuery,
                    label = { Text("Nom d'objet : M42, NGC 891, C22…") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )

                Text(
                    "Catalogues",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 10.dp),
                )
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(Catalog.MESSIER, Catalog.CALDWELL, Catalog.NGC_IC).forEach { catalog ->
                        FilterChip(
                            selected = catalog in state.explorerCatalogs,
                            onClick = { viewModel.toggleExplorerCatalog(catalog) },
                            label = {
                                Text(catalog.label, style = MaterialTheme.typography.labelSmall)
                            },
                        )
                    }
                }

                Text(
                    "Types",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 10.dp),
                )
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    ObjectType.entries.forEach { type ->
                        FilterChip(
                            selected = type in state.explorerTypes,
                            onClick = { viewModel.toggleExplorerType(type) },
                            label = {
                                Text(type.label, style = MaterialTheme.typography.labelSmall)
                            },
                        )
                    }
                }

                Button(
                    onClick = viewModel::searchExplorer,
                    modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
                    enabled = !state.explorerSearching,
                ) { Text(if (state.explorerSearching) "Recherche…" else "Rechercher") }
            }
        }

        if (state.explorerSearching) {
            item { CircularProgressIndicator(Modifier.padding(16.dp)) }
        }

        if (state.explorerResults.isNotEmpty()) {
            item {
                Text(
                    "${state.explorerResults.size} resultats, tries par magnitude",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            items(state.explorerResults, key = { it.catalog.name + it.id + it.designation }) { obj ->
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clickable {
                            viewModel.selectFromExplorer(obj)
                            showSheet = true
                        }
                        .padding(vertical = 6.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(obj.displayName, style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${obj.type.label} — ${obj.constellation} — ${obj.sizeLabel}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Text(
                        obj.magnitude?.let { "mag %.1f".format(it) } ?: "mag —",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

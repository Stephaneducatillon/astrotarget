package com.cielscore.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Insights
import androidx.compose.material.icons.filled.NightsStay
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.screens.AssistantScreen
import com.cielscore.app.ui.screens.DashboardScreen
import com.cielscore.app.ui.screens.EquipmentScreen
import com.cielscore.app.ui.screens.ExplorerScreen
import com.cielscore.app.ui.screens.InformationScreen
import com.cielscore.app.ui.screens.ProfileScreen
import com.cielscore.app.ui.screens.SessionsScreen
import com.cielscore.app.ui.screens.StatisticsScreen
import com.cielscore.app.ui.theme.CielScoreTheme

/**
 * Les huit onglets de l'application, dans l'ordre de la section 2.
 *
 * [title] reprend le nom exact du document, affiche en haut de l'ecran ;
 * [shortLabel] est la version courte, qui doit tenir sous une icone dans la
 * barre de navigation d'un telephone.
 *
 * L'acces « Connecte » de la section 2 est materialise par [requiresAccount].
 */
enum class AppTab(
    val title: String,
    val shortLabel: String,
    val icon: ImageVector,
    val requiresAccount: Boolean,
) {
    INFORMATION("Informations", "Infos", Icons.Filled.Info, false),
    DASHBOARD("Dashboard", "Cibles", Icons.Filled.Star, true),
    EXPLORER("Explorer", "Explorer", Icons.Filled.Explore, true),
    SESSIONS("Sessions", "Sessions", Icons.Filled.NightsStay, true),
    EQUIPMENT("Equipement", "Materiel", Icons.Filled.Visibility, true),
    STATISTICS("Statistiques", "Stats", Icons.Filled.Insights, true),
    ASSISTANT("Assistant IA", "Assistant", Icons.Filled.Chat, true),
    PROFILE("Profil", "Profil", Icons.Filled.Person, false),
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent { CielScoreApp() }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CielScoreApp() {
    val viewModel: AppViewModel = viewModel(factory = AppViewModel.Factory)
    val state by viewModel.state.collectAsState()

    CielScoreTheme(nightMode = state.nightMode) {
        var selectedTab by rememberSaveable { mutableIntStateOf(0) }
        val snackbar = remember { SnackbarHostState() }
        val tab = AppTab.entries[selectedTab]

        LaunchedEffect(state.message) {
            state.message?.let {
                snackbar.showSnackbar(it)
                viewModel.dismissMessage()
            }
        }

        Scaffold(
            snackbarHost = { SnackbarHost(snackbar) },
            // CenterAlignedTopAppBar et NavigationBar appliquent d'eux-memes les
            // marges de la barre d'etat et de la barre de navigation : le contenu
            // ne passe plus sous l'horloge ni sous la barre de gestes.
            topBar = {
                CenterAlignedTopAppBar(
                    title = { Text(tab.title, style = MaterialTheme.typography.titleMedium) },
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                        titleContentColor = MaterialTheme.colorScheme.onSurface,
                    ),
                )
            },
            // La navigation est en bas, a portee du pouce : l'application
            // s'utilise dehors, la nuit, souvent d'une seule main.
            bottomBar = {
                NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                    AppTab.entries.forEachIndexed { index, entry ->
                        NavigationBarItem(
                            selected = selectedTab == index,
                            onClick = { selectedTab = index },
                            icon = { Icon(entry.icon, contentDescription = entry.title) },
                            label = {
                                Text(
                                    entry.shortLabel,
                                    style = MaterialTheme.typography.labelSmall,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    textAlign = TextAlign.Center,
                                )
                            },
                            // Huit destinations : seul l'onglet actif porte son
                            // libelle, sans quoi rien ne tiendrait en largeur.
                            alwaysShowLabel = false,
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = MaterialTheme.colorScheme.onPrimaryContainer,
                                selectedTextColor = MaterialTheme.colorScheme.primary,
                                indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                                unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            ),
                        )
                    }
                }
            },
        ) { padding ->
            // imePadding : le clavier repousse le contenu au lieu de le masquer.
            Box(Modifier.fillMaxSize().padding(padding).imePadding()) {
                if (tab.requiresAccount && state.user == null) {
                    AccountRequired(tab.title) { selectedTab = AppTab.PROFILE.ordinal }
                } else {
                    when (tab) {
                        AppTab.INFORMATION -> InformationScreen(viewModel, state)
                        AppTab.DASHBOARD -> DashboardScreen(viewModel, state)
                        AppTab.EXPLORER -> ExplorerScreen(viewModel, state)
                        AppTab.SESSIONS -> SessionsScreen(viewModel, state)
                        AppTab.EQUIPMENT -> EquipmentScreen(state)
                        AppTab.STATISTICS -> StatisticsScreen(viewModel, state)
                        AppTab.ASSISTANT -> AssistantScreen(viewModel, state)
                        AppTab.PROFILE -> ProfileScreen(viewModel, state)
                    }
                }
            }
        }
    }
}

/** Ecran affiche pour un onglet « Connecte » lorsque aucun compte n'est ouvert. */
@Composable
private fun AccountRequired(tabTitle: String, onGoToProfile: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Filled.Person,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
        )
        Text(
            "L'onglet $tabTitle demande un compte",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 12.dp),
        )
        Text(
            "Creez un compte ou connectez-vous depuis l'onglet Profil. " +
                "Les onglets Informations et Profil restent accessibles sans connexion.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(top = 8.dp),
        )
        TextButton(onClick = onGoToProfile) { Text("Aller au Profil") }
    }
}

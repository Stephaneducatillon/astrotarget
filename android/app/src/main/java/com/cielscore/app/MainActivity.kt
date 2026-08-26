package com.cielscore.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
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
 * L'acces « Connecte » de la section 2 est materialise par [requiresAccount].
 */
enum class AppTab(
    val label: String,
    val icon: ImageVector,
    val requiresAccount: Boolean,
) {
    INFORMATION("Informations", Icons.Filled.Info, false),
    DASHBOARD("Dashboard", Icons.Filled.Star, true),
    EXPLORER("Explorer", Icons.Filled.Explore, true),
    SESSIONS("Sessions", Icons.Filled.NightsStay, true),
    EQUIPMENT("Equipement", Icons.Filled.Visibility, true),
    STATISTICS("Statistiques", Icons.Filled.Insights, true),
    ASSISTANT("Assistant IA", Icons.Filled.Chat, true),
    PROFILE("Profil", Icons.Filled.Person, false),
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent { CielScoreApp() }
    }
}

@Composable
private fun CielScoreApp() {
    val viewModel: AppViewModel = viewModel(factory = AppViewModel.Factory)
    val state by viewModel.state.collectAsState()

    CielScoreTheme(nightMode = state.nightMode) {
        var selectedTab by rememberSaveable { mutableIntStateOf(0) }
        val snackbar = remember { SnackbarHostState() }

        LaunchedEffect(state.message) {
            state.message?.let {
                snackbar.showSnackbar(it)
                viewModel.dismissMessage()
            }
        }

        Scaffold(
            snackbarHost = { SnackbarHost(snackbar) },
            topBar = {
                ScrollableTabRow(
                    selectedTabIndex = selectedTab,
                    edgePadding = 8.dp,
                    containerColor = MaterialTheme.colorScheme.surface,
                ) {
                    AppTab.entries.forEachIndexed { index, tab ->
                        Tab(
                            selected = selectedTab == index,
                            onClick = { selectedTab = index },
                            text = { Text(tab.label, style = MaterialTheme.typography.labelSmall) },
                            icon = { Icon(tab.icon, contentDescription = tab.label) },
                        )
                    }
                }
            },
        ) { padding ->
            val tab = AppTab.entries[selectedTab]
            Box(Modifier.fillMaxSize().padding(padding)) {
                if (tab.requiresAccount && state.user == null) {
                    AccountRequired(tab.label) { selectedTab = AppTab.PROFILE.ordinal }
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
private fun AccountRequired(tabLabel: String, onGoToProfile: () -> Unit) {
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
            "L'onglet $tabLabel demande un compte",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 12.dp),
        )
        Text(
            "Creez un compte ou connectez-vous depuis l'onglet Profil. " +
                "Les onglets Informations et Profil restent accessibles sans connexion.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp),
        )
        androidx.compose.material3.TextButton(onClick = onGoToProfile) {
            Text("Aller au Profil")
        }
    }
}

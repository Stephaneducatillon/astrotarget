package com.cielscore.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import com.cielscore.app.data.auth.AuthRepository
import com.cielscore.app.model.ApiKey
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.LabeledValue
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Profil — section 2.8 : connexion, inscription et recuperation.
 *
 *   Connexion        identifiant en minuscules, mot de passe de 8 caracteres minimum
 *   Inscription      prenom, nom, identifiant de 3 caracteres minimum, alphanumerique
 *   Code de recup.   genere a la creation, affiche une seule fois
 *   Reinitialisation identifiant + code de recuperation + nouveau mot de passe
 */
@Composable
fun ProfileScreen(viewModel: AppViewModel, state: AppUiState) {
    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
    ) {
        if (state.user != null) {
            item { AccountCard(viewModel, state) }
        } else {
            item { AuthCard(viewModel) }
        }
        item { KeysCard(viewModel, state) }
        item { DisplayCard(viewModel, state) }
        item { AboutCard() }
    }
}

@Composable
private fun AccountCard(viewModel: AppViewModel, state: AppUiState) {
    val user = state.user!!
    SectionCard(title = "Compte", subtitle = user.username) {
        LabeledValue("Prenom", user.firstName)
        LabeledValue("Nom", user.lastName)
        LabeledValue("Compte cree le", formatDateTime(user.createdAt))
        TextButton(onClick = viewModel::logout) { Text("Se deconnecter") }
    }
}

@Composable
private fun AuthCard(viewModel: AppViewModel) {
    var tab by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    var recoveryCode by remember { mutableStateOf<String?>(null) }

    SectionCard(title = "Acces") {
        TabRow(selectedTabIndex = tab, containerColor = MaterialTheme.colorScheme.surface) {
            listOf("Connexion", "Inscription", "Recuperation").forEachIndexed { index, label ->
                Tab(
                    selected = tab == index,
                    onClick = { tab = index; error = null; recoveryCode = null },
                    text = { Text(label, style = MaterialTheme.typography.labelSmall) },
                )
            }
        }

        Column(Modifier.padding(top = 12.dp)) {
            when (tab) {
                0 -> LoginForm { username, password ->
                    viewModel.login(username, password) { error = it }
                }
                1 -> RegisterForm { username, password, firstName, lastName ->
                    viewModel.register(username, password, firstName, lastName) { result ->
                        result
                            .onSuccess { recoveryCode = it.recoveryCode; error = null }
                            .onFailure { error = it.message }
                    }
                }
                else -> ResetForm { username, code, password ->
                    viewModel.resetPassword(username, code, password) {
                        error = it ?: "Mot de passe reinitialise. Vous pouvez vous connecter."
                    }
                }
            }

            error?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            // Le code de recuperation n'est affiche qu'une seule fois (section 2.8).
            recoveryCode?.let {
                Column(Modifier.padding(top = 12.dp)) {
                    Text("Code de recuperation", style = MaterialTheme.typography.titleMedium)
                    Text(
                        it,
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                    Text(
                        "Notez-le maintenant : il n'est affiche qu'une seule fois et n'est " +
                            "jamais conserve en clair. Il permet de reinitialiser votre mot de passe.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun LoginForm(onSubmit: (String, String) -> Unit) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    Column {
        OutlinedTextField(
            value = username,
            onValueChange = { username = it },
            label = { Text("Identifiant") },
            supportingText = { Text("Converti automatiquement en minuscules") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Mot de passe") },
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        Button(
            onClick = { onSubmit(username, password) },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
        ) { Text("Se connecter") }
    }
}

@Composable
private fun RegisterForm(onSubmit: (String, String, String, String) -> Unit) {
    var firstName by remember { mutableStateOf("") }
    var lastName by remember { mutableStateOf("") }
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    Column {
        OutlinedTextField(
            value = firstName,
            onValueChange = { firstName = it },
            label = { Text("Prenom") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = lastName,
            onValueChange = { lastName = it },
            label = { Text("Nom") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        OutlinedTextField(
            value = username,
            onValueChange = { username = it },
            label = { Text("Identifiant") },
            supportingText = {
                Text("${AuthRepository.MIN_USERNAME_LENGTH} caracteres minimum, alphanumerique")
            },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Mot de passe") },
            supportingText = {
                Text("${AuthRepository.MIN_PASSWORD_LENGTH} caracteres minimum")
            },
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        Button(
            onClick = { onSubmit(username, password, firstName, lastName) },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
        ) { Text("Creer le compte") }
    }
}

@Composable
private fun ResetForm(onSubmit: (String, String, String) -> Unit) {
    var username by remember { mutableStateOf("") }
    var code by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    Column {
        OutlinedTextField(
            value = username,
            onValueChange = { username = it },
            label = { Text("Identifiant") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = code,
            onValueChange = { code = it },
            label = { Text("Code de recuperation") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Nouveau mot de passe") },
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
        )
        Button(
            onClick = { onSubmit(username, code, password) },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
        ) { Text("Reinitialiser le mot de passe") }
    }
}

/** Cles d'API : NASA APOD et Mistral (section 8.2). */
@Composable
private fun KeysCard(viewModel: AppViewModel, state: AppUiState) {
    SectionCard(
        title = "Cles d'API",
        subtitle = "Conservees sur l'appareil, jamais transmises ailleurs",
    ) {
        // Les cles sont enregistrees a la frappe : aucun bouton a presser, donc
        // aucune saisie perdue en changeant d'onglet.
        ApiKeyField(
            label = "Cle NASA APOD",
            hint = "Image du jour de l'onglet Informations",
            value = state.nasaApiKey.orEmpty(),
            onValueChange = viewModel::setNasaKey,
        )
        ApiKeyField(
            label = "Cle Mistral",
            hint = "Guide objet, plan de soiree et assistant",
            value = state.mistralApiKey.orEmpty(),
            onValueChange = viewModel::setMistralKey,
            modifier = Modifier.padding(top = 10.dp),
        )

        Text(
            "Sans cle, le reste de l'application fonctionne normalement : seules les " +
                "fonctions concernees affichent un message explicite.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 10.dp),
        )
    }
}

/** Champ de cle d'API : saisie masquable, enregistrement immediat, etat visible. */
@Composable
private fun ApiKeyField(
    label: String,
    hint: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var revealed by remember { mutableStateOf(false) }

    Column(modifier) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            label = { Text(label) },
            supportingText = { Text(hint) },
            singleLine = true,
            visualTransformation = if (revealed) VisualTransformation.None
            else PasswordVisualTransformation(),
            trailingIcon = {
                IconButton(onClick = { revealed = !revealed }) {
                    Icon(
                        if (revealed) Icons.Filled.VisibilityOff else Icons.Filled.Visibility,
                        contentDescription = if (revealed) "Masquer la cle" else "Afficher la cle",
                    )
                }
            },
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            ApiKey.statusLabel(value),
            style = MaterialTheme.typography.labelSmall,
            color = if (value.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant
            else MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(start = 16.dp),
        )
    }
}

@Composable
private fun DisplayCard(viewModel: AppViewModel, state: AppUiState) {
    SectionCard(title = "Affichage") {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Mode nuit rouge", style = MaterialTheme.typography.bodyMedium)
                Text(
                    "Preserve la vision nocturne sur le terrain.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = state.nightMode, onCheckedChange = viewModel::setNightMode)
        }
    }
}

@Composable
private fun AboutCard() {
    SectionCard(title = "A propos", subtitle = "CielScore UX v0.6.4") {
        Text(
            "Planificateur d'observation astronomique. Calculs embarques : ephemerides, " +
                "crepuscules, brillance de surface et moteur de score a huit criteres.",
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            "Donnees : OpenNGC (catalogues), Open-Meteo (meteo), GFZ Potsdam et NOAA SWPC " +
                "(indice Kp), NASA (image du jour), The Space Devs (lancements), " +
                "CDS Strasbourg (Aladin Lite et hips2fits) et Mistral (IA). Les 34 869 " +
                "communes francaises et leur indice de Bortle sont embarques.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
}

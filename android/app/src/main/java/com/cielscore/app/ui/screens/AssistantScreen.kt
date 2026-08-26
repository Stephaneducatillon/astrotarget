package com.cielscore.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.cielscore.app.data.net.MistralApi
import com.cielscore.app.ui.AppUiState
import com.cielscore.app.ui.AppViewModel
import com.cielscore.app.ui.components.SectionCard

/**
 * Onglet Assistant IA — section 2.7 : chat libre avec injection du contexte
 * reel de la session en cours (lieu, instrument, ciel, cibles).
 */
@Composable
fun AssistantScreen(viewModel: AppViewModel, state: AppUiState) {
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    LaunchedEffect(state.chat.size) {
        if (state.chat.isNotEmpty()) listState.animateScrollToItem(state.chat.size - 1)
    }

    Column(Modifier.fillMaxSize()) {
        // Contexte transmis a l'IA, affiche pour transparence.
        SectionCard(
            title = "Contexte transmis",
            subtitle = "Ce que l'assistant sait de votre session",
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Text(
                state.aiContext().render(),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (state.chat.isNotEmpty()) {
                TextButton(onClick = viewModel::clearChat) { Text("Effacer la conversation") }
            }
        }

        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (state.chat.isEmpty()) {
                item {
                    Text(
                        "Posez votre question : « par ou commencer ce soir ? », " +
                            "« quel oculaire pour M13 ? », « pourquoi M33 n'apparait pas ? »",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            items(state.chat.size) { index ->
                ChatBubble(state.chat[index])
            }
            if (state.chatLoading) {
                item { CircularProgressIndicator(Modifier.padding(8.dp)) }
            }
        }

        Row(
            Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                placeholder = { Text("Votre question") },
                modifier = Modifier.weight(1f),
                maxLines = 4,
            )
            IconButton(
                onClick = {
                    if (input.isNotBlank()) {
                        viewModel.sendChatMessage(input.trim())
                        input = ""
                    }
                },
                enabled = !state.chatLoading,
            ) { Icon(Icons.Filled.Send, contentDescription = "Envoyer") }
        }
    }
}

@Composable
private fun ChatBubble(message: MistralApi.Message) {
    val isUser = message.role == "user"
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Box(
            Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(
                    if (isUser) MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.surfaceVariant
                )
                .padding(10.dp)
        ) {
            Text(message.content, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

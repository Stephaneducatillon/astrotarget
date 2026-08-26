package com.cielscore.app.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TimePicker
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cielscore.app.model.SessionInstant
import java.util.Date
import java.util.Locale

/**
 * Choix de la date et de l'heure de session (section 2.2 : « Date et heure
 * locale de la session », valeur par defaut « Maintenant »).
 *
 * La date se choisit dans un calendrier, l'heure dans une horloge : une seule
 * manipulation suffit pour viser, par exemple, le 28 a 4 h du matin.
 *
 * L'instant est manipule en millisecondes UTC, mais toujours presente et saisi
 * dans le fuseau de l'appareil.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DateTimeSelector(
    epochMillis: Long,
    onChange: (Long) -> Unit,
    modifier: Modifier = Modifier,
) {
    var showDatePicker by remember { mutableStateOf(false) }
    var showTimePicker by remember { mutableStateOf(false) }

    Column(modifier.fillMaxWidth()) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { showDatePicker = true },
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    Icons.Filled.CalendarMonth,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 6.dp),
                )
                Text(formatDate(epochMillis), maxLines = 1)
            }
            OutlinedButton(onClick = { showTimePicker = true }) {
                Icon(
                    Icons.Filled.Schedule,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 6.dp),
                )
                Text(formatTimeOfDay(epochMillis), maxLines = 1)
            }
        }

        Row(
            Modifier.padding(top = 2.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            TextButton(onClick = { onChange(System.currentTimeMillis()) }) {
                Text("Maintenant")
            }
            TextButton(onClick = { onChange(epochMillis - 3_600_000L) }) { Text("−1 h") }
            TextButton(onClick = { onChange(epochMillis + 3_600_000L) }) { Text("+1 h") }
        }
    }

    // ------------------------------------------------------------ Calendrier

    if (showDatePicker) {
        // Le calendrier raisonne en minuit UTC : on lui passe la date locale
        // convertie, et on refait le chemin inverse a la validation.
        val state = rememberDatePickerState(
            initialSelectedDateMillis = SessionInstant.localDateToUtcMidnight(epochMillis),
        )
        DatePickerDialog(
            onDismissRequest = { showDatePicker = false },
            confirmButton = {
                TextButton(
                    onClick = {
                        state.selectedDateMillis?.let {
                            onChange(SessionInstant.applyUtcMidnightDate(epochMillis, it))
                        }
                        showDatePicker = false
                    },
                ) { Text("Valider") }
            },
            dismissButton = {
                TextButton(onClick = { showDatePicker = false }) { Text("Annuler") }
            },
        ) {
            DatePicker(
                state = state,
                title = {
                    Text(
                        "Date de la session",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(start = 24.dp, top = 16.dp),
                    )
                },
            )
        }
    }

    // --------------------------------------------------------------- Horloge

    if (showTimePicker) {
        val state = rememberTimePickerState(
            initialHour = SessionInstant.localHour(epochMillis),
            initialMinute = SessionInstant.localMinute(epochMillis),
            is24Hour = true,
        )
        AlertDialog(
            onDismissRequest = { showTimePicker = false },
            title = { Text("Heure de la session") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    TimePicker(state = state)
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onChange(SessionInstant.applyTime(epochMillis, state.hour, state.minute))
                        showTimePicker = false
                    },
                ) { Text("Valider") }
            },
            dismissButton = {
                TextButton(onClick = { showTimePicker = false }) { Text("Annuler") }
            },
        )
    }
}

private fun formatDate(epochMillis: Long): String =
    java.text.SimpleDateFormat("EEE d MMM yyyy", Locale.FRANCE)
        .format(Date(epochMillis))
        .replaceFirstChar { it.uppercase() }

private fun formatTimeOfDay(epochMillis: Long): String =
    java.text.SimpleDateFormat("HH'h'mm", Locale.FRANCE).format(Date(epochMillis))

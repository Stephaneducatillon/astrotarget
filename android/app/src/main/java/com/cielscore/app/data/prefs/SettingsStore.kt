package com.cielscore.app.data.prefs

import android.content.Context
import androidx.datastore.preferences.core.MutablePreferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.cielscore.app.model.InstrumentType
import com.cielscore.app.model.ObservingSite
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "cielscore_settings")

/**
 * Preferences locales : session en cours, lieu, instrument et cles d'API.
 *
 * Les cles NASA APOD et Mistral (section 8.2) sont saisies par l'utilisateur
 * depuis le Profil et restent sur l'appareil.
 */
class SettingsStore(private val context: Context) {

    private object Keys {
        val CURRENT_USER = stringPreferencesKey("current_user")
        val NASA_KEY = stringPreferencesKey("nasa_api_key")
        val MISTRAL_KEY = stringPreferencesKey("mistral_api_key")

        val SITE_NAME = stringPreferencesKey("site_name")
        val SITE_DEPARTMENT = stringPreferencesKey("site_department")
        val SITE_LAT = doublePreferencesKey("site_lat")
        val SITE_LON = doublePreferencesKey("site_lon")
        val SITE_BORTLE = intPreferencesKey("site_bortle")
        val SITE_BORTLE_ESTIMATED = booleanPreferencesKey("site_bortle_estimated")

        val INSTRUMENT = stringPreferencesKey("instrument")
        val DIAMETER = doublePreferencesKey("diameter_mm")
        val FOCAL = doublePreferencesKey("focal_mm")
        val PUPIL = doublePreferencesKey("eye_pupil_mm")
        val CATALOGS = stringPreferencesKey("catalogs")
        val SMART_MODEL = stringPreferencesKey("smart_model")
        val SMART_EXPOSURE = doublePreferencesKey("smart_exposure_minutes")
        val NIGHT_MODE = booleanPreferencesKey("night_mode")
    }

    val currentUser: Flow<String?> = context.dataStore.data.map { it[Keys.CURRENT_USER] }
    val nasaApiKey: Flow<String?> = context.dataStore.data.map { it[Keys.NASA_KEY] }
    val mistralApiKey: Flow<String?> = context.dataStore.data.map { it[Keys.MISTRAL_KEY] }
    val nightMode: Flow<Boolean> = context.dataStore.data.map { it[Keys.NIGHT_MODE] ?: false }

    val site: Flow<ObservingSite?> = context.dataStore.data.map { prefs ->
        val name = prefs[Keys.SITE_NAME] ?: return@map null
        val lat = prefs[Keys.SITE_LAT] ?: return@map null
        val lon = prefs[Keys.SITE_LON] ?: return@map null
        ObservingSite(
            name = name,
            latitude = lat,
            longitude = lon,
            bortle = prefs[Keys.SITE_BORTLE] ?: 5,
            department = prefs[Keys.SITE_DEPARTMENT].orEmpty(),
            bortleEstimated = prefs[Keys.SITE_BORTLE_ESTIMATED] ?: true,
        )
    }

    val instrument: Flow<InstrumentType> = context.dataStore.data.map { prefs ->
        prefs[Keys.INSTRUMENT]?.let { name ->
            InstrumentType.entries.firstOrNull { it.name == name }
        } ?: InstrumentType.TELESCOPE
    }

    val diameterMm: Flow<Double> = context.dataStore.data.map { it[Keys.DIAMETER] ?: 130.0 }
    val focalMm: Flow<Double> = context.dataStore.data.map { it[Keys.FOCAL] ?: 650.0 }
    val eyePupilMm: Flow<Double> = context.dataStore.data.map { it[Keys.PUPIL] ?: 6.0 }
    val catalogs: Flow<String?> = context.dataStore.data.map { it[Keys.CATALOGS] }
    val smartModel: Flow<String?> = context.dataStore.data.map { it[Keys.SMART_MODEL] }
    val smartExposureMinutes: Flow<Double> =
        context.dataStore.data.map { it[Keys.SMART_EXPOSURE] ?: 60.0 }

    suspend fun setCurrentUser(username: String?) = edit { prefs ->
        if (username == null) prefs.remove(Keys.CURRENT_USER) else prefs[Keys.CURRENT_USER] = username
    }

    suspend fun setNasaApiKey(value: String) = edit { it[Keys.NASA_KEY] = value.trim() }
    suspend fun setMistralApiKey(value: String) = edit { it[Keys.MISTRAL_KEY] = value.trim() }
    suspend fun setNightMode(enabled: Boolean) = edit { it[Keys.NIGHT_MODE] = enabled }

    suspend fun setSite(site: ObservingSite) = edit { prefs ->
        prefs[Keys.SITE_NAME] = site.name
        prefs[Keys.SITE_LAT] = site.latitude
        prefs[Keys.SITE_LON] = site.longitude
        prefs[Keys.SITE_BORTLE] = site.bortle
        prefs[Keys.SITE_DEPARTMENT] = site.department
        prefs[Keys.SITE_BORTLE_ESTIMATED] = site.bortleEstimated
    }

    suspend fun setInstrument(type: InstrumentType) = edit { it[Keys.INSTRUMENT] = type.name }
    suspend fun setDiameter(mm: Double) = edit { it[Keys.DIAMETER] = mm }
    suspend fun setFocal(mm: Double) = edit { it[Keys.FOCAL] = mm }
    suspend fun setEyePupil(mm: Double) = edit { it[Keys.PUPIL] = mm }
    suspend fun setCatalogs(value: String) = edit { it[Keys.CATALOGS] = value }
    suspend fun setSmartModel(name: String?) = edit { prefs ->
        if (name == null) prefs.remove(Keys.SMART_MODEL) else prefs[Keys.SMART_MODEL] = name
    }
    suspend fun setSmartExposure(minutes: Double) = edit { it[Keys.SMART_EXPOSURE] = minutes }

    private suspend fun edit(block: (MutablePreferences) -> Unit) {
        context.dataStore.edit { prefs -> block(prefs) }
    }
}

package com.cielscore.app.data.prefs

import android.content.Context
import androidx.datastore.preferences.core.MutablePreferences
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.cielscore.app.model.ApiKey
import com.cielscore.app.model.InstrumentType
import com.cielscore.app.model.ObservingSite
import com.cielscore.app.util.Log
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import java.io.IOException

private val Context.dataStore by preferencesDataStore(name = "cielscore_settings")

/** Instantane de tout ce que l'application conserve entre deux lancements. */
data class StoredSettings(
    val currentUser: String? = null,
    val nasaApiKey: String? = null,
    val mistralApiKey: String? = null,
    val nightMode: Boolean = false,
    val site: ObservingSite? = null,
    val instrument: InstrumentType = InstrumentType.TELESCOPE,
    val diameterMm: Double = 130.0,
    val focalMm: Double = 650.0,
    val eyePupilMm: Double = 6.0,
    val catalogs: String? = null,
    val smartModel: String? = null,
    val smartExposureMinutes: Double = 60.0,
)

/**
 * Preferences locales : session en cours, lieu, instrument et cles d'API.
 *
 * Les cles NASA APOD et Mistral (section 8.2) sont saisies par l'utilisateur
 * depuis le Profil et restent sur l'appareil.
 *
 * DataStore propage les erreurs de lecture sous forme d'IOException dans son
 * flux : sans les intercepter, un incident de fichier ferait perdre d'un coup
 * l'ensemble des reglages. Toutes les lectures passent donc par [read], qui
 * retombe sur des preferences vides plutot que d'echouer, et qui ne parcourt
 * le fichier qu'une seule fois au lieu d'une fois par valeur.
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

    /** Lecture unique de toutes les preferences, sans jamais lever d'exception. */
    suspend fun read(): StoredSettings {
        val prefs: Preferences = context.dataStore.data
            .catch { error ->
                if (error is IOException) {
                    Log.w("Reglages", "Lecture des preferences impossible : ${error.message}")
                    emit(emptyPreferences())
                } else {
                    throw error
                }
            }
            .first()

        val site = run {
            val name = prefs[Keys.SITE_NAME]
            val lat = prefs[Keys.SITE_LAT]
            val lon = prefs[Keys.SITE_LON]
            if (name == null || lat == null || lon == null) null
            else ObservingSite(
                name = name,
                latitude = lat,
                longitude = lon,
                bortle = prefs[Keys.SITE_BORTLE] ?: 5,
                department = prefs[Keys.SITE_DEPARTMENT].orEmpty(),
                bortleEstimated = prefs[Keys.SITE_BORTLE_ESTIMATED] ?: true,
            )
        }

        val stored = StoredSettings(
            currentUser = prefs[Keys.CURRENT_USER],
            nasaApiKey = prefs[Keys.NASA_KEY],
            mistralApiKey = prefs[Keys.MISTRAL_KEY],
            nightMode = prefs[Keys.NIGHT_MODE] ?: false,
            site = site,
            instrument = prefs[Keys.INSTRUMENT]?.let { name ->
                InstrumentType.entries.firstOrNull { it.name == name }
            } ?: InstrumentType.TELESCOPE,
            diameterMm = prefs[Keys.DIAMETER] ?: 130.0,
            focalMm = prefs[Keys.FOCAL] ?: 650.0,
            eyePupilMm = prefs[Keys.PUPIL] ?: 6.0,
            catalogs = prefs[Keys.CATALOGS],
            smartModel = prefs[Keys.SMART_MODEL],
            smartExposureMinutes = prefs[Keys.SMART_EXPOSURE] ?: 60.0,
        )

        Log.i(
            "Reglages",
            "Preferences relues : ${prefs.asMap().size} entrees, " +
                "compte=${stored.currentUser ?: "aucun"}, " +
                "lieu=${stored.site?.name ?: "aucun"}, " +
                "cle NASA=${ApiKey.mask(stored.nasaApiKey).ifEmpty { "absente" }}, " +
                "cle Mistral=${ApiKey.mask(stored.mistralApiKey).ifEmpty { "absente" }}"
        )
        return stored
    }

    suspend fun setCurrentUser(username: String?) = edit { prefs ->
        if (username == null) prefs.remove(Keys.CURRENT_USER) else prefs[Keys.CURRENT_USER] = username
    }

    suspend fun setNasaApiKey(value: String) {
        edit { it[Keys.NASA_KEY] = value.trim() }
        Log.i("Reglages", "Cle NASA enregistree : ${ApiKey.mask(value).ifEmpty { "effacee" }}")
    }

    suspend fun setMistralApiKey(value: String) {
        edit { it[Keys.MISTRAL_KEY] = value.trim() }
        Log.i("Reglages", "Cle Mistral enregistree : ${ApiKey.mask(value).ifEmpty { "effacee" }}")
    }

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

    /** Une ecriture qui echoue est signalee, jamais avalee en silence. */
    private suspend fun edit(block: (MutablePreferences) -> Unit) {
        runCatching { context.dataStore.edit { prefs -> block(prefs) } }
            .onFailure { Log.e("Reglages", "Ecriture des preferences impossible", it) }
    }
}

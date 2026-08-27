package com.cielscore.app.data.prefs

import android.content.Context
import android.content.SharedPreferences
import com.cielscore.app.model.ApiKey
import com.cielscore.app.model.InstrumentType
import com.cielscore.app.model.ObservingSite
import com.cielscore.app.util.Log

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
 * POURQUOI SharedPreferences ET NON DataStore — la premiere version s'appuyait
 * sur DataStore Preferences, qui serialise ses donnees avec protobuf-lite et
 * lit ses champs par reflexion. Dans la version de release, ou R8 elague et
 * renomme, l'ensemble des reglages disparaissait a chaque relancement :
 * compte, lieu et cles d'API. SharedPreferences ne serialise rien, ne
 * reflechit rien, et se lit de maniere synchrone : le probleme ne peut pas se
 * reproduire, et l'etat est disponible des la construction de l'ecran, sans
 * course au demarrage.
 *
 * Les lectures sont synchrones et immediates ; les ecritures utilisent commit()
 * depuis une portee liee a l'application, pour que le resultat soit connu et
 * journalise.
 */
class SettingsStore(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences("cielscore_settings", Context.MODE_PRIVATE)

    private object Keys {
        const val CURRENT_USER = "current_user"
        const val NASA_KEY = "nasa_api_key"
        const val MISTRAL_KEY = "mistral_api_key"

        const val SITE_NAME = "site_name"
        const val SITE_DEPARTMENT = "site_department"
        const val SITE_LAT = "site_lat"
        const val SITE_LON = "site_lon"
        const val SITE_BORTLE = "site_bortle"
        const val SITE_BORTLE_ESTIMATED = "site_bortle_estimated"

        const val INSTRUMENT = "instrument"
        const val DIAMETER = "diameter_mm"
        const val FOCAL = "focal_mm"
        const val PUPIL = "eye_pupil_mm"
        const val CATALOGS = "catalogs"
        const val SMART_MODEL = "smart_model"
        const val SMART_EXPOSURE = "smart_exposure_minutes"
        const val NIGHT_MODE = "night_mode"
    }

    /**
     * Lecture de toutes les preferences. Synchrone : l'etat est pret avant que
     * le premier ecran se compose, ce qui supprime toute course au demarrage.
     */
    fun read(): StoredSettings {
        val site = run {
            val name = prefs.getString(Keys.SITE_NAME, null)
            if (name == null || !prefs.contains(Keys.SITE_LAT) || !prefs.contains(Keys.SITE_LON)) {
                null
            } else {
                ObservingSite(
                    name = name,
                    latitude = prefs.getDouble(Keys.SITE_LAT, 0.0),
                    longitude = prefs.getDouble(Keys.SITE_LON, 0.0),
                    bortle = prefs.getInt(Keys.SITE_BORTLE, 5),
                    department = prefs.getString(Keys.SITE_DEPARTMENT, "").orEmpty(),
                    bortleEstimated = prefs.getBoolean(Keys.SITE_BORTLE_ESTIMATED, true),
                )
            }
        }

        val stored = StoredSettings(
            currentUser = prefs.getString(Keys.CURRENT_USER, null),
            nasaApiKey = prefs.getString(Keys.NASA_KEY, null),
            mistralApiKey = prefs.getString(Keys.MISTRAL_KEY, null),
            nightMode = prefs.getBoolean(Keys.NIGHT_MODE, false),
            site = site,
            instrument = prefs.getString(Keys.INSTRUMENT, null)?.let { name ->
                InstrumentType.entries.firstOrNull { it.name == name }
            } ?: InstrumentType.TELESCOPE,
            diameterMm = prefs.getDouble(Keys.DIAMETER, 130.0),
            focalMm = prefs.getDouble(Keys.FOCAL, 650.0),
            eyePupilMm = prefs.getDouble(Keys.PUPIL, 6.0),
            catalogs = prefs.getString(Keys.CATALOGS, null),
            smartModel = prefs.getString(Keys.SMART_MODEL, null),
            smartExposureMinutes = prefs.getDouble(Keys.SMART_EXPOSURE, 60.0),
        )

        Log.i(
            "Reglages",
            "Preferences relues : ${prefs.all.size} entrees, " +
                "compte=${stored.currentUser ?: "aucun"}, " +
                "lieu=${stored.site?.name ?: "aucun"}, " +
                "cle NASA=${ApiKey.mask(stored.nasaApiKey).ifEmpty { "absente" }}, " +
                "cle Mistral=${ApiKey.mask(stored.mistralApiKey).ifEmpty { "absente" }}"
        )
        return stored
    }

    fun setCurrentUser(username: String?) = edit("compte") { editor ->
        if (username == null) editor.remove(Keys.CURRENT_USER)
        else editor.putString(Keys.CURRENT_USER, username)
    }

    fun setNasaApiKey(value: String) {
        val key = value.trim()
        edit("cle NASA ${ApiKey.mask(key).ifEmpty { "effacee" }}") {
            it.putString(Keys.NASA_KEY, key)
        }
    }

    fun setMistralApiKey(value: String) {
        val key = value.trim()
        edit("cle Mistral ${ApiKey.mask(key).ifEmpty { "effacee" }}") {
            it.putString(Keys.MISTRAL_KEY, key)
        }
    }

    fun setNightMode(enabled: Boolean) = edit("mode nuit") {
        it.putBoolean(Keys.NIGHT_MODE, enabled)
    }

    fun setSession(
        site: ObservingSite,
        instrument: InstrumentType,
        diameterMm: Double,
        focalMm: Double,
        eyePupilMm: Double,
        catalogs: String,
        smartModel: String?,
        smartExposureMinutes: Double,
    ) = edit("session") { editor ->
        editor.putString(Keys.SITE_NAME, site.name)
        editor.putDouble(Keys.SITE_LAT, site.latitude)
        editor.putDouble(Keys.SITE_LON, site.longitude)
        editor.putInt(Keys.SITE_BORTLE, site.bortle)
        editor.putString(Keys.SITE_DEPARTMENT, site.department)
        editor.putBoolean(Keys.SITE_BORTLE_ESTIMATED, site.bortleEstimated)
        editor.putString(Keys.INSTRUMENT, instrument.name)
        editor.putDouble(Keys.DIAMETER, diameterMm)
        editor.putDouble(Keys.FOCAL, focalMm)
        editor.putDouble(Keys.PUPIL, eyePupilMm)
        editor.putString(Keys.CATALOGS, catalogs)
        editor.putDouble(Keys.SMART_EXPOSURE, smartExposureMinutes)
        if (smartModel == null) editor.remove(Keys.SMART_MODEL)
        else editor.putString(Keys.SMART_MODEL, smartModel)
    }

    /**
     * commit() plutot qu'apply() : l'ecriture est confirmee sur le disque avant
     * de rendre la main, et son succes est journalise. Les appels viennent
     * d'une portee d'entrees-sorties, le blocage est donc sans consequence.
     */
    private fun edit(what: String, block: (SharedPreferences.Editor) -> Unit) {
        val editor = prefs.edit()
        block(editor)
        if (editor.commit()) {
            Log.i("Reglages", "Enregistre : $what")
        } else {
            Log.e("Reglages", "Enregistrement en echec : $what")
        }
    }
}

// SharedPreferences ne stocke pas de Double : on passe par la representation
// binaire d'un Long, qui est exacte et reversible.
private fun SharedPreferences.getDouble(key: String, defaultValue: Double): Double =
    if (!contains(key)) defaultValue
    else Double.fromBits(getLong(key, defaultValue.toRawBits()))

private fun SharedPreferences.Editor.putDouble(key: String, value: Double): SharedPreferences.Editor =
    putLong(key, value.toRawBits())

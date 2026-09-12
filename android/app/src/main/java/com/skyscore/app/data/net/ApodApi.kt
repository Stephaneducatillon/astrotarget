package com.skyscore.app.data.net

import com.skyscore.app.BuildConfig
import com.skyscore.app.data.cache.TtlCache
import org.json.JSONObject

/**
 * Image astronomique du jour de la NASA (sections 2.1 et 8.2).
 *
 * La cle API est embarquee dans l'APK (voir build.gradle.kts) : l'utilisateur
 * n'a rien a saisir. Repli de la section 8.4 : si le service repond en erreur
 * ou si le quota est epuise, l'application affiche un message explicite et le
 * reste continue de fonctionner.
 */
object ApodApi {

    data class Apod(
        val title: String,
        val explanation: String,
        val imageUrl: String?,
        val mediaType: String,
        val copyright: String?,
        val date: String,
    )

    /** Message de repli lorsque le service ne repond pas (section 8.4). */
    const val UNAVAILABLE_MESSAGE = "Image du jour indisponible pour le moment."

    /**
     * Message affiche si l'APK a ete construit sans cle. Il ne devrait jamais
     * apparaitre : il signale une construction fautive, pas une panne de la
     * NASA, et evite d'appeler le service avec « api_key= » vide.
     */
    const val MISSING_KEY_MESSAGE =
        "Image du jour indisponible : cette version a ete construite sans cle NASA."

    private val cache = TtlCache<Apod>(TtlCache.APOD_TTL)

    suspend fun today(): Result<Apod> {
        val apiKey = BuildConfig.NASA_API_KEY
        if (apiKey.isBlank()) return Result.failure(IllegalStateException(MISSING_KEY_MESSAGE))
        val key = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.FRANCE)
            .format(java.util.Date())
        val cached = cache.getOrPut(key) { fetch(apiKey) }
        return cached?.let { Result.success(it) }
            ?: Result.failure(IllegalStateException(UNAVAILABLE_MESSAGE))
    }

    private suspend fun fetch(apiKey: String): Apod? {
        val body = Http.getString("https://api.nasa.gov/planetary/apod?api_key=$apiKey") ?: return null
        return runCatching {
            val json = JSONObject(body)
            Apod(
                title = json.optString("title"),
                explanation = json.optString("explanation"),
                imageUrl = json.optString("url").takeIf { it.isNotBlank() },
                mediaType = json.optString("media_type", "image"),
                copyright = json.optString("copyright").takeIf { it.isNotBlank() },
                date = json.optString("date"),
            )
        }.getOrNull()
    }
}

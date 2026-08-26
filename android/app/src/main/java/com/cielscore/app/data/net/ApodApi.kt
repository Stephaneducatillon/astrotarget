package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import org.json.JSONObject

/**
 * Image astronomique du jour de la NASA (sections 2.1 et 8.2).
 *
 * Une cle API est requise. Repli de la section 8.4 : lorsque la cle manque ou
 * que le service repond en erreur, l'application affiche un message invitant a
 * configurer la cle.
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

    /** Message affiche lorsque la cle NASA n'est pas renseignee (section 8.4). */
    const val MISSING_KEY_MESSAGE =
        "Image du jour indisponible : renseignez une cle API NASA dans le Profil."

    private val cache = TtlCache<Apod>(TtlCache.APOD_TTL)

    suspend fun today(apiKey: String?): Result<Apod> {
        if (apiKey.isNullOrBlank()) return Result.failure(IllegalStateException(MISSING_KEY_MESSAGE))
        val key = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.FRANCE)
            .format(java.util.Date())
        val cached = cache.getOrPut(key) { fetch(apiKey) }
        return cached?.let { Result.success(it) }
            ?: Result.failure(IllegalStateException("Service APOD indisponible."))
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

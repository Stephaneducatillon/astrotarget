package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import org.json.JSONObject

/**
 * Prochains tirs spatiaux via The Space Devs (sections 2.1 et 8.2).
 * La section 2.1 en affiche cinq.
 */
object LaunchApi {

    data class Launch(
        val name: String,
        val provider: String,
        val padName: String,
        val country: String,
        val netMillis: Long?,
        val status: String,
    )

    private val cache = TtlCache<List<Launch>>(TtlCache.LAUNCHES_TTL)

    suspend fun upcoming(count: Int = 5): List<Launch> =
        cache.getOrPut("global") { fetch(count) }.orEmpty()

    private suspend fun fetch(count: Int): List<Launch>? {
        val body = Http.getString(
            "https://ll.thespacedevs.com/2.2.0/launch/upcoming/?limit=$count&mode=list"
        ) ?: return null
        return runCatching {
            val results = JSONObject(body).getJSONArray("results")
            val parser = java.text.SimpleDateFormat(
                "yyyy-MM-dd'T'HH:mm:ss'Z'", java.util.Locale.US
            ).apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
            (0 until results.length()).map { i ->
                val r = results.getJSONObject(i)
                Launch(
                    name = r.optString("name"),
                    provider = r.optJSONObject("launch_service_provider")?.optString("name").orEmpty(),
                    padName = r.optJSONObject("pad")?.optString("name").orEmpty(),
                    country = r.optJSONObject("pad")?.optString("country_code").orEmpty(),
                    netMillis = runCatching { parser.parse(r.optString("net"))?.time }.getOrNull(),
                    status = r.optJSONObject("status")?.optString("abbrev").orEmpty(),
                )
            }
        }.getOrNull()
    }
}

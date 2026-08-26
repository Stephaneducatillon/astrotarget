package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import com.cielscore.app.util.Log
import org.json.JSONArray

/**
 * Indice Kp et activite aurorale (section 2.1, « Aurores boreales »).
 *
 * Source principale : GFZ Potsdam. Repli de la section 8.4 : bascule sur
 * NOAA SWPC.
 */
object SpaceWeatherApi {

    /** Indice Kp courant et historique des 24 dernieres heures. */
    data class KpData(
        val current: Double,
        /** Couples (horodatage UTC en millisecondes, indice Kp). */
        val history: List<Pair<Long, Double>>,
        val source: String,
    ) {
        /** Interpretation courante de l'indice Kp. */
        val label: String
            get() = when {
                current < 3 -> "Activite calme"
                current < 5 -> "Activite moderee"
                current < 6 -> "Orage geomagnetique mineur"
                current < 7 -> "Orage geomagnetique modere"
                else -> "Orage geomagnetique fort"
            }
    }

    private val cache = TtlCache<KpData>(TtlCache.KP_TTL)

    suspend fun kp(): KpData? {
        val key = TtlCache.kpKey(System.currentTimeMillis())
        return cache.getOrPut(key) { fetchGfz() ?: fetchNoaa() }
    }

    private suspend fun fetchGfz(): KpData? {
        val now = System.currentTimeMillis()
        val start = now - 24 * 60 * 60 * 1000L
        val fmt = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", java.util.Locale.US)
            .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
        val url = "https://kp.gfz-potsdam.de/app/json/?start=${fmt.format(java.util.Date(start))}" +
            "&end=${fmt.format(java.util.Date(now))}&index=Kp"
        val body = Http.getString(url) ?: run {
            Log.w("SpaceWeatherApi", "GFZ Potsdam indisponible, bascule sur NOAA SWPC")
            return null
        }
        return runCatching {
            val json = org.json.JSONObject(body)
            val times = json.getJSONArray("datetime")
            val values = json.getJSONArray("Kp")
            val parser = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", java.util.Locale.US)
                .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
            val history = (0 until times.length()).mapNotNull { i ->
                val t = parser.parse(times.getString(i))?.time ?: return@mapNotNull null
                t to values.getDouble(i)
            }
            KpData(history.lastOrNull()?.second ?: 0.0, history, "GFZ Potsdam")
        }.getOrNull()
    }

    private suspend fun fetchNoaa(): KpData? {
        val body = Http.getString(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        ) ?: run {
            Log.w("SpaceWeatherApi", "NOAA SWPC indisponible")
            return null
        }
        return runCatching {
            val rows = JSONArray(body)
            val parser = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", java.util.Locale.US)
                .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
            val cutoff = System.currentTimeMillis() - 24 * 60 * 60 * 1000L
            val history = (1 until rows.length()).mapNotNull { i ->
                val row = rows.getJSONArray(i)
                val t = parser.parse(row.getString(0))?.time ?: return@mapNotNull null
                if (t < cutoff) null else t to row.getString(1).toDouble()
            }
            KpData(history.lastOrNull()?.second ?: 0.0, history, "NOAA SWPC")
        }.getOrNull()
    }
}

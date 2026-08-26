package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import com.cielscore.app.model.SkyConditions
import com.cielscore.app.util.Log
import org.json.JSONObject
import kotlin.math.abs

/**
 * Meteo horaire Open-Meteo (section 8.2 : gratuit, sans cle).
 *
 * Repli de la section 8.4 : en cas d'indisponibilite, l'application retient
 * 50 % de nuages, un seeing de 3 et l'indicateur ok = false.
 */
object WeatherApi {

    private val cache = TtlCache<SkyConditions>(TtlCache.WEATHER_TTL)

    suspend fun conditions(lat: Double, lon: Double, epochMillis: Long): SkyConditions {
        val key = TtlCache.weatherKey(lat, lon, epochMillis)
        return cache.getOrPut(key) { fetch(lat, lon, epochMillis) } ?: SkyConditions.FALLBACK
    }

    private suspend fun fetch(lat: Double, lon: Double, epochMillis: Long): SkyConditions? {
        val url = ApiUrls.openMeteoForecast(lat, lon)
        val body = Http.getString(url) ?: run {
            Log.w("WeatherApi", "Open-Meteo indisponible, repli sur les valeurs par defaut")
            return null
        }
        return runCatching {
            val hourly = JSONObject(body).getJSONObject("hourly")
            val times = hourly.getJSONArray("time")
            val targetSeconds = epochMillis / 1000

            var best = 0
            var bestDelta = Long.MAX_VALUE
            for (i in 0 until times.length()) {
                val delta = abs(times.getLong(i) - targetSeconds)
                if (delta < bestDelta) {
                    bestDelta = delta
                    best = i
                }
            }

            fun value(name: String, fallback: Double): Double {
                val arr = hourly.optJSONArray(name) ?: return fallback
                if (best >= arr.length() || arr.isNull(best)) return fallback
                return arr.optDouble(best, fallback)
            }

            SkyConditions(
                cloudCoverPercent = value("cloud_cover", 50.0),
                windSpeedKmh = value("wind_speed_10m", 10.0),
                humidityPercent = value("relative_humidity_2m", 70.0),
                visibilityMeters = value("visibility", 20_000.0),
                temperatureCelsius = value("temperature_2m", 10.0),
                ok = true,
            )
        }.getOrNull()
    }
}

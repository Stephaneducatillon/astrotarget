package com.cielscore.app.data.cache

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Cache a duree de vie — section 4.6 de la documentation.
 *
 *   Meteo Open-Meteo   1 heure     cle : lieu + heure arrondie
 *   Indice Kp          15 minutes  cle : horodatage 10 min
 *   Image APOD         12 heures   cle : date du jour
 *   Lancements         1 heure     cle : globale
 *   Guide IA           30 minutes  cle : nom objet + diametre
 */
class TtlCache<V>(private val ttlMillis: Long) {

    private data class Entry<V>(val value: V, val expiresAt: Long)

    private val mutex = Mutex()
    private val entries = HashMap<String, Entry<V>>()

    suspend fun get(key: String): V? = mutex.withLock {
        val e = entries[key] ?: return@withLock null
        if (System.currentTimeMillis() > e.expiresAt) {
            entries.remove(key)
            null
        } else e.value
    }

    suspend fun put(key: String, value: V) = mutex.withLock {
        entries[key] = Entry(value, System.currentTimeMillis() + ttlMillis)
    }

    /** Renvoie la valeur en cache ou la produit puis la memorise. */
    suspend fun getOrPut(key: String, producer: suspend () -> V?): V? {
        get(key)?.let { return it }
        val produced = producer() ?: return null
        put(key, produced)
        return produced
    }

    suspend fun clear() = mutex.withLock { entries.clear() }

    companion object {
        const val WEATHER_TTL = 60 * 60 * 1000L
        const val KP_TTL = 15 * 60 * 1000L
        const val APOD_TTL = 12 * 60 * 60 * 1000L
        const val LAUNCHES_TTL = 60 * 60 * 1000L
        const val AI_GUIDE_TTL = 30 * 60 * 1000L
        const val GEO_TTL = 24 * 60 * 60 * 1000L

        /** Cle meteo : lieu arrondi au centieme de degre + heure pleine. */
        fun weatherKey(lat: Double, lon: Double, epochMillis: Long): String {
            val hour = epochMillis / (60 * 60 * 1000L)
            return "%.2f_%.2f_%d".format(lat, lon, hour)
        }

        /** Cle Kp : horodatage arrondi a 10 minutes. */
        fun kpKey(epochMillis: Long): String = (epochMillis / (10 * 60 * 1000L)).toString()

        /** Cle du guide IA : nom de l'objet + diametre de l'instrument. */
        fun aiGuideKey(objectName: String, diameterMm: Double): String =
            "$objectName@${diameterMm.toInt()}"
    }
}

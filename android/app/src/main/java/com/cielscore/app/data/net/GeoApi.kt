package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import com.cielscore.app.model.ObservingSite
import org.json.JSONArray

/**
 * Recherche de commune francaise via l'API Geo de l'Etat
 * (geo.api.gouv.fr, gratuite et sans cle).
 *
 * La documentation de reference s'appuie sur un fichier communes_bortle.csv de
 * 34 870 lignes embarque dans l'application. Ce fichier n'etant pas fourni,
 * l'application interroge l'API Geo pour obtenir nom, departement et
 * coordonnees, et estime l'indice de Bortle a partir de la population.
 *
 * L'estimation reste modifiable par l'utilisateur : l'indice retenu prime des
 * qu'il a ete ajuste a la main.
 */
object GeoApi {

    private val cache = TtlCache<List<ObservingSite>>(TtlCache.GEO_TTL)

    private const val FIELDS = "nom,code,codeDepartement,centre,population"

    /** Recherche par nom, resultats classes par population decroissante. */
    suspend fun searchCommunes(query: String, limit: Int = 20): List<ObservingSite> {
        val trimmed = query.trim()
        if (trimmed.length < 2) return emptyList()
        val encoded = java.net.URLEncoder.encode(trimmed, "UTF-8")
        return cache.getOrPut("nom:$encoded:$limit") {
            val body = Http.getString(
                "https://geo.api.gouv.fr/communes?nom=$encoded&fields=$FIELDS" +
                    "&boost=population&limit=$limit"
            ) ?: return@getOrPut null
            parse(body)
        }.orEmpty()
    }

    /** Commune la plus proche d'une position GPS. */
    suspend fun communeAt(latitude: Double, longitude: Double): ObservingSite? {
        val key = "geo:%.4f:%.4f".format(latitude, longitude)
        return cache.getOrPut(key) {
            val body = Http.getString(
                "https://geo.api.gouv.fr/communes?lat=%.6f&lon=%.6f&fields=%s"
                    .format(latitude, longitude, FIELDS)
            ) ?: return@getOrPut null
            parse(body)
        }?.firstOrNull()
    }

    private fun parse(body: String): List<ObservingSite>? = runCatching {
        val arr = JSONArray(body)
        (0 until arr.length()).mapNotNull { i ->
            val o = arr.getJSONObject(i)
            val centre = o.optJSONObject("centre") ?: return@mapNotNull null
            val coords = centre.optJSONArray("coordinates") ?: return@mapNotNull null
            if (coords.length() < 2) return@mapNotNull null
            val population = o.optInt("population", 0)
            ObservingSite(
                name = o.optString("nom"),
                // GeoJSON : [longitude, latitude]
                longitude = coords.getDouble(0),
                latitude = coords.getDouble(1),
                bortle = estimateBortle(population),
                department = o.optString("codeDepartement"),
                bortleEstimated = true,
            )
        }
    }.getOrNull()

    /**
     * Estimation de l'indice de Bortle a partir de la population communale.
     *
     * INTERPRETATION — le fichier communes_bortle.csv de la section 8.1 n'est
     * pas fourni avec la documentation. Cette table de correspondance en tient
     * lieu ; elle donne un ordre de grandeur et reste ajustable par
     * l'utilisateur depuis le Dashboard.
     *
     * La valeur par defaut hors commune reste Bortle 5 (RG-INFO-01).
     */
    fun estimateBortle(population: Int): Int = when {
        population >= 500_000 -> 9
        population >= 150_000 -> 8
        population >= 50_000 -> 7
        population >= 15_000 -> 6
        population >= 3_000 -> 5
        population >= 500 -> 4
        else -> 3
    }
}

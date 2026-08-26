package com.cielscore.app.catalog

import com.cielscore.app.model.ObservingSite
import java.text.Normalizer
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Une commune francaise du fichier communes_bortle.csv (section 8.1).
 *
 *   code_insee;commune;code_departement;departement;population;lat;lng;
 *   bortle_estime;description_ciel
 */
data class Commune(
    val inseeCode: String,
    val name: String,
    /** Nom normalise, calcule au chargement, qui sert a la recherche. */
    val searchKey: String,
    val departmentCode: String,
    val departmentName: String,
    val population: Int,
    val latitude: Double,
    val longitude: Double,
    val bortle: Int,
)

/**
 * Analyse et interrogation du fichier des 34 869 communes francaises
 * (section 8.1 : « Nom, departement, lat, lng, Bortle estime »,
 * « Bortle automatique France entiere »).
 *
 * Toutes les fonctions de ce fichier sont pures : la recherche et le
 * rattachement a la commune la plus proche fonctionnent hors ligne et sont
 * directement testables.
 */
object CommuneIndex {

    /**
     * Normalise un nom de commune ou une saisie utilisateur : minuscules, sans
     * accent, sans separateur. « L'Abergement-Clemenciat » devient
     * « labergementclemenciat », ce qui rend la recherche tolerante aux
     * apostrophes, aux traits d'union et aux accents.
     */
    fun normalize(value: String): String {
        val decomposed = Normalizer.normalize(value, Normalizer.Form.NFD)
        val builder = StringBuilder(decomposed.length)
        for (c in decomposed) {
            when {
                c in 'a'..'z' || c in '0'..'9' -> builder.append(c)
                c in 'A'..'Z' -> builder.append(c + 32)
                // Les marques diacritiques et les separateurs sont ecartes.
                else -> Unit
            }
        }
        return builder.toString()
    }

    /** Analyse une ligne du fichier. Renvoie null pour l'en-tete ou une ligne invalide. */
    fun parseLine(line: String): Commune? {
        if (line.isBlank()) return null
        val f = line.split(';')
        if (f.size < 8) return null
        if (f[0] == "code_insee") return null
        val latitude = f[5].toDoubleOrNull() ?: return null
        val longitude = f[6].toDoubleOrNull() ?: return null
        val bortle = f[7].toIntOrNull() ?: return null
        val name = f[1].trim()
        if (name.isEmpty()) return null
        return Commune(
            inseeCode = f[0],
            name = name,
            searchKey = normalize(name),
            departmentCode = f[2],
            departmentName = f[3],
            population = f[4].toIntOrNull() ?: 0,
            latitude = latitude,
            longitude = longitude,
            bortle = bortle.coerceIn(1, 9),
        )
    }

    /**
     * Recherche par nom, tolerante aux accents et a la ponctuation.
     *
     * Les communes dont le nom commence par la saisie passent devant celles qui
     * la contiennent seulement ; a rang egal, la plus peuplee est proposee en
     * premier, pour que « Paris » ou « Lyon » arrivent en tete.
     */
    fun search(communes: List<Commune>, query: String, limit: Int = 20): List<Commune> {
        val key = normalize(query)
        if (key.length < 2) return emptyList()
        val matches = ArrayList<Pair<Int, Commune>>()
        for (commune in communes) {
            val rank = when {
                commune.searchKey == key -> 0
                commune.searchKey.startsWith(key) -> 1
                commune.searchKey.contains(key) -> 2
                else -> continue
            }
            matches.add(rank to commune)
        }
        matches.sortWith(
            compareBy<Pair<Int, Commune>> { it.first }
                .thenByDescending { it.second.population }
                .thenBy { it.second.name }
        )
        return matches.take(limit).map { it.second }
    }

    /**
     * Commune la plus proche d'une position GPS, ce qui remplace tout appel de
     * geocodage inverse : le rattachement se fait hors ligne.
     */
    fun nearest(communes: List<Commune>, latitude: Double, longitude: Double): Commune? {
        var best: Commune? = null
        var bestDistance = Double.MAX_VALUE
        for (commune in communes) {
            val d = distanceKm(latitude, longitude, commune.latitude, commune.longitude)
            if (d < bestDistance) {
                bestDistance = d
                best = commune
            }
        }
        return best
    }

    /** Distance orthodromique entre deux points, en kilometres (formule de haversine). */
    fun distanceKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val earthRadiusKm = 6371.0088
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2) * sin(dLat / 2) +
            cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
            sin(dLon / 2) * sin(dLon / 2)
        return 2 * earthRadiusKm * asin(min(1.0, sqrt(a)))
    }
}

/**
 * Une commune devient un lieu d'observation : coordonnees et indice de Bortle
 * sont fixes automatiquement (section 2.2, « Commune selectionnee »).
 *
 * L'indice reste marque comme estime tant que l'utilisateur ne l'a pas ajuste :
 * la colonne du fichier s'appelle bien bortle_estime.
 */
fun Commune.toObservingSite(): ObservingSite = ObservingSite(
    name = name,
    latitude = latitude,
    longitude = longitude,
    bortle = bortle,
    department = departmentCode,
    bortleEstimated = true,
)

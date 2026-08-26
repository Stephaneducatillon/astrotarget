package com.cielscore.app.catalog

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Etoile brillante du fond de carte (section 3.3). */
data class Star(
    val id: String,
    val name: String,
    val raDeg: Double,
    val decDeg: Double,
    val magnitude: Double,
    val constellation: String,
)

/** Figure de constellation : une liste de segments entre identifiants d'etoiles. */
data class Constellation(
    val abbreviation: String,
    val frenchName: String,
    val segments: List<Pair<String, String>>,
)

/**
 * Fond d'etoiles et figures de constellations de la Carte du ciel (section 3.3).
 *
 * Les fichiers stars.csv et constellations.csv sont produits par
 * tools/build_stars.py. Les regles d'affichage de la section 3.3 sont
 * appliquees a la projection, pas au chargement.
 */
class StarCatalog(private val context: Context) {

    private val mutex = Mutex()
    private var stars: List<Star>? = null
    private var starsById: Map<String, Star>? = null
    private var constellations: List<Constellation>? = null

    suspend fun stars(): List<Star> {
        ensureLoaded()
        return stars.orEmpty()
    }

    suspend fun starsById(): Map<String, Star> {
        ensureLoaded()
        return starsById.orEmpty()
    }

    suspend fun constellations(): List<Constellation> {
        ensureLoaded()
        return constellations.orEmpty()
    }

    private suspend fun ensureLoaded() = mutex.withLock {
        if (stars != null) return@withLock
        withContext(Dispatchers.IO) {
            val loaded = context.assets.open("stars.csv").bufferedReader().useLines { lines ->
                lines.drop(1).mapNotNull { line ->
                    val f = line.split(';')
                    if (f.size < 6) return@mapNotNull null
                    Star(
                        id = f[0],
                        name = f[1],
                        raDeg = f[2].toDoubleOrNull() ?: return@mapNotNull null,
                        decDeg = f[3].toDoubleOrNull() ?: return@mapNotNull null,
                        magnitude = f[4].toDoubleOrNull() ?: return@mapNotNull null,
                        constellation = f[5],
                    )
                }.toList()
            }
            val figures = context.assets.open("constellations.csv").bufferedReader()
                .useLines { lines ->
                    lines.drop(1).mapNotNull { line ->
                        val f = line.split(';')
                        if (f.size < 3) return@mapNotNull null
                        Constellation(
                            abbreviation = f[0],
                            frenchName = f[1],
                            segments = f[2].split(',')
                                .mapNotNull { seg ->
                                    val parts = seg.split('>')
                                    if (parts.size == 2) parts[0] to parts[1] else null
                                },
                        )
                    }.toList()
                }
            stars = loaded
            starsById = loaded.associateBy { it.id }
            constellations = figures
        }
    }
}

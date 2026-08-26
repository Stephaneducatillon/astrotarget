package com.cielscore.app.catalog

import android.content.Context
import com.cielscore.app.model.ObservingSite
import com.cielscore.app.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Les 34 869 communes francaises embarquees (section 8.1).
 *
 * Le fichier communes_bortle.csv est livre dans les assets : la recherche par
 * nom et le rattachement d'une position GPS a la commune la plus proche
 * fonctionnent donc entierement hors ligne, indice de Bortle compris
 * (« Bortle automatique France entiere »).
 *
 * Le chargement est paresseux et mis en cache : il n'a lieu qu'a la premiere
 * recherche, sur le dispatcher d'entrees-sorties.
 */
class CommuneRepository(private val context: Context) {

    private val mutex = Mutex()
    private var communes: List<Commune>? = null

    /** Charge le fichier au premier appel, puis renvoie la liste mise en cache. */
    suspend fun communes(): List<Commune> {
        communes?.let { return it }
        return mutex.withLock {
            communes ?: load().also { communes = it }
        }
    }

    private suspend fun load(): List<Commune> = withContext(Dispatchers.IO) {
        val started = System.currentTimeMillis()
        val parsed = ArrayList<Commune>(35_000)
        runCatching {
            context.assets.open(ASSET).bufferedReader().use { reader ->
                while (true) {
                    val line = reader.readLine() ?: break
                    CommuneIndex.parseLine(line)?.let(parsed::add)
                }
            }
        }.onFailure {
            Log.e("Communes", "Chargement de $ASSET impossible", it)
        }
        Log.i(
            "Communes",
            "${parsed.size} communes chargees en ${System.currentTimeMillis() - started} ms"
        )
        parsed
    }

    /** Recherche par nom, tolerante aux accents et a la ponctuation. */
    suspend fun search(query: String, limit: Int = 20): List<ObservingSite> {
        if (query.trim().length < 2) return emptyList()
        val all = communes()
        return withContext(Dispatchers.Default) {
            CommuneIndex.search(all, query, limit).map { it.toObservingSite() }
        }
    }

    /** Commune la plus proche d'une position GPS. */
    suspend fun nearest(latitude: Double, longitude: Double): ObservingSite? {
        val all = communes()
        return withContext(Dispatchers.Default) {
            CommuneIndex.nearest(all, latitude, longitude)?.toObservingSite()
        }
    }

    private companion object {
        const val ASSET = "communes_bortle.csv"
    }
}

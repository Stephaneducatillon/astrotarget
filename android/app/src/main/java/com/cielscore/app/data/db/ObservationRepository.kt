package com.cielscore.app.data.db

import com.cielscore.app.util.Log
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map

/**
 * Carnet d'observation (section 2.4) et statistiques (section 2.6).
 */
class ObservationRepository(private val db: CielScoreDatabase) {

    private val dao = db.observationDao()

    /** Les 100 dernieres observations du carnet. */
    fun recent(username: String): Flow<List<ObservationEntity>> = dao.recent(username)

    suspend fun save(observation: ObservationEntity): Long {
        val id = dao.insert(observation)
        Log.i("Carnet", "Observation enregistree : ${observation.objectName}")
        return id
    }

    suspend fun delete(id: Long, username: String) = dao.delete(id, username)

    suspend fun all(username: String): List<ObservationEntity> = dao.all(username)

    /** Indicateurs de la section 2.6. */
    data class Stats(
        val sessions: Int,
        val observations: Int,
        val distinctObjects: Int,
        val averageScore: Double,
    )

    fun stats(username: String): Flow<Stats> = combine(
        dao.totalSessions(username),
        dao.totalObservations(username),
        dao.distinctObjects(username),
        dao.averageScore(username),
    ) { sessions, observations, objects, average ->
        Stats(sessions, observations, objects, average ?: 0.0)
    }

    fun favourites(username: String): Flow<List<CountedLabel>> = dao.favourites(username)
    fun byType(username: String): Flow<List<CountedLabel>> = dao.byType(username)
    fun bySite(username: String): Flow<List<CountedLabel>> = dao.bySite(username)

    /** Heatmap sur douze mois : nombre d'observations par jour. */
    fun activity(username: String): Flow<Map<String, Int>> {
        val since = System.currentTimeMillis() - 365L * 24 * 60 * 60 * 1000
        return dao.activitySince(username, since).map { rows ->
            rows.associate { it.label to it.total }
        }
    }

    /**
     * Progression Messier (n/110) et Caldwell (n/109), section 2.6.
     * La correspondance se fait sur l'identifiant court enregistre dans le carnet.
     */
    suspend fun progress(username: String): Pair<Int, Int> {
        val names = dao.observedObjectNames(username)
        val messier = names.mapNotNull { messierNumber(it) }.toSet().size
        val caldwell = names.mapNotNull { caldwellNumber(it) }.toSet().size
        return messier to caldwell
    }

    private fun messierNumber(label: String): Int? {
        val id = label.substringBefore(' ').substringBefore('—').trim()
        if (!id.startsWith("M", ignoreCase = true)) return null
        return id.drop(1).toIntOrNull()?.takeIf { it in 1..110 }
    }

    private fun caldwellNumber(label: String): Int? {
        val id = label.substringBefore(' ').substringBefore('—').trim()
        if (!id.startsWith("C", ignoreCase = true)) return null
        return id.drop(1).toIntOrNull()?.takeIf { it in 1..109 }
    }
}

package com.cielscore.app.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface UserDao {

    @Query("SELECT * FROM users WHERE username = :username LIMIT 1")
    suspend fun findByUsername(username: String): UserEntity?

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(user: UserEntity)

    @Update
    suspend fun update(user: UserEntity)

    @Query("SELECT COUNT(*) FROM users")
    suspend fun count(): Int
}

/** Une observation groupee, pour les statistiques de la section 2.6. */
data class CountedLabel(val label: String, val total: Int)

@Dao
interface ObservationDao {

    @Insert
    suspend fun insert(observation: ObservationEntity): Long

    @Query("DELETE FROM observations WHERE id = :id AND username = :username")
    suspend fun delete(id: Long, username: String)

    /** Carnet : les 100 dernieres observations (section 2.4). */
    @Query(
        "SELECT * FROM observations WHERE username = :username " +
            "ORDER BY observation_date DESC LIMIT :limit"
    )
    fun recent(username: String, limit: Int = 100): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations WHERE username = :username ORDER BY observation_date DESC")
    suspend fun all(username: String): List<ObservationEntity>

    @Query("SELECT COUNT(*) FROM observations WHERE username = :username")
    fun totalObservations(username: String): Flow<Int>

    @Query(
        "SELECT COUNT(DISTINCT date(observation_date / 1000, 'unixepoch')) " +
            "FROM observations WHERE username = :username"
    )
    fun totalSessions(username: String): Flow<Int>

    @Query("SELECT COUNT(DISTINCT objet) FROM observations WHERE username = :username")
    fun distinctObjects(username: String): Flow<Int>

    @Query("SELECT AVG(score) FROM observations WHERE username = :username")
    fun averageScore(username: String): Flow<Double?>

    @Query("SELECT DISTINCT objet FROM observations WHERE username = :username")
    suspend fun observedObjectNames(username: String): List<String>

    @Query(
        "SELECT objet AS label, COUNT(*) AS total FROM observations " +
            "WHERE username = :username GROUP BY objet ORDER BY total DESC LIMIT :limit"
    )
    fun favourites(username: String, limit: Int = 10): Flow<List<CountedLabel>>

    @Query(
        "SELECT type AS label, COUNT(*) AS total FROM observations " +
            "WHERE username = :username AND type <> '' GROUP BY type ORDER BY total DESC"
    )
    fun byType(username: String): Flow<List<CountedLabel>>

    @Query(
        "SELECT site AS label, COUNT(*) AS total FROM observations " +
            "WHERE username = :username GROUP BY site ORDER BY total DESC"
    )
    fun bySite(username: String): Flow<List<CountedLabel>>

    /** Activite sur douze mois : un compte par jour (heatmap de la section 2.6). */
    @Query(
        "SELECT date(observation_date / 1000, 'unixepoch') AS label, COUNT(*) AS total " +
            "FROM observations WHERE username = :username AND observation_date >= :since " +
            "GROUP BY label"
    )
    fun activitySince(username: String, since: Long): Flow<List<CountedLabel>>
}

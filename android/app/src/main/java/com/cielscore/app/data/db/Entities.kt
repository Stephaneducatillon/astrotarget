package com.cielscore.app.data.db

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Table users — section 9.2.
 *
 *   username, password_hash, first_name, last_name, recovery_hash, created_at
 */
@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey
    @ColumnInfo(name = "username") val username: String,
    @ColumnInfo(name = "password_hash") val passwordHash: String,
    @ColumnInfo(name = "first_name") val firstName: String,
    @ColumnInfo(name = "last_name") val lastName: String,
    @ColumnInfo(name = "recovery_hash") val recoveryHash: String,
    @ColumnInfo(name = "created_at") val createdAt: Long,
)

/**
 * Table observations — section 9.2.
 *
 *   id, username, date, objet, site, instrument, bortle, lune, nuages,
 *   seeing, score, notes
 *
 * Index sur username et observation_date, pour accelerer les requetes du carnet
 * et des statistiques.
 */
@Entity(
    tableName = "observations",
    indices = [
        Index(value = ["username"]),
        Index(value = ["observation_date"]),
    ],
)
data class ObservationEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id") val id: Long = 0,
    @ColumnInfo(name = "username") val username: String,
    @ColumnInfo(name = "observation_date") val observationDate: Long,
    @ColumnInfo(name = "objet") val objectName: String,
    @ColumnInfo(name = "site") val site: String,
    @ColumnInfo(name = "instrument") val instrument: String,
    @ColumnInfo(name = "bortle") val bortle: Int,
    @ColumnInfo(name = "lune") val moonPhasePercent: Double,
    @ColumnInfo(name = "nuages") val cloudCoverPercent: Double,
    @ColumnInfo(name = "seeing") val seeingIndex: Int,
    @ColumnInfo(name = "score") val score: Double,
    @ColumnInfo(name = "notes") val notes: String,
    /** Type d'objet, utilise par la repartition des statistiques (section 2.6). */
    @ColumnInfo(name = "type") val objectType: String = "",
)

package com.cielscore.app.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.cielscore.app.util.Log

/**
 * Base SQLite locale de l'application (section 9.2).
 *
 * La documentation de reference restaure la base depuis un depot distant au
 * demarrage (section 9.3) ; l'application Android conserve ses donnees
 * localement, sur l'appareil, et s'appuie sur la sauvegarde Android
 * (backup_rules.xml) pour la persistance entre reinstallations.
 */
@Database(
    entities = [UserEntity::class, ObservationEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class CielScoreDatabase : RoomDatabase() {

    abstract fun userDao(): UserDao
    abstract fun observationDao(): ObservationDao

    companion object {
        @Volatile
        private var instance: CielScoreDatabase? = null

        fun get(context: Context): CielScoreDatabase =
            instance ?: synchronized(this) {
                instance ?: build(context).also { instance = it }
            }

        private fun build(context: Context): CielScoreDatabase {
            Log.i("Database", "Initialisation de la base cielscore.db")
            return Room.databaseBuilder(
                context.applicationContext,
                CielScoreDatabase::class.java,
                "cielscore.db",
            ).build()
        }
    }
}

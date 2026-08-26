package com.cielscore.app.di

import android.content.Context
import com.cielscore.app.catalog.CatalogRepository
import com.cielscore.app.catalog.StarCatalog
import com.cielscore.app.data.auth.AuthRepository
import com.cielscore.app.data.db.CielScoreDatabase
import com.cielscore.app.data.db.ObservationRepository
import com.cielscore.app.data.prefs.SettingsStore

/** Assemblage des dependances de l'application. */
class AppContainer(context: Context) {

    private val appContext = context.applicationContext

    val database: CielScoreDatabase by lazy { CielScoreDatabase.get(appContext) }
    val auth: AuthRepository by lazy { AuthRepository(database) }
    val observations: ObservationRepository by lazy { ObservationRepository(database) }
    val catalogs: CatalogRepository by lazy { CatalogRepository(appContext) }
    val stars: StarCatalog by lazy { StarCatalog(appContext) }
    val settings: SettingsStore by lazy { SettingsStore(appContext) }
}

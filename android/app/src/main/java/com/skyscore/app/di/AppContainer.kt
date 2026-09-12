package com.skyscore.app.di

import android.content.Context
import com.skyscore.app.catalog.CatalogRepository
import com.skyscore.app.catalog.CommuneRepository
import com.skyscore.app.catalog.StarCatalog
import com.skyscore.app.data.auth.AuthRepository
import com.skyscore.app.data.db.SkyScoreDatabase
import com.skyscore.app.data.db.ObservationRepository
import com.skyscore.app.data.prefs.SettingsStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/** Assemblage des dependances de l'application. */
class AppContainer(context: Context) {

    private val appContext = context.applicationContext

    val database: SkyScoreDatabase by lazy { SkyScoreDatabase.get(appContext) }
    val auth: AuthRepository by lazy { AuthRepository(database) }
    val observations: ObservationRepository by lazy { ObservationRepository(database) }
    val catalogs: CatalogRepository by lazy { CatalogRepository(appContext) }
    val communes: CommuneRepository by lazy { CommuneRepository(appContext) }
    val stars: StarCatalog by lazy { StarCatalog(appContext) }
    val settings: SettingsStore by lazy { SettingsStore(appContext) }

    /**
     * Portee liee a l'application, pas a l'ecran.
     *
     * Les ecritures de preferences ne doivent pas dependre du cycle de vie du
     * ViewModel : une cle saisie juste avant de quitter l'application verrait
     * sinon son enregistrement annule avant d'atteindre le disque.
     */
    val persistenceScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
}

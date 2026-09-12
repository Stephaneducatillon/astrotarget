package com.skyscore.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.skyscore.app.SkyScoreApplication
import com.skyscore.app.astro.AstroMath
import com.skyscore.app.astro.SolarSystem
import com.skyscore.app.astro.Twilight
import com.skyscore.app.catalog.Catalog
import com.skyscore.app.catalog.CatalogRepository
import com.skyscore.app.catalog.ObjectType
import com.skyscore.app.catalog.SkyObject
import com.skyscore.app.data.auth.AuthRepository
import com.skyscore.app.data.db.ObservationEntity
import com.skyscore.app.data.net.ApodApi
import com.skyscore.app.data.net.LaunchApi
import com.skyscore.app.data.net.SpaceWeatherApi
import com.skyscore.app.data.net.WeatherApi
import com.skyscore.app.data.prefs.StoredSettings
import com.skyscore.app.model.MoonState
import com.skyscore.app.model.ObservingSite
import com.skyscore.app.model.SessionParams
import com.skyscore.app.model.SignedInUser
import com.skyscore.app.model.SkyConditions
import com.skyscore.app.model.SmartTelescope
import com.skyscore.app.scoring.Formulas
import com.skyscore.app.scoring.ScoringEngine
import com.skyscore.app.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Etat global de l'application, partage par les huit onglets. */
data class AppUiState(
    val user: SignedInUser? = null,
    val params: SessionParams = SessionParams(),
    val conditions: SkyConditions = SkyConditions.FALLBACK,
    val night: Twilight.NightInfo? = null,
    val moon: MoonState? = null,
    val sunAltitudeDeg: Double = 0.0,
    val sunRise: Long? = null,
    val sunSet: Long? = null,
    val moonRise: Long? = null,
    val moonSet: Long? = null,

    /** Top des cibles du Dashboard (section 2.2). */
    val topTargets: List<ScoringEngine.Scored> = emptyList(),
    val selected: ScoringEngine.Scored? = null,
    val altitudeCurve: List<Pair<Long, Double>> = emptyList(),
    val calculating: Boolean = false,
    val lastComputedAt: Long? = null,

    /** Explorer (section 2.3). */
    val explorerQuery: String = "",
    val explorerCatalogs: Set<Catalog> = setOf(Catalog.MESSIER, Catalog.CALDWELL, Catalog.NGC_IC),
    val explorerTypes: Set<ObjectType> = ObjectType.entries.toSet(),
    val explorerResults: List<SkyObject> = emptyList(),
    val explorerSearching: Boolean = false,

    /** Recherche de commune (section 2.2). */
    val communeQuery: String = "",
    val communeResults: List<ObservingSite> = emptyList(),

    /** Informations (section 2.1). */
    val apod: ApodApi.Apod? = null,
    val apodError: String? = null,
    val kp: SpaceWeatherApi.KpData? = null,
    val launches: List<LaunchApi.Launch> = emptyList(),

    val nightMode: Boolean = false,
    val message: String? = null,
) {
    val magnitudeLimit: Double get() = params.limitingMagnitude
    val surfaceBrightnessLimit: Double get() = Formulas.surfaceBrightnessLimit(params.site.bortle)

}

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val container = (application as SkyScoreApplication).container

    private val _state = MutableStateFlow(AppUiState())
    val state: StateFlow<AppUiState> = _state.asStateFlow()

    val observations get() = container.observations
    val starCatalog get() = container.stars

    init {
        // Les preferences se lisent de maniere synchrone : l'etat est complet
        // avant que le premier ecran se compose. Plus aucune course entre la
        // restauration et l'onglet Informations, qui reclamait une cle d'API
        // alors qu'une cle etait bien enregistree.
        restoreSettings()
        refreshSkyState()
    }

    // ------------------------------------------------------------- Preferences

    private fun restoreSettings(): StoredSettings {
        val stored = container.settings.read()
        _state.value = _state.value.copy(
            // La session est un fait local : elle est retablie ici meme, sans
            // attendre la moindre lecture en base. L'utilisateur qui s'est
            // connecte une fois reste connecte.
            user = stored.signedInUser,
            params = SessionParams(
                site = stored.site ?: ObservingSite.DEFAULT,
                instrument = stored.instrument,
                diameterMm = stored.diameterMm,
                focalMm = stored.focalMm,
                eyePupilMm = stored.eyePupilMm,
                smartTelescope = SmartTelescope.CATALOG.firstOrNull { it.name == stored.smartModel },
                smartExposureMinutes = stored.smartExposureMinutes,
                catalogs = decodeCatalogs(stored.catalogs),
            ),
            nightMode = stored.nightMode,
        )
        return stored
    }

    private fun decodeCatalogs(raw: String?): Set<Catalog> {
        if (raw.isNullOrBlank()) {
            return setOf(Catalog.MESSIER, Catalog.CALDWELL, Catalog.NGC_IC, Catalog.PLANETS)
        }
        return raw.split(',').mapNotNull { name ->
            Catalog.entries.firstOrNull { it.name == name }
        }.toSet()
    }

    // -------------------------------------------------------- Parametres session

    fun updateParams(transform: (SessionParams) -> SessionParams) {
        val updated = transform(_state.value.params)
        _state.value = _state.value.copy(params = updated)
        container.persistenceScope.launch {
            container.settings.setSession(
                site = updated.site,
                instrument = updated.instrument,
                diameterMm = updated.diameterMm,
                focalMm = updated.focalMm,
                eyePupilMm = updated.eyePupilMm,
                catalogs = updated.catalogs.joinToString(",") { it.name },
                smartModel = updated.smartTelescope?.name,
                smartExposureMinutes = updated.smartExposureMinutes,
            )
        }
        refreshSkyState()
    }

    fun setNightMode(enabled: Boolean) {
        _state.value = _state.value.copy(nightMode = enabled)
        container.persistenceScope.launch { container.settings.setNightMode(enabled) }
    }

    fun dismissMessage() {
        _state.value = _state.value.copy(message = null)
    }

    // ---------------------------------------------------------------- Ephemerides

    /** Recalcule Soleil, Lune et crepuscules pour les parametres courants. */
    fun refreshSkyState() {
        viewModelScope.launch {
            val params = _state.value.params
            val computed = withContext(Dispatchers.Default) {
                val jd = AstroMath.julianDay(params.epochMillis)
                val lat = params.site.latitude
                val lon = params.site.longitude
                val night = Twilight.computeNight(params.epochMillis, lat, lon)
                val moonPos = SolarSystem.moon(jd)
                val lst = AstroMath.localSiderealTimeDeg(jd, lon)
                val horiz = AstroMath.equatorialToHorizontal(
                    moonPos.raDeg, moonPos.decDeg, lat, lst
                )
                val moon = MoonState(
                    raDeg = moonPos.raDeg,
                    decDeg = moonPos.decDeg,
                    altitudeDeg = horiz.altitudeDeg,
                    azimuthDeg = horiz.azimuthDeg,
                    phasePercent = moonPos.illuminatedFraction * 100.0,
                    phaseName = SolarSystem.moonPhaseName(jd),
                )
                val (moonRise, moonSet) = Twilight.riseSet(
                    params.epochMillis, lat, lon, 0.125
                ) { j -> SolarSystem.moon(j).let { com.skyscore.app.astro.RaDec(it.raDeg, it.decDeg) } }
                Triple(night, moon, moonRise to moonSet)
            }
            val (night, moon, moonTimes) = computed
            Log.d(
                "Ephemerides",
                "Soleil a %.1f degres, score de nuit %.0f".format(
                    night.sunAltitudeDeg, Twilight.nightScore(night.sunAltitudeDeg)
                )
            )
            _state.value = _state.value.copy(
                night = night,
                moon = moon,
                sunAltitudeDeg = night.sunAltitudeDeg,
                sunRise = night.sunrise,
                sunSet = night.sunset,
                moonRise = moonTimes.first,
                moonSet = moonTimes.second,
            )
        }
    }

    // --------------------------------------------------------------- Dashboard

    /** Lance le calcul du Top des cibles (section 2.2). */
    fun computeSession() {
        val params = _state.value.params
        _state.value = _state.value.copy(calculating = true, message = null)
        Log.i("Dashboard", "Calcul lance pour ${params.site.name}")
        viewModelScope.launch {
            val conditions = WeatherApi.conditions(
                params.site.latitude, params.site.longitude, params.epochMillis
            )
            Log.d(
                "Meteo",
                "%.0f %% de nuages, vent %.0f km/h, seeing %d"
                    .format(conditions.cloudCoverPercent, conditions.windSpeedKmh, conditions.seeingIndex)
            )

            val deepSky = container.catalogs.deepSky(params.catalogs - Catalog.PLANETS)
            val result = withContext(Dispatchers.Default) {
                val ctx = ScoringEngine.buildContext(params, conditions)
                val solar = if (Catalog.PLANETS in params.catalogs && !params.isSmartMode) {
                    CatalogRepository.solarSystemObjects(ctx.jd)
                } else emptyList()
                ScoringEngine.topTargets(deepSky + solar, ctx, limit = 20) to ctx
            }
            val (top, ctx) = result
            val selected = top.firstOrNull()
            val curve = selected?.let {
                withContext(Dispatchers.Default) { ScoringEngine.altitudeCurve(it.target, ctx) }
            }.orEmpty()

            _state.value = _state.value.copy(
                conditions = conditions,
                topTargets = top,
                selected = selected,
                altitudeCurve = curve,
                calculating = false,
                lastComputedAt = System.currentTimeMillis(),
                moon = ctx.moon,
                sunAltitudeDeg = ctx.sunAltitudeDeg,
                message = if (top.isEmpty()) messageForEmptyResult(ctx) else null,
            )
            Log.i("Dashboard", "Calcul termine : ${top.size} cibles retenues")
        }
    }

    private fun messageForEmptyResult(ctx: ScoringEngine.Context): String = when {
        ctx.conditions.ok && ctx.conditions.cloudCoverPercent > 90.0 ->
            "RG-F-04 : couverture nuageuse superieure a 90 %, aucune observation possible."
        ctx.sunAltitudeDeg > 0.0 ->
            "Le Soleil est au-dessus de l'horizon : seule la Lune peut etre proposee (RG-P-02)."
        ctx.limits.deepSkyLimit == null ->
            "Le ciel est encore trop clair pour le ciel profond (section 4.2). Attendez le crepuscule nautique."
        else ->
            "Aucune cible ne passe les filtres pour ces parametres. Essayez un instrument plus grand ou un ciel plus sombre."
    }

    /**
     * RG-DASH-01 — le clic sur une cible ouvre la fiche detaillee sans
     * recalculer la session.
     */
    fun selectTarget(scored: ScoringEngine.Scored) {
        _state.value = _state.value.copy(selected = scored)
        viewModelScope.launch {
            val params = _state.value.params
            val conditions = _state.value.conditions
            val curve = withContext(Dispatchers.Default) {
                val ctx = ScoringEngine.buildContext(params, conditions)
                ScoringEngine.altitudeCurve(scored.target, ctx)
            }
            _state.value = _state.value.copy(altitudeCurve = curve)
        }
    }

    /** Evalue un objet issu de l'Explorer avec les conditions du soir. */
    fun selectFromExplorer(target: SkyObject) {
        viewModelScope.launch {
            val params = _state.value.params
            val conditions = _state.value.conditions
            val (scored, curve) = withContext(Dispatchers.Default) {
                val ctx = ScoringEngine.buildContext(params, conditions)
                ScoringEngine.score(target, ctx) to ScoringEngine.altitudeCurve(target, ctx)
            }
            _state.value = _state.value.copy(selected = scored, altitudeCurve = curve)
        }
    }

    // ---------------------------------------------------------------- Explorer

    fun setExplorerQuery(query: String) {
        _state.value = _state.value.copy(explorerQuery = query)
    }

    fun toggleExplorerCatalog(catalog: Catalog) {
        val current = _state.value.explorerCatalogs
        val updated = if (catalog in current) current - catalog else current + catalog
        _state.value = _state.value.copy(explorerCatalogs = updated)
    }

    fun toggleExplorerType(type: ObjectType) {
        val current = _state.value.explorerTypes
        val updated = if (type in current) current - type else current + type
        _state.value = _state.value.copy(explorerTypes = updated)
    }

    /** Recherche libre dans les catalogues (section 2.3). */
    fun searchExplorer() {
        val s = _state.value
        _state.value = s.copy(explorerSearching = true)
        viewModelScope.launch {
            val pool = container.catalogs.deepSky(s.explorerCatalogs)
            val query = s.explorerQuery.trim().lowercase()
            val results = withContext(Dispatchers.Default) {
                pool.asSequence()
                    .filter { it.type in s.explorerTypes }
                    .filter { obj ->
                        query.isBlank() ||
                            obj.id.lowercase().contains(query) ||
                            obj.designation.lowercase().contains(query) ||
                            obj.commonName.lowercase().contains(query)
                    }
                    // Resultats tries par magnitude, les plus brillants d'abord.
                    .sortedBy { it.magnitude ?: Double.MAX_VALUE }
                    .take(200)
                    .toList()
            }
            _state.value = _state.value.copy(explorerResults = results, explorerSearching = false)
        }
    }

    // ----------------------------------------------------------------- Commune

    fun setCommuneQuery(query: String) {
        _state.value = _state.value.copy(communeQuery = query)
        if (query.trim().length < 2) {
            _state.value = _state.value.copy(communeResults = emptyList())
            return
        }
        viewModelScope.launch {
            val results = container.communes.search(query)
            // La saisie a pu changer pendant le chargement du fichier.
            if (_state.value.communeQuery == query) {
                _state.value = _state.value.copy(communeResults = results)
            }
        }
    }

    fun selectCommune(site: ObservingSite) {
        _state.value = _state.value.copy(communeResults = emptyList(), communeQuery = "")
        updateParams { it.copy(site = site) }
    }

    /**
     * Position GPS de l'appareil, rattachee hors ligne a la commune la plus
     * proche : le nom, le departement et l'indice de Bortle en decoulent, mais
     * les coordonnees retenues restent celles du telephone.
     */
    fun useDeviceLocation(latitude: Double, longitude: Double) {
        viewModelScope.launch {
            val commune = container.communes.nearest(latitude, longitude)
            val site = commune?.copy(latitude = latitude, longitude = longitude)
                ?: ObservingSite(
                    name = "Position GPS",
                    latitude = latitude,
                    longitude = longitude,
                    bortle = _state.value.params.site.bortle,
                )
            updateParams { it.copy(site = site) }
        }
    }

    fun setBortle(bortle: Int) {
        updateParams {
            it.copy(site = it.site.copy(bortle = bortle.coerceIn(1, 9), bortleEstimated = false))
        }
    }

    // ---------------------------------------------------------------- Sessions

    /** Pre-remplit le formulaire d'observation depuis la session Dashboard. */
    fun observationDraft(): ObservationEntity? {
        val s = _state.value
        val user = s.user ?: return null
        val target = s.selected ?: s.topTargets.firstOrNull() ?: return null
        return ObservationEntity(
            username = user.username,
            observationDate = s.params.epochMillis,
            objectName = target.target.displayName,
            site = s.params.site.name,
            instrument = s.params.smartTelescope?.name
                ?: "${s.params.instrument.label} ${s.params.effectiveDiameterMm.toInt()} mm",
            bortle = s.params.site.bortle,
            moonPhasePercent = s.moon?.phasePercent ?: 0.0,
            cloudCoverPercent = s.conditions.cloudCoverPercent,
            seeingIndex = s.conditions.seeingIndex,
            score = target.score,
            notes = "",
            objectType = target.target.type.label,
        )
    }

    fun saveObservation(observation: ObservationEntity) {
        viewModelScope.launch {
            container.observations.save(observation)
            _state.value = _state.value.copy(message = "Observation enregistree dans le carnet.")
        }
    }

    fun deleteObservation(id: Long) {
        val user = _state.value.user ?: return
        viewModelScope.launch { container.observations.delete(id, user.username) }
    }

    // ------------------------------------------------------------ Informations

    fun loadInformationTab() {
        viewModelScope.launch { loadApod() }
        viewModelScope.launch {
            _state.value = _state.value.copy(kp = SpaceWeatherApi.kp())
        }
        viewModelScope.launch {
            _state.value = _state.value.copy(launches = LaunchApi.upcoming(5))
        }
    }

    private suspend fun loadApod() {
        ApodApi.today()
            .onSuccess { _state.value = _state.value.copy(apod = it, apodError = null) }
            .onFailure {
                _state.value = _state.value.copy(
                    apod = null, apodError = it.message ?: ApodApi.UNAVAILABLE_MESSAGE
                )
            }
    }

    // ----------------------------------------------------------------- Comptes

    fun login(username: String, password: String, onResult: (String?) -> Unit) {
        viewModelScope.launch {
            container.auth.login(username, password)
                .onSuccess { entity ->
                    val session = SignedInUser(
                        username = entity.username,
                        firstName = entity.firstName,
                        lastName = entity.lastName,
                        createdAt = entity.createdAt,
                    )
                    container.persistenceScope.launch {
                        container.settings.setSignedInUser(session)
                    }
                    _state.value = _state.value.copy(user = session)
                    onResult(null)
                }
                .onFailure { onResult(it.message) }
        }
    }

    fun register(
        username: String,
        password: String,
        firstName: String,
        lastName: String,
        onResult: (Result<AuthRepository.Registration>) -> Unit,
    ) {
        viewModelScope.launch {
            val result = container.auth.register(username, password, firstName, lastName)
            result.onSuccess { registration ->
                val session = SignedInUser(
                    username = registration.user.username,
                    firstName = registration.user.firstName,
                    lastName = registration.user.lastName,
                    createdAt = registration.user.createdAt,
                )
                container.persistenceScope.launch {
                    container.settings.setSignedInUser(session)
                }
                _state.value = _state.value.copy(user = session)
            }
            onResult(result)
        }
    }

    fun resetPassword(
        username: String,
        recoveryCode: String,
        newPassword: String,
        onResult: (String?) -> Unit,
    ) {
        viewModelScope.launch {
            container.auth.resetPassword(username, recoveryCode, newPassword)
                .onSuccess { onResult(null) }
                .onFailure { onResult(it.message) }
        }
    }

    fun logout() {
        container.persistenceScope.launch { container.settings.setSignedInUser(null) }
        _state.value = _state.value.copy(user = null)
    }

    companion object {
        val Factory: ViewModelProvider.Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: androidx.lifecycle.viewmodel.CreationExtras,
            ): T {
                val app = extras[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY]!!
                return AppViewModel(app) as T
            }
        }
    }
}

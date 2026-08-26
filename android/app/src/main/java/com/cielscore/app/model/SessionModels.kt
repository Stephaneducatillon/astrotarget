package com.cielscore.app.model

import com.cielscore.app.catalog.Catalog
import com.cielscore.app.scoring.Formulas

/** Types d'instrument proposes au Dashboard (section 2.2). */
enum class InstrumentType(val label: String) {
    TELESCOPE("Telescope"),
    REFRACTOR("Lunette"),
    BINOCULARS("Jumelles"),
    NAKED_EYE("Oeil nu"),
    SMART_TELESCOPE("Smart telescope"),
}

/** Lieu d'observation : commune choisie ou position GPS. */
data class ObservingSite(
    val name: String,
    val latitude: Double,
    val longitude: Double,
    val bortle: Int,
    val department: String = "",
    /** Vrai lorsque l'indice de Bortle est une estimation et non un choix de l'utilisateur. */
    val bortleEstimated: Boolean = true,
) {
    companion object {
        /**
         * RG-INFO-01 — sans commune selectionnee, l'application retient une
         * estimation France en Bortle 5.
         */
        val DEFAULT = ObservingSite(
            name = "France (estimation)",
            latitude = 46.6,
            longitude = 2.5,
            bortle = 5,
            bortleEstimated = true,
        )
    }
}

/**
 * Parametres de session du Dashboard (section 2.2), valeurs par defaut comprises.
 */
data class SessionParams(
    val site: ObservingSite = ObservingSite.DEFAULT,
    val instrument: InstrumentType = InstrumentType.TELESCOPE,
    /** Diametre en millimetres, de 50 a 600 (defaut 130). */
    val diameterMm: Double = 130.0,
    /** Focale en millimetres, de 300 a 3000 (defaut 650). */
    val focalMm: Double = 650.0,
    /** Pupille maximale de l'oeil, de 4 a 8 mm (defaut 6). */
    val eyePupilMm: Double = 6.0,
    /** Instant de la session, en millisecondes UTC. */
    val epochMillis: Long = System.currentTimeMillis(),
    val catalogs: Set<Catalog> = setOf(
        Catalog.MESSIER, Catalog.CALDWELL, Catalog.NGC_IC, Catalog.PLANETS
    ),
    /** Modele de smart telescope actif, ou null hors mode smart telescope. */
    val smartTelescope: SmartTelescope? = null,
    /** Duree de pose cumulee en minutes, utilisee par RG-I-05. */
    val smartExposureMinutes: Double = 60.0,
) {
    val isSmartMode: Boolean get() = smartTelescope != null

    /** Diametre effectivement utilise pour les calculs. */
    val effectiveDiameterMm: Double
        get() = when {
            smartTelescope != null -> smartTelescope.diameterMm
            instrument == InstrumentType.NAKED_EYE -> Formulas.NAKED_EYE_MAX_DIAMETER_MM
            else -> diameterMm
        }

    val effectiveFocalMm: Double
        get() = smartTelescope?.focalMm ?: focalMm

    /**
     * Magnitude limite de l'instrument (section 5.3), majoree pour un smart
     * telescope selon la duree de pose cumulee (RG-I-05).
     */
    val limitingMagnitude: Double
        get() = smartTelescope?.let {
            Formulas.smartTelescopeLimitingMagnitude(
                it.diameterMm, smartExposureMinutes * 60.0, site.bortle
            )
        } ?: Formulas.limitingMagnitude(effectiveDiameterMm, site.bortle)

    val focalRatio: Double
        get() = Formulas.focalRatio(effectiveFocalMm, effectiveDiameterMm)
}

/** Conditions de ciel issues d'Open-Meteo (sections 2.2, 8.2 et 8.4). */
data class SkyConditions(
    /** Couverture nuageuse en pourcentage. */
    val cloudCoverPercent: Double = 50.0,
    val windSpeedKmh: Double = 10.0,
    val humidityPercent: Double = 70.0,
    val visibilityMeters: Double = 20_000.0,
    val temperatureCelsius: Double = 10.0,
    /**
     * Faux lorsque Open-Meteo est indisponible : l'application retombe alors sur
     * 50 % de nuages et un seeing de 3 (section 8.4).
     */
    val ok: Boolean = false,
) {
    val seeingIndex: Int get() = if (ok) Formulas.seeingIndex(windSpeedKmh) else 3
    val seeingLabel: String get() = Formulas.seeingLabel(seeingIndex)

    companion object {
        /** Strategie de repli de la section 8.4. */
        val FALLBACK = SkyConditions(cloudCoverPercent = 50.0, windSpeedKmh = 10.0, ok = false)
    }
}

/** Etat de la Lune au moment de la session (sections 4.4 et 6.1). */
data class MoonState(
    val raDeg: Double,
    val decDeg: Double,
    val altitudeDeg: Double,
    val azimuthDeg: Double,
    /** Fraction illuminee, en pourcentage. */
    val phasePercent: Double,
    val phaseName: String,
) {
    /** RG-L-01 — Lune sous l'horizon : impact nul sur le score. */
    val isBelowHorizon: Boolean get() = altitudeDeg < 0.0
}

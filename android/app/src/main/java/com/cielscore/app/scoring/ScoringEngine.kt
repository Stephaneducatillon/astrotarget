package com.cielscore.app.scoring

import com.cielscore.app.astro.AstroMath
import com.cielscore.app.astro.SolarSystem
import com.cielscore.app.astro.Twilight
import com.cielscore.app.catalog.Catalog
import com.cielscore.app.catalog.ObjectType
import com.cielscore.app.catalog.SkyObject
import com.cielscore.app.model.MoonState
import com.cielscore.app.model.SessionParams
import com.cielscore.app.model.SkyConditions
import kotlin.math.abs

/**
 * Moteur de score — sections 4 (regles de gestion) et 6 (scoring).
 */
object ScoringEngine {

    /** Codes des filtres eliminatoires de la section 4.1. */
    enum class Rejection(val code: String, val reason: String) {
        LOW_ALTITUDE("RG-F-01", "Altitude inferieure a 5 degres"),
        SURFACE_BRIGHTNESS("RG-F-02", "Objet trop diffus pour emerger du fond de ciel"),
        MAGNITUDE("RG-F-03", "Trop faible pour l'instrument ou le ciel"),
        CLOUDS("RG-F-04", "Couverture nuageuse superieure a 90 %"),
        NO_ANGULAR_SIZE("RG-F-02", "Dimensions angulaires inconnues"),
        SOLAR_SYSTEM_RULE("RG-P", "Astre non proposable a cette heure"),
        SMART_MODE_PLANET("RG-P-05", "Planetes desactivees en mode smart telescope"),
    }

    /** Detail des huit criteres ponderes (section 6.1). */
    data class Breakdown(
        val altitude: Double = 0.0,
        val window: Double = 0.0,
        val seeing: Double = 0.0,
        val transparency: Double = 0.0,
        val bortle: Double = 0.0,
        val moon: Double = 0.0,
        val surfaceBrightness: Double = 0.0,
        val night: Double = 0.0,
        /** Criteres propres au mode smart telescope (section 6.4). */
        val focalRatio: Double = 0.0,
        val fieldMatch: Double = 0.0,
        /** Critere de magnitude, propre aux planetes (section 6.3). */
        val magnitude: Double = 0.0,
    )

    /** Un objet evalue pour la session en cours. */
    data class Scored(
        val target: SkyObject,
        val score: Double,
        val altitudeDeg: Double,
        val azimuthDeg: Double,
        /** Duree, en minutes, pendant laquelle l'objet reste exploitable. */
        val windowMinutes: Double,
        val surfaceBrightness: Double?,
        val moonDistanceDeg: Double,
        val breakdown: Breakdown,
        val rejection: Rejection? = null,
    ) {
        val isObservable: Boolean get() = rejection == null && score > 0.0
    }

    /** Contexte partage par tous les objets d'une meme session. */
    data class Context(
        val params: SessionParams,
        val conditions: SkyConditions,
        val moon: MoonState,
        val sunAltitudeDeg: Double,
        val limits: Formulas.DarknessLimits,
        val lstDeg: Double,
        val jd: Double,
        /** Instants d'echantillonnage de la fenetre d'observation. */
        val sampleLstDeg: DoubleArray,
        val sampleSunAltDeg: DoubleArray,
        val sampleStepMinutes: Double,
    )

    /**
     * Duree, en heures, sur laquelle est evaluee la fenetre d'observation.
     * La section 2.2 trace la courbe d'altitude « sur 10 h » : la meme fenetre
     * sert de base au critere « Fenetre » de la section 6.1.
     */
    const val WINDOW_HOURS = 10

    /**
     * Altitude a partir de laquelle un objet est considere exploitable.
     * Section 2.2 : « seuil optimal 30 degres », egalement le plateau du critere
     * d'altitude de la section 6.1.
     */
    const val OPTIMAL_ALTITUDE_DEG = 30.0

    private const val SAMPLE_STEP_MINUTES = 10.0

    /** Prepare le contexte de session : Soleil, Lune, temps sideral, echantillons. */
    fun buildContext(params: SessionParams, conditions: SkyConditions): Context {
        val jd = AstroMath.julianDay(params.epochMillis)
        val lat = params.site.latitude
        val lon = params.site.longitude

        val moonPos = SolarSystem.moon(jd)
        val lst = AstroMath.localSiderealTimeDeg(jd, lon)
        val moonHoriz = AstroMath.equatorialToHorizontal(moonPos.raDeg, moonPos.decDeg, lat, lst)
        val moon = MoonState(
            raDeg = moonPos.raDeg,
            decDeg = moonPos.decDeg,
            altitudeDeg = moonHoriz.altitudeDeg,
            azimuthDeg = moonHoriz.azimuthDeg,
            phasePercent = moonPos.illuminatedFraction * 100.0,
            phaseName = SolarSystem.moonPhaseName(jd),
        )

        val sunAlt = SolarSystem.sunAltitudeDeg(jd, lat, lon)
        val limits = Formulas.darknessLimits(sunAlt, params.limitingMagnitude)

        val sampleCount = (WINDOW_HOURS * 60 / SAMPLE_STEP_MINUTES).toInt()
        val sampleLst = DoubleArray(sampleCount)
        val sampleSun = DoubleArray(sampleCount)
        for (i in 0 until sampleCount) {
            val ms = params.epochMillis + (i * SAMPLE_STEP_MINUTES * 60_000L).toLong()
            val sampleJd = AstroMath.julianDay(ms)
            sampleLst[i] = AstroMath.localSiderealTimeDeg(sampleJd, lon)
            sampleSun[i] = SolarSystem.sunAltitudeDeg(sampleJd, lat, lon)
        }

        return Context(
            params = params,
            conditions = conditions,
            moon = moon,
            sunAltitudeDeg = sunAlt,
            limits = limits,
            lstDeg = lst,
            jd = jd,
            sampleLstDeg = sampleLst,
            sampleSunAltDeg = sampleSun,
            sampleStepMinutes = SAMPLE_STEP_MINUTES,
        )
    }

    /** Evalue un objet dans le contexte donne. */
    fun score(target: SkyObject, ctx: Context): Scored {
        val lat = ctx.params.site.latitude
        val horiz = AstroMath.equatorialToHorizontal(
            target.raDeg, target.decDeg, lat, ctx.lstDeg
        )
        val alt = horiz.altitudeDeg
        val moonDistance = AstroMath.angularSeparationDeg(
            target.raDeg, target.decDeg, ctx.moon.raDeg, ctx.moon.decDeg
        )
        val sb = target.surfaceBrightness

        fun rejected(rule: Rejection) = Scored(
            target = target,
            score = 0.0,
            altitudeDeg = alt,
            azimuthDeg = horiz.azimuthDeg,
            windowMinutes = 0.0,
            surfaceBrightness = sb,
            moonDistanceDeg = moonDistance,
            breakdown = Breakdown(),
            rejection = rule,
        )

        // ---- RG-F-04 : ciel bouche, aucune observation possible.
        if (ctx.conditions.ok && ctx.conditions.cloudCoverPercent > 90.0) {
            return rejected(Rejection.CLOUDS)
        }

        return if (target.isSolarSystem) scoreSolarSystem(target, ctx, alt, horiz.azimuthDeg, moonDistance)
        else scoreDeepSky(target, ctx, alt, horiz.azimuthDeg, moonDistance, sb)
    }

    // ------------------------------------------------------------ Ciel profond

    private fun scoreDeepSky(
        target: SkyObject,
        ctx: Context,
        alt: Double,
        az: Double,
        moonDistance: Double,
        sb: Double?,
    ): Scored {
        fun rejected(rule: Rejection) = Scored(
            target, 0.0, alt, az, 0.0, sb, moonDistance, Breakdown(), rule
        )

        // RG-F-01 : sous l'horizon ou dans la turbulence basse.
        if (alt < 5.0) return rejected(Rejection.LOW_ALTITUDE)

        // Sans dimensions angulaires, ni RG-F-02 ni le critere « Brillance surf. »
        // ne peuvent etre evalues : l'objet reste consultable dans l'Explorer
        // mais n'entre pas dans le classement du Dashboard.
        if (sb == null) return rejected(Rejection.NO_ANGULAR_SIZE)

        // RG-F-03 : filtrage dynamique selon l'obscurite (section 4.2), conserve
        // en v2.0 : la magnitude limite retenue depend de la hauteur du Soleil.
        val magLimit = ctx.limits.deepSkyLimit ?: return rejected(Rejection.MAGNITUDE)
        val mag = target.magnitude ?: return rejected(Rejection.MAGNITUDE)
        if (mag > magLimit) return rejected(Rejection.MAGNITUDE)

        // RG-F-02, reecrit par les regles v2.0 (§ 4 et § 7) : un objet diffus
        // n'est ecarte que s'il est a la fois noye dans le fond de ciel et hors
        // d'atteinte en magnitude integree. Le correctif des objets brillants
        // etendus (M31, M42, M45) tient dans cette seule conjonction.
        val sbAssessment = assessSurfaceBrightness(
            type = target.type,
            surfaceBrightness = sb,
            magnitude = mag,
            limitingMagnitude = ctx.params.limitingMagnitude,
            bortle = ctx.params.site.bortle,
        )
        if (sbAssessment.factor <= 0.0 && !sbAssessment.magnitudeAccessible) {
            return rejected(Rejection.SURFACE_BRIGHTNESS)
        }

        val windowMinutes = observationWindowMinutes(target, ctx)

        return if (ctx.params.isSmartMode) {
            smartScore(target, ctx, alt, az, moonDistance, sb, sbAssessment, windowMinutes)
        } else {
            visualScore(target, ctx, alt, az, moonDistance, sb, sbAssessment, windowMinutes)
        }
    }

    /**
     * Section 6.1 — score visuel, huit criteres ponderes :
     *
     *     Score = 0.25*alt + 0.15*fenetre + 0.11*seeing + 0.13*transparence
     *           + 0.08*bortle + 0.06*lune + 0.15*SB + 0.07*nuit
     */
    private fun visualScore(
        target: SkyObject,
        ctx: Context,
        alt: Double,
        az: Double,
        moonDistance: Double,
        sb: Double,
        sbAssessment: SurfaceBrightnessAssessment,
        windowMinutes: Double,
    ): Scored {
        val b = Breakdown(
            altitude = altitudeScore(alt),
            window = windowScore(windowMinutes),
            seeing = seeingScore(ctx.conditions.seeingIndex),
            transparency = transparencyScore(ctx.conditions),
            bortle = bortleScore(ctx.params.site.bortle),
            moon = moonScore(ctx.moon, moonDistance),
            surfaceBrightness = sbAssessment.score,
            night = Twilight.nightScore(ctx.sunAltitudeDeg),
        )
        val weighted = 0.25 * b.altitude + 0.15 * b.window + 0.11 * b.seeing +
            0.13 * b.transparency + 0.08 * b.bortle + 0.06 * b.moon +
            0.15 * b.surfaceBrightness + 0.07 * b.night
        val score = weighted * sbAssessment.factor
        return Scored(target, score.coerceIn(0.0, 100.0), alt, az, windowMinutes, sb, moonDistance, b)
    }

    /**
     * Section 6.4 — score smart telescope :
     *
     *     Score = 0.25*alt + 0.20*transp. + 0.15*seeing + 0.15*bortle
     *           + 0.15*lune + 0.05*F/D + 0.05*champ
     *
     * Regles v2.0, § 6 — en mode smart telescope le filtre de brillance de
     * surface reste applique en amont, mais f_sb n'intervient pas comme
     * multiplicateur du score final : la pose longue integre le flux.
     */
    private fun smartScore(
        target: SkyObject,
        ctx: Context,
        alt: Double,
        az: Double,
        moonDistance: Double,
        sb: Double,
        sbAssessment: SurfaceBrightnessAssessment,
        windowMinutes: Double,
    ): Scored {
        val scope = ctx.params.smartTelescope!!
        val b = Breakdown(
            altitude = altitudeScore(alt),
            transparency = transparencyScore(ctx.conditions),
            seeing = seeingScore(ctx.conditions.seeingIndex),
            bortle = bortleScore(ctx.params.site.bortle),
            moon = moonScore(ctx.moon, moonDistance),
            focalRatio = focalRatioScore(scope.focalRatio),
            fieldMatch = fieldMatchScore(target.sizeArcmin, scope.fieldWidthArcmin),
            surfaceBrightness = sbAssessment.score,
            window = windowScore(windowMinutes),
            night = Twilight.nightScore(ctx.sunAltitudeDeg),
        )
        val score = 0.25 * b.altitude + 0.20 * b.transparency + 0.15 * b.seeing +
            0.15 * b.bortle + 0.15 * b.moon + 0.05 * b.focalRatio + 0.05 * b.fieldMatch
        return Scored(target, score.coerceIn(0.0, 100.0), alt, az, windowMinutes, sb, moonDistance, b)
    }

    // --------------------------------------------------------- Systeme solaire

    /**
     * Section 6.3 — score des planetes :
     *
     *     Score = 0.40*altitude + 0.30*magnitude + 0.20*transparence + 0.10*lune
     *
     * Les regles RG-P-01 a RG-P-05 de la section 4.3 s'appliquent en amont.
     */
    private fun scoreSolarSystem(
        target: SkyObject,
        ctx: Context,
        alt: Double,
        az: Double,
        moonDistance: Double,
    ): Scored {
        fun rejected(rule: Rejection) = Scored(
            target, 0.0, alt, az, 0.0, null, moonDistance, Breakdown(), rule
        )

        val body = target.body!!
        val isMoon = body == SolarSystem.Body.MOON

        // RG-P-05 : planetes desactivees en mode smart telescope.
        if (ctx.params.isSmartMode) return rejected(Rejection.SMART_MODE_PLANET)

        // RG-P-02 : la Lune est proposee des 2 degres d'altitude, meme de jour.
        // Les autres corps restent soumis a RG-F-01.
        val minAltitude = if (isMoon) 2.0 else 5.0
        if (alt < minAltitude) return rejected(Rejection.LOW_ALTITUDE)

        val mag = target.magnitude
        if (!isMoon) {
            if (mag == null || mag.isNaN()) return rejected(Rejection.MAGNITUDE)
            val sunAlt = ctx.sunAltitudeDeg
            // RG-P-03 : Venus et Jupiter (mag < -1) apparaissent des le crepuscule
            // civil, a partir de -3 degres de hauteur du Soleil.
            // RG-P-04 : les autres planetes attendent le crepuscule nautique (-6).
            val threshold = if (mag < -1.0) -3.0 else -6.0
            if (sunAlt > threshold) return rejected(Rejection.SOLAR_SYSTEM_RULE)

            // Filtrage dynamique de la section 4.2 lorsqu'une limite est active.
            val planetLimit = ctx.limits.planetLimit
            if (planetLimit != null && mag > planetLimit) return rejected(Rejection.MAGNITUDE)
        }

        val magnitudeScore = if (mag == null || mag.isNaN()) 0.0
        else Formulas.clip((8.0 - mag) / 8.0 * 80.0 + 20.0, 0.0, 100.0)

        val b = Breakdown(
            altitude = altitudeScore(alt),
            magnitude = magnitudeScore,
            transparency = transparencyScore(ctx.conditions),
            // Section 6.3 : 100 - phase, sauf si l'objet est la Lune.
            moon = if (isMoon) 100.0 else 100.0 - ctx.moon.phasePercent,
            night = Twilight.nightScore(ctx.sunAltitudeDeg),
            seeing = seeingScore(ctx.conditions.seeingIndex),
            bortle = bortleScore(ctx.params.site.bortle),
        )
        val score = 0.40 * b.altitude + 0.30 * b.magnitude + 0.20 * b.transparency + 0.10 * b.moon
        val windowMinutes = observationWindowMinutes(target, ctx)
        return Scored(target, score.coerceIn(0.0, 100.0), alt, az, windowMinutes, null, moonDistance, b)
    }

    // -------------------------------------------------------- Criteres unitaires

    /** Altitude, 25 % : clip((alt-5)/25, 0, 1) * 100 — plateau atteint a 30 degres. */
    fun altitudeScore(altitudeDeg: Double): Double =
        Formulas.clip((altitudeDeg - 5.0) / 25.0) * 100.0

    /** Fenetre, 15 % : min(duree_min/240, 1) * 100 — maximum a 4 heures. */
    fun windowScore(durationMinutes: Double): Double =
        kotlin.math.min(durationMinutes / 240.0, 1.0) * 100.0

    /** Seeing, 11 % : (seeing-1)/4 * 100, indice 1 a 5 deduit du vent. */
    fun seeingScore(seeingIndex: Int): Double = (seeingIndex - 1) / 4.0 * 100.0

    /** Transparence, 13 % : 100 - nuages_%. */
    fun transparencyScore(conditions: SkyConditions): Double =
        (100.0 - conditions.cloudCoverPercent).coerceIn(0.0, 100.0)

    /** Bortle, 8 % : (9-Bortle)/8 * 100. */
    fun bortleScore(bortle: Int): Double = (9.0 - bortle.coerceIn(1, 9)) / 8.0 * 100.0

    /**
     * Lune, 6 % : (1 - phase/100 * (1 - dist/180)) * 100.
     *
     * RG-L-01 — Lune sous l'horizon : impact nul, quelle que soit la phase.
     * RG-L-02 — l'impact combine phase et distance angulaire a l'objet.
     * RG-L-03 — pleine Lune proche de l'objet : penalite maximale.
     * RG-L-04 — nouvelle Lune : aucune penalite.
     */
    fun moonScore(moon: MoonState, distanceDeg: Double): Double {
        if (moon.isBelowHorizon) return 100.0
        val d = distanceDeg.coerceIn(0.0, 180.0)
        return ((1.0 - moon.phasePercent / 100.0 * (1.0 - d / 180.0)) * 100.0).coerceIn(0.0, 100.0)
    }

    /**
     * Resultat du critere de brillance de surface des regles v2.0 (§ 7).
     *
     * @param score  sous-score s_sb, sur 100, pesant 15 % du score visuel.
     * @param factor multiplicateur f_sb applique au score visuel final ; il vaut
     *   toujours 1 pour un amas resolu ou un objet rattrape par le correctif des
     *   objets brillants etendus.
     * @param magnitudeAccessible vrai si la magnitude integree suffit a elle
     *   seule a rendre l'objet visible (mag < mag_limite - 2).
     */
    data class SurfaceBrightnessAssessment(
        val score: Double,
        val factor: Double,
        val magnitudeAccessible: Boolean,
    )

    /**
     * Critere de brillance de surface, 15 % — regles v2.0, § 7.
     *
     * Le probleme corrige : la brillance de surface repartit le flux sur toute
     * l'etendue angulaire de l'objet. Pour M31 (190' x 60'), la SB calculee
     * atteint 22.2 mag/arcsec2, au-dela du plafond d'un ciel Bortle 7 ; l'oeil,
     * lui, integre le flux total et la galaxie reste visible. La version 1.0
     * lui attribuait donc un score nul.
     *
     *     f_sb        = clip((sb_lim + 3.5 - sb) / 3.5, 0, 1)
     *     mag_access. = magnitude < mag_lim - 2
     *
     *     amas resolu    ->  s_sb = 100, f_sb = 1  (TYPES_RESOLUS, § 1)
     *     sinon          ->  s_sb_diff = clip((sb_lim + 3.5 - sb) / 8.5, 0, 1) * 100
     *       si mag_access.  s_sb_mag  = clip((mag_lim - magnitude) / 6, 0, 1) * 100
     *                       poids_mag = clip((mag_lim - 2 - magnitude) / (mag_lim - 2), 0, 1)
     *                       s_sb      = s_sb_diff * (1 - poids) + s_sb_mag * poids
     *                       f_sb      = 1
     *       sinon           s_sb = s_sb_diff, f_sb conserve sa valeur
     *
     * @param limitingMagnitude magnitude limite INSTRUMENTALE, et non la limite
     *   crepusculaire de la section 4.2 : le correctif juge de ce que
     *   l'instrument peut atteindre, le filtrage nocturne restant un filtre
     *   d'exclusion distinct, applique en amont.
     */
    fun assessSurfaceBrightness(
        type: ObjectType,
        surfaceBrightness: Double,
        magnitude: Double?,
        limitingMagnitude: Double,
        bortle: Int,
    ): SurfaceBrightnessAssessment {
        val magnitudeAccessible = magnitude != null && magnitude < limitingMagnitude - 2.0

        // TYPES_RESOLUS : un amas est resolu en etoiles, la SB moyenne est sans objet.
        if (type.isResolved) {
            return SurfaceBrightnessAssessment(100.0, 1.0, magnitudeAccessible)
        }

        val sbLimit = Formulas.surfaceBrightnessLimit(bortle)
        val ceiling = sbLimit + Formulas.SB_TOLERANCE
        val diffuseScore = Formulas.clip(
            (ceiling - surfaceBrightness) / (5.0 + Formulas.SB_TOLERANCE)
        ) * 100.0
        val factor = Formulas.surfaceBrightnessFactor(surfaceBrightness, bortle)

        if (!magnitudeAccessible) {
            return SurfaceBrightnessAssessment(diffuseScore, factor, false)
        }

        val mag = magnitude!!
        val magScore = Formulas.clip((limitingMagnitude - mag) / 6.0) * 100.0
        val threshold = limitingMagnitude - 2.0
        val weight = if (threshold <= 0.0) 1.0 else Formulas.clip((threshold - mag) / threshold)
        val blended = diffuseScore * (1.0 - weight) + magScore * weight
        return SurfaceBrightnessAssessment(blended, 1.0, true)
    }

    /**
     * Critere F/D du score smart telescope (section 6.4, 5 %).
     *
     * INTERPRETATION — la documentation enonce le principe (« bonus F/D court :
     * un rapport F/D faible collecte la lumiere plus vite ») sans donner de
     * formule. Retenu ici : score plein a F/D 2 et nul a partir de F/D 8, ce qui
     * couvre la plage reelle des smart telescopes (F/2.2 a F/5).
     */
    fun focalRatioScore(focalRatio: Double): Double =
        Formulas.clip((8.0 - focalRatio) / 6.0) * 100.0

    /**
     * Critere d'adequation champ / objet du score smart telescope
     * (section 6.4, 5 %).
     *
     * INTERPRETATION — la documentation enonce le principe (« grand champ
     * favorable aux nebuleuses etendues ») sans donner de formule. Retenu ici :
     * un objet occupant la moitie du champ obtient 100, un objet ponctuel ou
     * plus grand que le champ obtient 0, avec une variation lineaire entre les
     * deux.
     */
    fun fieldMatchScore(objectSizeArcmin: Double?, fieldArcmin: Double): Double {
        if (objectSizeArcmin == null || objectSizeArcmin <= 0.0 || fieldArcmin <= 0.0) return 0.0
        val ratio = objectSizeArcmin / fieldArcmin
        return when {
            ratio >= 1.0 -> 0.0
            ratio <= 0.5 -> Formulas.clip(ratio / 0.5) * 100.0
            else -> Formulas.clip((1.0 - ratio) / 0.5) * 100.0
        }
    }

    /**
     * Duree, en minutes, pendant laquelle l'objet reste exploitable.
     *
     * INTERPRETATION — la section 6.1 definit le critere « Fenetre » par
     * min(duree_min/240, 1) sans preciser ce que mesure duree_min. Retenu ici :
     * le temps passe au-dessus du seuil optimal de 30 degres (section 2.2)
     * pendant que le Soleil est sous l'horizon, evalue sur les 10 heures qui
     * suivent l'heure de session — la meme fenetre que la courbe d'altitude.
     */
    fun observationWindowMinutes(target: SkyObject, ctx: Context): Double {
        val lat = ctx.params.site.latitude
        var minutes = 0.0
        for (i in ctx.sampleLstDeg.indices) {
            if (ctx.sampleSunAltDeg[i] >= 0.0) continue
            val alt = AstroMath.equatorialToHorizontal(
                target.raDeg, target.decDeg, lat, ctx.sampleLstDeg[i]
            ).altitudeDeg
            if (alt >= OPTIMAL_ALTITUDE_DEG) minutes += ctx.sampleStepMinutes
        }
        return minutes
    }

    /**
     * Calcule le Top des cibles observables (section 2.2).
     *
     * @param limit nombre d'objets conserves ; le Dashboard en affiche 20.
     */
    fun topTargets(
        targets: List<SkyObject>,
        ctx: Context,
        limit: Int = 20,
    ): List<Scored> {
        val out = ArrayList<Scored>(256)
        for (t in targets) {
            if (t.catalog == Catalog.PLANETS && ctx.params.isSmartMode) continue
            val scored = score(t, ctx)
            if (scored.isObservable) out.add(scored)
        }
        out.sortByDescending { it.score }
        return if (out.size > limit) out.subList(0, limit).toList() else out
    }

    // ------------------------------------------- Validation deterministe (v2.0)

    /**
     * Motifs de rejet de la validation deterministe des regles v2.0 (§ 8).
     *
     * Ces criteres sont volontairement plus severes que les filtres
     * eliminatoires : ils ne decident pas de l'observabilite d'un objet, mais
     * de son eligibilite a etre propose comme cible du soir a l'assistant IA.
     */
    enum class AiVeto(val reason: String) {
        LOW_ALTITUDE("Altitude inferieure a 20 degres : trop d'atmosphere a traverser"),
        MAGNITUDE("Magnitude au-dela de la limite instrumentale"),
        CLOUDS("Plus de 80 % de couverture nuageuse"),
        MOON_GLARE("Lune a moins de 30 degres et eclairee a plus de 60 %"),
        SURFACE_BRIGHTNESS("Fond de ciel plus lumineux que l'objet (SB > sb_lim + 3.5)"),
    }

    /**
     * Regles v2.0, § 8 — valider_top_ia().
     *
     * Verifie qu'une cible retenue tient debout independamment du score : c'est
     * ce garde-fou qui empeche l'assistant de recommander un objet rasant
     * l'horizon, noye dans les nuages ou colle a une Lune brillante.
     *
     * @return la liste des motifs de rejet ; vide si la cible est validee.
     */
    fun aiVetoes(scored: Scored, ctx: Context): List<AiVeto> {
        val vetoes = ArrayList<AiVeto>(2)
        if (scored.altitudeDeg < 20.0) vetoes.add(AiVeto.LOW_ALTITUDE)

        val mag = scored.target.magnitude
        if (mag == null || mag.isNaN() || mag > ctx.params.limitingMagnitude) {
            vetoes.add(AiVeto.MAGNITUDE)
        }
        if (ctx.conditions.ok && ctx.conditions.cloudCoverPercent > 80.0) {
            vetoes.add(AiVeto.CLOUDS)
        }
        if (!ctx.moon.isBelowHorizon &&
            scored.moonDistanceDeg < 30.0 &&
            ctx.moon.phasePercent > 60.0
        ) {
            vetoes.add(AiVeto.MOON_GLARE)
        }

        val sb = scored.surfaceBrightness
        if (sb != null && !scored.target.type.isResolved) {
            val ceiling = Formulas.surfaceBrightnessLimit(ctx.params.site.bortle) +
                Formulas.SB_TOLERANCE
            if (sb > ceiling) vetoes.add(AiVeto.SURFACE_BRIGHTNESS)
        }
        return vetoes
    }

    /** Vrai si la cible passe la validation deterministe des regles v2.0 (§ 8). */
    fun validateForAi(scored: Scored, ctx: Context): Boolean = aiVetoes(scored, ctx).isEmpty()

    /**
     * Cibles du Top validees par le § 8, dans l'ordre du score.
     *
     * Utilisee pour construire le contexte envoye a l'assistant : celui-ci ne
     * doit proposer que des objets reellement confortables a observer.
     */
    fun validatedTargets(targets: List<Scored>, ctx: Context): List<Scored> =
        targets.filter { validateForAi(it, ctx) }

    /** Courbe d'altitude d'un objet sur les 10 heures suivantes (section 2.2). */
    fun altitudeCurve(
        target: SkyObject,
        ctx: Context,
        points: Int = 121,
    ): List<Pair<Long, Double>> {
        val lat = ctx.params.site.latitude
        val lon = ctx.params.site.longitude
        val totalMs = WINDOW_HOURS * 3_600_000L
        return (0 until points).map { i ->
            val ms = ctx.params.epochMillis + totalMs * i / (points - 1)
            val jd = AstroMath.julianDay(ms)
            val lst = AstroMath.localSiderealTimeDeg(jd, lon)
            val eq = if (target.isSolarSystem) {
                val p = if (target.body == SolarSystem.Body.MOON) SolarSystem.moon(jd)
                else SolarSystem.planet(target.body!!, jd)
                p.raDeg to p.decDeg
            } else target.raDeg to target.decDeg
            ms to AstroMath.equatorialToHorizontal(eq.first, eq.second, lat, lst).altitudeDeg
        }
    }

    /** Vrai si la somme des poids de la section 6.1 vaut bien 100 %. */
    fun visualWeightsSum(): Double =
        0.25 + 0.15 + 0.11 + 0.13 + 0.08 + 0.06 + 0.15 + 0.07

    /** Vrai si la somme des poids de la section 6.4 vaut bien 100 %. */
    fun smartWeightsSum(): Double = 0.25 + 0.20 + 0.15 + 0.15 + 0.15 + 0.05 + 0.05

    /** Vrai si la somme des poids de la section 6.3 vaut bien 100 %. */
    fun planetWeightsSum(): Double = 0.40 + 0.30 + 0.20 + 0.10

    internal fun approximatelyOne(value: Double) = abs(value - 1.0) < 1e-9
}

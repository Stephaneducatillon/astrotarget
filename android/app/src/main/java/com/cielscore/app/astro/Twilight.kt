package com.cielscore.app.astro

/**
 * Crepuscules et nuit astronomique — section 7 de la documentation.
 */
object Twilight {

    /** Phases definies en section 7.1, avec l'habillage d'affichage de la section 7.5. */
    enum class Phase(
        val label: String,
        val message: String,
        /** Couleur ARGB de la section 7.5. */
        val colorArgb: Long,
    ) {
        NIGHT("Nuit astronomique", "Ciel parfaitement noir", 0xFF2E7D32),
        ASTRONOMICAL("Crepuscule astronomique", "Ciel presque noir", 0xFF9CCC65),
        NAUTICAL("Crepuscule nautique", "Attendre la nuit", 0xFFF57C00),
        CIVIL("Crepuscule civil", "Ciel encore trop clair", 0xFFE64A19),
        DAY("Jour", "Observation nocturne impossible", 0xFFC62828),
    }

    /** Phase courante deduite de la hauteur du Soleil (section 7.1). */
    fun phaseOf(sunAltitudeDeg: Double): Phase = when {
        sunAltitudeDeg <= -18.0 -> Phase.NIGHT
        sunAltitudeDeg <= -12.0 -> Phase.ASTRONOMICAL
        sunAltitudeDeg <= -6.0 -> Phase.NAUTICAL
        sunAltitudeDeg <= 0.0 -> Phase.CIVIL
        else -> Phase.DAY
    }

    /**
     * Score de nuit, section 7.2 :
     *
     *     100 si alt <= -18   ->  nuit noire
     *      70 si alt <= -12   ->  crepuscule astronomique
     *      40 si alt <= -6    ->  crepuscule nautique
     *      10 si alt <= 0     ->  crepuscule civil
     *       0 si alt > 0      ->  jour
     */
    fun nightScore(sunAltitudeDeg: Double): Double = when {
        sunAltitudeDeg <= -18.0 -> 100.0
        sunAltitudeDeg <= -12.0 -> 70.0
        sunAltitudeDeg <= -6.0 -> 40.0
        sunAltitudeDeg <= 0.0 -> 10.0
        else -> 0.0
    }

    /** Intervalle horaire ; [startMillis] et [endMillis] sont nuls si l'evenement n'existe pas. */
    data class Window(val startMillis: Long?, val endMillis: Long?) {
        val exists: Boolean get() = startMillis != null && endMillis != null
        val durationMinutes: Double
            get() = if (exists) (endMillis!! - startMillis!!) / 60_000.0 else 0.0
    }

    /** Les quatre fenetres crepusculaires d'une nuit, plus lever et coucher du Soleil. */
    data class NightInfo(
        val sunset: Long?,
        val sunrise: Long?,
        val civil: Window,
        val nautical: Window,
        val astronomical: Window,
        val currentPhase: Phase,
        val sunAltitudeDeg: Double,
    )

    /**
     * Calcule les fenetres crepusculaires de la nuit qui suit [referenceMillis].
     *
     * Balayage par pas de deux minutes sur 24 heures, puis affinage par
     * dichotomie sur le passage du seuil. Les fenetres inexistantes (nuits d'ete
     * au-dela de 49 degres de latitude, section 7.4) sont renvoyees vides.
     */
    fun computeNight(
        referenceMillis: Long,
        latitudeDeg: Double,
        longitudeDeg: Double,
    ): NightInfo {
        val stepMs = 2 * 60_000L
        val steps = (24 * 60 / 2)
        val alt = DoubleArray(steps + 1) { idx ->
            SolarSystem.sunAltitudeDeg(
                AstroMath.julianDay(referenceMillis + idx * stepMs), latitudeDeg, longitudeDeg
            )
        }

        fun crossing(threshold: Double, descending: Boolean): Long? {
            for (i in 0 until steps) {
                val a = alt[i]
                val b = alt[i + 1]
                val isCross = if (descending) a > threshold && b <= threshold
                else a <= threshold && b > threshold
                if (isCross) {
                    var lo = referenceMillis + i * stepMs
                    var hi = referenceMillis + (i + 1) * stepMs
                    repeat(20) {
                        val mid = (lo + hi) / 2
                        val m = SolarSystem.sunAltitudeDeg(
                            AstroMath.julianDay(mid), latitudeDeg, longitudeDeg
                        )
                        val below = m <= threshold
                        if (below == descending) hi = mid else lo = mid
                    }
                    return (lo + hi) / 2
                }
            }
            return null
        }

        fun window(threshold: Double) = Window(crossing(threshold, true), crossing(threshold, false))

        // Le lever/coucher tient compte du demi-diametre solaire et de la refraction.
        val horizon = -0.833
        val currentAlt = alt[0]
        return NightInfo(
            sunset = crossing(horizon, true),
            sunrise = crossing(horizon, false),
            civil = window(-6.0),
            nautical = window(-12.0),
            astronomical = window(-18.0),
            currentPhase = phaseOf(currentAlt),
            sunAltitudeDeg = currentAlt,
        )
    }

    /**
     * Lever et coucher d'un corps du systeme solaire (section 2.1 : Soleil & Lune).
     */
    fun riseSet(
        referenceMillis: Long,
        latitudeDeg: Double,
        longitudeDeg: Double,
        horizonDeg: Double = -0.833,
        positionAt: (Double) -> RaDec,
    ): Pair<Long?, Long?> {
        val stepMs = 5 * 60_000L
        val steps = 24 * 60 / 5
        fun altitudeAt(ms: Long): Double {
            val jd = AstroMath.julianDay(ms)
            val eq = positionAt(jd)
            val lst = AstroMath.localSiderealTimeDeg(jd, longitudeDeg)
            return AstroMath.equatorialToHorizontal(eq.raDeg, eq.decDeg, latitudeDeg, lst).altitudeDeg
        }

        var rise: Long? = null
        var set: Long? = null
        var prev = altitudeAt(referenceMillis)
        for (i in 1..steps) {
            val ms = referenceMillis + i * stepMs
            val cur = altitudeAt(ms)
            if (prev <= horizonDeg && cur > horizonDeg && rise == null) {
                rise = refine(referenceMillis + (i - 1) * stepMs, ms, horizonDeg, true, ::altitudeAt)
            }
            if (prev > horizonDeg && cur <= horizonDeg && set == null) {
                set = refine(referenceMillis + (i - 1) * stepMs, ms, horizonDeg, false, ::altitudeAt)
            }
            prev = cur
            if (rise != null && set != null) break
        }
        return rise to set
    }

    private fun refine(
        loMs: Long,
        hiMs: Long,
        threshold: Double,
        ascending: Boolean,
        altitudeAt: (Long) -> Double,
    ): Long {
        var lo = loMs
        var hi = hiMs
        repeat(18) {
            val mid = (lo + hi) / 2
            val above = altitudeAt(mid) > threshold
            if (above == ascending) hi = mid else lo = mid
        }
        return (lo + hi) / 2
    }
}

package com.cielscore.app.astro

import kotlin.math.abs

/**
 * Calendrier astronomique sur 60 jours — section 2.1 : phases, solstices et
 * essaims de meteores.
 *
 * Les phases et les solstices sont calcules ; les essaims proviennent d'une
 * table de dates de maximum, comme le prevoit la documentation
 * (« PyEphem + table »).
 */
object SkyCalendar {

    enum class Kind { MOON_PHASE, SEASON, METEOR_SHOWER }

    data class Event(
        val epochMillis: Long,
        val kind: Kind,
        val title: String,
        val detail: String,
    )

    /** Essaims de meteores : jour du maximum et taux horaire zenithal indicatif. */
    private data class Shower(
        val name: String,
        val month: Int,
        val day: Int,
        val zhr: Int,
    )

    private val SHOWERS = listOf(
        Shower("Quadrantides", 1, 3, 110),
        Shower("Lyrides", 4, 22, 18),
        Shower("Eta Aquarides", 5, 6, 50),
        Shower("Delta Aquarides", 7, 30, 25),
        Shower("Perseides", 8, 12, 100),
        Shower("Draconides", 10, 8, 10),
        Shower("Orionides", 10, 21, 20),
        Shower("Taurides du Sud", 11, 5, 5),
        Shower("Leonides", 11, 17, 15),
        Shower("Geminides", 12, 14, 150),
        Shower("Ursides", 12, 22, 10),
    )

    /**
     * Evenements des [days] prochains jours a partir de [fromMillis].
     */
    fun upcoming(fromMillis: Long, days: Int = 60): List<Event> {
        val events = ArrayList<Event>()
        val until = fromMillis + days * 86_400_000L

        events += moonPhases(fromMillis, until)
        events += seasons(fromMillis, until)
        events += meteorShowers(fromMillis, until)

        return events.sortedBy { it.epochMillis }
    }

    /**
     * Nouvelles Lunes, premiers quartiers, pleines Lunes et derniers quartiers.
     * Detectes par le passage de la difference de longitude Lune - Soleil par
     * 0, 90, 180 et 270 degres.
     */
    private fun moonPhases(fromMillis: Long, untilMillis: Long): List<Event> {
        val out = ArrayList<Event>()
        val stepMs = 3 * 3_600_000L
        val targets = listOf(
            0.0 to "Nouvelle Lune",
            90.0 to "Premier quartier",
            180.0 to "Pleine Lune",
            270.0 to "Dernier quartier",
        )

        fun elongation(ms: Long): Double {
            val jd = AstroMath.julianDay(ms)
            val moon = SolarSystem.moon(jd)
            val moonLon = eclipticLongitude(moon.raDeg, moon.decDeg)
            val sun = SolarSystem.sun(jd)
            val sunLon = eclipticLongitude(sun.raDeg, sun.decDeg)
            return AstroMath.norm360(moonLon - sunLon)
        }

        var t = fromMillis
        var previous = elongation(t)
        while (t < untilMillis) {
            val next = t + stepMs
            val current = elongation(next)
            targets.forEach { (angle, label) ->
                if (crosses(previous, current, angle)) {
                    val instant = refine(t, next, angle, ::elongation)
                    out.add(Event(instant, Kind.MOON_PHASE, label, "Phase lunaire"))
                }
            }
            previous = current
            t = next
        }
        return out
    }

    /** Equinoxes et solstices : passage de la longitude solaire par 0, 90, 180 et 270. */
    private fun seasons(fromMillis: Long, untilMillis: Long): List<Event> {
        val out = ArrayList<Event>()
        val stepMs = 12 * 3_600_000L
        val targets = listOf(
            0.0 to "Equinoxe de printemps",
            90.0 to "Solstice d'ete",
            180.0 to "Equinoxe d'automne",
            270.0 to "Solstice d'hiver",
        )

        fun sunLongitude(ms: Long): Double {
            val sun = SolarSystem.sun(AstroMath.julianDay(ms))
            return AstroMath.norm360(eclipticLongitude(sun.raDeg, sun.decDeg))
        }

        var t = fromMillis
        var previous = sunLongitude(t)
        while (t < untilMillis) {
            val next = t + stepMs
            val current = sunLongitude(next)
            targets.forEach { (angle, label) ->
                if (crosses(previous, current, angle)) {
                    val instant = refine(t, next, angle, ::sunLongitude)
                    out.add(Event(instant, Kind.SEASON, label, "Changement de saison"))
                }
            }
            previous = current
            t = next
        }
        return out
    }

    private fun meteorShowers(fromMillis: Long, untilMillis: Long): List<Event> {
        val out = ArrayList<Event>()
        val calendar = java.util.Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"))
        calendar.timeInMillis = fromMillis
        val startYear = calendar.get(java.util.Calendar.YEAR)

        for (year in startYear..startYear + 1) {
            SHOWERS.forEach { shower ->
                val c = java.util.Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"))
                c.clear()
                c.set(year, shower.month - 1, shower.day, 2, 0, 0)
                val ms = c.timeInMillis
                if (ms in fromMillis..untilMillis) {
                    out.add(
                        Event(
                            ms,
                            Kind.METEOR_SHOWER,
                            shower.name,
                            "Maximum, environ ${shower.zhr} meteores par heure au zenith",
                        )
                    )
                }
            }
        }
        return out
    }

    private fun eclipticLongitude(raDeg: Double, decDeg: Double): Double {
        val eps = AstroMath.OBLIQUITY_J2000 * AstroMath.DEG
        val ra = raDeg * AstroMath.DEG
        val dec = decDeg * AstroMath.DEG
        val lambda = kotlin.math.atan2(
            kotlin.math.sin(ra) * kotlin.math.cos(eps) + kotlin.math.tan(dec) * kotlin.math.sin(eps),
            kotlin.math.cos(ra),
        )
        return AstroMath.norm360(lambda * AstroMath.RAD)
    }

    /** Vrai si l'angle [target] est franchi entre [previous] et [current]. */
    private fun crosses(previous: Double, current: Double, target: Double): Boolean {
        val a = AstroMath.norm360(previous - target)
        val b = AstroMath.norm360(current - target)
        // Passage par zero dans le sens croissant.
        return a > 270.0 && b < 90.0
    }

    private fun refine(
        loMs: Long,
        hiMs: Long,
        target: Double,
        angleAt: (Long) -> Double,
    ): Long {
        var lo = loMs
        var hi = hiMs
        repeat(24) {
            val mid = (lo + hi) / 2
            val delta = AstroMath.norm360(angleAt(mid) - target)
            if (delta > 180.0 || abs(delta - 360.0) < 1e-9) lo = mid else hi = mid
        }
        return (lo + hi) / 2
    }
}

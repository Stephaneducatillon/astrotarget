package com.cielscore.app.astro

import com.cielscore.app.astro.AstroMath.DEG
import com.cielscore.app.astro.AstroMath.RAD
import com.cielscore.app.astro.AstroMath.norm360
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Ephemerides du systeme solaire : Soleil, Lune et les cinq planetes visibles a
 * l'oeil nu (section 8.1, « Planetes : 6 corps »).
 *
 * Les positions planetaires utilisent les elements keplerians approches du JPL
 * (Standish, valables 1800-2050) ; le Soleil suit les formules de Meeus
 * chapitre 25 et la Lune la serie tronquee du chapitre 47.
 */
object SolarSystem {

    /** Corps du systeme solaire proposes par l'application. */
    enum class Body(val frenchName: String) {
        MOON("Lune"),
        MERCURY("Mercure"),
        VENUS("Venus"),
        MARS("Mars"),
        JUPITER("Jupiter"),
        SATURN("Saturne"),
    }

    /** Position geocentrique apparente d'un corps. */
    data class Position(
        val raDeg: Double,
        val decDeg: Double,
        /** Distance a la Terre, en unites astronomiques (rayons terrestres pour la Lune). */
        val distanceAu: Double,
        /** Magnitude visuelle apparente. */
        val magnitude: Double,
        /** Fraction illuminee, de 0 a 1. Pertinente pour la Lune et les planetes interieures. */
        val illuminatedFraction: Double,
    )

    // ------------------------------------------------------------------ Soleil

    /** Longitude ecliptique apparente du Soleil, en degres (Meeus 25). */
    private fun sunEclipticLongitude(t: Double): Double {
        val l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
        val m = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
        val mr = m * DEG
        val c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * sin(mr) +
            (0.019993 - 0.000101 * t) * sin(2 * mr) +
            0.000289 * sin(3 * mr)
        return norm360(l0 + c)
    }

    /** Distance Terre-Soleil en unites astronomiques. */
    private fun sunDistanceAu(t: Double): Double {
        val m = (357.52911 + 35999.05029 * t - 0.0001537 * t * t) * DEG
        val e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
        val c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * sin(m) +
            (0.019993 - 0.000101 * t) * sin(2 * m) +
            0.000289 * sin(3 * m)
        val v = m * RAD + c
        return 1.000001018 * (1 - e * e) / (1 + e * cos(v * DEG))
    }

    /** Position equatoriale apparente du Soleil. */
    fun sun(jd: Double): RaDec {
        val t = AstroMath.centuriesSinceJ2000(jd)
        return AstroMath.eclipticToEquatorial(sunEclipticLongitude(t), 0.0)
    }

    /** Altitude du Soleil au-dessus de l'horizon, en degres. Base du filtrage 4.2 et 7.2. */
    fun sunAltitudeDeg(jd: Double, latDeg: Double, lonDeg: Double): Double {
        val eq = sun(jd)
        val lst = AstroMath.localSiderealTimeDeg(jd, lonDeg)
        return AstroMath.equatorialToHorizontal(eq.raDeg, eq.decDeg, latDeg, lst).altitudeDeg
    }

    // -------------------------------------------------------------------- Lune

    /**
     * Position geocentrique de la Lune (Meeus 47, termes principaux).
     * Precision de l'ordre de la minute d'arc, largement suffisante pour le
     * calcul d'altitude, de phase et de distance angulaire aux objets.
     */
    fun moon(jd: Double): Position {
        val t = AstroMath.centuriesSinceJ2000(jd)

        val lp = norm360(218.3164477 + 481267.88123421 * t - 0.0015786 * t * t +
            t * t * t / 538841.0 - t * t * t * t / 65194000.0)
        val d = norm360(297.8501921 + 445267.1114034 * t - 0.0018819 * t * t +
            t * t * t / 545868.0 - t * t * t * t / 113065000.0)
        val m = norm360(357.5291092 + 35999.0502909 * t - 0.0001536 * t * t +
            t * t * t / 24490000.0)
        val mp = norm360(134.9633964 + 477198.8675055 * t + 0.0087414 * t * t +
            t * t * t / 69699.0 - t * t * t * t / 14712000.0)
        val f = norm360(93.2720950 + 483202.0175233 * t - 0.0036539 * t * t -
            t * t * t / 3526000.0 + t * t * t * t / 863310000.0)

        val e = 1.0 - 0.002516 * t - 0.0000074 * t * t
        val dr = d * DEG
        val mr = m * DEG
        val mpr = mp * DEG
        val fr = f * DEG

        // Longitude (millioniemes de degre) et distance (km) - termes majeurs.
        var sumL = 0.0
        var sumR = 0.0
        for (term in LONGITUDE_TERMS) {
            val arg = term.d * dr + term.m * mr + term.mp * mpr + term.f * fr
            var ecc = 1.0
            if (abs(term.m) == 1) ecc = e
            if (abs(term.m) == 2) ecc = e * e
            sumL += term.sl * ecc * sin(arg)
            sumR += term.cr * ecc * cos(arg)
        }
        var sumB = 0.0
        for (term in LATITUDE_TERMS) {
            val arg = term.d * dr + term.m * mr + term.mp * mpr + term.f * fr
            var ecc = 1.0
            if (abs(term.m) == 1) ecc = e
            if (abs(term.m) == 2) ecc = e * e
            sumB += term.sl * ecc * sin(arg)
        }

        val lambda = norm360(lp + sumL / 1_000_000.0)
        val beta = sumB / 1_000_000.0
        val distanceKm = 385_000.56 + sumR / 1_000.0

        val eq = AstroMath.eclipticToEquatorial(lambda, beta)

        // Phase : elongation geocentrique Soleil-Lune.
        val sunLon = sunEclipticLongitude(t)
        val elongation = kotlin.math.acos(
            (cos((lambda - sunLon) * DEG) * cos(beta * DEG)).coerceIn(-1.0, 1.0)
        ) * RAD
        val sunDistKm = sunDistanceAu(t) * AU_KM
        val phaseAngle = atan2(
            sunDistKm * sin(elongation * DEG),
            distanceKm - sunDistKm * cos(elongation * DEG)
        ) * RAD
        val illuminated = (1.0 + cos(phaseAngle * DEG)) / 2.0

        // Magnitude approchee de la Lune selon l'angle de phase.
        val magnitude = -12.73 + 0.026 * abs(phaseAngle) + 4.0e-9 * Math.pow(abs(phaseAngle), 4.0)

        return Position(eq.raDeg, eq.decDeg, distanceKm / AU_KM, magnitude, illuminated)
    }

    /** Fraction illuminee de la Lune, exprimee en pourcentage (0 = nouvelle, 100 = pleine). */
    fun moonPhasePercent(jd: Double): Double = moon(jd).illuminatedFraction * 100.0

    /**
     * Nom francais de la phase lunaire courante, base sur l'age de la lunaison.
     */
    fun moonPhaseName(jd: Double): String {
        val t = AstroMath.centuriesSinceJ2000(jd)
        val lambda = moonEclipticLongitude(jd)
        val diff = norm360(lambda - sunEclipticLongitude(t))
        return when {
            diff < 22.5 || diff >= 337.5 -> "Nouvelle Lune"
            diff < 67.5 -> "Premier croissant"
            diff < 112.5 -> "Premier quartier"
            diff < 157.5 -> "Gibbeuse croissante"
            diff < 202.5 -> "Pleine Lune"
            diff < 247.5 -> "Gibbeuse decroissante"
            diff < 292.5 -> "Dernier quartier"
            else -> "Dernier croissant"
        }
    }

    private fun moonEclipticLongitude(jd: Double): Double {
        val p = moon(jd)
        val eps = AstroMath.OBLIQUITY_J2000 * DEG
        val ra = p.raDeg * DEG
        val dec = p.decDeg * DEG
        val lambda = atan2(sin(ra) * cos(eps) + kotlin.math.tan(dec) * sin(eps), cos(ra))
        return norm360(lambda * RAD)
    }

    // ---------------------------------------------------------------- Planetes

    private class Elements(
        val a0: Double, val aDot: Double,
        val e0: Double, val eDot: Double,
        val i0: Double, val iDot: Double,
        val l0: Double, val lDot: Double,
        val peri0: Double, val periDot: Double,
        val node0: Double, val nodeDot: Double,
    )

    // Elements keplerians moyens, JPL (Standish), epoque J2000, valables 1800-2050.
    private val EARTH = Elements(
        1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
        100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0
    )
    private val ELEMENTS = mapOf(
        Body.MERCURY to Elements(
            0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
            252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081
        ),
        Body.VENUS to Elements(
            0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
            181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418
        ),
        Body.MARS to Elements(
            1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
            -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343
        ),
        Body.JUPITER to Elements(
            5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
            34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106
        ),
        Body.SATURN to Elements(
            9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
            49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794
        ),
    )

    private data class Vec3(val x: Double, val y: Double, val z: Double)

    /** Coordonnees ecliptiques heliocentriques d'un jeu d'elements, en unites astronomiques. */
    private fun heliocentric(el: Elements, t: Double): Vec3 {
        val a = el.a0 + el.aDot * t
        val e = el.e0 + el.eDot * t
        val i = (el.i0 + el.iDot * t) * DEG
        val l = el.l0 + el.lDot * t
        val peri = el.peri0 + el.periDot * t
        val node = (el.node0 + el.nodeDot * t) * DEG

        val m = l - peri
        val eAnom = AstroMath.solveKepler(m, e)
        val xv = a * (cos(eAnom) - e)
        val yv = a * sqrt(1 - e * e) * sin(eAnom)

        val argPeri = (peri - el.node0 - el.nodeDot * t) * DEG
        val cosW = cos(argPeri)
        val sinW = sin(argPeri)
        val cosO = cos(node)
        val sinO = sin(node)
        val cosI = cos(i)
        val sinI = sin(i)

        val x = (cosW * cosO - sinW * sinO * cosI) * xv + (-sinW * cosO - cosW * sinO * cosI) * yv
        val y = (cosW * sinO + sinW * cosO * cosI) * xv + (-sinW * sinO + cosW * cosO * cosI) * yv
        val z = (sinW * sinI) * xv + (cosW * sinI) * yv
        return Vec3(x, y, z)
    }

    /** Position geocentrique apparente d'une planete, magnitude comprise. */
    fun planet(body: Body, jd: Double): Position {
        if (body == Body.MOON) return moon(jd)
        val el = ELEMENTS.getValue(body)
        val t = AstroMath.centuriesSinceJ2000(jd)

        val p = heliocentric(el, t)
        val earth = heliocentric(EARTH, t)
        val gx = p.x - earth.x
        val gy = p.y - earth.y
        val gz = p.z - earth.z

        val delta = AstroMath.hypot3(gx, gy, gz)          // distance a la Terre
        val r = AstroMath.hypot3(p.x, p.y, p.z)            // distance au Soleil
        val rEarth = AstroMath.hypot3(earth.x, earth.y, earth.z)

        val lambda = norm360(atan2(gy, gx) * RAD)
        val beta = asin((gz / delta).coerceIn(-1.0, 1.0)) * RAD
        val eq = AstroMath.eclipticToEquatorial(lambda, beta)

        // Angle de phase (Soleil-planete-Terre).
        val cosPhase = ((r * r + delta * delta - rEarth * rEarth) / (2 * r * delta))
            .coerceIn(-1.0, 1.0)
        val phaseAngle = kotlin.math.acos(cosPhase) * RAD
        val illuminated = (1.0 + cosPhase) / 2.0

        val magnitude = apparentMagnitude(body, r, delta, phaseAngle, lambda, beta, t)
        return Position(eq.raDeg, eq.decDeg, delta, magnitude, illuminated)
    }

    /**
     * Magnitude visuelle apparente (Astronomical Almanac / Meeus 41).
     * Le terme des anneaux de Saturne suit Meeus chapitre 45.
     */
    private fun apparentMagnitude(
        body: Body,
        r: Double,
        delta: Double,
        i: Double,
        lambda: Double,
        beta: Double,
        t: Double,
    ): Double {
        val base = 5.0 * log10(r * delta)
        return when (body) {
            Body.MERCURY -> -0.42 + base + 0.0380 * i - 0.000273 * i * i + 2.0e-6 * i * i * i
            Body.VENUS -> -4.40 + base + 0.0009 * i + 2.39e-4 * i * i - 6.5e-7 * i * i * i
            Body.MARS -> -1.52 + base + 0.016 * i
            Body.JUPITER -> -9.40 + base + 0.005 * i
            Body.SATURN -> {
                val ringInc = (28.075216 - 0.012998 * t + 0.000004 * t * t) * DEG
                val ringNode = (169.508470 + 1.394681 * t + 0.000412 * t * t) * DEG
                val b = asin(
                    (sin(ringInc) * cos(beta * DEG) * sin(lambda * DEG - ringNode) -
                        cos(ringInc) * sin(beta * DEG)).coerceIn(-1.0, 1.0)
                )
                -8.88 + base - 2.60 * abs(sin(b)) + 1.25 * sin(b) * sin(b)
            }
            // La magnitude lunaire est calculee directement dans moon().
            Body.MOON -> Double.NaN
        }
    }

    private const val AU_KM = 149_597_870.7

    private class Term(val d: Int, val m: Int, val mp: Int, val f: Int, val sl: Double, val cr: Double)

    // Meeus 47, table 47.A tronquee aux termes superieurs a 10000 (0.01 degre).
    private val LONGITUDE_TERMS = listOf(
        Term(0, 0, 1, 0, 6288774.0, -20905355.0),
        Term(2, 0, -1, 0, 1274027.0, -3699111.0),
        Term(2, 0, 0, 0, 658314.0, -2955968.0),
        Term(0, 0, 2, 0, 213618.0, -569925.0),
        Term(0, 1, 0, 0, -185116.0, 48888.0),
        Term(0, 0, 0, 2, -114332.0, -3149.0),
        Term(2, 0, -2, 0, 58793.0, 246158.0),
        Term(2, -1, -1, 0, 57066.0, -152138.0),
        Term(2, 0, 1, 0, 53322.0, -170733.0),
        Term(2, -1, 0, 0, 45758.0, -204586.0),
        Term(0, 1, -1, 0, -40923.0, -129620.0),
        Term(1, 0, 0, 0, -34720.0, 108743.0),
        Term(0, 1, 1, 0, -30383.0, 104755.0),
        Term(2, 0, 0, -2, 15327.0, 10321.0),
        Term(0, 0, 1, 2, -12528.0, 0.0),
        Term(0, 0, 1, -2, 10980.0, 79661.0),
        Term(4, 0, -1, 0, 10675.0, -34782.0),
        Term(0, 0, 3, 0, 10034.0, -23210.0),
        Term(4, 0, -2, 0, 8548.0, -21636.0),
        Term(2, 1, -1, 0, -7888.0, 24208.0),
        Term(2, 1, 0, 0, -6766.0, 30824.0),
        Term(1, 0, -1, 0, -5163.0, -8379.0),
        Term(1, 1, 0, 0, 4987.0, -16675.0),
        Term(2, -1, 1, 0, 4036.0, -12831.0),
        Term(2, 0, 2, 0, 3994.0, -10445.0),
        Term(4, 0, 0, 0, 3861.0, -11650.0),
        Term(2, 0, -3, 0, 3665.0, 14403.0),
        Term(0, 1, -2, 0, -2689.0, -7003.0),
        Term(2, 0, -1, 2, -2602.0, 0.0),
        Term(2, -1, -2, 0, 2390.0, 10056.0),
        Term(1, 0, 1, 0, -2348.0, 6322.0),
        Term(2, -2, 0, 0, 2236.0, -9884.0),
        Term(0, 1, 2, 0, -2120.0, 5751.0),
        Term(0, 2, 0, 0, -2069.0, 0.0),
        Term(2, -2, -1, 0, 2048.0, -4950.0),
    )

    // Meeus 47, table 47.B tronquee.
    private val LATITUDE_TERMS = listOf(
        Term(0, 0, 0, 1, 5128122.0, 0.0),
        Term(0, 0, 1, 1, 280602.0, 0.0),
        Term(0, 0, 1, -1, 277693.0, 0.0),
        Term(2, 0, 0, -1, 173237.0, 0.0),
        Term(2, 0, -1, 1, 55413.0, 0.0),
        Term(2, 0, -1, -1, 46271.0, 0.0),
        Term(2, 0, 0, 1, 32573.0, 0.0),
        Term(0, 0, 2, 1, 17198.0, 0.0),
        Term(2, 0, 1, -1, 9266.0, 0.0),
        Term(0, 0, 2, -1, 8822.0, 0.0),
        Term(2, -1, 0, -1, 8216.0, 0.0),
        Term(2, 0, -2, -1, 4324.0, 0.0),
        Term(2, 0, 1, 1, 4200.0, 0.0),
        Term(2, 1, 0, -1, -3359.0, 0.0),
        Term(2, -1, -1, 1, 2463.0, 0.0),
        Term(2, -1, 0, 1, 2211.0, 0.0),
        Term(2, -1, -1, -1, 2065.0, 0.0),
        Term(0, 1, -1, -1, -1870.0, 0.0),
        Term(4, 0, -1, -1, 1828.0, 0.0),
        Term(0, 1, 0, 1, -1794.0, 0.0),
        Term(0, 0, 0, 3, -1749.0, 0.0),
    )
}

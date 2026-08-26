package com.cielscore.app.astro

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Primitives de calcul astronomique.
 *
 * Les conversions reprennent litteralement les formules de la section 3.5 de la
 * documentation CielScore :
 *
 *     H = temps_sideral_local - RA
 *     sin(alt) = sin(Dec)*sin(lat) + cos(Dec)*cos(lat)*cos(H)
 *     tan(az)  = sin(H) / (cos(H)*sin(lat) - tan(Dec)*cos(lat))
 */
object AstroMath {

    const val DEG = Math.PI / 180.0
    const val RAD = 180.0 / Math.PI

    /** Obliquite moyenne de l'ecliptique a J2000, en degres. */
    const val OBLIQUITY_J2000 = 23.4392911

    /** Jour julien a partir d'un instant UTC en millisecondes depuis l'epoque Unix. */
    fun julianDay(epochMillis: Long): Double = epochMillis / 86_400_000.0 + 2_440_587.5

    /** Instant UTC (ms) correspondant a un jour julien. */
    fun epochMillis(jd: Double): Long = ((jd - 2_440_587.5) * 86_400_000.0).toLong()

    /** Siecles julien ecoules depuis J2000.0. */
    fun centuriesSinceJ2000(jd: Double): Double = (jd - 2_451_545.0) / 36_525.0

    fun norm360(deg: Double): Double {
        val d = deg % 360.0
        return if (d < 0) d + 360.0 else d
    }

    fun norm180(deg: Double): Double {
        var d = norm360(deg)
        if (d > 180.0) d -= 360.0
        return d
    }

    /**
     * Temps sideral moyen de Greenwich, en degres (Meeus, eq. 12.4).
     */
    fun greenwichMeanSiderealTimeDeg(jd: Double): Double {
        val t = centuriesSinceJ2000(jd)
        val theta = 280.46061837 +
            360.98564736629 * (jd - 2_451_545.0) +
            0.000387933 * t * t -
            t * t * t / 38_710_000.0
        return norm360(theta)
    }

    /** Temps sideral local, en degres. Longitude positive vers l'est. */
    fun localSiderealTimeDeg(jd: Double, longitudeDeg: Double): Double =
        norm360(greenwichMeanSiderealTimeDeg(jd) + longitudeDeg)

    /**
     * Conversion equatorial -> horizontal (section 3.5).
     *
     * L'azimut renvoye est compte depuis le Nord vers l'Est (0 = N, 90 = E), ce
     * qui est la convention attendue par la projection de la section 3.4
     * (x = cx + r*sin(az), y = cy - r*cos(az)). La formule documentee donne
     * l'azimut depuis le Sud : on ajoute donc 180 degres.
     */
    fun equatorialToHorizontal(
        raDeg: Double,
        decDeg: Double,
        latitudeDeg: Double,
        lstDeg: Double,
    ): AltAz {
        val h = (lstDeg - raDeg) * DEG
        val dec = decDeg * DEG
        val lat = latitudeDeg * DEG

        val sinAlt = sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(h)
        val alt = asin(sinAlt.coerceIn(-1.0, 1.0)) * RAD

        // atan2 leve l'indetermination de la tangente ; azimut depuis le Sud.
        val azSouth = atan2(sin(h), cos(h) * sin(lat) - kotlin.math.tan(dec) * cos(lat)) * RAD
        return AltAz(alt, norm360(azSouth + 180.0))
    }

    /** Distance angulaire entre deux directions equatoriales, en degres. */
    fun angularSeparationDeg(ra1: Double, dec1: Double, ra2: Double, dec2: Double): Double {
        val d1 = dec1 * DEG
        val d2 = dec2 * DEG
        val dRa = (ra1 - ra2) * DEG
        val cosSep = sin(d1) * sin(d2) + cos(d1) * cos(d2) * cos(dRa)
        return kotlin.math.acos(cosSep.coerceIn(-1.0, 1.0)) * RAD
    }

    /** Conversion de coordonnees ecliptiques (deg) vers equatoriales (deg). */
    fun eclipticToEquatorial(lambdaDeg: Double, betaDeg: Double): RaDec {
        val eps = OBLIQUITY_J2000 * DEG
        val l = lambdaDeg * DEG
        val b = betaDeg * DEG
        val ra = atan2(sin(l) * cos(eps) - kotlin.math.tan(b) * sin(eps), cos(l))
        val dec = asin((sin(b) * cos(eps) + cos(b) * sin(eps) * sin(l)).coerceIn(-1.0, 1.0))
        return RaDec(norm360(ra * RAD), dec * RAD)
    }

    /**
     * Resolution de l'equation de Kepler M = E - e*sin(E) par iteration de Newton.
     * @param mDeg anomalie moyenne en degres, [e] excentricite.
     */
    fun solveKepler(mDeg: Double, e: Double): Double {
        val m = norm180(mDeg) * DEG
        var eAnom = m + e * sin(m)
        repeat(12) {
            val dE = (eAnom - e * sin(eAnom) - m) / (1.0 - e * cos(eAnom))
            eAnom -= dE
            if (abs(dE) < 1e-12) return eAnom
        }
        return eAnom
    }

    /** Refraction atmospherique approchee (Bennett), en degres, pour une altitude apparente. */
    fun refractionDeg(altitudeDeg: Double): Double {
        if (altitudeDeg < -1.0) return 0.0
        val r = 1.02 / kotlin.math.tan((altitudeDeg + 10.3 / (altitudeDeg + 5.11)) * DEG)
        return r / 60.0
    }

    fun hypot3(x: Double, y: Double, z: Double): Double = sqrt(x * x + y * y + z * z)

    /** Partie fractionnaire positive. */
    fun frac(x: Double): Double = x - floor(x)
}

/** Couple altitude / azimut en degres. Azimut compte depuis le Nord vers l'Est. */
data class AltAz(val altitudeDeg: Double, val azimuthDeg: Double)

/** Couple ascension droite / declinaison en degres (J2000). */
data class RaDec(val raDeg: Double, val decDeg: Double)

package com.cielscore.app.astro

import kotlin.math.cos
import kotlin.math.sin

/**
 * Projection azimutale equidistante de la Carte du ciel (section 3.4).
 *
 *     r = (90 - altitude) / 90 * R
 *     x = centre_x + r * sin(azimut)
 *     y = centre_y - r * cos(azimut)
 *
 * Le zenith est au centre, l'horizon sur le cercle exterieur ; la projection
 * conserve les distances angulaires depuis le zenith.
 */
object SkyProjection {

    /** Point projete en coordonnees ecran (pixels). */
    data class Point(val x: Float, val y: Float)

    fun project(
        altitudeDeg: Double,
        azimuthDeg: Double,
        centerX: Float,
        centerY: Float,
        radius: Float,
    ): Point {
        val r = ((90.0 - altitudeDeg) / 90.0) * radius
        val az = azimuthDeg * AstroMath.DEG
        return Point(
            (centerX + r * sin(az)).toFloat(),
            (centerY - r * cos(az)).toFloat(),
        )
    }

    /** Rayon, en pixels, du cercle d'altitude constante utilise pour les reperes 30 et 60 degres. */
    fun altitudeCircleRadius(altitudeDeg: Double, radius: Float): Float =
        (((90.0 - altitudeDeg) / 90.0) * radius).toFloat()
}

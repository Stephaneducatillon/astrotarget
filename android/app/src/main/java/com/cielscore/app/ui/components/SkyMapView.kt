package com.cielscore.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import com.cielscore.app.astro.AltAz
import com.cielscore.app.astro.AstroMath
import com.cielscore.app.astro.SkyProjection
import com.cielscore.app.astro.SolarSystem
import com.cielscore.app.catalog.Constellation
import com.cielscore.app.catalog.SkyObject
import com.cielscore.app.catalog.Star
import kotlin.math.min

/**
 * Vue hemispherique depuis le lieu d'observation — section 3.2, premiere vue.
 *
 * Elements et regles d'affichage de la section 3.3 :
 *
 *   Etoiles          affichees si altitude > 2 degres
 *   Constellations   segment visible si ses deux extremites sont > 0 degre
 *   Planetes         affichees si altitude > 5 degres (Lune > 2 degres)
 *   Cible            cercle dore + halo + ligne guide, si altitude > 2 degres
 *   Cercles altitude 30 et 60 degres en pointilles, toujours
 *   Cardinaux        N / E / S / O sur le bord, toujours
 *   Horizon          cercle exterieur, altitude 0 degre, toujours
 *
 * La projection est azimutale equidistante (section 3.4) : le zenith est au
 * centre, l'horizon sur le cercle exterieur.
 */
@Composable
fun SkyMapView(
    stars: List<Star>,
    starsById: Map<String, Star>,
    constellations: List<Constellation>,
    latitude: Double,
    longitude: Double,
    epochMillis: Long,
    target: SkyObject?,
    modifier: Modifier = Modifier,
    nightMode: Boolean = false,
) {
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    val jd = AstroMath.julianDay(epochMillis)
    val lst = AstroMath.localSiderealTimeDeg(jd, longitude)

    fun horizontal(raDeg: Double, decDeg: Double): AltAz =
        AstroMath.equatorialToHorizontal(raDeg, decDeg, latitude, lst)

    val skyLine = if (nightMode) Color(0xFF7A2A20) else Color(0xFF2B3A5C)
    val starColor = if (nightMode) Color(0xFFFF8A70) else Color(0xFFE8EEFF)
    val figureColor = if (nightMode) Color(0xFF803026) else Color(0xFF3D5480)
    val cardinalColor = if (nightMode) Color(0xFFFF6B5A) else Color(0xFF9FB4DA)
    val targetColor = Color(0xFFF2C14E)

    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .background(if (nightMode) Color(0xFF0A0201) else Color(0xFF070B16))
            .pointerInput(Unit) {
                detectTransformGestures { _, panChange, zoomChange, _ ->
                    zoom = (zoom * zoomChange).coerceIn(1f, 5f)
                    pan = if (zoom <= 1.01f) Offset.Zero else pan + panChange
                }
            }
    ) {
        Canvas(modifier = Modifier.fillMaxWidth().aspectRatio(1f)) {
            val cx = size.width / 2f + pan.x
            val cy = size.height / 2f + pan.y
            val radius = min(size.width, size.height) / 2f * 0.92f * zoom

            drawHorizon(cx, cy, radius, skyLine)
            drawAltitudeCircles(cx, cy, radius, skyLine)
            drawCardinals(cx, cy, radius, cardinalColor)

            // Constellations : un segment n'est trace que si ses deux extremites
            // sont au-dessus de l'horizon (altitude > 0 degre).
            constellations.forEach { figure ->
                figure.segments.forEach { (aId, bId) ->
                    val a = starsById[aId] ?: return@forEach
                    val b = starsById[bId] ?: return@forEach
                    val altAzA = horizontal(a.raDeg, a.decDeg)
                    val altAzB = horizontal(b.raDeg, b.decDeg)
                    if (altAzA.altitudeDeg > 0.0 && altAzB.altitudeDeg > 0.0) {
                        val pa = SkyProjection.project(altAzA.altitudeDeg, altAzA.azimuthDeg, cx, cy, radius)
                        val pb = SkyProjection.project(altAzB.altitudeDeg, altAzB.azimuthDeg, cx, cy, radius)
                        drawLine(
                            color = figureColor,
                            start = Offset(pa.x, pa.y),
                            end = Offset(pb.x, pb.y),
                            strokeWidth = 1.2f,
                        )
                    }
                }
            }

            // Etoiles : affichees si altitude > 2 degres.
            stars.forEach { star ->
                val altAz = horizontal(star.raDeg, star.decDeg)
                if (altAz.altitudeDeg <= 2.0) return@forEach
                val p = SkyProjection.project(altAz.altitudeDeg, altAz.azimuthDeg, cx, cy, radius)
                val r = starRadius(star.magnitude) * zoom.coerceAtMost(2f)
                drawCircle(starColor, radius = r, center = Offset(p.x, p.y))
            }

            // Planetes : altitude > 5 degres. Lune : altitude > 2 degres.
            SolarSystem.Body.entries.forEach { body ->
                val pos = if (body == SolarSystem.Body.MOON) SolarSystem.moon(jd)
                else SolarSystem.planet(body, jd)
                val altAz = horizontal(pos.raDeg, pos.decDeg)
                val threshold = if (body == SolarSystem.Body.MOON) 2.0 else 5.0
                if (altAz.altitudeDeg <= threshold) return@forEach
                val p = SkyProjection.project(altAz.altitudeDeg, altAz.azimuthDeg, cx, cy, radius)
                val color = if (body == SolarSystem.Body.MOON) Color(0xFFF5F0DC) else Color(0xFFFFC773)
                val r = if (body == SolarSystem.Body.MOON) 7f else 4.5f
                drawCircle(color, radius = r * zoom.coerceAtMost(2f), center = Offset(p.x, p.y))
                drawLabel(body.frenchName, p.x + 9f, p.y + 4f, cardinalColor, 22f)
            }

            // Cible : cercle dore, halo et ligne guide depuis le zenith.
            if (target != null) {
                val eq = if (target.isSolarSystem) {
                    val pos = if (target.body == SolarSystem.Body.MOON) SolarSystem.moon(jd)
                    else SolarSystem.planet(target.body!!, jd)
                    pos.raDeg to pos.decDeg
                } else target.raDeg to target.decDeg
                val altAz = horizontal(eq.first, eq.second)
                if (altAz.altitudeDeg > 2.0) {
                    val p = SkyProjection.project(altAz.altitudeDeg, altAz.azimuthDeg, cx, cy, radius)
                    drawLine(
                        color = targetColor.copy(alpha = 0.35f),
                        start = Offset(cx, cy),
                        end = Offset(p.x, p.y),
                        strokeWidth = 1f,
                    )
                    drawCircle(targetColor.copy(alpha = 0.18f), radius = 22f, center = Offset(p.x, p.y))
                    drawCircle(
                        color = targetColor,
                        radius = 11f,
                        center = Offset(p.x, p.y),
                        style = Stroke(width = 2.2f),
                    )
                    drawLabel(target.id, p.x + 15f, p.y - 12f, targetColor, 26f)
                }
            }
        }
    }
}

private fun starRadius(magnitude: Double): Float = when {
    magnitude < 0.0 -> 4.2f
    magnitude < 1.0 -> 3.6f
    magnitude < 2.0 -> 3.0f
    magnitude < 3.0 -> 2.4f
    magnitude < 4.0 -> 1.8f
    else -> 1.3f
}

private fun DrawScope.drawHorizon(cx: Float, cy: Float, radius: Float, color: Color) {
    // Cercle exterieur = altitude 0 degre.
    drawCircle(color, radius = radius, center = Offset(cx, cy), style = Stroke(width = 2f))
}

private fun DrawScope.drawAltitudeCircles(cx: Float, cy: Float, radius: Float, color: Color) {
    val dashed = PathEffect.dashPathEffect(floatArrayOf(6f, 8f), 0f)
    listOf(30.0, 60.0).forEach { alt ->
        drawCircle(
            color = color.copy(alpha = 0.6f),
            radius = SkyProjection.altitudeCircleRadius(alt, radius),
            center = Offset(cx, cy),
            style = Stroke(width = 1f, pathEffect = dashed),
        )
    }
}

private fun DrawScope.drawCardinals(cx: Float, cy: Float, radius: Float, color: Color) {
    // Azimut compte depuis le Nord vers l'Est, conformement a la projection 3.4.
    val cardinals = listOf("N" to 0.0, "E" to 90.0, "S" to 180.0, "O" to 270.0)
    cardinals.forEach { (label, az) ->
        val p = SkyProjection.project(-4.0, az, cx, cy, radius)
        drawLabel(label, p.x - 7f, p.y + 8f, color, 30f)
    }
}

private fun DrawScope.drawLabel(text: String, x: Float, y: Float, color: Color, size: Float) {
    drawContext.canvas.nativeCanvas.drawText(
        text,
        x,
        y,
        android.graphics.Paint().apply {
            this.color = color.toArgb()
            textSize = size
            isAntiAlias = true
        },
    )
}

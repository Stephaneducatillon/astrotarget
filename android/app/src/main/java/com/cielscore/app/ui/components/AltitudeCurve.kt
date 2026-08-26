package com.cielscore.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import com.cielscore.app.scoring.ScoringEngine

/**
 * Courbe d'altitude — section 2.2 : trajectoire de l'objet sur 10 heures, avec
 * le seuil optimal a 30 degres.
 */
@Composable
fun AltitudeCurve(
    points: List<Pair<Long, Double>>,
    modifier: Modifier = Modifier,
    lineColor: Color = Color(0xFF7FB3FF),
    thresholdColor: Color = Color(0xFFF2C14E),
) {
    if (points.size < 2) return
    val gridColor = Color(0xFF2B3A5C)
    val labelColor = Color(0xFF9FB4DA)

    Box(
        modifier
            .fillMaxWidth()
            .height(170.dp)
            .background(Color(0xFF0A0F1E))
    ) {
        Canvas(Modifier.fillMaxWidth().height(170.dp)) {
            val left = 34f
            val right = size.width - 8f
            val top = 12f
            val bottom = size.height - 20f
            val plotWidth = right - left
            val plotHeight = bottom - top

            // Echelle verticale : de -10 a +90 degres.
            fun yFor(alt: Double): Float =
                (bottom - ((alt + 10.0) / 100.0) * plotHeight).toFloat()

            fun xFor(index: Int): Float =
                left + plotWidth * index / (points.size - 1).toFloat()

            val paint = android.graphics.Paint().apply {
                color = labelColor.toArgb()
                textSize = 22f
                isAntiAlias = true
            }

            // Graduations horizontales tous les 30 degres.
            listOf(0.0, 30.0, 60.0, 90.0).forEach { alt ->
                val y = yFor(alt)
                drawLine(gridColor, Offset(left, y), Offset(right, y), strokeWidth = 1f)
                drawContext.canvas.nativeCanvas.drawText("${alt.toInt()}", 4f, y + 7f, paint)
            }

            // Seuil optimal de 30 degres (section 2.2), en pointilles.
            val dashed = PathEffect.dashPathEffect(floatArrayOf(8f, 8f), 0f)
            drawLine(
                thresholdColor.copy(alpha = 0.8f),
                Offset(left, yFor(ScoringEngine.OPTIMAL_ALTITUDE_DEG)),
                Offset(right, yFor(ScoringEngine.OPTIMAL_ALTITUDE_DEG)),
                strokeWidth = 1.4f,
                pathEffect = dashed,
            )

            // Horizon.
            drawLine(
                Color(0xFF6B7A99),
                Offset(left, yFor(0.0)),
                Offset(right, yFor(0.0)),
                strokeWidth = 1.6f,
            )

            // Trajectoire.
            val path = Path()
            points.forEachIndexed { index, (_, alt) ->
                val x = xFor(index)
                val y = yFor(alt)
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, lineColor, style = Stroke(width = 2.4f))

            // Repere horaire toutes les deux heures.
            val formatter = java.text.SimpleDateFormat("HH'h'", java.util.Locale.FRANCE)
            val step = (points.size - 1) / 5
            for (i in 0..5) {
                val index = (i * step).coerceAtMost(points.size - 1)
                val x = xFor(index)
                drawLine(gridColor, Offset(x, top), Offset(x, bottom), strokeWidth = 0.8f)
                drawContext.canvas.nativeCanvas.drawText(
                    formatter.format(java.util.Date(points[index].first)),
                    x - 12f,
                    size.height - 2f,
                    paint,
                )
            }
        }
    }
}

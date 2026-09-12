package com.skyscore.app.export

import android.content.Context
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import com.skyscore.app.model.SessionParams
import com.skyscore.app.model.SkyConditions
import com.skyscore.app.scoring.ScoringEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Export PDF de la feuille de route de la soiree (section 2.4) : conditions de
 * la session, puis tableau des cibles calculees.
 */
object PdfExporter {

    private const val PAGE_WIDTH = 595   // A4 a 72 dpi
    private const val PAGE_HEIGHT = 842
    private const val MARGIN = 40f

    suspend fun exportEveningPlan(
        context: Context,
        targets: List<ScoringEngine.Scored>,
        params: SessionParams,
        conditions: SkyConditions,
    ): android.net.Uri = withContext(Dispatchers.IO) {
        val document = PdfDocument()
        val title = Paint().apply {
            textSize = 20f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val heading = Paint().apply {
            textSize = 13f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val body = Paint().apply { textSize = 10.5f }
        val muted = Paint().apply {
            textSize = 9f
            color = android.graphics.Color.DKGRAY
        }

        var pageNumber = 1
        var page = document.startPage(
            PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, pageNumber).create()
        )
        var canvas = page.canvas
        var y = MARGIN + 20f

        fun newPage() {
            document.finishPage(page)
            pageNumber++
            page = document.startPage(
                PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, pageNumber).create()
            )
            canvas = page.canvas
            y = MARGIN + 20f
        }

        fun line(text: String, paint: Paint, indent: Float = 0f) {
            if (y > PAGE_HEIGHT - MARGIN) newPage()
            canvas.drawText(text, MARGIN + indent, y, paint)
            y += paint.textSize + 5f
        }

        val dateLabel = java.text.SimpleDateFormat(
            "EEEE d MMMM yyyy 'a' HH'h'mm", java.util.Locale.FRANCE
        ).format(java.util.Date(params.epochMillis))

        line("SkyScore — Plan de soiree", title)
        line(dateLabel.replaceFirstChar { it.uppercase() }, muted)
        line(
            "%s — Bortle %d — %s de %.0f mm — magnitude limite %.1f".format(
                params.site.name, params.site.bortle,
                params.smartTelescope?.name ?: params.instrument.label,
                params.effectiveDiameterMm, params.limitingMagnitude
            ),
            muted
        )
        line(
            "Nuages %.0f %% — seeing %s — %s".format(
                conditions.cloudCoverPercent, conditions.seeingLabel,
                if (conditions.ok) "donnees Open-Meteo" else "valeurs de repli"
            ),
            muted
        )
        y += 10f

        y += 6f
        line("Tableau des cibles", heading)
        line(
            "%-26s %6s %7s %9s %8s".format("Objet", "Score", "Alt.", "Magnitude", "Fenetre"),
            muted
        )
        targets.forEach { s ->
            line(
                "%-26s %6.0f %6.0f° %9s %7.0f min".format(
                    s.target.displayName.take(26),
                    s.score,
                    s.altitudeDeg,
                    s.target.magnitude?.let { "%.1f".format(it) } ?: "—",
                    s.windowMinutes,
                ),
                body
            )
        }

        document.finishPage(page)

        val dir = File(context.cacheDir, "exports").apply { mkdirs() }
        val stamp = java.text.SimpleDateFormat("yyyyMMdd-HHmm", java.util.Locale.FRANCE)
            .format(java.util.Date(params.epochMillis))
        val file = File(dir, "skyscore-plan-$stamp.pdf")
        file.outputStream().use { document.writeTo(it) }
        document.close()

        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }
}

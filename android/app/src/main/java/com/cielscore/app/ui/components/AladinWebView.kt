package com.cielscore.app.ui.components

import android.annotation.SuppressLint
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.cielscore.app.catalog.SkyObject
import com.cielscore.app.data.net.SkyImagery

/**
 * Vue Aladin Lite v3 — section 3.2, deuxieme vue : image reelle de l'objet avec
 * cercle du champ oculaire, choix du releve, zoom et mode nuit.
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun AladinWebView(
    target: SkyObject,
    survey: SkyImagery.Survey,
    fieldDeg: Double,
    fieldCircleArcmin: Double?,
    nightMode: Boolean,
    modifier: Modifier = Modifier,
) {
    val html = SkyImagery.aladinHtml(target, survey, fieldDeg, fieldCircleArcmin, nightMode)
    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.loadWithOverviewMode = true
                settings.useWideViewPort = true
                settings.cacheMode = WebSettings.LOAD_DEFAULT
                setBackgroundColor(android.graphics.Color.BLACK)
            }
        },
        update = { webView ->
            webView.loadDataWithBaseURL(
                "https://aladin.cds.unistra.fr/",
                html,
                "text/html",
                "utf-8",
                null,
            )
        },
    )
}

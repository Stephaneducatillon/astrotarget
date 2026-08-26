package com.cielscore.app.data.net

import com.cielscore.app.catalog.SkyObject

/**
 * Imagerie du ciel — sections 2.2, 3.2 et 8.2.
 *
 *   hips2fits (CDS)   images reelles DSS2 / PanSTARRS
 *   Aladin Lite v3    carte interactive, cercle de champ, mode nuit
 *   Stellarium Web    ouverture externe avec position, date et objet
 */
object SkyImagery {

    /** Releves proposes par la vue Aladin (section 3.2). */
    enum class Survey(val label: String, val hipsId: String) {
        DSS2_COLOR("DSS2 couleur", "CDS/P/DSS2/color"),
        DSS2_RED("DSS2 rouge", "CDS/P/DSS2/red"),
        PANSTARRS("PanSTARRS DR1", "CDS/P/PanSTARRS/DR1/color-z-zg-g"),
        TWO_MASS("2MASS couleur", "CDS/P/2MASS/color"),
    }

    /**
     * Vignette hips2fits d'un objet.
     *
     * @param fieldDeg champ couvert par l'image, en degres.
     */
    fun thumbnailUrl(
        raDeg: Double,
        decDeg: Double,
        fieldDeg: Double,
        survey: Survey = Survey.DSS2_COLOR,
        pixels: Int = 512,
    ): String {
        return ApiUrls.hips2fits(survey.hipsId, raDeg, decDeg, fieldDeg, pixels)
    }

    /** Champ d'affichage adapte a la taille angulaire de l'objet. */
    fun suggestedFieldDeg(target: SkyObject): Double {
        val sizeArcmin = target.majorAxisArcmin ?: 15.0
        return ((sizeArcmin * 2.5) / 60.0).coerceIn(0.15, 4.0)
    }

    /**
     * Page Aladin Lite v3 autonome, chargee dans une WebView.
     *
     * @param fieldCircleArcmin diametre du cercle de champ oculaire a tracer,
     *   ou null pour ne pas en tracer.
     */
    fun aladinHtml(
        target: SkyObject,
        survey: Survey,
        fieldDeg: Double,
        fieldCircleArcmin: Double?,
        nightMode: Boolean,
    ): String {
        val circle = fieldCircleArcmin?.let {
            """
            var overlay = A.graphicOverlay({color: '#f2c14e', lineWidth: 2});
            aladin.addOverlay(overlay);
            overlay.add(A.circle(${target.raDeg}, ${target.decDeg}, ${it / 120.0}));
            """.trimIndent()
        }.orEmpty()

        val filter = if (nightMode) "filter: url(#nightmode);" else ""
        return """
            <!DOCTYPE html>
            <html lang="fr">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
              <link rel="stylesheet" href="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.css">
              <style>
                html, body { margin:0; padding:0; height:100%; background:#000; }
                #aladin-lite-div { width:100%; height:100%; $filter }
              </style>
            </head>
            <body>
              <svg width="0" height="0" style="position:absolute">
                <filter id="nightmode">
                  <feColorMatrix type="matrix"
                    values="0.9 0.3 0.2 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/>
                </filter>
              </svg>
              <div id="aladin-lite-div"></div>
              <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
              <script>
                let aladin;
                A.init.then(() => {
                  aladin = A.aladin('#aladin-lite-div', {
                    survey: '${survey.hipsId}',
                    target: '${target.raDeg} ${target.decDeg}',
                    fov: $fieldDeg,
                    showReticle: true,
                    showZoomControl: true,
                    showFullscreenControl: false,
                    showLayersControl: false,
                    showGotoControl: false,
                    showCooGrid: false,
                    cooFrame: 'ICRS'
                  });
                  $circle
                });
              </script>
            </body>
            </html>
        """.trimIndent()
    }

    /**
     * Lien externe Stellarium Web charge sur la position, la date et l'objet
     * courants (section 3.2, troisieme vue).
     */
    fun stellariumWebUrl(
        target: SkyObject,
        latitude: Double,
        longitude: Double,
        epochMillis: Long,
    ): String {
        val iso = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", java.util.Locale.US)
            .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
            .format(java.util.Date(epochMillis))
        // Stellarium Web attend la position en degres et l'instant en ISO 8601.
        return ApiUrls.stellariumWeb(
            target.designation.ifBlank { target.id }, latitude, longitude, iso
        )
    }
}

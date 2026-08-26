package com.cielscore.app.model

import com.cielscore.app.scoring.Formulas

/**
 * Smart telescope integre au module Equipement (section 2.5) et score par la
 * formule dediee de la section 6.4.
 *
 * Les sept modeles ci-dessous sont ceux detailles par le tableau de la
 * section 5.9. Les diametres et les magnitudes limites qui en decoulent sont
 * repris tels quels de la documentation.
 */
data class SmartTelescope(
    val brand: String,
    val model: String,
    val diameterMm: Double,
    val focalMm: Double,
    val sensorWidthMm: Double,
    val sensorHeightMm: Double,
    val pixelSizeUm: Double,
) {
    val name: String get() = "$brand $model"

    val focalRatio: Double get() = Formulas.focalRatio(focalMm, diameterMm)

    /** Champ couvert, en arcminutes, sur le grand cote du capteur. */
    val fieldWidthArcmin: Double
        get() = Formulas.sensorFieldDeg(sensorWidthMm, focalMm) * 60.0

    /** Champ couvert, en arcminutes, sur le petit cote du capteur. */
    val fieldHeightArcmin: Double
        get() = Formulas.sensorFieldDeg(sensorHeightMm, focalMm) * 60.0

    /** Section 5.9 — magnitude limite selon la duree de pose et le Bortle. */
    fun limitingMagnitude(exposureMinutes: Double, bortle: Int): Double =
        Formulas.smartTelescopeLimitingMagnitude(diameterMm, exposureMinutes * 60.0, bortle)

    companion object {
        /**
         * Les sept modeles du tableau de la section 5.9, dans l'ordre du document.
         *
         * ATTENTION — la documentation ne fournit que le DIAMETRE de chaque
         * modele. La focale, les dimensions du capteur et la taille de pixel
         * proviennent des specifications constructeur : elles sont necessaires
         * au module Equipement (section 2.5 : « ouverture, focale, capteur,
         * champ ») et aux criteres F/D et champ du score smart telescope
         * (section 6.4), mais elles ne sont PAS issues du document.
         * Ce tableau est volontairement isole pour etre corrige d'un seul geste.
         *
         *   marque, modele, diametre mm, focale mm, capteur L mm, capteur H mm, pixel um
         */
        val CATALOG: List<SmartTelescope> = listOf(
            SmartTelescope("ZWO", "Seestar S30", 30.0, 150.0, 5.6, 3.2, 2.9),
            SmartTelescope("ZWO", "Seestar S50", 50.0, 250.0, 5.6, 3.2, 2.9),
            SmartTelescope("Vaonis", "Vespera II", 50.0, 250.0, 8.4, 4.7, 2.9),
            SmartTelescope("Vaonis", "Stellina", 80.0, 400.0, 7.4, 5.0, 2.4),
            SmartTelescope("Unistellar", "Odyssey Pro", 85.0, 320.0, 7.4, 4.2, 2.9),
            SmartTelescope("Unistellar", "eVscope 2", 114.0, 450.0, 7.4, 4.2, 2.9),
            SmartTelescope("Celestron", "Origin Mk II", 152.0, 335.0, 7.4, 5.0, 2.4),
        )
    }
}

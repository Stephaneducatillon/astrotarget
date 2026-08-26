package com.cielscore.app.scoring

import kotlin.math.PI
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

/**
 * Formules de calcul — section 5 de la documentation CielScore.
 *
 * Toutes les fonctions de ce fichier sont pures : elles sont directement
 * testables et ne dependent d'aucun contexte Android.
 */
object Formulas {

    /** Diametre, en millimetres, au-dela duquel on considere un instrument optique (RG-I-01). */
    const val NAKED_EYE_MAX_DIAMETER_MM = 7.0

    // ------------------------------------------------- 5.1 / 5.3 Magnitude limite

    /**
     * Section 5.1 — magnitude limite d'un instrument :
     *
     *     mag_limite = 2.1 + 5 * log10(D_mm)
     */
    fun instrumentLimitingMagnitude(diameterMm: Double): Double =
        2.1 + 5.0 * log10(diameterMm)

    /**
     * Section 5.2 — magnitude limite a l'oeil nu (NELM) selon l'indice de Bortle.
     */
    fun nakedEyeLimitingMagnitude(bortle: Int): Double = when (bortle.coerceIn(1, 9)) {
        1 -> 7.6
        2 -> 7.1
        3 -> 6.6
        4 -> 6.1
        5 -> 5.6
        6 -> 5.1
        7 -> 4.6
        8 -> 4.1
        else -> 3.6
    }

    /** Description du ciel associee a l'indice de Bortle (section 5.2). */
    fun bortleDescription(bortle: Int): String = when (bortle.coerceIn(1, 9)) {
        1 -> "Ciel vierge exceptionnel"
        2 -> "Ciel vierge"
        3 -> "Ciel rural"
        4 -> "Transition rural / peri-urbain"
        5 -> "Ciel peri-urbain"
        6 -> "Banlieue lumineuse"
        7 -> "Transition banlieue / ville"
        8 -> "Ciel urbain"
        else -> "Centre-ville"
    }

    /**
     * Section 5.3 — magnitude limite reelle.
     *
     *     Oeil nu (D <= 7 mm)   : mag_limite = NELM(Bortle)
     *     Instrument (D > 7 mm) : mag_limite = 2.1 + 5 * log10(D_mm)
     *
     * Pour un instrument optique la limite ne depend que du diametre (RG-I-02) ;
     * l'effet du ciel pollue est traite separement par la brillance de surface
     * (RG-I-03).
     */
    fun limitingMagnitude(diameterMm: Double, bortle: Int): Double =
        if (diameterMm <= NAKED_EYE_MAX_DIAMETER_MM) nakedEyeLimitingMagnitude(bortle)
        else instrumentLimitingMagnitude(diameterMm)

    // ------------------------------------------------- 5.4 / 5.5 Brillance de surface

    /**
     * Section 5.4 — brillance de surface, en mag/arcsec carre :
     *
     *     SB = magnitude + 2.5 * log10( PI * (a/2) * (b/2) * 3600 )
     *
     * @param majorAxisArcmin grand axe en arcminutes
     * @param minorAxisArcmin petit axe en arcminutes
     * @return la brillance de surface, ou null si les dimensions sont inconnues.
     */
    fun surfaceBrightness(
        magnitude: Double?,
        majorAxisArcmin: Double?,
        minorAxisArcmin: Double?,
    ): Double? {
        if (magnitude == null || majorAxisArcmin == null || majorAxisArcmin <= 0.0) return null
        val b = if (minorAxisArcmin == null || minorAxisArcmin <= 0.0) majorAxisArcmin
        else minorAxisArcmin
        val areaArcsec2 = PI * (majorAxisArcmin / 2.0) * (b / 2.0) * 3600.0
        return magnitude + 2.5 * log10(areaArcsec2)
    }

    /**
     * Section 5.5 — limite de brillance de surface par indice de Bortle,
     * en mag/arcsec carre.
     */
    fun surfaceBrightnessLimit(bortle: Int): Double = when (bortle.coerceIn(1, 9)) {
        1 -> 22.0
        2 -> 21.8
        3 -> 21.5
        4 -> 21.0
        5 -> 20.5
        6 -> 19.5
        7 -> 18.5
        8 -> 17.5
        else -> 17.0
    }

    /** Exemple de lieu associe a chaque limite (section 5.5). */
    fun bortlePlaceExample(bortle: Int): String = when (bortle.coerceIn(1, 9)) {
        1 -> "Ciel vierge"
        2 -> "Desert, haute montagne"
        3 -> "Campagne isolee"
        4 -> "Rural"
        5 -> "Peri-urbain"
        6 -> "Banlieue"
        7 -> "Ville moyenne"
        8 -> "Grande ville"
        else -> "Centre-ville"
    }

    // ------------------------------------------------------------- 5.7 Oculaires

    /** Grossissement = focale_instrument / focale_oculaire. */
    fun magnification(instrumentFocalMm: Double, eyepieceFocalMm: Double): Double =
        instrumentFocalMm / eyepieceFocalMm

    /** Champ reel = champ_apparent / grossissement, en degres. */
    fun trueFieldDeg(apparentFieldDeg: Double, magnification: Double): Double =
        apparentFieldDeg / magnification

    /** Pupille de sortie = focale_oculaire / (F/D), en millimetres. */
    fun exitPupilMm(eyepieceFocalMm: Double, focalRatio: Double): Double =
        eyepieceFocalMm / focalRatio

    /** Grossissement minimal utile = D / pupille_oeil (RG-I-04). */
    fun minMagnification(diameterMm: Double, eyePupilMm: Double): Double =
        diameterMm / eyePupilMm

    /** Grossissement ideal = D / 3. */
    fun idealMagnification(diameterMm: Double): Double = diameterMm / 3.0

    /** Grossissement maximal = D * 1.5. */
    fun maxMagnification(diameterMm: Double): Double = diameterMm * 1.5

    /** Rapport d'ouverture F/D. */
    fun focalRatio(focalMm: Double, diameterMm: Double): Double = focalMm / diameterMm

    /**
     * Diagnostic d'un oculaire selon le grossissement obtenu, exprime avec les
     * bornes de la section 5.7.
     */
    fun eyepieceDiagnosis(
        magnification: Double,
        diameterMm: Double,
        eyePupilMm: Double,
    ): String {
        val gMin = minMagnification(diameterMm, eyePupilMm)
        val gIdeal = idealMagnification(diameterMm)
        val gMax = maxMagnification(diameterMm)
        return when {
            magnification < gMin -> "Sous le grossissement minimal : une partie du faisceau depasse la pupille de l'oeil"
            magnification > gMax -> "Au-dela du grossissement maximal : image sombre et empatee"
            magnification in (gIdeal * 0.7)..(gIdeal * 1.4) -> "Proche du grossissement ideal pour le ciel profond"
            magnification > gIdeal -> "Grossissement eleve : planetaire et objets serres"
            else -> "Grossissement modere : grands champs et amas ouverts"
        }
    }

    // ------------------------------------------------------ 5.8 Astrophotographie

    /** F/D effectif = (focale * barlow) / diametre. */
    fun effectiveFocalRatio(focalMm: Double, barlow: Double, diameterMm: Double): Double =
        (focalMm * barlow) / diameterMm

    /** Echantillonnage = (taille_pixel / focale_eff) * 206.265, en secondes d'arc par pixel. */
    fun samplingArcsecPerPixel(pixelSizeUm: Double, effectiveFocalMm: Double): Double =
        (pixelSizeUm / effectiveFocalMm) * 206.265

    /** F/D minimal recommande = taille_pixel * 3.5 (critere de Shannon). */
    fun minRecommendedFocalRatio(pixelSizeUm: Double): Double = pixelSizeUm * 3.5

    /** F/D ideal = taille_pixel * 5.0. */
    fun idealFocalRatio(pixelSizeUm: Double): Double = pixelSizeUm * 5.0

    /** F/D maximal = taille_pixel * 8.0. */
    fun maxRecommendedFocalRatio(pixelSizeUm: Double): Double = pixelSizeUm * 8.0

    /** Champ couvert par un capteur, en degres, pour une dimension donnee en millimetres. */
    fun sensorFieldDeg(sensorSizeMm: Double, effectiveFocalMm: Double): Double =
        Math.toDegrees(2.0 * kotlin.math.atan(sensorSizeMm / (2.0 * effectiveFocalMm)))

    /**
     * Conseil de la section 5.8 : l'echantillonnage optimal en planetaire se
     * situe entre 0.5 et 2.0 secondes d'arc par pixel.
     */
    fun samplingDiagnosis(arcsecPerPixel: Double): String = when {
        arcsecPerPixel < 0.5 -> "Sur-echantillonnage : image molle, poses inutilement longues"
        arcsecPerPixel <= 2.0 -> "Echantillonnage optimal en planetaire"
        else -> "Sous-echantillonnage : perte de detail"
    }

    // --------------------------------------------------- 5.9 Smart telescopes

    /**
     * Section 5.9 — magnitude limite d'un smart telescope :
     *
     *     mag_limite = 2.1 + 5*log10(D_mm) + 2.5*log10(T_sec/60) - (Bortle-1)*0.55
     *
     * @param exposureSeconds duree de pose cumulee, en secondes (RG-I-05).
     */
    fun smartTelescopeLimitingMagnitude(
        diameterMm: Double,
        exposureSeconds: Double,
        bortle: Int,
    ): Double = 2.1 + 5.0 * log10(diameterMm) +
        2.5 * log10(max(exposureSeconds, 1.0) / 60.0) -
        (bortle.coerceIn(1, 9) - 1) * 0.55

    // ------------------------------------------- 4.2 Filtrage dynamique nocturne

    /** Resultat du filtrage dynamique de la section 4.2. */
    data class DarknessLimits(
        /** Magnitude limite retenue pour les objets du ciel profond ; null = aucun DSO. */
        val deepSkyLimit: Double?,
        /** Magnitude limite retenue pour les planetes ; null = aucune planete. */
        val planetLimit: Double?,
        /** Vrai si seule la Lune est proposee (RG-P-02). */
        val moonOnly: Boolean,
        val phaseLabel: String,
    )

    /**
     * Section 4.2 — la magnitude limite retenue depend de la hauteur du Soleil.
     *
     *     > 0        Jour                 aucun DSO, Lune seule
     *     0 a -6     Crepuscule civil     aucun DSO ; Lune, Venus, Jupiter
     *     -6 a -12   Crepuscule nautique  DSO mag <= 2, planetes mag <= 3
     *     -12 a -18  Crepuscule astro.    ouverture progressive, toutes planetes
     *     < -18      Nuit noire           magnitude limite instrument
     *
     * Interpolation en crepuscule astronomique :
     *
     *     fraction = (-alt_soleil - 12) / 6
     *     mag_limite = 3 + fraction * (mag_limite_instrument - 3)
     */
    fun darknessLimits(sunAltitudeDeg: Double, instrumentLimit: Double): DarknessLimits = when {
        sunAltitudeDeg > 0.0 ->
            DarknessLimits(null, null, moonOnly = true, phaseLabel = "Jour")

        sunAltitudeDeg > -6.0 ->
            // Seules la Lune, Venus et Jupiter (mag < -1) passent : RG-P-03.
            DarknessLimits(null, -1.0, moonOnly = false, phaseLabel = "Crepuscule civil")

        sunAltitudeDeg > -12.0 ->
            DarknessLimits(2.0, 3.0, moonOnly = false, phaseLabel = "Crepuscule nautique")

        sunAltitudeDeg > -18.0 -> {
            val fraction = ((-sunAltitudeDeg - 12.0) / 6.0).coerceIn(0.0, 1.0)
            val limit = 3.0 + fraction * (instrumentLimit - 3.0)
            DarknessLimits(limit, null, moonOnly = false, phaseLabel = "Crepuscule astronomique")
        }

        else ->
            DarknessLimits(instrumentLimit, null, moonOnly = false, phaseLabel = "Nuit noire")
    }

    // ------------------------------------------------------------- 6.2 Seeing

    /** Section 6.2 — indice de seeing (1 a 5) deduit du vent, en km/h. */
    fun seeingIndex(windKmh: Double): Int = when {
        windKmh < 5.0 -> 5
        windKmh < 15.0 -> 4
        windKmh < 25.0 -> 3
        windKmh <= 40.0 -> 2
        else -> 1
    }

    /** Libelle de l'indice de seeing (section 6.2). */
    fun seeingLabel(index: Int): String = when (index.coerceIn(1, 5)) {
        5 -> "Excellent"
        4 -> "Bon"
        3 -> "Correct"
        2 -> "Mauvais"
        else -> "Tres mauvais"
    }

    // -------------------------------------------------------- 6.5 Lecture du score

    /** Section 6.5 — interpretation du score. */
    fun scoreInterpretation(score: Double): Pair<String, String> = when {
        score >= 85 -> "Conditions excellentes" to "Cible prioritaire de la soiree"
        score >= 70 -> "Tres favorable" to "A programmer sans hesitation"
        score >= 50 -> "Observable" to "Correct, viser une meilleure altitude si possible"
        score >= 25 -> "Difficile" to "Reserve aux observateurs experimentes"
        score >= 1 -> "Tres difficile" to "Conditions marginales"
        else -> "Non observable" to "Filtre eliminatoire declenche"
    }

    /** clip(x, lo, hi) utilise par les formules de scoring. */
    fun clip(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
        min(max(value, lo), hi)
}

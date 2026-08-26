package com.cielscore.app

import com.cielscore.app.astro.AstroMath
import com.cielscore.app.astro.SolarSystem
import com.cielscore.app.astro.Twilight
import com.cielscore.app.scoring.Formulas
import com.cielscore.app.scoring.ScoringEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

/**
 * Conformite du moteur de calcul a la documentation CielScore.
 *
 * Chaque test rejoue un tableau ou un exemple chiffre du document ; les
 * references de section sont indiquees dans les noms de methode.
 */
class DocumentationConformanceTest {

    private fun utc(y: Int, month: Int, d: Int, h: Int, min: Int): Long {
        val c = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
        c.clear()
        c.set(y, month - 1, d, h, min, 0)
        return c.timeInMillis
    }

    // ------------------------------------------------ 5.1 Magnitude limite

    @Test
    fun `section 5-1 magnitude limite instrument`() {
        assertEquals(6.3, Formulas.instrumentLimitingMagnitude(7.0), 0.05)
        assertEquals(10.6, Formulas.instrumentLimitingMagnitude(50.0), 0.05)
        assertEquals(11.3, Formulas.instrumentLimitingMagnitude(70.0), 0.05)
        assertEquals(12.7, Formulas.instrumentLimitingMagnitude(130.0), 0.05)
        assertEquals(13.6, Formulas.instrumentLimitingMagnitude(200.0), 0.05)
        assertEquals(14.5, Formulas.instrumentLimitingMagnitude(300.0), 0.05)
        assertEquals(16.0, Formulas.instrumentLimitingMagnitude(600.0), 0.05)
    }

    @Test
    fun `section 5-2 NELM par indice de Bortle`() {
        val expected = listOf(7.6, 7.1, 6.6, 6.1, 5.6, 5.1, 4.6, 4.1, 3.6)
        expected.forEachIndexed { index, value ->
            assertEquals(value, Formulas.nakedEyeLimitingMagnitude(index + 1), 1e-9)
        }
    }

    @Test
    fun `section 5-3 oeil nu suit le Bortle, instrument suit le diametre`() {
        // RG-I-01 : oeil nu (D <= 7 mm), la limite depend du Bortle.
        assertEquals(4.6, Formulas.limitingMagnitude(7.0, 7), 1e-9)
        // RG-I-02 : instrument optique, la limite ne depend que du diametre.
        assertEquals(
            Formulas.instrumentLimitingMagnitude(130.0),
            Formulas.limitingMagnitude(130.0, 9),
            1e-9,
        )
        assertEquals(
            Formulas.limitingMagnitude(130.0, 1),
            Formulas.limitingMagnitude(130.0, 9),
            1e-9,
        )
    }

    // -------------------------------------------- 5.4 / 5.5 / 5.6 Brillance

    @Test
    fun `section 5-4 exemple M91`() {
        assertEquals(22.24, Formulas.surfaceBrightness(10.2, 5.4, 4.3)!!, 0.01)
    }

    @Test
    fun `section 5-6 tableau des exemples d objets`() {
        assertEquals(17.8, Formulas.surfaceBrightness(8.8, 1.4, 1.0)!!, 0.05)
        assertEquals(20.9, Formulas.surfaceBrightness(5.8, 20.0, 20.0)!!, 0.05)
        assertEquals(22.0, Formulas.surfaceBrightness(6.9, 27.0, 14.0)!!, 0.05)
        assertEquals(22.2, Formulas.surfaceBrightness(3.4, 190.0, 60.0)!!, 0.05)
        assertEquals(23.0, Formulas.surfaceBrightness(5.7, 71.0, 42.0)!!, 0.05)
        assertEquals(23.8, Formulas.surfaceBrightness(7.9, 29.0, 27.0)!!, 0.05)
    }

    @Test
    fun `section 5-5 limites de brillance de surface`() {
        val expected = listOf(22.0, 21.8, 21.5, 21.0, 20.5, 19.5, 18.5, 17.5, 17.0)
        expected.forEachIndexed { index, value ->
            assertEquals(value, Formulas.surfaceBrightnessLimit(index + 1), 1e-9)
        }
    }

    @Test
    fun `dimensions inconnues, la brillance de surface n est pas calculable`() {
        assertEquals(null, Formulas.surfaceBrightness(8.0, null, null))
        assertEquals(null, Formulas.surfaceBrightness(null, 5.0, 5.0))
        // Petit axe absent : l'objet est traite comme circulaire.
        assertEquals(
            Formulas.surfaceBrightness(8.0, 4.0, 4.0)!!,
            Formulas.surfaceBrightness(8.0, 4.0, null)!!,
            1e-9,
        )
    }

    // ---------------------------------------------------- 5.7 / 5.8 Optique

    @Test
    fun `section 5-7 oculaires`() {
        assertEquals(65.0, Formulas.magnification(650.0, 10.0), 1e-9)
        assertEquals(0.8, Formulas.trueFieldDeg(52.0, 65.0), 1e-9)
        assertEquals(2.0, Formulas.exitPupilMm(10.0, 5.0), 1e-9)
        assertEquals(130.0 / 6.0, Formulas.minMagnification(130.0, 6.0), 1e-9)
        assertEquals(130.0 / 3.0, Formulas.idealMagnification(130.0), 1e-9)
        assertEquals(195.0, Formulas.maxMagnification(130.0), 1e-9)
    }

    @Test
    fun `section 5-8 astrophotographie`() {
        assertEquals(10.0, Formulas.effectiveFocalRatio(650.0, 2.0, 130.0), 1e-9)
        // Echantillonnage = (taille_pixel / focale_eff) * 206.265
        assertEquals(1.2376, Formulas.samplingArcsecPerPixel(3.0, 500.0), 1e-4)
        assertEquals(10.5, Formulas.minRecommendedFocalRatio(3.0), 1e-9)
        assertEquals(15.0, Formulas.idealFocalRatio(3.0), 1e-9)
        assertEquals(24.0, Formulas.maxRecommendedFocalRatio(3.0), 1e-9)
        // Conseil : optimal entre 0.5 et 2.0 "/px
        assertTrue(Formulas.samplingDiagnosis(1.0).contains("optimal"))
        assertTrue(Formulas.samplingDiagnosis(0.2).contains("Sur-echantillonnage"))
        assertTrue(Formulas.samplingDiagnosis(3.0).contains("Sous-echantillonnage"))
    }

    @Test
    fun `section 5-9 magnitude limite smart telescope`() {
        // 2.1 + 5*log10(50) + 2.5*log10(3600/60) - (7-1)*0.55
        assertEquals(11.74, Formulas.smartTelescopeLimitingMagnitude(50.0, 3600.0, 7), 0.02)
        // Un ciel plus sombre releve la limite de 3 * 0.55 = 1.65 magnitude.
        assertEquals(
            Formulas.smartTelescopeLimitingMagnitude(50.0, 3600.0, 7) + 1.65,
            Formulas.smartTelescopeLimitingMagnitude(50.0, 3600.0, 4),
            1e-9,
        )
    }

    // ----------------------------------------------------- 6 Moteur de score

    @Test
    fun `section 6 la somme des ponderations vaut exactement 100 pourcent`() {
        assertEquals(1.0, ScoringEngine.visualWeightsSum(), 1e-9)
        assertEquals(1.0, ScoringEngine.planetWeightsSum(), 1e-9)
        assertEquals(1.0, ScoringEngine.smartWeightsSum(), 1e-9)
    }

    @Test
    fun `section 6-1 critere d altitude, plateau atteint a 30 degres`() {
        assertEquals(0.0, ScoringEngine.altitudeScore(5.0), 1e-9)
        assertEquals(60.0, ScoringEngine.altitudeScore(20.0), 1e-9)
        assertEquals(100.0, ScoringEngine.altitudeScore(30.0), 1e-9)
        assertEquals(100.0, ScoringEngine.altitudeScore(80.0), 1e-9)
    }

    @Test
    fun `section 6-1 critere de fenetre, maximum a 4 heures`() {
        assertEquals(0.0, ScoringEngine.windowScore(0.0), 1e-9)
        assertEquals(50.0, ScoringEngine.windowScore(120.0), 1e-9)
        assertEquals(100.0, ScoringEngine.windowScore(240.0), 1e-9)
        assertEquals(100.0, ScoringEngine.windowScore(600.0), 1e-9)
    }

    @Test
    fun `section 6-2 bareme du seeing deduit du vent`() {
        assertEquals(5, Formulas.seeingIndex(3.0))
        assertEquals(4, Formulas.seeingIndex(10.0))
        assertEquals(3, Formulas.seeingIndex(20.0))
        assertEquals(2, Formulas.seeingIndex(30.0))
        assertEquals(1, Formulas.seeingIndex(50.0))

        assertEquals(100.0, ScoringEngine.seeingScore(5), 1e-9)
        assertEquals(75.0, ScoringEngine.seeingScore(4), 1e-9)
        assertEquals(50.0, ScoringEngine.seeingScore(3), 1e-9)
        assertEquals(25.0, ScoringEngine.seeingScore(2), 1e-9)
        assertEquals(0.0, ScoringEngine.seeingScore(1), 1e-9)
    }

    @Test
    fun `section 6-1 critere de Bortle`() {
        assertEquals(100.0, ScoringEngine.bortleScore(1), 1e-9)
        assertEquals(25.0, ScoringEngine.bortleScore(7), 1e-9)
        assertEquals(0.0, ScoringEngine.bortleScore(9), 1e-9)
    }

    @Test
    fun `section 4-4 regles lunaires`() {
        val fullMoonHigh = com.cielscore.app.model.MoonState(
            0.0, 0.0, 45.0, 180.0, 100.0, "Pleine Lune"
        )
        val newMoonHigh = fullMoonHigh.copy(phasePercent = 0.0)
        val fullMoonBelow = fullMoonHigh.copy(altitudeDeg = -10.0)

        // RG-L-03 : pleine Lune proche de l'objet, penalite maximale.
        assertEquals(0.0, ScoringEngine.moonScore(fullMoonHigh, 0.0), 1e-9)
        // RG-L-04 : nouvelle Lune, aucune penalite.
        assertEquals(100.0, ScoringEngine.moonScore(newMoonHigh, 0.0), 1e-9)
        // RG-L-01 : Lune sous l'horizon, impact nul quelle que soit la phase.
        assertEquals(100.0, ScoringEngine.moonScore(fullMoonBelow, 0.0), 1e-9)
        // RG-L-02 : l'impact combine phase et distance angulaire.
        assertEquals(50.0, ScoringEngine.moonScore(fullMoonHigh, 90.0), 1e-9)
    }

    @Test
    fun `section 6-5 lecture du score`() {
        assertEquals("Conditions excellentes", Formulas.scoreInterpretation(90.0).first)
        assertEquals("Tres favorable", Formulas.scoreInterpretation(75.0).first)
        assertEquals("Observable", Formulas.scoreInterpretation(60.0).first)
        assertEquals("Difficile", Formulas.scoreInterpretation(30.0).first)
        assertEquals("Tres difficile", Formulas.scoreInterpretation(10.0).first)
        assertEquals("Non observable", Formulas.scoreInterpretation(0.0).first)
    }

    // -------------------------------------------------- 4.2 / 7 Crepuscules

    @Test
    fun `section 7-2 score de nuit`() {
        assertEquals(100.0, Twilight.nightScore(-45.0), 1e-9)
        assertEquals(70.0, Twilight.nightScore(-15.0), 1e-9)
        assertEquals(40.0, Twilight.nightScore(-9.0), 1e-9)
        assertEquals(10.0, Twilight.nightScore(-3.0), 1e-9)
        assertEquals(0.0, Twilight.nightScore(5.0), 1e-9)
    }

    @Test
    fun `section 7-1 phases du crepuscule`() {
        assertEquals(Twilight.Phase.NIGHT, Twilight.phaseOf(-20.0))
        assertEquals(Twilight.Phase.ASTRONOMICAL, Twilight.phaseOf(-15.0))
        assertEquals(Twilight.Phase.NAUTICAL, Twilight.phaseOf(-9.0))
        assertEquals(Twilight.Phase.CIVIL, Twilight.phaseOf(-3.0))
        assertEquals(Twilight.Phase.DAY, Twilight.phaseOf(2.0))
    }

    @Test
    fun `section 4-2 filtrage dynamique selon l obscurite`() {
        val instrument = Formulas.instrumentLimitingMagnitude(130.0)

        // Jour : aucun objet diffus, la Lune seule.
        val day = Formulas.darknessLimits(5.0, instrument)
        assertEquals(null, day.deepSkyLimit)
        assertTrue(day.moonOnly)

        // Crepuscule civil : toujours aucun DSO.
        assertEquals(null, Formulas.darknessLimits(-3.0, instrument).deepSkyLimit)

        // Crepuscule nautique : DSO mag <= 2, planetes mag <= 3.
        val nautical = Formulas.darknessLimits(-9.0, instrument)
        assertEquals(2.0, nautical.deepSkyLimit!!, 1e-9)
        assertEquals(3.0, nautical.planetLimit!!, 1e-9)

        // Crepuscule astronomique : ouverture progressive.
        // fraction = (-alt - 12) / 6 ; mag = 3 + fraction * (mag_instrument - 3)
        assertEquals(3.0, Formulas.darknessLimits(-12.001, instrument).deepSkyLimit!!, 0.01)
        assertEquals(
            3.0 + 0.5 * (instrument - 3.0),
            Formulas.darknessLimits(-15.0, instrument).deepSkyLimit!!,
            0.01,
        )

        // Nuit noire : magnitude limite de l'instrument.
        assertEquals(instrument, Formulas.darknessLimits(-20.0, instrument).deepSkyLimit!!, 1e-9)
    }

    @Test
    fun `section 7-4 pas de nuit astronomique en ete au-dela de 49 degres nord`() {
        // Exemple du document : latitude 50,4 N.
        val summer = Twilight.computeNight(utc(2025, 5, 30, 18, 0), 50.4, 4.0)
        assertFalse(summer.astronomical.exists)

        val winter = Twilight.computeNight(utc(2025, 12, 21, 12, 0), 50.4, 4.0)
        assertTrue(winter.astronomical.exists)
        // Le document annonce environ 12 heures de nuit astronomique le 21 decembre.
        assertEquals(12.0, winter.astronomical.durationMinutes / 60.0, 0.6)
    }

    // ------------------------------------------------------- 3.4 / 3.5 Carte

    @Test
    fun `section 3-5 conversion equatorial vers horizontal`() {
        // Polaris reste a une altitude proche de la latitude de l'observateur.
        val jd = AstroMath.julianDay(utc(2025, 12, 21, 21, 0))
        val lst = AstroMath.localSiderealTimeDeg(jd, 2.35)
        val polaris = AstroMath.equatorialToHorizontal(37.9546, 89.2641, 48.85, lst)
        assertEquals(48.85, polaris.altitudeDeg, 1.0)
        assertEquals(0.0, AstroMath.norm180(polaris.azimuthDeg), 3.0)
    }

    @Test
    fun `section 3-4 projection azimutale equidistante`() {
        val radius = 100f
        // Le zenith est au centre.
        val zenith = com.cielscore.app.astro.SkyProjection.project(90.0, 0.0, 0f, 0f, radius)
        assertEquals(0.0, zenith.x.toDouble(), 1e-4)
        assertEquals(0.0, zenith.y.toDouble(), 1e-4)
        // L'horizon est sur le cercle exterieur, au Nord.
        val north = com.cielscore.app.astro.SkyProjection.project(0.0, 0.0, 0f, 0f, radius)
        assertEquals(0.0, north.x.toDouble(), 1e-4)
        assertEquals(-100.0, north.y.toDouble(), 1e-4)
        // L'Est est a droite.
        val east = com.cielscore.app.astro.SkyProjection.project(0.0, 90.0, 0f, 0f, radius)
        assertEquals(100.0, east.x.toDouble(), 1e-4)
        // Distances angulaires conservees depuis le zenith.
        assertEquals(
            50f,
            com.cielscore.app.astro.SkyProjection.altitudeCircleRadius(45.0, radius),
            1e-4f,
        )
    }

    // -------------------------------------------------------- Ephemerides

    @Test
    fun `ephemerides du Soleil aux solstices et equinoxes`() {
        val solstice = SolarSystem.sun(AstroMath.julianDay(utc(2025, 12, 21, 15, 3)))
        assertEquals(-23.44, solstice.decDeg, 0.05)
        val equinox = SolarSystem.sun(AstroMath.julianDay(utc(2025, 3, 20, 9, 1)))
        assertEquals(0.0, equinox.decDeg, 0.05)
    }

    @Test
    fun `ephemerides lunaires aux syzygies`() {
        // Pleine Lune du 17 octobre 2024 a 11h26 TU.
        val full = SolarSystem.moon(AstroMath.julianDay(utc(2024, 10, 17, 11, 26)))
        assertEquals(100.0, full.illuminatedFraction * 100.0, 1.0)
        // Nouvelle Lune du 1er novembre 2024 a 12h47 TU.
        val new = SolarSystem.moon(AstroMath.julianDay(utc(2024, 11, 1, 12, 47)))
        assertEquals(0.0, new.illuminatedFraction * 100.0, 1.0)
    }

    // ------------------------------------------------ 10.2 Exemple complet

    @Test
    fun `section 10-2 exemple complet M57 depuis un site Bortle 7`() {
        val magLimit = Formulas.limitingMagnitude(130.0, 7)
        assertEquals(12.7, magLimit, 0.05)

        val sb = Formulas.surfaceBrightness(8.8, 1.4, 1.0)!!
        assertEquals(17.8, sb, 0.05)

        val sbLimit = Formulas.surfaceBrightnessLimit(7)
        assertEquals(18.5, sbLimit, 1e-9)

        // Filtre brillance : 17.8 <= 18.5, passe.
        assertTrue(sb <= sbLimit)
        // Filtre magnitude : 8.8 <= 12.7, passe.
        assertTrue(8.8 <= magLimit)
        // Score altitude a 35 degres.
        assertEquals(100.0, ScoringEngine.altitudeScore(35.0), 1e-9)
        // Score brillance.
        assertEquals(14.0, ScoringEngine.surfaceBrightnessScore(sb, 7), 1.0)
        // Score de nuit en decembre, Soleil a -45 degres.
        assertEquals(100.0, Twilight.nightScore(-45.0), 1e-9)

        // Somme ponderee : le document annonce environ 78 sur 100.
        val total = 0.25 * ScoringEngine.altitudeScore(35.0) +
            0.15 * ScoringEngine.windowScore(240.0) +
            0.11 * ScoringEngine.seeingScore(4) +
            0.13 * 100.0 +
            0.08 * ScoringEngine.bortleScore(7) +
            0.06 * 100.0 +
            0.15 * ScoringEngine.surfaceBrightnessScore(sb, 7) +
            0.07 * Twilight.nightScore(-45.0)
        assertEquals(78.0, total, 1.0)
    }
}

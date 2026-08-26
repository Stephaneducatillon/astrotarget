package com.cielscore.app

import com.cielscore.app.model.SmartTelescope
import com.cielscore.app.scoring.Formulas
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Catalogue des smart telescopes.
 *
 * Le tableau de la section 5.9 fournit uniquement le diametre de chaque
 * modele ; le reste vient des specifications constructeur. Ces controles
 * verrouillent la coherence du tableau et la seule grandeur reellement
 * documentee, la magnitude limite.
 */
class SmartTelescopeCatalogTest {

    private val catalog = SmartTelescope.CATALOG

    @Test
    fun `les sept modeles de la section 5-9 sont presents avec leur diametre`() {
        val documented = mapOf(
            "Seestar S30" to 30.0,
            "Seestar S50" to 50.0,
            "Vespera II" to 50.0,
            "Stellina" to 80.0,
            "Odyssey Pro" to 85.0,
            "eVscope 2" to 114.0,
            "Origin Mk II" to 152.0,
        )
        documented.forEach { (model, diameter) ->
            val scope = catalog.firstOrNull { it.model == model }
            assertTrue("modele absent du catalogue : $model", scope != null)
            assertEquals("diametre errone pour $model", diameter, scope!!.diameterMm, 1e-9)
        }
    }

    @Test
    fun `les deux modeles Pro ajoutes sont presents`() {
        val s30 = catalog.first { it.model == "Seestar S30 Pro" }
        assertEquals(30.0, s30.diameterMm, 1e-9)
        assertEquals(150.0, s30.focalMm, 1e-9)
        assertEquals("IMX585", s30.sensorName)

        val s50 = catalog.first { it.model == "Seestar S50 Pro" }
        assertEquals(50.0, s50.diameterMm, 1e-9)
        assertEquals(260.0, s50.focalMm, 1e-9)
        assertEquals("IMX585", s50.sensorName)
    }

    @Test
    fun `les dimensions de l IMX585 decoulent de sa definition`() {
        // 3856 x 2180 pixels au pas de 2,9 um, soit une diagonale de 12,85 mm
        // qui correspond au format 1/1,2 pouce.
        catalog.filter { it.sensorName == "IMX585" }.forEach { scope ->
            assertEquals(3856 * 0.0029, scope.sensorWidthMm, 0.01)
            assertEquals(2180 * 0.0029, scope.sensorHeightMm, 0.01)
            val diagonal = kotlin.math.hypot(scope.sensorWidthMm, scope.sensorHeightMm)
            assertEquals(12.85, diagonal, 0.05)
        }
    }

    @Test
    fun `la magnitude limite ne depend que du diametre, de la pose et du Bortle`() {
        // Deux modeles de meme diametre partagent la meme magnitude limite,
        // quelles que soient leur focale et leur optique (section 5.9).
        val s50 = catalog.first { it.model == "Seestar S50" }
        val s50Pro = catalog.first { it.model == "Seestar S50 Pro" }
        assertEquals(
            s50.limitingMagnitude(60.0, 6),
            s50Pro.limitingMagnitude(60.0, 6),
            1e-9,
        )
        // Valeur de reference : 2.1 + 5*log10(50) + 2.5*log10(60) - 5*0.55
        assertEquals(
            Formulas.smartTelescopeLimitingMagnitude(50.0, 3600.0, 6),
            s50Pro.limitingMagnitude(60.0, 6),
            1e-9,
        )
    }

    @Test
    fun `le catalogue est coherent`() {
        assertEquals("noms de modeles dupliques", catalog.size, catalog.map { it.name }.toSet().size)
        catalog.forEach { scope ->
            assertTrue("${scope.name} : diametre invalide", scope.diameterMm > 0)
            assertTrue("${scope.name} : focale invalide", scope.focalMm > 0)
            assertTrue("${scope.name} : capteur invalide", scope.sensorWidthMm > 0)
            assertTrue("${scope.name} : capteur invalide", scope.sensorHeightMm > 0)
            assertTrue("${scope.name} : pixel invalide", scope.pixelSizeUm > 0)
            assertTrue("${scope.name} : capteur non paysage", scope.sensorWidthMm >= scope.sensorHeightMm)
            // Les smart telescopes sont des instruments ouverts, entre f/2 et f/6.
            assertTrue(
                "${scope.name} : rapport F/D improbable (f/%.1f)".format(scope.focalRatio),
                scope.focalRatio in 2.0..6.0,
            )
            assertTrue("${scope.name} : champ nul", scope.fieldWidthArcmin > 0)
        }
    }
}

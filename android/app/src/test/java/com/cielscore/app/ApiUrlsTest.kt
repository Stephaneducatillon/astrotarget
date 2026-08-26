package com.cielscore.app

import com.cielscore.app.data.net.ApiUrls
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.Locale

/**
 * Construction des URL appelees par l'application.
 *
 * Ces tests tournent sous une locale francaise, celle des appareils vises :
 * c'est precisement la que String.format produit une virgule decimale et
 * casse silencieusement les appels reseau.
 */
class ApiUrlsTest {

    private lateinit var original: Locale

    @Before
    fun useFrenchLocale() {
        original = Locale.getDefault()
        Locale.setDefault(Locale.FRANCE)
    }

    @After
    fun restoreLocale() {
        Locale.setDefault(original)
    }

    @Test
    fun `la locale de l appareil produit bien une virgule, d ou le risque`() {
        // Constat de depart : c'est ce comportement que ApiUrls neutralise.
        assertEquals("50,3784", "%.4f".format(50.3784))
    }

    @Test
    fun `le formatage des URL impose le point decimal`() {
        assertEquals("50.3784", ApiUrls.decimal(50.3784, 4))
        assertEquals("3.05", ApiUrls.decimal(3.0514, 2))
        assertEquals("-1.2500", ApiUrls.decimal(-1.25, 4))
    }

    @Test
    fun `l URL Open-Meteo porte des coordonnees a point decimal`() {
        // Coordonnees reelles de Cuincy, dans le Nord.
        val url = ApiUrls.openMeteoForecast(50.3784, 3.0514)
        assertTrue(url, url.contains("latitude=50.3784"))
        assertTrue(url, url.contains("longitude=3.0514"))
        // La liste hourly= separe ses variables par des virgules : seules les
        // valeurs numeriques doivent en etre exemptes.
        val coordinates = url.substringBefore("&hourly=")
        assertFalse("virgule decimale dans $coordinates", coordinates.contains(","))
    }

    @Test
    fun `l URL Open-Meteo demande bien les grandeurs utiles au scoring`() {
        val url = ApiUrls.openMeteoForecast(48.85, 2.35)
        // Couverture nuageuse : filtre RG-F-04 et critere de transparence.
        assertTrue(url.contains("cloud_cover"))
        // Vent : barème du seeing de la section 6.2.
        assertTrue(url.contains("wind_speed_10m"))
        assertTrue(url.contains("wind_speed_unit=kmh"))
        // Les instants sont demandes en UTC, comme le reste du moteur.
        assertTrue(url.contains("timezone=UTC"))
        assertTrue(url.contains("timeformat=unixtime"))
    }

    @Test
    fun `l URL hips2fits porte des coordonnees a point decimal`() {
        val url = ApiUrls.hips2fits("CDS/P/DSS2/color", 283.3959, 33.0286, 0.5, 512)
        assertTrue(url, url.contains("ra=283.395900"))
        assertTrue(url, url.contains("dec=33.028600"))
        assertTrue(url, url.contains("fov=0.5000"))
        assertFalse("virgule decimale dans l'URL : $url", url.contains(","))
        // L'identifiant du releve est encode : les barres obliques passent.
        assertTrue(url, url.contains("hips=CDS%2FP%2FDSS2%2Fcolor"))
    }

    @Test
    fun `le champ hips2fits reste dans les bornes du service`() {
        assertTrue(ApiUrls.hips2fits("x", 0.0, 0.0, 42.0, 256).contains("fov=5.0000"))
        assertTrue(ApiUrls.hips2fits("x", 0.0, 0.0, 0.0001, 256).contains("fov=0.0500"))
    }

    @Test
    fun `l URL Stellarium porte des coordonnees a point decimal`() {
        val url = ApiUrls.stellariumWeb("NGC 6720", 50.3784, 3.0514, "2026-08-28T02:00:00Z")
        assertTrue(url, url.contains("lat=50.3784"))
        assertTrue(url, url.contains("lng=3.0514"))
        assertFalse("virgule decimale dans l'URL : $url", url.contains(","))
        // L'espace du nom d'objet est encode.
        assertTrue(url, url.contains("NGC+6720"))
    }

    @Test
    fun `la cle du cache meteo est stable d une locale a l autre`() {
        val french = ApiUrls.weatherCacheKey(50.3784, 3.0514, 1_756_000_000_000L)
        Locale.setDefault(Locale.US)
        val english = ApiUrls.weatherCacheKey(50.3784, 3.0514, 1_756_000_000_000L)
        assertEquals(french, english)
        assertTrue(french, french.startsWith("50.38_3.05_"))
    }

    @Test
    fun `la cle du cache meteo change d une heure a l autre`() {
        val hour = 60 * 60 * 1000L
        val base = 1_756_000_000_000L
        assertEquals(
            ApiUrls.weatherCacheKey(50.0, 3.0, base),
            ApiUrls.weatherCacheKey(50.0, 3.0, base + 60_000L),
        )
        assertTrue(
            ApiUrls.weatherCacheKey(50.0, 3.0, base) !=
                ApiUrls.weatherCacheKey(50.0, 3.0, base + hour),
        )
    }

    @Test
    fun `aucune URL construite ne contient d espace`() {
        listOf(
            ApiUrls.openMeteoForecast(50.3784, 3.0514),
            ApiUrls.hips2fits("CDS/P/DSS2/color", 10.0, -20.0, 1.0, 512),
            ApiUrls.stellariumWeb("Ring Nebula", 50.0, 3.0, "2026-08-28T02:00:00Z"),
        ).forEach { url ->
            // Une virgule decimale echappee du formatage se lirait ici.
            assertFalse("virgule suspecte : $url", Regex("=[-0-9]+,[0-9]").containsMatchIn(url))
            assertFalse("espace dans l'URL : $url", url.contains(" "))
            assertTrue("URL non https : $url", url.startsWith("https://"))
        }
    }
}

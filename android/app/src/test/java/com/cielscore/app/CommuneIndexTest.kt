package com.cielscore.app

import com.cielscore.app.catalog.Commune
import com.cielscore.app.catalog.CommuneIndex
import com.cielscore.app.catalog.toObservingSite
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Recherche de commune et rattachement GPS, a partir du fichier
 * communes_bortle.csv de la section 8.1.
 */
class CommuneIndexTest {

    /** Quelques lignes reelles du fichier embarque. */
    private val sample = listOf(
        "01001;L'Abergement-Clémenciat;01;Ain;860;46.1331;4.99858;4;Transition rural/périurbain",
        "05150;Saint-Véran;05;Hautes-Alpes;160;44.73984;6.81323;2;Ciel vraiment noir",
        "13055;Marseille;13;Bouches-du-Rhône;886040;43.30348;5.39555;8;Ciel urbain",
        "18165;Nançay;18;Cher;748;47.31851;2.20935;2;Ciel vraiment noir",
        "69123;Lyon;69;Rhône;519127;45.75889;4.83531;9;Ciel de centre-ville",
        "75056;Paris;75;Paris;2103778;48.86029;2.34443;9;Ciel de centre-ville",
    ).mapNotNull(CommuneIndex::parseLine)

    @Test
    fun `l en-tete et les lignes invalides sont ecartees`() {
        assertNull(
            CommuneIndex.parseLine(
                "code_insee;commune;code_departement;departement;population;lat;lng;" +
                    "bortle_estime;description_ciel"
            )
        )
        assertNull(CommuneIndex.parseLine(""))
        assertNull(CommuneIndex.parseLine("75056;Paris"))
        // Coordonnees illisibles.
        assertNull(CommuneIndex.parseLine("75056;Paris;75;Paris;2103778;abc;2.34;9;x"))
    }

    @Test
    fun `une ligne est analysee dans son integralite`() {
        val paris = CommuneIndex.parseLine(
            "75056;Paris;75;Paris;2103778;48.86029;2.34443;9;Ciel de centre-ville"
        )!!
        assertEquals("75056", paris.inseeCode)
        assertEquals("Paris", paris.name)
        assertEquals("75", paris.departmentCode)
        assertEquals("Paris", paris.departmentName)
        assertEquals(2_103_778, paris.population)
        assertEquals(48.86029, paris.latitude, 1e-6)
        assertEquals(2.34443, paris.longitude, 1e-6)
        assertEquals(9, paris.bortle)
    }

    @Test
    fun `la normalisation ignore accents, apostrophes et traits d union`() {
        assertEquals("saintveran", CommuneIndex.normalize("Saint-Véran"))
        assertEquals("labergementclemenciat", CommuneIndex.normalize("L'Abergement-Clémenciat"))
        assertEquals("nancay", CommuneIndex.normalize("Nançay"))
        assertEquals("paris", CommuneIndex.normalize("  PARIS  "))
    }

    @Test
    fun `la recherche tolere l absence d accent et de ponctuation`() {
        assertEquals("Saint-Véran", CommuneIndex.search(sample, "saint veran").first().name)
        assertEquals("Saint-Véran", CommuneIndex.search(sample, "SAINT-VÉRAN").first().name)
        assertEquals("Nançay", CommuneIndex.search(sample, "nancay").first().name)
        // Recherche sur un fragment interne.
        assertEquals("L'Abergement-Clémenciat", CommuneIndex.search(sample, "clemenciat").first().name)
    }

    @Test
    fun `a rang egal la commune la plus peuplee vient en tete`() {
        // « pari » ne prefixe que Paris dans l'echantillon.
        assertEquals("Paris", CommuneIndex.search(sample, "pari").first().name)
        // Une saisie trop courte ne renvoie rien.
        assertTrue(CommuneIndex.search(sample, "p").isEmpty())
    }

    @Test
    fun `le prefixe passe devant la simple occurrence`() {
        val communes = listOf(
            "10001;Ville-sous-Anjou;38;Isere;900;45.4;4.8;3;Ciel rural",
            "10002;Anjou;38;Isere;100;45.5;4.9;3;Ciel rural",
        ).mapNotNull(CommuneIndex::parseLine)
        // Anjou est moins peuple, mais son nom commence par la saisie.
        assertEquals("Anjou", CommuneIndex.search(communes, "anjou").first().name)
    }

    @Test
    fun `le rattachement GPS retient la commune la plus proche`() {
        // Position au coeur de Paris.
        assertEquals("Paris", CommuneIndex.nearest(sample, 48.8566, 2.3522)!!.name)
        // Position dans le Queyras, a quelques kilometres de Saint-Veran.
        assertEquals("Saint-Véran", CommuneIndex.nearest(sample, 44.75, 6.80)!!.name)
        assertNotNull(CommuneIndex.nearest(sample, 45.0, 5.0))
        assertNull(CommuneIndex.nearest(emptyList(), 45.0, 5.0))
    }

    @Test
    fun `la distance orthodromique est correcte`() {
        // Paris - Marseille : environ 660 km.
        val d = CommuneIndex.distanceKm(48.86029, 2.34443, 43.30348, 5.39555)
        assertEquals(660.0, d, 15.0)
        assertEquals(0.0, CommuneIndex.distanceKm(45.0, 5.0, 45.0, 5.0), 1e-9)
    }

    @Test
    fun `un lieu d observation herite du Bortle de la commune`() {
        val site = sample.first { it.name == "Saint-Véran" }.toObservingSite()
        assertEquals("Saint-Véran", site.name)
        assertEquals(2, site.bortle)
        assertEquals("05", site.department)
        // La colonne du fichier s'appelle bortle_estime : l'indice reste modifiable.
        assertTrue(site.bortleEstimated)
    }

    @Test
    fun `le Bortle est ramene dans la plage 1 a 9`() {
        val out = CommuneIndex.parseLine("99999;Test;99;Test;10;45.0;5.0;42;x")!!
        assertEquals(9, out.bortle)
    }
}

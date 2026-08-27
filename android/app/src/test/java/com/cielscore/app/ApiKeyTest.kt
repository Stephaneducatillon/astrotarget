package com.cielscore.app

import com.cielscore.app.model.ApiKey
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Affichage des cles d'API dans le Profil (section 8.2).
 *
 * Les cles restent sur l'appareil, mais l'ecran n'en montre que la fin :
 * assez pour reconnaitre celle qui est en place, pas assez pour la relever
 * par-dessus l'epaule.
 */
class ApiKeyTest {

    @Test
    fun `seule la fin de la cle est revelee`() {
        assertEquals("••••wxyz", ApiKey.mask("abcdefghijklmnopqrstuvwxyz"))
        assertEquals("••••3456", ApiKey.mask("DEMO_KEY_123456"))
    }

    @Test
    fun `le masque ne laisse pas fuir le debut de la cle`() {
        val key = "sk-tres-secret-0000-final"
        val masked = ApiKey.mask(key)
        assertFalse(masked.contains("secret"))
        assertFalse(masked.contains("sk-"))
        assertTrue(masked.endsWith("inal"))
    }

    @Test
    fun `une cle absente ou vide ne revele rien`() {
        assertEquals("", ApiKey.mask(null))
        assertEquals("", ApiKey.mask(""))
        assertEquals("", ApiKey.mask("   "))
    }

    @Test
    fun `une cle tres courte est entierement masquee`() {
        assertEquals("••••", ApiKey.mask("abcd"))
        assertEquals("••", ApiKey.mask("ab"))
    }

    @Test
    fun `le libelle d etat distingue la presence d une cle`() {
        assertEquals("Aucune cle enregistree", ApiKey.statusLabel(null))
        assertEquals("Aucune cle enregistree", ApiKey.statusLabel(""))
        assertTrue(ApiKey.statusLabel("DEMO_KEY_123456").startsWith("Cle enregistree"))
        assertTrue(ApiKey.statusLabel("DEMO_KEY_123456").endsWith("3456"))
    }
}

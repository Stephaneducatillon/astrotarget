package com.cielscore.app.model

/**
 * Presentation d'une cle d'API enregistree (section 8.2).
 *
 * Les cles restent sur l'appareil, mais rien n'oblige a les afficher en clair
 * dans le Profil : seule leur fin sert a reconnaitre celle qui est en place.
 */
object ApiKey {

    /**
     * Masque une cle en n'en laissant que les derniers caracteres.
     * Une cle vide, ou trop courte pour etre masquee utilement, ne revele rien.
     */
    fun mask(key: String?, visibleChars: Int = 4): String {
        val trimmed = key?.trim().orEmpty()
        if (trimmed.isEmpty()) return ""
        if (trimmed.length <= visibleChars) return "•".repeat(trimmed.length)
        return "•".repeat(4) + trimmed.takeLast(visibleChars)
    }

    /** Libelle d'etat affiche sous le champ de saisie. */
    fun statusLabel(key: String?): String {
        val masked = mask(key)
        return if (masked.isEmpty()) "Aucune cle enregistree"
        else "Cle enregistree — $masked"
    }
}

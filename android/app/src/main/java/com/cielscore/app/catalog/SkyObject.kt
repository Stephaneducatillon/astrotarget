package com.cielscore.app.catalog

import com.cielscore.app.astro.SolarSystem
import com.cielscore.app.scoring.Formulas

/** Catalogues proposes au Dashboard et a l'Explorer (sections 2.2, 2.3, 8.1). */
enum class Catalog(val label: String) {
    MESSIER("Messier"),
    CALDWELL("Caldwell"),
    NGC_IC("NGC/IC"),
    PLANETS("Planetes"),
}

/** Les cinq types d'objets du filtre Explorer (section 2.3). */
enum class ObjectType(val label: String) {
    NEBULA("Nebuleuse"),
    GALAXY("Galaxie"),
    OPEN_CLUSTER("Amas ouvert"),
    GLOBULAR_CLUSTER("Amas globulaire"),
    OTHER("Autre");

    companion object {
        fun fromAsset(raw: String): ObjectType = when (raw.trim()) {
            "Nebuleuse" -> NEBULA
            "Galaxie" -> GALAXY
            "Amas ouvert" -> OPEN_CLUSTER
            "Amas globulaire" -> GLOBULAR_CLUSTER
            else -> OTHER
        }
    }
}

/**
 * Un objet du ciel, ciel profond ou corps du systeme solaire.
 *
 * Les coordonnees des objets de catalogue sont fixes (J2000) ; celles des
 * planetes et de la Lune sont recalculees a chaque session par [SolarSystem].
 */
data class SkyObject(
    val id: String,
    val designation: String,
    val type: ObjectType,
    val catalog: Catalog,
    val raDeg: Double,
    val decDeg: Double,
    val magnitude: Double?,
    val majorAxisArcmin: Double?,
    val minorAxisArcmin: Double?,
    val constellation: String,
    val commonName: String,
    /** Renseigne uniquement pour les corps du systeme solaire. */
    val body: SolarSystem.Body? = null,
) {
    val isSolarSystem: Boolean get() = body != null

    /** Nom affiche : identifiant court, complete du nom usuel quand il existe. */
    val displayName: String
        get() = if (commonName.isBlank()) id else "$id — $commonName"

    /**
     * Brillance de surface (section 5.4), ou null lorsque les dimensions
     * angulaires sont inconnues. Les objets sans dimension restent consultables
     * dans l'Explorer mais sont ecartes du Top du Dashboard, faute de pouvoir
     * evaluer le filtre RG-F-02 et le critere « Brillance surf. » (15 %).
     */
    val surfaceBrightness: Double?
        get() = Formulas.surfaceBrightness(magnitude, majorAxisArcmin, minorAxisArcmin)

    /** Taille angulaire principale, en arcminutes. */
    val sizeArcmin: Double? get() = majorAxisArcmin

    /** Libelle de taille pour les fiches objet. */
    val sizeLabel: String
        get() {
            val a = majorAxisArcmin ?: return "—"
            val b = minorAxisArcmin
            return if (b == null || kotlin.math.abs(a - b) < 0.01) String.format("%.1f'", a)
            else String.format("%.1f' x %.1f'", a, b)
        }
}

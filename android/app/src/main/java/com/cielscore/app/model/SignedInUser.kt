package com.cielscore.app.model

/**
 * Identite de la session ouverte (section 2.8).
 *
 * Volontairement distincte de l'entite Room : une session est un fait local,
 * conserve dans les preferences, et n'a pas a dependre d'une lecture en base
 * pour etre retablie au lancement. La base reste la reference pour les
 * identifiants, jamais pour savoir qui est connecte.
 *
 * Aucune donnee secrete n'y figure : ni mot de passe, ni empreinte, ni code de
 * recuperation.
 */
data class SignedInUser(
    val username: String,
    val firstName: String,
    val lastName: String,
    val createdAt: Long,
)

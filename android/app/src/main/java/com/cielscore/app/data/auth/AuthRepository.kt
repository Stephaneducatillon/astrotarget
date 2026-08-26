package com.cielscore.app.data.auth

import com.cielscore.app.data.db.CielScoreDatabase
import com.cielscore.app.data.db.UserEntity
import com.cielscore.app.util.Log

/**
 * Comptes utilisateur — sections 2.8 et 9.1.
 *
 * Regles de la section 2.8 :
 *   Connexion    identifiant en minuscules, mot de passe de 8 caracteres minimum
 *   Inscription  prenom, nom, identifiant de 3 caracteres minimum, alphanumerique
 *   Recuperation code genere a la creation, affiche une seule fois
 *   Reinitialisation identifiant + code de recuperation + nouveau mot de passe
 */
class AuthRepository(private val db: CielScoreDatabase) {

    /** Resultat d'une inscription : le code de recuperation n'est montre qu'ici. */
    data class Registration(val user: UserEntity, val recoveryCode: String)

    sealed interface AuthError {
        data class Message(val text: String) : AuthError
    }

    companion object {
        const val MIN_PASSWORD_LENGTH = 8
        const val MIN_USERNAME_LENGTH = 3

        /** Identifiant en minuscules (section 2.8). */
        fun normalizeUsername(raw: String): String = raw.trim().lowercase()

        fun validateUsername(username: String): String? = when {
            username.length < MIN_USERNAME_LENGTH ->
                "L'identifiant doit comporter au moins $MIN_USERNAME_LENGTH caracteres."
            !username.all { it.isLetterOrDigit() } ->
                "L'identifiant doit etre alphanumerique, sans espace ni signe de ponctuation."
            else -> null
        }

        fun validatePassword(password: String): String? =
            if (password.length < MIN_PASSWORD_LENGTH)
                "Le mot de passe doit comporter au moins $MIN_PASSWORD_LENGTH caracteres."
            else null
    }

    /** Inscription. Le code de recuperation renvoye n'est jamais stocke en clair. */
    suspend fun register(
        rawUsername: String,
        password: String,
        firstName: String,
        lastName: String,
    ): Result<Registration> {
        val username = normalizeUsername(rawUsername)
        validateUsername(username)?.let { return Result.failure(IllegalArgumentException(it)) }
        validatePassword(password)?.let { return Result.failure(IllegalArgumentException(it)) }
        if (firstName.isBlank() || lastName.isBlank()) {
            return Result.failure(IllegalArgumentException("Le prenom et le nom sont obligatoires."))
        }
        if (db.userDao().findByUsername(username) != null) {
            return Result.failure(IllegalArgumentException("Cet identifiant est deja utilise."))
        }

        val recoveryCode = PasswordHasher.generateRecoveryCode()
        val user = UserEntity(
            username = username,
            passwordHash = PasswordHasher.hash(password),
            firstName = firstName.trim(),
            lastName = lastName.trim(),
            recoveryHash = PasswordHasher.hash(PasswordHasher.normalizeRecoveryCode(recoveryCode)),
            createdAt = System.currentTimeMillis(),
        )
        return runCatching {
            db.userDao().insert(user)
            Log.i("Auth", "Compte cree : $username")
            Registration(user, recoveryCode)
        }
    }

    /** Connexion. */
    suspend fun login(rawUsername: String, password: String): Result<UserEntity> {
        val username = normalizeUsername(rawUsername)
        val user = db.userDao().findByUsername(username)
        if (user == null || !PasswordHasher.verify(password, user.passwordHash)) {
            Log.w("Auth", "Echec de connexion pour $username")
            return Result.failure(IllegalArgumentException("Identifiant ou mot de passe incorrect."))
        }
        // Rehachage silencieux si l'empreinte date d'un reglage anterieur.
        if (PasswordHasher.needsUpgrade(user.passwordHash)) {
            db.userDao().update(user.copy(passwordHash = PasswordHasher.hash(password)))
        }
        Log.i("Auth", "Connexion reussie : $username")
        return Result.success(user)
    }

    /** Reinitialisation : identifiant + code de recuperation + nouveau mot de passe. */
    suspend fun resetPassword(
        rawUsername: String,
        recoveryCode: String,
        newPassword: String,
    ): Result<Unit> {
        val username = normalizeUsername(rawUsername)
        validatePassword(newPassword)?.let { return Result.failure(IllegalArgumentException(it)) }
        val user = db.userDao().findByUsername(username)
            ?: return Result.failure(IllegalArgumentException("Identifiant ou code de recuperation incorrect."))
        val normalized = PasswordHasher.normalizeRecoveryCode(recoveryCode)
        if (!PasswordHasher.verify(normalized, user.recoveryHash)) {
            Log.w("Auth", "Code de recuperation invalide pour $username")
            return Result.failure(IllegalArgumentException("Identifiant ou code de recuperation incorrect."))
        }
        db.userDao().update(user.copy(passwordHash = PasswordHasher.hash(newPassword)))
        Log.i("Auth", "Mot de passe reinitialise : $username")
        return Result.success(Unit)
    }

    suspend fun findUser(username: String): UserEntity? =
        db.userDao().findByUsername(normalizeUsername(username))
}

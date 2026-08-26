package com.cielscore.app.data.auth

import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

/**
 * Hachage des mots de passe — section 9.1.
 *
 *   Hachage         PBKDF2-SHA256, 260 000 iterations (recommandation OWASP 2024)
 *   Format stocke   algorithme $ iterations $ sel $ empreinte
 *   Sel             16 octets aleatoires par utilisateur
 *   Comparaison     comparaison a temps constant, resistante aux attaques temporelles
 *
 * Retrocompatibilite : le nombre d'iterations etant porte par l'empreinte
 * elle-meme, les hachages produits avec un reglage anterieur restent
 * verifiables.
 */
object PasswordHasher {

    private const val ALGORITHM = "pbkdf2_sha256"
    private const val PBKDF2 = "PBKDF2WithHmacSHA256"
    const val ITERATIONS = 260_000
    private const val SALT_BYTES = 16
    private const val KEY_BITS = 256

    private val random = SecureRandom()

    /** Produit une empreinte au format « algorithme$iterations$sel$empreinte ». */
    fun hash(password: String, iterations: Int = ITERATIONS): String {
        val salt = ByteArray(SALT_BYTES).also { random.nextBytes(it) }
        val digest = pbkdf2(password, salt, iterations)
        return "$ALGORITHM$${iterations}$${encode(salt)}$${encode(digest)}"
    }

    /**
     * Verifie un mot de passe contre une empreinte stockee.
     * La comparaison finale est a temps constant.
     */
    fun verify(password: String, stored: String): Boolean {
        val parts = stored.split('$')
        if (parts.size != 4) return false
        if (parts[0] != ALGORITHM) return false
        val iterations = parts[1].toIntOrNull() ?: return false
        val salt = decode(parts[2]) ?: return false
        val expected = decode(parts[3]) ?: return false
        val actual = pbkdf2(password, salt, iterations)
        return MessageDigest.isEqual(expected, actual)
    }

    /** Vrai lorsque l'empreinte a ete produite avec moins d'iterations qu'aujourd'hui. */
    fun needsUpgrade(stored: String): Boolean {
        val parts = stored.split('$')
        if (parts.size != 4) return true
        val iterations = parts[1].toIntOrNull() ?: return true
        return iterations < ITERATIONS
    }

    /**
     * Code de recuperation (section 2.8) : genere a la creation du compte,
     * affiche une seule fois, jamais stocke en clair (section 9.1).
     */
    fun generateRecoveryCode(): String {
        val alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        val raw = CharArray(16) { alphabet[random.nextInt(alphabet.length)] }
        return String(raw).chunked(4).joinToString("-")
    }

    /** Normalise un code de recuperation saisi par l'utilisateur. */
    fun normalizeRecoveryCode(code: String): String =
        code.uppercase().filter { it.isLetterOrDigit() }

    private fun pbkdf2(password: String, salt: ByteArray, iterations: Int): ByteArray {
        val spec = PBEKeySpec(password.toCharArray(), salt, iterations, KEY_BITS)
        return SecretKeyFactory.getInstance(PBKDF2).generateSecret(spec).encoded
    }

    private fun encode(bytes: ByteArray): String =
        android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)

    private fun decode(value: String): ByteArray? =
        runCatching { android.util.Base64.decode(value, android.util.Base64.NO_WRAP) }.getOrNull()
}

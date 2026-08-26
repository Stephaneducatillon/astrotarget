package com.cielscore.app.util

/**
 * Journalisation — section 9.4.
 *
 *     Format : horodatage | niveau | module | message
 *
 *   INFO     demarrage, connexion reussie, calcul lance, observation enregistree
 *   WARNING  echec de connexion, service meteo indisponible, sauvegarde echouee
 *   ERROR    erreur IA, echec d'initialisation de la base
 *   DEBUG    details meteo, hauteur du Soleil, score de nuit
 */
object Log {

    private const val TAG = "CielScore"

    private fun format(level: String, module: String, message: String): String {
        val ts = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.FRANCE)
            .format(java.util.Date())
        return "$ts | $level | $module | $message"
    }

    fun i(module: String, message: String) =
        android.util.Log.i(TAG, format("INFO", module, message))

    fun w(module: String, message: String) =
        android.util.Log.w(TAG, format("WARNING", module, message))

    fun e(module: String, message: String, error: Throwable? = null) =
        android.util.Log.e(TAG, format("ERROR", module, message), error)

    fun d(module: String, message: String) =
        android.util.Log.d(TAG, format("DEBUG", module, message))
}

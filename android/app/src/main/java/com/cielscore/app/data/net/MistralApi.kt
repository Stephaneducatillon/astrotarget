package com.cielscore.app.data.net

import com.cielscore.app.data.cache.TtlCache
import com.cielscore.app.scoring.ScoringEngine
import com.cielscore.app.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Intelligence artificielle Mistral (sections 2.4, 2.7 et 8.2).
 *
 * Trois usages : guide objet, plan de soiree et assistant conversationnel.
 * Repli de la section 8.4 : en cas d'echec, un message d'erreur explicite est
 * renvoye et le reste de l'application continue de fonctionner.
 */
object MistralApi {

    private const val ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
    private const val MODEL = "mistral-large-latest"
    private val JSON = "application/json; charset=utf-8".toMediaType()

    /** Message affiche lorsque la cle Mistral n'est pas renseignee. */
    const val MISSING_KEY_MESSAGE =
        "Fonctions IA indisponibles : renseignez une cle API Mistral dans le Profil."

    private val guideCache = TtlCache<String>(TtlCache.AI_GUIDE_TTL)

    /** Un tour de conversation. */
    data class Message(val role: String, val content: String)

    /**
     * Contexte reel injecte a l'IA — tableau de la section 2.7.
     */
    data class SessionContext(
        val siteName: String,
        val bortle: Int,
        val instrumentLabel: String,
        val diameterMm: Double,
        val limitingMagnitude: Double,
        val surfaceBrightnessLimit: Double,
        val moonPhasePercent: Double,
        val cloudCoverPercent: Double,
        val seeingLabel: String,
        val topTargets: List<ScoringEngine.Scored>,
    ) {
        fun render(): String = buildString {
            appendLine("Lieu : $siteName, indice Bortle $bortle.")
            appendLine(
                "Instrument : %s de %.0f mm, magnitude limite calculee %.1f."
                    .format(instrumentLabel, diameterMm, limitingMagnitude)
            )
            appendLine(
                "Ciel : brillance de surface limite %.1f mag/arcsec2, Lune eclairee a %.0f %%, %.0f %% de nuages, seeing %s."
                    .format(surfaceBrightnessLimit, moonPhasePercent, cloudCoverPercent, seeingLabel)
            )
            if (topTargets.isEmpty()) {
                appendLine("Aucune cible calculee pour l'instant.")
            } else {
                appendLine("Cibles du soir (les 5 meilleures, score sur 100) :")
                topTargets.take(5).forEach {
                    appendLine(
                        "  - %s : score %.0f, altitude %.0f degres."
                            .format(it.target.displayName, it.score, it.altitudeDeg)
                    )
                }
            }
        }
    }

    /**
     * Prompt systeme commun.
     *
     * ATTENTION (section 2.7) — l'assistant ne doit jamais qualifier un objet
     * d'impossible sans verifier son score. La regle est inscrite ici, dans le
     * prompt systeme, comme le prevoit la documentation.
     */
    private fun systemPrompt(context: SessionContext?): String = buildString {
        appendLine("Tu es l'assistant de CielScore, un planificateur d'observation astronomique.")
        appendLine("Tu reponds en francais, de maniere concise, concrete et bienveillante.")
        appendLine("Tu t'adresses a un astronome amateur qui observe depuis son jardin.")
        appendLine()
        appendLine("REGLE IMPERATIVE : ne qualifie jamais un objet d'impossible ou d'invisible")
        appendLine("sans avoir verifie son score dans le contexte fourni. Un objet present dans")
        appendLine("la liste des cibles a deja passe les filtres eliminatoires de l'application")
        appendLine("(altitude, brillance de surface, magnitude, couverture nuageuse) : il est")
        appendLine("donc observable. Si tu n'as pas l'information, dis-le plutot que de conclure.")
        appendLine()
        appendLine("N'invente jamais de valeur numerique : utilise celles du contexte.")
        if (context != null) {
            appendLine()
            appendLine("Contexte reel de la session en cours :")
            append(context.render())
        }
    }

    /** Guide d'observation d'un objet (fiche objet du Dashboard, section 2.2). */
    suspend fun objectGuide(
        apiKey: String?,
        objectName: String,
        diameterMm: Double,
        context: SessionContext?,
    ): Result<String> {
        if (apiKey.isNullOrBlank()) return Result.failure(IllegalStateException(MISSING_KEY_MESSAGE))
        val key = TtlCache.aiGuideKey(objectName, diameterMm)
        guideCache.get(key)?.let { return Result.success(it) }

        val prompt = """
            Redige un guide d'observation court pour $objectName avec un instrument de
            ${diameterMm.toInt()} mm de diametre. Structure ta reponse en quatre paragraphes brefs :
            ce qu'est l'objet, comment le reperer dans le ciel, ce que l'on voit reellement
            a l'oculaire avec ce diametre, et un conseil pratique.
        """.trimIndent()

        return chat(apiKey, systemPrompt(context), listOf(Message("user", prompt)))
            .onSuccess { guideCache.put(key, it) }
    }

    /**
     * Plan de soiree (section 2.4) : quatre sections — ordre horodate, oculaires,
     * ce que vous verrez, conseil astrophoto.
     */
    suspend fun eveningPlan(
        apiKey: String?,
        context: SessionContext,
        startLabel: String,
    ): Result<String> {
        if (apiKey.isNullOrBlank()) return Result.failure(IllegalStateException(MISSING_KEY_MESSAGE))
        val prompt = """
            Construis le plan de ma soiree d'observation, qui commence a $startLabel.
            Utilise uniquement les cibles listees dans le contexte, dans un ordre qui
            tient compte de leur altitude. Rends exactement quatre sections, avec ces titres :

            1. Ordre horodate
            2. Oculaires
            3. Ce que vous verrez
            4. Conseil astrophoto

            Reste factuel et bref : le plan doit tenir sur un ecran de telephone.
        """.trimIndent()
        return chat(apiKey, systemPrompt(context), listOf(Message("user", prompt)))
    }

    /** Assistant conversationnel (section 2.7). */
    suspend fun assistant(
        apiKey: String?,
        context: SessionContext?,
        history: List<Message>,
    ): Result<String> {
        if (apiKey.isNullOrBlank()) return Result.failure(IllegalStateException(MISSING_KEY_MESSAGE))
        return chat(apiKey, systemPrompt(context), history)
    }

    private suspend fun chat(
        apiKey: String,
        system: String,
        messages: List<Message>,
    ): Result<String> {
        val payload = JSONObject().apply {
            put("model", MODEL)
            put("temperature", 0.4)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", system))
                messages.forEach {
                    put(JSONObject().put("role", it.role).put("content", it.content))
                }
            })
        }

        val body = Http.postString(
            ENDPOINT,
            payload.toString().toRequestBody(JSON),
            mapOf("Authorization" to "Bearer $apiKey"),
        ) ?: run {
            Log.e("MistralApi", "Appel Mistral en echec")
            return Result.failure(
                IllegalStateException("Service Mistral indisponible. Reessayez dans un instant.")
            )
        }

        return runCatching {
            JSONObject(body)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
        }.recoverCatching {
            throw IllegalStateException("Reponse Mistral illisible.")
        }
    }
}

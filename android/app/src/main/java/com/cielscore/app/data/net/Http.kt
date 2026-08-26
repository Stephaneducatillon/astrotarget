package com.cielscore.app.data.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import java.util.concurrent.TimeUnit

/**
 * Client HTTP partage par les interfaces externes de la section 8.2.
 *
 * Toutes les erreurs sont remontees sous forme de null : chaque appelant
 * applique ensuite sa propre strategie de repli (section 8.4).
 */
object Http {

    val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .callTimeout(35, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private const val USER_AGENT = "CielScore-Android/0.6.4 (+https://github.com/Stephaneducatillon/astrotarget)"

    suspend fun getString(url: String): String? = withContext(Dispatchers.IO) {
        runCatching {
            val request = Request.Builder()
                .url(url)
                .header("User-Agent", USER_AGENT)
                .header("Accept", "application/json")
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@use null
                response.body?.string()
            }
        }.getOrNull()
    }

    suspend fun postString(
        url: String,
        body: RequestBody,
        headers: Map<String, String> = emptyMap(),
    ): String? = withContext(Dispatchers.IO) {
        runCatching {
            val builder = Request.Builder()
                .url(url)
                .header("User-Agent", USER_AGENT)
                .post(body)
            headers.forEach { (k, v) -> builder.header(k, v) }
            client.newCall(builder.build()).execute().use { response ->
                if (!response.isSuccessful) return@use null
                response.body?.string()
            }
        }.getOrNull()
    }
}

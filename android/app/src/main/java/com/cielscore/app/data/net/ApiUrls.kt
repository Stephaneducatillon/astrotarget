package com.cielscore.app.data.net

import java.net.URLEncoder
import java.util.Locale

/**
 * Construction des URL appelees par l'application (section 8.2).
 *
 * ATTENTION AU FORMATAGE DES NOMBRES — String.format applique par defaut la
 * locale de l'appareil. Sur un telephone francais, "%.4f".format(50.3784)
 * produit « 50,3784 », avec une virgule : l'URL devient latitude=50,3784 et le
 * service la rejette. Toutes les valeurs numeriques destinees a une URL ou a
 * une cle de cache passent donc par [decimal], qui impose Locale.ROOT et donc
 * le point decimal.
 *
 * Ce fichier ne depend ni d'Android ni du reseau : il est directement testable.
 */
object ApiUrls {

    /** Formate un nombre pour une URL : point decimal, quelle que soit la locale. */
    fun decimal(value: Double, decimals: Int): String =
        String.format(Locale.ROOT, "%.${decimals}f", value)

    /** Meteo horaire Open-Meteo (gratuit, sans cle). */
    fun openMeteoForecast(latitude: Double, longitude: Double): String =
        "https://api.open-meteo.com/v1/forecast" +
            "?latitude=" + decimal(latitude, 4) +
            "&longitude=" + decimal(longitude, 4) +
            "&hourly=cloud_cover,wind_speed_10m,relative_humidity_2m,visibility,temperature_2m" +
            "&forecast_days=3&timeformat=unixtime&timezone=UTC&wind_speed_unit=kmh"

    /** Vignette hips2fits du CDS. */
    fun hips2fits(
        hipsId: String,
        raDeg: Double,
        decDeg: Double,
        fieldDeg: Double,
        pixels: Int,
    ): String {
        val fov = fieldDeg.coerceIn(0.05, 5.0)
        return "https://alasky.u-strasbg.fr/hips-image-services/hips2fits" +
            "?hips=" + URLEncoder.encode(hipsId, "UTF-8") +
            "&width=" + pixels +
            "&height=" + pixels +
            "&fov=" + decimal(fov, 4) +
            "&projection=TAN&coordsys=icrs&rotation_angle=0.0" +
            "&ra=" + decimal(raDeg, 6) +
            "&dec=" + decimal(decDeg, 6) +
            "&format=jpg"
    }

    /** Ouverture externe de Stellarium Web sur un objet, un lieu et un instant. */
    fun stellariumWeb(
        objectName: String,
        latitude: Double,
        longitude: Double,
        isoDate: String,
    ): String =
        "https://stellarium-web.org/skysource/" +
            URLEncoder.encode(objectName, "UTF-8") +
            "?fov=5&date=" + isoDate +
            "&lat=" + decimal(latitude, 4) +
            "&lng=" + decimal(longitude, 4) +
            "&elev=0"

    /**
     * Cle du cache meteo (section 4.6) : lieu au centieme de degre et heure
     * pleine. Le point decimal garantit une cle stable d'une locale a l'autre.
     */
    fun weatherCacheKey(latitude: Double, longitude: Double, epochMillis: Long): String =
        decimal(latitude, 2) + "_" + decimal(longitude, 2) + "_" +
            (epochMillis / (60 * 60 * 1000L))
}

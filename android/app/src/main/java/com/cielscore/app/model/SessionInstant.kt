package com.cielscore.app.model

import java.util.Calendar
import java.util.TimeZone

/**
 * Manipulation de l'instant de session (section 2.2 : « Date et heure locale de
 * la session »).
 *
 * L'application raisonne en millisecondes UTC, mais l'utilisateur saisit et lit
 * toujours une date et une heure locales. Les conversions sont regroupees ici,
 * a l'ecart de l'interface, pour etre testees.
 */
object SessionInstant {

    private val UTC: TimeZone get() = TimeZone.getTimeZone("UTC")

    private fun local(epochMillis: Long): Calendar =
        Calendar.getInstance().apply { timeInMillis = epochMillis }

    /**
     * Minuit UTC correspondant a la date LOCALE de [epochMillis].
     *
     * Le calendrier de Material 3 raisonne en minuit UTC : c'est la valeur
     * qu'il attend en entree.
     */
    fun localDateToUtcMidnight(epochMillis: Long): Long {
        val l = local(epochMillis)
        return Calendar.getInstance(UTC).apply {
            clear()
            set(l.get(Calendar.YEAR), l.get(Calendar.MONTH), l.get(Calendar.DAY_OF_MONTH))
        }.timeInMillis
    }

    /**
     * Reporte la date choisie dans le calendrier sur [epochMillis], en
     * conservant l'heure locale deja retenue.
     *
     * @param utcMidnightMillis valeur renvoyee par le calendrier, minuit UTC.
     */
    fun applyUtcMidnightDate(epochMillis: Long, utcMidnightMillis: Long): Long {
        val picked = Calendar.getInstance(UTC).apply { timeInMillis = utcMidnightMillis }
        return local(epochMillis).apply {
            set(Calendar.YEAR, picked.get(Calendar.YEAR))
            set(Calendar.MONTH, picked.get(Calendar.MONTH))
            set(Calendar.DAY_OF_MONTH, picked.get(Calendar.DAY_OF_MONTH))
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    /** Reporte l'heure choisie sur [epochMillis], en conservant la date locale. */
    fun applyTime(epochMillis: Long, hour: Int, minute: Int): Long =
        local(epochMillis).apply {
            set(Calendar.HOUR_OF_DAY, hour)
            set(Calendar.MINUTE, minute)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

    /** Composantes locales d'un instant, pour alimenter les selecteurs. */
    fun localHour(epochMillis: Long): Int = local(epochMillis).get(Calendar.HOUR_OF_DAY)

    fun localMinute(epochMillis: Long): Int = local(epochMillis).get(Calendar.MINUTE)

    fun localDayOfMonth(epochMillis: Long): Int = local(epochMillis).get(Calendar.DAY_OF_MONTH)

    /** Mois local, de 1 a 12. */
    fun localMonth(epochMillis: Long): Int = local(epochMillis).get(Calendar.MONTH) + 1

    fun localYear(epochMillis: Long): Int = local(epochMillis).get(Calendar.YEAR)
}

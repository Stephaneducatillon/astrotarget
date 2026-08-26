package com.cielscore.app

import com.cielscore.app.model.SessionInstant
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

/**
 * Choix de la date et de l'heure de session (section 2.2).
 *
 * L'application stocke un instant UTC, l'utilisateur saisit une date et une
 * heure locales : ces tests verrouillent la conversion dans plusieurs fuseaux,
 * y compris de part et d'autre d'un changement d'heure.
 */
class SessionInstantTest {

    private lateinit var original: TimeZone

    @Before
    fun captureTimeZone() {
        original = TimeZone.getDefault()
    }

    @After
    fun restoreTimeZone() {
        TimeZone.setDefault(original)
    }

    private fun useTimeZone(id: String) = TimeZone.setDefault(TimeZone.getTimeZone(id))

    /** Instant correspondant a une date et une heure LOCALES. */
    private fun localInstant(y: Int, month: Int, d: Int, h: Int, min: Int): Long =
        Calendar.getInstance().apply {
            clear()
            set(y, month - 1, d, h, min, 0)
        }.timeInMillis

    private fun describe(epochMillis: Long): String = "%04d-%02d-%02d %02d:%02d".format(
        SessionInstant.localYear(epochMillis),
        SessionInstant.localMonth(epochMillis),
        SessionInstant.localDayOfMonth(epochMillis),
        SessionInstant.localHour(epochMillis),
        SessionInstant.localMinute(epochMillis),
    )

    @Test
    fun `le calendrier recoit minuit UTC de la date locale`() {
        useTimeZone("Europe/Paris")
        // 26 aout a 23h30 locale : en UTC on est deja le 26 a 21h30, mais la
        // date locale reste le 26.
        val instant = localInstant(2026, 8, 26, 23, 30)
        val utcMidnight = SessionInstant.localDateToUtcMidnight(instant)

        val utc = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            timeInMillis = utcMidnight
        }
        assertEquals(2026, utc.get(Calendar.YEAR))
        assertEquals(8, utc.get(Calendar.MONTH) + 1)
        assertEquals(26, utc.get(Calendar.DAY_OF_MONTH))
        assertEquals(0, utc.get(Calendar.HOUR_OF_DAY))
        assertEquals(0, utc.get(Calendar.MINUTE))
    }

    @Test
    fun `une date locale tardive ne glisse pas au lendemain`() {
        // Fuseau tres a l'est : 26 aout a 8h locale = 25 aout 22h UTC.
        useTimeZone("Pacific/Auckland")
        val instant = localInstant(2026, 8, 26, 8, 0)
        val utc = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            timeInMillis = SessionInstant.localDateToUtcMidnight(instant)
        }
        assertEquals(26, utc.get(Calendar.DAY_OF_MONTH))
        assertEquals(8, utc.get(Calendar.MONTH) + 1)
    }

    @Test
    fun `choisir une date conserve l heure locale`() {
        useTimeZone("Europe/Paris")
        // Session reglee au 26 aout a 4h du matin.
        val instant = localInstant(2026, 8, 26, 4, 0)
        // L'utilisateur choisit le 28 dans le calendrier.
        val picked = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            clear()
            set(2026, 7, 28)
        }.timeInMillis

        val result = SessionInstant.applyUtcMidnightDate(instant, picked)
        assertEquals("2026-08-28 04:00", describe(result))
    }

    @Test
    fun `choisir une heure conserve la date locale`() {
        useTimeZone("Europe/Paris")
        val instant = localInstant(2026, 8, 26, 21, 45)
        val result = SessionInstant.applyTime(instant, 4, 0)
        assertEquals("2026-08-26 04:00", describe(result))
    }

    @Test
    fun `le cas demande, le 28 a 4h du matin, aboutit bien`() {
        useTimeZone("Europe/Paris")
        // Point de depart : maintenant, le 26 aout en fin d'apres-midi.
        var instant = localInstant(2026, 8, 26, 16, 7)

        // L'utilisateur choisit le 28 dans le calendrier...
        val picked = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            clear()
            set(2026, 7, 28)
        }.timeInMillis
        instant = SessionInstant.applyUtcMidnightDate(instant, picked)
        // ... puis 4h00 dans l'horloge.
        instant = SessionInstant.applyTime(instant, 4, 0)

        assertEquals("2026-08-28 04:00", describe(instant))
    }

    @Test
    fun `l ordre de saisie est indifferent`() {
        useTimeZone("Europe/Paris")
        val start = localInstant(2026, 8, 26, 16, 7)
        val picked = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            clear()
            set(2026, 7, 28)
        }.timeInMillis

        val dateThenTime = SessionInstant.applyTime(
            SessionInstant.applyUtcMidnightDate(start, picked), 4, 0
        )
        val timeThenDate = SessionInstant.applyUtcMidnightDate(
            SessionInstant.applyTime(start, 4, 0), picked
        )
        assertEquals(describe(dateThenTime), describe(timeThenDate))
    }

    @Test
    fun `l aller-retour par le calendrier est stable`() {
        listOf("Europe/Paris", "Indian/Reunion", "America/Cayenne", "Pacific/Noumea")
            .forEach { zone ->
                useTimeZone(zone)
                listOf(0, 4, 12, 23).forEach { hour ->
                    val instant = localInstant(2026, 8, 26, hour, 30)
                    val roundTrip = SessionInstant.applyUtcMidnightDate(
                        instant, SessionInstant.localDateToUtcMidnight(instant)
                    )
                    assertEquals(
                        "aller-retour instable en $zone a ${hour}h",
                        describe(instant),
                        describe(roundTrip),
                    )
                }
            }
    }

    @Test
    fun `le passage a l heure d hiver est traverse sans derive de date`() {
        useTimeZone("Europe/Paris")
        // Nuit du 24 au 25 octobre 2026 : recul d'une heure a 3h du matin.
        val before = localInstant(2026, 10, 24, 22, 0)
        val picked = Calendar.getInstance(TimeZone.getTimeZone("UTC")).apply {
            clear()
            set(2026, 9, 25)
        }.timeInMillis

        val result = SessionInstant.applyTime(
            SessionInstant.applyUtcMidnightDate(before, picked), 4, 0
        )
        assertEquals("2026-10-25 04:00", describe(result))
    }
}

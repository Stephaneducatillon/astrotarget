package com.cielscore.app.catalog

import android.content.Context
import com.cielscore.app.astro.AstroMath
import com.cielscore.app.astro.SolarSystem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.BufferedReader

/**
 * Chargement des catalogues embarques (section 8.1).
 *
 *   messier.csv    110 objets, dimensions angulaires incluses
 *   caldwell.csv   109 objets, construits depuis OpenNGC
 *   ngcic.csv      13 308 objets issus d'OpenNGC
 *
 * Les fichiers sont generes par tools/build_catalogs.py a partir de NGC.csv.
 * Le chargement est paresseux : Messier et Caldwell sont lus au demarrage,
 * NGC/IC seulement lorsqu'un ecran le demande.
 */
class CatalogRepository(private val context: Context) {

    private val mutex = Mutex()
    private var messier: List<SkyObject>? = null
    private var caldwell: List<SkyObject>? = null
    private var ngcIc: List<SkyObject>? = null

    suspend fun messier(): List<SkyObject> = load(Catalog.MESSIER)
    suspend fun caldwell(): List<SkyObject> = load(Catalog.CALDWELL)
    suspend fun ngcIc(): List<SkyObject> = load(Catalog.NGC_IC)

    /** Objets des catalogues demandes, planetes exclues. */
    suspend fun deepSky(catalogs: Set<Catalog>): List<SkyObject> = buildList {
        if (Catalog.MESSIER in catalogs) addAll(messier())
        if (Catalog.CALDWELL in catalogs) addAll(caldwell())
        if (Catalog.NGC_IC in catalogs) addAll(ngcIc())
    }

    private suspend fun load(catalog: Catalog): List<SkyObject> = mutex.withLock {
        when (catalog) {
            Catalog.MESSIER -> messier ?: read("messier.csv", catalog).also { messier = it }
            Catalog.CALDWELL -> caldwell ?: read("caldwell.csv", catalog).also { caldwell = it }
            Catalog.NGC_IC -> ngcIc ?: read("ngcic.csv", catalog).also { ngcIc = it }
            Catalog.PLANETS -> emptyList()
        }
    }

    private suspend fun read(asset: String, catalog: Catalog): List<SkyObject> =
        withContext(Dispatchers.IO) {
            context.assets.open(asset).bufferedReader().use { reader ->
                parse(reader, catalog)
            }
        }

    private fun parse(reader: BufferedReader, catalog: Catalog): List<SkyObject> {
        val out = ArrayList<SkyObject>(if (catalog == Catalog.NGC_IC) 14_000 else 128)
        reader.readLine() // en-tete
        while (true) {
            val line = reader.readLine() ?: break
            if (line.isBlank()) continue
            val f = line.split(';')
            if (f.size < 10) continue
            val ra = f[3].toDoubleOrNull() ?: continue
            val dec = f[4].toDoubleOrNull() ?: continue
            out.add(
                SkyObject(
                    id = f[0],
                    designation = f[1],
                    type = ObjectType.fromAsset(f[2]),
                    catalog = catalog,
                    raDeg = ra,
                    decDeg = dec,
                    magnitude = f[5].toDoubleOrNull(),
                    majorAxisArcmin = f[6].toDoubleOrNull(),
                    minorAxisArcmin = f[7].toDoubleOrNull(),
                    constellation = f[8],
                    commonName = f[9],
                )
            )
        }
        return out
    }

    companion object {
        /**
         * Les six corps du systeme solaire proposes (section 8.1). Le Soleil
         * n'est jamais propose : RG-P-01, observation dangereuse sans filtration
         * specialisee.
         */
        fun solarSystemObjects(jd: Double): List<SkyObject> =
            SolarSystem.Body.entries.map { body ->
                val p = if (body == SolarSystem.Body.MOON) SolarSystem.moon(jd)
                else SolarSystem.planet(body, jd)
                SkyObject(
                    id = body.frenchName,
                    designation = body.frenchName,
                    type = ObjectType.OTHER,
                    catalog = Catalog.PLANETS,
                    raDeg = p.raDeg,
                    decDeg = p.decDeg,
                    magnitude = p.magnitude,
                    majorAxisArcmin = null,
                    minorAxisArcmin = null,
                    constellation = "",
                    commonName = "",
                    body = body,
                )
            }

        /** Position instantanee d'un corps, utilisee pour les courbes d'altitude. */
        fun bodyPosition(body: SolarSystem.Body, epochMillis: Long): SolarSystem.Position {
            val jd = AstroMath.julianDay(epochMillis)
            return if (body == SolarSystem.Body.MOON) SolarSystem.moon(jd)
            else SolarSystem.planet(body, jd)
        }
    }
}

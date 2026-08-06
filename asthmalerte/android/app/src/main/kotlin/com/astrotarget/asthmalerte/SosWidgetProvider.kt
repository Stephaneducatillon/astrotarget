package com.astrotarget.asthmalerte

import android.appwidget.AppWidgetManager
import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.widget.RemoteViews
import es.antonborri.home_widget.HomeWidgetLaunchIntent
import es.antonborri.home_widget.HomeWidgetProvider

/**
 * Widget « SOS » de l'écran d'accueil.
 *
 * Un appui ouvre l'application directement sur l'écran d'alerte via l'URI
 * asthmalerte://sos, sans passer par l'accueil : c'est le chemin le plus
 * court entre la crise et le SMS aux proches.
 */
class SosWidgetProvider : HomeWidgetProvider() {

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
        widgetData: SharedPreferences
    ) {
        appWidgetIds.forEach { widgetId ->
            val views = RemoteViews(context.packageName, R.layout.sos_widget).apply {
                val pendingIntent = HomeWidgetLaunchIntent.getActivity(
                    context,
                    MainActivity::class.java,
                    Uri.parse("asthmalerte://sos")
                )
                setOnClickPendingIntent(R.id.widget_root, pendingIntent)

                val subtitle = widgetData.getString("sos_subtitle", null)
                    ?: context.getString(R.string.widget_default_subtitle)
                setTextViewText(R.id.widget_subtitle, subtitle)
            }
            appWidgetManager.updateAppWidget(widgetId, views)
        }
    }
}

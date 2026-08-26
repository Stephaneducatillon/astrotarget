package com.cielscore.app

import android.app.Application
import com.cielscore.app.di.AppContainer
import com.cielscore.app.util.Log

class CielScoreApplication : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        Log.i("App", "Demarrage de CielScore ${BuildConfig.VERSION_NAME}")
    }
}

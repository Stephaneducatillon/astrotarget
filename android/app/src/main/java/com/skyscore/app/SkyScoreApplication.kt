package com.skyscore.app

import android.app.Application
import com.skyscore.app.di.AppContainer
import com.skyscore.app.util.Log

class SkyScoreApplication : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        Log.i("App", "Demarrage de SkyScore ${BuildConfig.VERSION_NAME}")
    }
}

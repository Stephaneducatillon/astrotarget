plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.cielscore.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.cielscore.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.6.4"
    }

    /**
     * Signature de la version de release.
     *
     * Deux cas, dans cet ordre :
     *
     *  1. Une cle de distribution est fournie par l'environnement
     *     (RELEASE_STORE_FILE, RELEASE_STORE_PASSWORD, RELEASE_KEY_ALIAS,
     *     RELEASE_KEY_PASSWORD). C'est le chemin a suivre pour une vraie
     *     publication : la cle privee ne quitte jamais le coffre de secrets.
     *
     *  2. Sinon, l'APK est signe avec la cle de TEST versionnee dans le depot,
     *     keystore/cielscore-test.jks. Ses identifiants sont publics, exactement
     *     comme ceux du debug.keystore fourni avec le SDK Android : elle sert a
     *     produire un APK installable et surtout MISE A JOUR d'une version a
     *     l'autre, jamais a publier l'application.
     */
    signingConfigs {
        create("release") {
            val envStore = System.getenv("RELEASE_STORE_FILE")
            if (!envStore.isNullOrBlank()) {
                storeFile = file(envStore)
                storePassword = System.getenv("RELEASE_STORE_PASSWORD")
                keyAlias = System.getenv("RELEASE_KEY_ALIAS")
                keyPassword = System.getenv("RELEASE_KEY_PASSWORD")
            } else {
                storeFile = rootProject.file("keystore/cielscore-test.jks")
                storePassword = "cielscore"
                keyAlias = "cielscore-test"
                keyPassword = "cielscore"
            }
            enableV1Signing = true
            enableV2Signing = true
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.androidx.navigation.compose)

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.webkit)
    implementation(libs.okhttp)
    implementation(libs.coil.compose)
    implementation(libs.play.services.location)

    testImplementation(libs.junit)
}

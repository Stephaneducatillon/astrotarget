# Regles R8 pour la version de release.
#
# L'application n'utilise ni Gson, ni Moshi, ni aucune analyse JSON par
# reflexion : les reponses des interfaces externes sont lues champ par champ
# avec org.json. Le risque de suppression abusive est donc faible, mais les
# regles ci-dessous le reduisent encore et rendent les rapports de plantage
# exploitables.

# Conserver les numeros de ligne : une pile d'appel reste lisible.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Ne pas obfusquer nos propres classes : un plantage remonte des noms parlants,
# ce qui compte davantage que les quelques kilo-octets economises.
-keepnames class com.cielscore.app.** { *; }

# Modeles de donnees et entites Room : champs conserves.
-keepclassmembers class com.cielscore.app.data.db.** { <fields>; }
-keepclassmembers class com.cielscore.app.model.** { <fields>; }
-keepclassmembers class com.cielscore.app.catalog.** { <fields>; }

# Point d'entree declares dans le manifeste.
-keep class com.cielscore.app.CielScoreApplication
-keep class com.cielscore.app.MainActivity

# La WebView Aladin appelle du JavaScript : conserver les interfaces exposees.
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Enumerations : valueOf et values sont resolus par nom.
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Dependances tierces.
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, AnnotationDefault
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
-dontwarn okhttp3.internal.platform.**

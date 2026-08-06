# AsthmAlerte

Application mobile **Android + iOS** pour les personnes asthmatiques : en un
appui, elle prévient les proches enregistrés avec la **position** de la personne
et propose l'**appel** dans la foulée. Le bouton d'alerte est accessible depuis
l'**écran d'accueil du téléphone**, sans ouvrir l'application.

Écrit en Flutter : un seul code source pour les deux plateformes, plus un widget
natif de chaque côté (AppWidget Android, WidgetKit iOS).

---

## Ce que fait l'application

### Le parcours d'urgence

1. **Un appui** sur le bouton SOS (dans l'app, sur le widget d'accueil, ou via
   le raccourci de l'icône).
2. **Décompte de 5 secondes**, annulable d'un gros bouton — réglable de 0 à 15 s.
   À 0, l'alerte part immédiatement.
3. **Localisation** : position GPS + adresse postale, avec repli sur la dernière
   position connue si le GPS traîne (12 s maximum, on n'attend pas la
   perfection en pleine crise).
4. **SMS pré-rempli** à tous les proches : nom, adresse, lien Google Maps,
   traitement de crise, heure.
5. **Appel** du proche prioritaire lancé automatiquement après le SMS.
6. **Écran de suivi** : appel des secours (112), rappel d'un proche, carte,
   renvoi de l'alerte, et un bouton « Je vais mieux » qui rassure tout le monde.

### Le reste

- **Mes proches** — liste réordonnable ; le premier est celui qu'on appelle.
  Chacun peut être exclu du SMS ou des appels.
- **Fiche médicale** — traitement, allergies, médecin. Affichée en grand pendant
  l'alerte (lisible par un passant ou un secouriste) et résumée dans le SMS.
- **Historique** — les alertes passées, avec position : utile en consultation.
- **Réglages** — délai, partage de position, infos médicales, appel automatique,
  SMS groupé ou individuel, vibration, et texte du message personnalisable
  (variables `{nom}` `{adresse}` `{lien}` `{heure}` `{medical}`).

### Accès depuis l'écran d'accueil

Trois chemins, tous en un appui :

| Chemin | Android | iOS |
|---|---|---|
| Widget « SOS » rouge posé sur l'écran d'accueil | AppWidget (`SosWidgetProvider`) | WidgetKit (`SosWidget`) |
| Appui long sur l'icône → « Alerte SOS » | raccourci dynamique | Quick Action |
| Ouverture normale de l'app | bouton plein écran | bouton plein écran |

Les trois passent par l'URI `asthmalerte://sos`, qui ouvre l'écran d'alerte
directement — jamais l'accueil.

### Choix d'interface

L'écran est pensé pour être utilisable en manquant d'air : bouton circulaire de
360 px maximum au centre, rouge sur fond clair, pulsation lente, texte large,
cibles tactiles ≥ 56 px, écran maintenu allumé pendant l'alerte, retour haptique
à chaque seconde du décompte, et pendant le décompte le retour arrière système
est neutralisé pour qu'un geste involontaire n'annule pas l'alerte.

---

## Compiler et installer

Les projets Android et iOS sont versionnés : il n'y a rien à générer.
Chaîne validée avec **Flutter 3.44.8** (Dart 3.12) — `flutter analyze` ne
remonte aucun problème et les 30 tests passent.

```bash
cd asthmalerte
flutter pub get
flutter analyze
flutter test
flutter run          # sur un appareil connecté
```

### Android

`android/app/build.gradle.kts` est déjà configuré : `applicationId`
`com.astrotarget.asthmalerte`, `minSdk 24`, `targetSdk 36` — au-dessus des
minimums exigés par geolocator et permission_handler.

Le widget fonctionne sans étape supplémentaire : appui long sur l'écran
d'accueil → **Widgets** → **AsthmAlerte** → glisser le carré rouge « SOS ».

Générer l'APK :

```bash
flutter build apk --release          # installation directe
flutter build appbundle --release    # publication Play Store
```

### Réglages iOS (le widget demande Xcode)

Le fichier `ios/SosWidget/SosWidget.swift` est prêt, mais une extension WidgetKit
doit être déclarée dans Xcode :

1. `open ios/Runner.xcworkspace`
2. **File → New → Target… → Widget Extension**, nommée **`SosWidget`**,
   décocher « Include Configuration Intent ».
3. Remplacer les fichiers générés par ceux de `ios/SosWidget/`
   (`SosWidget.swift`, `Info.plist`).
4. **Signing & Capabilities** → ajouter **App Groups** — la même valeur sur la
   cible `Runner` **et** sur `SosWidget` :
   `group.com.astrotarget.asthmalerte`
   (elle doit correspondre à `HomeEntryService.iosAppGroupId` côté Dart).
5. Ajouter le paquet `home_widget` à la cible `SosWidget` si Xcode le demande.

Puis :

```bash
flutter build ios --release
```

L'envoi sur un iPhone nécessite un compte développeur Apple (99 $/an pour
l'App Store ; un compte gratuit suffit pour un test 7 jours sur son propre
téléphone).

---

## Ce que l'application ne fait pas (et pourquoi)

- **Le SMS n'est pas envoyé sans confirmation.** L'application ouvre la
  messagerie avec le texte et les destinataires déjà remplis ; il reste un appui
  sur « Envoyer ». C'est le seul comportement accepté sur l'App Store, et il
  évite la permission Android `SEND_SMS`, très mal vue en validation. Pour un
  envoi réellement automatique, il faut une passerelle SMS côté serveur
  (voir ci-dessous).
- **Aucune donnée ne quitte le téléphone.** Contacts, fiche médicale et
  historique sont stockés localement (`shared_preferences`). Pas de compte, pas
  de serveur, pas de traçage.
- **Ce n'est pas un dispositif médical** et cela ne remplace pas un appel au
  15 ou au 112. Le rappel figure dans les réglages et l'écran d'alerte propose
  l'appel des secours en premier.

## Aller plus loin

- **Envoi 100 % automatique** : un petit service (Twilio, OVH SMS, Vonage) et un
  backend qui reçoit `{position, contacts, message}` — l'appui devient réellement
  unique. Coût : quelques centimes par SMS.
- **Détection automatique de crise** via montre connectée (SpO₂, fréquence
  cardiaque) avec `health` / HealthKit.
- **Suivi de la qualité de l'air** et alerte préventive les jours à risque.
- **Bouton physique** : déclenchement par triple appui sur le bouton
  d'alimentation (Android : service d'accessibilité).

---

## Organisation du code

```
asthmalerte/
├── lib/
│   ├── main.dart                    point d'entrée, chargement de l'état
│   ├── app.dart                     MaterialApp + AppScope (état & services)
│   ├── theme.dart                   palette haute lisibilité
│   ├── models/                      contact, fiche médicale, réglages, alerte
│   ├── services/
│   │   ├── alert_message.dart       construction du SMS (pure, testée)
│   │   ├── alert_service.dart       orchestration position → SMS → appel
│   │   ├── location_service.dart    GPS + adresse, avec repli et délais
│   │   ├── messaging_service.dart   URI sms:/tel: selon la plateforme
│   │   ├── home_entry_service.dart  widget d'accueil + raccourcis
│   │   └── storage_service.dart     persistance locale
│   ├── screens/                     accueil, SOS, proches, fiche, réglages…
│   └── widgets/                     bouton SOS, cartes
├── android/app/src/main/            manifeste, widget Kotlin, layout
├── ios/                             Info.plist, extension WidgetKit
└── test/                            tests unitaires
```

Les 30 tests (`flutter test`) couvrent :

- la construction du SMS — position présente ou absente, options désactivées,
  modèle personnalisé, nom manquant ;
- la normalisation des numéros de téléphone et la sérialisation des contacts ;
- le rendu de chaque écran, avec un service d'alerte simulé ;
- le parcours d'urgence complet : décompte, annulation, envoi immédiat,
  écran de résultat et message « je vais mieux ».

## État de la vérification

| Vérifié | Comment |
|---|---|
| Analyse statique de tout le code Dart | `flutter analyze` — 0 problème |
| Logique métier et rendu de tous les écrans | `flutter test` — 30 tests au vert |
| Manifeste, layouts et plists | validation XML / plist |
| Compilation de l'APK et du binaire iOS | **pas encore faite** — à lancer sur votre machine |

L'APK n'a pas pu être compilé dans l'environnement où le projet a été écrit
(SDK Android inaccessible). Le premier `flutter build apk` peut donc encore
révéler un détail de configuration Gradle ; le reste de la chaîne est vérifié.

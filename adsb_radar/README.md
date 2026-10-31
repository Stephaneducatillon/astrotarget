# Radar ADS-B

Carte du trafic aérien en temps réel avec la liste des avions visibles dans la zone affichée.

- **Ouvrir** : double-cliquez `index.html` (aucune installation, un seul fichier).
- **Sources** (« Source et réglages ») : adsb.lol, airplanes.live, OpenSky Network
  (monde entier quand on dézoome), récepteur local dump1090/readsb (`aircraft.json`),
  simulation, ou import d'un instantané JSON.
- Positions extrapolées entre deux rafraîchissements (vitesse sol + route),
  codes 7500/7600/7700 mis en avant, distance/relèvement depuis « Ma position ».

Fichiers : `app.src.html` (source), `map-data.json` (traits de côte et frontières Natural Earth 1:50m,
via world-atlas), `build.js` (`node build.js` régénère `index.html`).

// Données de référence : planètes, observatoires, instruments, orientations.

export const SIGNATURE = 'CielScore Satellites — calculs CielScore';
export const YEAR_MIN = 2025;
export const YEAR_MAX = 2060;
export const DEFAULT_TZ = 'Europe/Paris';

// Aplatissement (pck00011), pôle nord IAU (ICRF J2000, degrés et degrés/siècle)
export const PLANETS = {
  Saturn: {
    key: 'Saturn', name: 'Saturne', body: 'Saturn', flattening: 0.09796,
    pole: [40.589, -0.036, 83.537, -0.004], ringOuter: 136780 / 60268,
  },
  Jupiter: {
    key: 'Jupiter', name: 'Jupiter', body: 'Jupiter', flattening: 0.06487,
    pole: [268.056595, -0.006499, 64.495303, 0.002413], ringOuter: 0,
  },
  Uranus: {
    key: 'Uranus', name: 'Uranus', body: 'Uranus', flattening: 0.02293,
    pole: [257.311, 0, -15.175, 0], ringOuter: 0,
  },
};

export const SITES = [
  { name: 'Calern — C2PU (OCA)', lat: 43.7536, lon: 6.9222, elevation: 1270 },
  { name: 'Cantin', lat: 50.3089, lon: 3.1269, elevation: 40 },
  { name: 'Observatoire de Haute-Provence', lat: 43.9317, lon: 5.7128, elevation: 650 },
  { name: 'Pic du Midi', lat: 42.9364, lon: 0.1425, elevation: 2877 },
  { name: 'Paris — Observatoire', lat: 48.8363, lon: 2.3363, elevation: 67 },
];

export class Instrument {
  constructor(name, focalMm, sensorW, sensorH, pixelUm) {
    Object.assign(this, { name, focalMm, sensorW, sensorH, pixelUm });
  }

  get scale() { return 206.264806 * this.pixelUm / this.focalMm; }

  get field() {
    const k = 206264.806 / this.focalMm;
    return [this.sensorW * k, this.sensorH * k];
  }
}

export const INSTRUMENTS = [
  new Instrument('C2PU 1,04 m f/12,5 — APS-C 23,5×15,7 mm', 13000, 23.5, 15.7, 3.76),
  new Instrument('C2PU 1,04 m f/12,5 — capteur 1″ 13,2×8,8 mm', 13000, 13.2, 8.8, 2.4),
  new Instrument('C14 f/11 — APS-C', 3910, 23.5, 15.7, 3.76),
  new Instrument('Lunette 120 f/7,5 — APS-C', 900, 23.5, 15.7, 3.76),
];

// Orientation du schéma : [rotation (°), miroir gauche-droite] ; référence « ciel » :
// nord en haut, est à gauche.
export const ORIENTATIONS = [
  { label: 'Ciel : nord en haut, est à gauche', rot: 0, mirror: false },
  { label: 'Newton / sans renvoi : sud en haut, est à droite', rot: 180, mirror: false },
  { label: 'Lunette ou SCT avec renvoi coudé : nord en haut, est à droite', rot: 0, mirror: true },
  { label: 'Rotation libre', rot: null, mirror: false },
];

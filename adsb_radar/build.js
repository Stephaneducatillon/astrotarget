// Assemble la page : node build.js  →  index.html (autonome) + artefact (sans squelette)
const fs = require("fs"), path = require("path");
const dir = __dirname;
const src = fs.readFileSync(path.join(dir, "app.src.html"), "utf8");
const map = fs.readFileSync(path.join(dir, "map-data.json"), "utf8").trim();
const body = src.replace("/*MAPDATA*/null", () => map);
const standalone = `<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}</style>
</head><body>
${body}
</body></html>
`;
fs.writeFileSync(path.join(dir, "index.html"), standalone);
const out = process.argv[2];
if (out) fs.writeFileSync(out, body);
console.log("index.html", standalone.length, out ? "+ " + out : "");

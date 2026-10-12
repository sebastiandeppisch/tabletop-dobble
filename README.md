# tabletop-dobble

Web-App, die das gemeinsame Symbol zweier (oder mehrerer) Dobble-Karten findet – live über die Kamera oder per Foto. Läuft komplett im Browser mit [OpenCV.js](https://docs.opencv.org/4.12.0/d5/d10/tutorial_js_root.html) in einem Web Worker.

## Pipeline

1. **Karten finden** – Weißanteil (`V − S` in HSV) per Otsu, Distanztransformation + Watershed trennt sich berührende Karten, Ellipsen-Fit nur auf dem äußeren Rand (robust gegen Überlappung und Perspektive).
2. **Symbole segmentieren** – pro Karte: Abweichung von der lokalen Kartenhelligkeit, Farbabstand (Lab) zum Kartenweiß und Kantenstärke; Closing + Hole-Fill ergibt Silhouetten.
3. **Normalisieren** – jede Karte wird per Affinität von Ellipse auf Kreis entzerrt, jedes Symbol auf Fläche und Schwerpunkt normiert und in 36 Rotationen als 64×64-Patch (Lab + Maske) abgetastet.
4. **Matchen** – für jedes Kartenpaar wird jedes Symbolpaar über alle Rotationen verglichen (Masken-Überdeckung + Farbabstand). Das beste Paar ist der Treffer, der Abstand zum Zweitbesten die angezeigte Sicherheit.

## Entwicklung

```bash
npm install
npm run dev      # Dev-Server
npm test         # Erkennung gegen tests/fixtures prüfen
npm run build    # Produktions-Build nach dist/
```

Testbilder erzeugen (synthetische Karten mit Noto-Emoji, Rotation, Licht-Gradient, Perspektive):

```bash
python3 scripts/generate_cards.py 16 tests/fixtures
npx tsx scripts/debug.ts /tmp/out tests/fixtures/*.jpg   # Overlays rendern
FIXTURES=/tmp/big npm test                               # gegen anderes Set testen
```

## Deployment

`.github/workflows/pages.yml` testet und baut bei jedem Push/PR und deployt `main` auf GitHub Pages.
Einmalig nötig: **Settings → Pages → Source: GitHub Actions**.

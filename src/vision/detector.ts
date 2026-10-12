import type * as CV from '@techstark/opencv-js'
import { extractFeatures, matchFeatures, type CardFeatures, type KeypointMatch } from './keypoints'
import type { CardMatch, DetectedCard, DetectedSymbol, DetectionResult, MatchRegion, Point, RgbaImage } from './types'

type Cv = typeof CV
type Mat = CV.Mat
type Affine = [number, number, number, number, number, number]

const WORK_SIZE = 960
const PATCH = 64
const PATCH_FILL = 0.28
const ROTATIONS = 36
const MIN_CARD_AREA = 0.02
const MIN_SYMBOL_AREA = 0.004
const CARD_SEARCH_SIZE = 480
const EDGE_THRESHOLD = 32
const MAX_SYMBOL_AREA = 0.25

interface Disposable {
  delete(): void
}

class Scope {
  private items: Disposable[] = []

  add<T extends Disposable>(item: T): T {
    this.items.push(item)
    return item
  }

  dispose(): void {
    for (const item of this.items.splice(0)) item.delete()
  }
}

interface CardRegion {
  mask: Mat
  rect: CV.Rect
  center: Point
  size: { width: number; height: number }
  angle: number
  rectify: Affine
}

interface SymbolRegion {
  id: number
  centroid: Point
  area: number
  contour: Point[]
  center: Point
  radius: number
  bounds: CV.Rect
}

interface Patch {
  mask: Uint8Array
  lab: Uint8Array
}

export function detect(cv: Cv, image: RgbaImage): DetectionResult {
  const started = performance.now()
  const scope = new Scope()
  try {
    const scale = Math.min(1, WORK_SIZE / Math.max(image.width, image.height))
    const rgb = loadRgb(cv, image, scale, scope)
    const lab = scope.add(new cv.Mat())
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab)

    const cardRegions = findCards(cv, rgb, scope)
    const labels = scope.add(cv.Mat.zeros(rgb.rows, rgb.cols, cv.CV_8UC1))
    const background = lightnessBackground(cv, lab, cardRegions, scope)

    let nextId = 1
    const cardSymbols = cardRegions.map((card) => {
      const symbols = findSymbols(cv, lab, background, card, labels, nextId, scope)
      nextId += symbols.length
      return symbols
    })

    const gray = scope.add(new cv.Mat())
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY)
    const chroma = chromaSampler(lab)
    const features = cardRegions.map((card) => cardFeatures(cv, gray, chroma, card, scope))
    let patches: Patch[][][] | null = null

    const matches: CardMatch[] = []
    for (let a = 0; a < cardRegions.length; a++) {
      for (let b = a + 1; b < cardRegions.length; b++) {
        const keypoints = matchFeatures(cv, features[a], features[b], (item) => scope.add(item))
        if (keypoints) {
          matches.push({
            a: keypointRegion(a, keypoints.inliersA, cardSymbols[a], cardRadius(cardRegions[a])),
            b: keypointRegion(b, keypoints.inliersB, cardSymbols[b], cardRadius(cardRegions[b])),
            confidence: keypointConfidence(keypoints),
            method: 'keypoints',
          })
          continue
        }

        patches ??= cardRegions.map((card, i) => cardSymbols[i].map((symbol) => symbolPatches(cv, lab, labels, card, symbol, scope)))
        const match = matchCards(patches[a], patches[b])
        if (match) {
          matches.push({
            a: symbolRegion(a, cardSymbols[a][match.a]),
            b: symbolRegion(b, cardSymbols[b][match.b]),
            confidence: match.confidence,
            method: 'patches',
          })
        }
      }
    }

    const toImage = (p: Point): Point => ({ x: p.x / scale, y: p.y / scale })
    const regionToImage = (region: MatchRegion): MatchRegion => ({
      card: region.card,
      center: toImage(region.center),
      radius: region.radius / scale,
      contour: region.contour?.map(toImage),
    })
    const cards: DetectedCard[] = cardRegions.map((card, i) => ({
      center: toImage(card.center),
      size: { width: card.size.width / scale, height: card.size.height / scale },
      angle: card.angle,
      symbols: cardSymbols[i].map<DetectedSymbol>((s) => ({
        center: toImage(s.center),
        radius: s.radius / scale,
        contour: s.contour.map(toImage),
      })),
    }))

    return {
      width: image.width,
      height: image.height,
      cards,
      matches: matches.map((m) => ({ ...m, a: regionToImage(m.a), b: regionToImage(m.b) })),
      durationMs: performance.now() - started,
    }
  } finally {
    scope.dispose()
  }
}

function loadRgb(cv: Cv, image: RgbaImage, scale: number, scope: Scope): Mat {
  const rgba = scope.add(new cv.Mat(image.height, image.width, cv.CV_8UC4))
  rgba.data.set(image.data)
  const rgb = scope.add(new cv.Mat())
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB)
  if (scale >= 1) return rgb
  const resized = scope.add(new cv.Mat())
  cv.resize(rgb, resized, new cv.Size(Math.round(image.width * scale), Math.round(image.height * scale)), 0, 0, cv.INTER_AREA)
  return resized
}

function findCards(cv: Cv, full: Mat, scope: Scope): CardRegion[] {
  const factor = Math.min(1, CARD_SEARCH_SIZE / Math.max(full.rows, full.cols))
  const rgb = scope.add(new cv.Mat())
  cv.resize(full, rgb, new cv.Size(Math.round(full.cols * factor), Math.round(full.rows * factor)), 0, 0, cv.INTER_AREA)
  const imageArea = rgb.rows * rgb.cols
  const hsv = scope.add(new cv.Mat())
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV)
  const channels = scope.add(new cv.MatVector())
  cv.split(hsv, channels)
  const saturation = scope.add(channels.get(1))
  const value = scope.add(channels.get(2))

  const white = scope.add(new cv.Mat())
  cv.subtract(value, saturation, white)
  cv.GaussianBlur(white, white, new cv.Size(5, 5), 0)
  cv.threshold(white, white, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  const kernel = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5)))
  cv.morphologyEx(white, white, cv.MORPH_OPEN, kernel)

  // Bridges symbols touching the rim so the card outline stays closed.
  const bridge = Math.round(Math.max(rgb.rows, rgb.cols) * 0.04) | 1
  const closing = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(bridge, bridge)))
  cv.morphologyEx(white, white, cv.MORPH_CLOSE, closing)

  const filled = scope.add(cv.Mat.zeros(rgb.rows, rgb.cols, cv.CV_8UC1))
  forEachContour(cv, white, scope, (contour, index, all) => {
    if (cv.contourArea(contour) > imageArea * MIN_CARD_AREA) cv.drawContours(filled, all, index, new cv.Scalar(255), cv.FILLED)
  })

  const dist = scope.add(new cv.Mat())
  cv.distanceTransform(filled, dist, cv.DIST_L2, 5)
  const maxDist = cv.minMaxLoc(dist, scope.add(new cv.Mat())).maxVal
  if (maxDist < 5) return []

  const peaks = scope.add(new cv.Mat())
  cv.threshold(dist, peaks, maxDist * 0.5, 255, cv.THRESH_BINARY)
  peaks.convertTo(peaks, cv.CV_8UC1)

  const seeds: { x: number; y: number; r: number }[] = []
  forEachContour(cv, peaks, scope, (contour) => {
    const m = cv.moments(contour)
    if (!m.m00) return
    const x = m.m10 / m.m00
    const y = m.m01 / m.m00
    seeds.push({ x, y, r: dist.floatAt(Math.round(y), Math.round(x)) })
  })
  if (!seeds.length) return []

  // Power-diagram split: each pixel goes to the card whose estimated rim is closest.
  const labels = new Int32Array(imageArea)
  const filledData = filled.data
  for (let y = 0; y < rgb.rows; y++) {
    for (let x = 0; x < rgb.cols; x++) {
      const i = y * rgb.cols + x
      if (!filledData[i]) continue
      let best = 0
      let bestScore = Infinity
      seeds.forEach((seed, k) => {
        const score = Math.hypot(x - seed.x, y - seed.y) - seed.r
        if (score < bestScore) {
          bestScore = score
          best = k + 1
        }
      })
      labels[i] = best
    }
  }

  const regions: CardRegion[] = []
  for (let label = 1; label <= seeds.length; label++) {
    const small = scope.add(cv.Mat.zeros(rgb.rows, rgb.cols, cv.CV_8UC1))
    const smallData = small.data
    let area = 0
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === label) {
        smallData[i] = 255
        area++
      }
    }
    if (area < imageArea * MIN_CARD_AREA) continue

    const outline = largestContour(cv, small, scope)
    if (!outline) continue
    const rim = outerRimPoints(cv, outline, dist, scope)
    if (rim.rows < 20) continue

    const ellipse = cv.fitEllipse(rim)
    const ellipseArea = (Math.PI * ellipse.size.width * ellipse.size.height) / 4
    if (Math.min(ellipse.size.width, ellipse.size.height) / Math.max(ellipse.size.width, ellipse.size.height) < 0.35) continue
    if (area / ellipseArea < 0.55 || area / ellipseArea > 1.15) continue

    const mask = scope.add(new cv.Mat())
    cv.resize(small, mask, new cv.Size(full.cols, full.rows), 0, 0, cv.INTER_NEAREST)
    const center = { x: ellipse.center.x / factor, y: ellipse.center.y / factor }
    const width = ellipse.size.width / factor
    const height = ellipse.size.height / factor
    const box = cv.boundingRect(outline)
    const x = Math.max(0, Math.floor(box.x / factor) - 2)
    const y = Math.max(0, Math.floor(box.y / factor) - 2)
    const rect = new cv.Rect(x, y, Math.min(full.cols - x, Math.ceil(box.width / factor) + 4), Math.min(full.rows - y, Math.ceil(box.height / factor) + 4))

    regions.push({
      mask,
      rect,
      center,
      size: { width, height },
      angle: ellipse.angle,
      rectify: ellipseToCircle(center, width, height, ellipse.angle),
    })
  }

  return regions.sort((a, b) => a.center.x - b.center.x || a.center.y - b.center.y)
}

// Contour points on the outer, convex card rim. Drops cut lines between touching cards
// and dents from symbols touching the edge so the ellipse fit stays true.
function outerRimPoints(cv: Cv, contour: Mat, dist: Mat, scope: Scope): Mat {
  const hull = scope.add(new cv.Mat())
  cv.convexHull(contour, hull, false, true)
  const points: number[] = []
  const data = contour.data32S
  for (let i = 0; i < data.length; i += 2) {
    if (dist.floatAt(data[i + 1], data[i]) > 3) continue
    if (Math.abs(cv.pointPolygonTest(hull, new cv.Point(data[i], data[i + 1]), true)) > 2) continue
    points.push(data[i], data[i + 1])
  }
  return scope.add(cv.matFromArray(points.length / 2, 1, cv.CV_32SC2, points))
}

function ellipseToCircle(center: Point, width: number, height: number, angleDeg: number): Affine {
  const radius = Math.max(width, height) / 2
  const sx = radius / (width / 2)
  const sy = radius / (height / 2)
  const t = (angleDeg * Math.PI) / 180
  const c = Math.cos(t)
  const s = Math.sin(t)
  const a = c * c * sx + s * s * sy
  const b = c * s * (sx - sy)
  const d = b
  const e = s * s * sx + c * c * sy
  return [a, b, center.x - a * center.x - b * center.y, d, e, center.y - d * center.x - e * center.y]
}

function lightnessBackground(cv: Cv, lab: Mat, cards: CardRegion[], scope: Scope): Mat {
  const channels = scope.add(new cv.MatVector())
  cv.split(lab, channels)
  const lightness = scope.add(channels.get(0))
  const factor = 8
  const small = scope.add(new cv.Mat())
  cv.resize(lightness, small, new cv.Size(Math.ceil(lab.cols / factor), Math.ceil(lab.rows / factor)), 0, 0, cv.INTER_AREA)

  const radius = cards.length ? Math.min(...cards.map((c) => Math.max(c.size.width, c.size.height) / 2)) : lab.cols / 4
  const k = Math.max(3, Math.round((radius * 0.6) / factor)) | 1
  const kernel = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k)))
  cv.dilate(small, small, kernel)
  cv.GaussianBlur(small, small, new cv.Size(k, k), 0)

  const background = scope.add(new cv.Mat())
  cv.resize(small, background, new cv.Size(lab.cols, lab.rows), 0, 0, cv.INTER_LINEAR)
  return background
}

function innerMask(cv: Cv, card: CardRegion, scope: Scope): Mat {
  const { x, y, width, height } = card.rect
  const inner = scope.add(cv.Mat.zeros(height, width, cv.CV_8UC1))
  cv.ellipse1(
    inner,
    new cv.RotatedRect(new cv.Point(card.center.x - x, card.center.y - y), new cv.Size(card.size.width * 0.93, card.size.height * 0.93), card.angle),
    new cv.Scalar(255),
    cv.FILLED,
    cv.LINE_8,
  )
  cv.bitwise_and(inner, scope.add(card.mask.roi(card.rect)), inner)
  return inner
}

function cardFeatures(cv: Cv, gray: Mat, chroma: ChromaSampler, card: CardRegion, scope: Scope): CardFeatures {
  const r = card.rectify
  return extractFeatures(
    cv,
    {
      gray: crop(cv, gray, card.rect, scope),
      mask: innerMask(cv, card, scope),
      offset: { x: card.rect.x, y: card.rect.y },
      radius: cardRadius(card),
      rectify: (p) => ({ x: r[0] * p.x + r[1] * p.y + r[2], y: r[3] * p.x + r[4] * p.y + r[5] }),
      chroma,
    },
    (item) => scope.add(item),
  )
}

type ChromaSampler = (p: Point, radius: number) => [number, number]

function chromaSampler(lab: Mat): ChromaSampler {
  const data = lab.data
  const { rows, cols } = lab
  return (p, radius) => {
    const r = Math.round(radius)
    const x0 = Math.max(0, Math.round(p.x) - r)
    const x1 = Math.min(cols - 1, Math.round(p.x) + r)
    const y0 = Math.max(0, Math.round(p.y) - r)
    const y1 = Math.min(rows - 1, Math.round(p.y) + r)
    let a = 0
    let b = 0
    let n = 0
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = (y * cols + x) * 3
        a += data[i + 1]
        b += data[i + 2]
        n++
      }
    }
    return n ? [a / n, b / n] : [128, 128]
  }
}

function cardRadius(card: CardRegion): number {
  return Math.max(card.size.width, card.size.height) / 2
}

function findSymbols(
  cv: Cv,
  lab: Mat,
  background: Mat,
  card: CardRegion,
  labels: Mat,
  firstId: number,
  scope: Scope,
): SymbolRegion[] {
  const { x: ox, y: oy, width: w, height: h } = card.rect
  const radius = cardRadius(card)
  const labRoi = crop(cv, lab, card.rect, scope)
  const bgRoi = crop(cv, background, card.rect, scope)
  const inner = innerMask(cv, card, scope)

  const px = labRoi.data
  const bg = bgRoi.data
  const mask = inner.data
  let sumA = 0
  let sumB = 0
  let count = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || px[i * 3] < bg[i] * 0.95) continue
    sumA += px[i * 3 + 1]
    sumB += px[i * 3 + 2]
    count++
  }
  if (count === 0) return []
  const a0 = sumA / count
  const b0 = sumB / count

  const edges = edgeStrength(cv, labRoi, scope)
  const foreground = scope.add(cv.Mat.zeros(h, w, cv.CV_8UC1))
  const fg = foreground.data
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const darker = bg[i] - px[i * 3] > bg[i] * 0.1
    const chroma = Math.hypot(px[i * 3 + 1] - a0, px[i * 3 + 2] - b0) > 16
    if (darker || chroma || edges[i] > EDGE_THRESHOLD) fg[i] = 255
  }

  const small = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3)))
  cv.morphologyEx(foreground, foreground, cv.MORPH_OPEN, small)
  const k = Math.max(3, Math.round(radius * 0.04)) | 1
  const grow = scope.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k)))
  const silhouettes = scope.add(new cv.Mat())
  cv.dilate(foreground, silhouettes, grow)
  forEachContour(cv, silhouettes, scope, (_, index, all) => cv.drawContours(silhouettes, all, index, new cv.Scalar(255), cv.FILLED))
  cv.erode(silhouettes, silhouettes, grow)
  cv.bitwise_and(silhouettes, inner, silhouettes)

  const cardArea = Math.PI * radius * radius
  const symbols: SymbolRegion[] = []
  const offset = new cv.Point(ox, oy)
  forEachContour(cv, silhouettes, scope, (outline, index, all) => {
    const moments = cv.moments(outline)
    if (moments.m00 < cardArea * MIN_SYMBOL_AREA || moments.m00 > cardArea * MAX_SYMBOL_AREA) return
    const id = firstId + symbols.length
    if (id > 255) return
    cv.drawContours(labels, all, index, new cv.Scalar(id), cv.FILLED, cv.LINE_8, scope.add(new cv.Mat()), 0, offset)

    const circle = cv.minEnclosingCircle(outline)
    const simplified = scope.add(new cv.Mat())
    cv.approxPolyDP(outline, simplified, 1.5, true)
    const bounds = cv.boundingRect(outline)
    symbols.push({
      id,
      centroid: { x: moments.m10 / moments.m00 + ox, y: moments.m01 / moments.m00 + oy },
      area: moments.m00,
      contour: toPoints(simplified).map((p) => ({ x: p.x + ox, y: p.y + oy })),
      center: { x: circle.center.x + ox, y: circle.center.y + oy },
      radius: circle.radius,
      bounds: new cv.Rect(bounds.x + ox, bounds.y + oy, bounds.width, bounds.height),
    })
  })
  return symbols
}

function edgeStrength(cv: Cv, lab: Mat, scope: Scope): Uint8Array {
  const channels = scope.add(new cv.MatVector())
  cv.split(lab, channels)
  const lightness = scope.add(channels.get(0))
  cv.GaussianBlur(lightness, lightness, new cv.Size(5, 5), 0)
  const gx = scope.add(new cv.Mat())
  const gy = scope.add(new cv.Mat())
  cv.Sobel(lightness, gx, cv.CV_16S, 1, 0, 3)
  cv.Sobel(lightness, gy, cv.CV_16S, 0, 1, 3)
  cv.convertScaleAbs(gx, gx)
  cv.convertScaleAbs(gy, gy)
  const magnitude = scope.add(new cv.Mat())
  cv.addWeighted(gx, 0.5, gy, 0.5, 0, magnitude)
  return magnitude.data
}

function symbolPatches(cv: Cv, lab: Mat, labels: Mat, card: CardRegion, symbol: SymbolRegion, scope: Scope): Patch[] {
  const r = card.rectify
  const det = Math.abs(r[0] * r[4] - r[1] * r[3])
  const cx = r[0] * symbol.centroid.x + r[1] * symbol.centroid.y + r[2]
  const cy = r[3] * symbol.centroid.x + r[4] * symbol.centroid.y + r[5]
  const scale = Math.sqrt((PATCH * PATCH * PATCH_FILL) / (symbol.area * det))
  const size = new cv.Size(PATCH, PATCH)
  const { x: ox, y: oy } = symbol.bounds
  const labSrc = scope.add(lab.roi(symbol.bounds))
  const labelSrc = scope.add(labels.roi(symbol.bounds))
  const labOut = scope.add(new cv.Mat())
  const maskOut = scope.add(new cv.Mat())

  const patches: Patch[] = []
  for (let step = 0; step < ROTATIONS; step++) {
    const t = (step * 2 * Math.PI) / ROTATIONS
    const c = Math.cos(t) * scale
    const s = Math.sin(t) * scale
    const half = PATCH / 2
    // patch = scale * R(t) * (rectify(roiOrigin + q) - centroid) + half
    const m00 = c * r[0] - s * r[3]
    const m01 = c * r[1] - s * r[4]
    const m10 = s * r[0] + c * r[3]
    const m11 = s * r[1] + c * r[4]
    const m: Affine = [
      m00,
      m01,
      c * (r[2] - cx) - s * (r[5] - cy) + half + m00 * ox + m01 * oy,
      m10,
      m11,
      s * (r[2] - cx) + c * (r[5] - cy) + half + m10 * ox + m11 * oy,
    ]
    const affine = cv.matFromArray(2, 3, cv.CV_64FC1, m)
    cv.warpAffine(labSrc, labOut, affine, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0))
    cv.warpAffine(labelSrc, maskOut, affine, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0))
    affine.delete()

    const labelData = maskOut.data
    const mask = new Uint8Array(PATCH * PATCH)
    for (let i = 0; i < mask.length; i++) mask[i] = labelData[i] === symbol.id ? 1 : 0
    patches.push({ mask, lab: new Uint8Array(labOut.data) })
  }
  return patches
}

function patchDistance(a: Patch, b: Patch): number {
  let union = 0
  let cost = 0
  for (let i = 0; i < a.mask.length; i++) {
    const ma = a.mask[i]
    const mb = b.mask[i]
    if (!ma && !mb) continue
    union++
    if (ma && mb) {
      const j = i * 3
      const dl = a.lab[j] - b.lab[j]
      const da = a.lab[j + 1] - b.lab[j + 1]
      const db = a.lab[j + 2] - b.lab[j + 2]
      cost += Math.min(Math.sqrt(dl * dl + da * da + db * db) / 60, 1)
    } else {
      cost += 1
    }
  }
  return union ? cost / union : 1
}

function symbolRegion(card: number, symbol: SymbolRegion): MatchRegion {
  return { card, center: symbol.center, radius: symbol.radius, contour: symbol.contour }
}

// Prefers the segmented symbol holding most inliers; falls back to a circle around the inliers.
function keypointRegion(card: number, inliers: Point[], symbols: SymbolRegion[], radius: number): MatchRegion {
  const owner = symbols.find((s) => inliers.filter((p) => insidePolygon(p, s.contour)).length >= inliers.length * 0.6)
  if (owner && owner.radius < radius * 0.4) return symbolRegion(card, owner)

  const center = {
    x: inliers.reduce((sum, p) => sum + p.x, 0) / inliers.length,
    y: inliers.reduce((sum, p) => sum + p.y, 0) / inliers.length,
  }
  const spread = Math.max(...inliers.map((p) => Math.hypot(p.x - center.x, p.y - center.y)))
  return { card, center, radius: Math.max(spread * 1.1, radius * 0.12) }
}

function keypointConfidence(match: KeypointMatch): number {
  return Math.max(0, Math.min(1, 1 - match.runnerUp / match.inliers))
}

function insidePolygon(p: Point, polygon: Point[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i]
    const b = polygon[j]
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside
  }
  return inside
}

function matchCards(a: Patch[][], b: Patch[][]): { a: number; b: number; score: number; confidence: number } | null {
  if (!a.length || !b.length) return null
  const scores: { a: number; b: number; score: number }[] = []
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      let best = Infinity
      for (const rotated of b[j]) best = Math.min(best, patchDistance(a[i][0], rotated))
      scores.push({ a: i, b: j, score: best })
    }
  }
  scores.sort((x, y) => x.score - y.score)
  const [best, second] = scores
  const confidence = second ? Math.max(0, Math.min(1, (second.score - best.score) / second.score)) : 0
  return { ...best, confidence }
}

function forEachContour(cv: Cv, binary: Mat, scope: Scope, fn: (contour: Mat, index: number, all: CV.MatVector) => void): void {
  const contours = scope.add(new cv.MatVector())
  const hierarchy = scope.add(new cv.Mat())
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  for (let i = 0; i < contours.size(); i++) fn(scope.add(contours.get(i)), i, contours)
}

function largestContour(cv: Cv, binary: Mat, scope: Scope): Mat | null {
  let best: Mat | null = null
  let bestArea = 0
  forEachContour(cv, binary, scope, (contour) => {
    const area = cv.contourArea(contour)
    if (area > bestArea) {
      best = contour
      bestArea = area
    }
  })
  return best
}

// Mat.roi().clone() keeps the parent stride in opencv.js, copyTo yields a continuous Mat.
function crop(cv: Cv, mat: Mat, rect: CV.Rect, scope: Scope): Mat {
  const out = scope.add(new cv.Mat())
  scope.add(mat.roi(rect)).copyTo(out)
  return out
}

function toPoints(contour: Mat): Point[] {
  const data = contour.data32S
  const points: Point[] = []
  for (let i = 0; i < data.length; i += 2) points.push({ x: data[i], y: data[i + 1] })
  return points
}

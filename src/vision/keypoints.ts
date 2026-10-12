import type * as CV from '@techstark/opencv-js'
import type { Point } from './types'

type Cv = typeof CV
type Mat = CV.Mat

const FEATURES = 3000
const RATIO = 0.85
const ITERATIONS = 1500
const MIN_INLIERS = 5
const ANGLE_TOLERANCE = (25 * Math.PI) / 180
const COLOR_TOLERANCE = 18

export interface CardFeatures {
  points: Point[]
  rectified: Point[]
  angles: number[]
  colors: [number, number][]
  descriptors: Mat
  radius: number
}

export interface CardInput {
  gray: Mat
  mask: Mat
  offset: Point
  radius: number
  rectify: (p: Point) => Point
  chroma: (p: Point, radius: number) => [number, number]
}

type Track = <T extends { delete(): void }>(item: T) => T

export interface KeypointMatch {
  inliersA: Point[]
  inliersB: Point[]
  inliers: number
  runnerUp: number
}

interface Correspondence {
  a: number
  b: number
}

interface Similarity {
  c: number
  s: number
  tx: number
  ty: number
}

export function extractFeatures(cv: Cv, input: CardInput, track: Track): CardFeatures {
  const detector = track(new cv.ORB(FEATURES))
  const keypoints = track(new cv.KeyPointVector())
  const descriptors = track(new cv.Mat())
  detector.detectAndCompute(input.gray, input.mask, keypoints, descriptors)

  const features: CardFeatures = { points: [], rectified: [], angles: [], colors: [], descriptors, radius: input.radius }
  for (let i = 0; i < keypoints.size(); i++) {
    const kp = keypoints.get(i)
    const point = { x: kp.pt.x + input.offset.x, y: kp.pt.y + input.offset.y }
    features.points.push(point)
    features.rectified.push(input.rectify(point))
    features.angles.push((kp.angle * Math.PI) / 180)
    features.colors.push(input.chroma(point, Math.max(3, kp.size / 4)))
  }
  return features
}

export function matchFeatures(cv: Cv, a: CardFeatures, b: CardFeatures, track: Track): KeypointMatch | null {
  if (a.descriptors.rows < 2 || b.descriptors.rows < 2) return null

  const matcher = track(new cv.BFMatcher(cv.NORM_HAMMING, false))
  const knn = track(new cv.DMatchVectorVector())
  matcher.knnMatch(a.descriptors, b.descriptors, knn, 2)

  const pairs: Correspondence[] = []
  for (let i = 0; i < knn.size(); i++) {
    const candidates = knn.get(i)
    if (candidates.size() < 2) continue
    const best = candidates.get(0)
    const second = candidates.get(1)
    if (best.distance >= RATIO * second.distance) continue
    const [ca, cb] = [a.colors[best.queryIdx], b.colors[best.trainIdx]]
    if (Math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= COLOR_TOLERANCE) pairs.push({ a: best.queryIdx, b: best.trainIdx })
  }
  if (pairs.length < MIN_INLIERS) return null

  const random = seededRandom(pairs.length)
  const first = ransac(a, b, pairs, random)
  if (!first || first.length < MIN_INLIERS) return null
  const used = new Set(first)
  const second = ransac(a, b, pairs.filter((_, i) => !used.has(i)), random)

  return {
    inliersA: first.map((i) => a.points[pairs[i].a]),
    inliersB: first.map((i) => b.points[pairs[i].b]),
    inliers: first.length,
    runnerUp: second?.length ?? 0,
  }
}

function ransac(a: CardFeatures, b: CardFeatures, pairs: Correspondence[], random: () => number): number[] | null {
  if (pairs.length < 2) return null
  const tolerance = b.radius * 0.04
  const reach = Math.max(a.radius, b.radius) * 0.5
  let best: number[] = []

  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const i = Math.floor(random() * pairs.length)
    const j = Math.floor(random() * pairs.length)
    if (i === j) continue
    const pi = pairs[i]
    const pj = pairs[j]
    const ai = a.rectified[pi.a]
    const aj = a.rectified[pj.a]
    if (Math.hypot(ai.x - aj.x, ai.y - aj.y) > reach) continue

    const model = fitSimilarity(ai, aj, b.rectified[pi.b], b.rectified[pj.b])
    if (!model) continue
    const scale = Math.hypot(model.c, model.s)
    if (scale < 0.3 || scale > 3.3) continue
    const rotation = Math.atan2(model.s, model.c)

    const inliers: number[] = []
    for (let k = 0; k < pairs.length; k++) {
      const p = a.rectified[pairs[k].a]
      const q = b.rectified[pairs[k].b]
      const x = model.c * p.x - model.s * p.y + model.tx
      const y = model.s * p.x + model.c * p.y + model.ty
      if (Math.hypot(x - q.x, y - q.y) > tolerance) continue
      if (angleDistance(b.angles[pairs[k].b] - a.angles[pairs[k].a], rotation) > ANGLE_TOLERANCE) continue
      inliers.push(k)
    }
    if (inliers.length > best.length && compact(a, pairs, inliers, reach)) best = inliers
  }
  return best.length ? best : null
}

// mulberry32, so identical frames give identical results
function seededRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function fitSimilarity(p1: Point, p2: Point, q1: Point, q2: Point): Similarity | null {
  const dx = p2.x - p1.x
  const dy = p2.y - p1.y
  const ex = q2.x - q1.x
  const ey = q2.y - q1.y
  const norm = dx * dx + dy * dy
  if (norm < 4) return null
  const c = (dx * ex + dy * ey) / norm
  const s = (dx * ey - dy * ex) / norm
  return { c, s, tx: q1.x - (c * p1.x - s * p1.y), ty: q1.y - (s * p1.x + c * p1.y) }
}

function angleDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % (2 * Math.PI)
  return d > Math.PI ? 2 * Math.PI - d : d
}

function compact(a: CardFeatures, pairs: Correspondence[], inliers: number[], reach: number): boolean {
  const points = inliers.map((k) => a.rectified[pairs[k].a])
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length
  return points.every((p) => Math.hypot(p.x - cx, p.y - cy) < reach)
}

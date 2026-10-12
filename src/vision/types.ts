export interface Point {
  x: number
  y: number
}

export interface DetectedSymbol {
  center: Point
  radius: number
  contour: Point[]
}

export interface DetectedCard {
  center: Point
  size: { width: number; height: number }
  angle: number
  symbols: DetectedSymbol[]
}

export interface MatchRegion {
  card: number
  center: Point
  radius: number
  contour?: Point[]
}

export interface CardMatch {
  a: MatchRegion
  b: MatchRegion
  confidence: number
  method: 'keypoints' | 'patches'
}

export interface DetectionResult {
  width: number
  height: number
  cards: DetectedCard[]
  matches: CardMatch[]
  durationMs: number
}

export interface RgbaImage {
  data: Uint8ClampedArray | Uint8Array
  width: number
  height: number
}

export type WorkerRequest = { id: number; image: RgbaImage }

export type WorkerResponse = { type: 'ready' } | { type: 'result'; id: number; result: DetectionResult } | { type: 'error'; id?: number; message: string }

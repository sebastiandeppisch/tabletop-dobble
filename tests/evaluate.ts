import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import jpeg from 'jpeg-js'
import type { DetectionResult, Point } from '../src/vision/types'

export interface Truth {
  shared: string
  cards: { cx: number; cy: number; r: number; symbols: { char: string; x: number; y: number; r: number }[] }[]
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

export function loadCase(dir: string, name: string) {
  return {
    truth: JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8')) as Truth,
    image: jpeg.decode(readFileSync(join(dir, `${name}.jpg`)), { useTArray: true }),
  }
}

// Returns null when the detected match covers the shared symbol on both cards.
export function evaluate(result: DetectionResult, truth: Truth): string | null {
  if (result.cards.length !== truth.cards.length) return `found ${result.cards.length} cards`
  if (result.matches.length !== 1) return `found ${result.matches.length} matches`
  const [match] = result.matches
  for (const region of [match.a, match.b]) {
    const card = result.cards[region.card]
    const truthCard = truth.cards.reduce((best, c) =>
      distance(card.center, { x: c.cx, y: c.cy }) < distance(card.center, { x: best.cx, y: best.cy }) ? c : best,
    )
    const expected = truthCard.symbols.find((s) => s.char === truth.shared)!
    if (distance(region.center, expected) > expected.r) return `wrong symbol on card ${region.card}`
  }
  return null
}

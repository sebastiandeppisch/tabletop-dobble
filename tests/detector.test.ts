import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import jpeg from 'jpeg-js'
import assert from 'node:assert/strict'
import { before, describe, it } from 'node:test'
import { detect } from '../src/vision/detector'
import { loadOpenCv, type OpenCv } from '../src/vision/opencv'
import type { Point } from '../src/vision/types'

interface Truth {
  shared: string
  cards: { cx: number; cy: number; r: number; symbols: { char: string; x: number; y: number; r: number }[] }[]
}

const fixtures = process.env.FIXTURES ?? join(import.meta.dirname, 'fixtures')
const cases = readdirSync(fixtures)
  .filter((f) => f.endsWith('.jpg'))
  .map((f) => f.replace('.jpg', ''))

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

let cv: OpenCv

before(async () => {
  cv = (await loadOpenCv()).cv
})

describe('detect', () => {
  for (const name of cases) {
  it(`finds the shared symbol in ${name}`, () => {
    const truth: Truth = JSON.parse(readFileSync(join(fixtures, `${name}.json`), 'utf8'))
    const image = jpeg.decode(readFileSync(join(fixtures, `${name}.jpg`)), { useTArray: true })
    const result = detect(cv, image)

    assert.equal(result.cards.length, truth.cards.length, 'card count')
    const truthCards = result.cards.map((card) =>
      truth.cards.reduce((best, c) => (distance(card.center, { x: c.cx, y: c.cy }) < distance(card.center, { x: best.cx, y: best.cy }) ? c : best)),
    )
    result.cards.forEach((card, i) => assert.equal(card.symbols.length, truthCards[i].symbols.length, `symbols on card ${i}`))

    assert.equal(result.matches.length, 1, 'match count')
    const [match] = result.matches
    for (const ref of [match.a, match.b]) {
      const found = result.cards[ref.card].symbols[ref.symbol].center
      const expected = truthCards[ref.card].symbols.find((s) => s.char === truth.shared)!
      assert.ok(distance(found, expected) < expected.r, `wrong symbol on card ${ref.card}`)
    }
  })
  }
})

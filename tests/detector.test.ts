import assert from 'node:assert/strict'
import { readdirSync } from 'node:fs'
import { join } from 'node:path'
import { test } from 'node:test'
import { detect } from '../src/vision/detector'
import { loadOpenCv } from '../src/vision/opencv'
import { evaluate, loadCase } from './evaluate'

const MIN_ACCURACY = 0.9

const fixtures = process.env.FIXTURES ?? join(import.meta.dirname, 'fixtures')
const cases = readdirSync(fixtures)
  .filter((f) => f.endsWith('.jpg'))
  .map((f) => f.replace('.jpg', ''))

test('finds the shared symbol on synthetic card photos', async () => {
  const { cv } = await loadOpenCv()
  const failures = cases.flatMap((name) => {
    const { truth, image } = loadCase(fixtures, name)
    const error = evaluate(detect(cv, image), truth)
    return error ? [`${name}: ${error}`] : []
  })
  const accuracy = 1 - failures.length / cases.length
  assert.ok(accuracy >= MIN_ACCURACY, `accuracy ${(accuracy * 100).toFixed(0)} %\n${failures.join('\n')}`)
})

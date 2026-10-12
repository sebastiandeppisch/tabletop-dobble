import { readdirSync } from 'node:fs'
import { evaluate, loadCase } from '../tests/evaluate'
import { detect } from '../src/vision/detector'
import { loadOpenCv } from '../src/vision/opencv'

const dir = process.argv[2]
const { cv } = await loadOpenCv()
const names = readdirSync(dir)
  .filter((f) => f.endsWith('.jpg'))
  .map((f) => f.replace('.jpg', ''))

const methods: Record<string, number> = {}
let correct = 0
let time = 0
for (const name of names) {
  const { truth, image } = loadCase(dir, name)
  const result = detect(cv, image)
  time += result.durationMs
  for (const m of result.matches) methods[m.method] = (methods[m.method] ?? 0) + 1
  const error = evaluate(result, truth)
  if (error) console.log(name, error, result.matches.map((m) => `${m.method} ${m.confidence.toFixed(2)}`).join())
  else correct++
}
console.log(`${correct}/${names.length} correct, avg ${(time / names.length).toFixed(0)} ms`, methods)

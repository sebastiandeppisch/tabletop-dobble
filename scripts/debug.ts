import { readFileSync, writeFileSync } from 'node:fs'
import { basename } from 'node:path'
import jpeg from 'jpeg-js'
import { detect } from '../src/vision/detector'
import { loadOpenCv } from '../src/vision/opencv'

const [outDir, ...files] = process.argv.slice(2)
const { cv } = await loadOpenCv()

for (const file of files) {
  const image = jpeg.decode(readFileSync(file), { useTArray: true })
  const result = detect(cv, image)
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC4)
  mat.data.set(image.data)
  result.cards.forEach((card) => {
    const rect = new cv.RotatedRect(new cv.Point(card.center.x, card.center.y), new cv.Size(card.size.width, card.size.height), card.angle)
    cv.ellipse1(mat, rect, new cv.Scalar(0, 120, 255, 255), 3, cv.LINE_8)
    card.symbols.forEach((s) => {
      const pts = cv.matFromArray(s.contour.length, 1, cv.CV_32SC2, s.contour.flatMap((p) => [Math.round(p.x), Math.round(p.y)]))
      const vec = new cv.MatVector()
      vec.push_back(pts)
      cv.polylines(mat, vec, true, new cv.Scalar(0, 200, 0, 255), 2)
      vec.delete()
      pts.delete()
    })
  })
  for (const m of result.matches) {
    for (const region of [m.a, m.b]) cv.circle(mat, new cv.Point(region.center.x, region.center.y), region.radius + 8, new cv.Scalar(255, 0, 200, 255), 6)
  }
  console.log(basename(file), result.cards.map((c) => c.symbols.length), result.matches.map((m) => `${m.method} conf ${m.confidence.toFixed(2)}`), `${result.durationMs.toFixed(0)}ms`)
  writeFileSync(`${outDir}/${basename(file)}`, jpeg.encode({ data: Buffer.from(mat.data), width: image.width, height: image.height }, 80).data)
  mat.delete()
}

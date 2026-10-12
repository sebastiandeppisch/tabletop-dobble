import type { DetectionResult } from './vision/types'

export type Drawable = HTMLVideoElement | ImageBitmap

const MATCH_COLORS = ['#ff2d95', '#00e5ff', '#ffd60a', '#7cff4f', '#b388ff', '#ff9100']

export function sourceSize(source: Drawable): { width: number; height: number } {
  return source instanceof HTMLVideoElement
    ? { width: source.videoWidth, height: source.videoHeight }
    : { width: source.width, height: source.height }
}

export function matchColor(index: number): string {
  return MATCH_COLORS[index % MATCH_COLORS.length]
}

export function drawScene(ctx: CanvasRenderingContext2D, source: Drawable, result: DetectionResult | null, showDetails: boolean): void {
  const { width, height } = ctx.canvas
  const size = sourceSize(source)
  ctx.clearRect(0, 0, width, height)
  if (!size.width || !size.height) return

  const fit = Math.min(width / size.width, height / size.height)
  const dx = (width - size.width * fit) / 2
  const dy = (height - size.height * fit) / 2
  ctx.drawImage(source, dx, dy, size.width * fit, size.height * fit)
  if (!result) return

  const k = (fit * size.width) / result.width
  const px = (pixels: number) => (pixels * devicePixelRatio) / k
  ctx.save()
  ctx.translate(dx, dy)
  ctx.scale(k, k)

  for (const card of result.cards) {
    ctx.beginPath()
    ctx.ellipse(card.center.x, card.center.y, card.size.width / 2, card.size.height / 2, (card.angle * Math.PI) / 180, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)'
    ctx.lineWidth = px(2)
    ctx.stroke()

    if (!showDetails) continue
    for (const symbol of card.symbols) {
      ctx.beginPath()
      symbol.contour.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)))
      ctx.closePath()
      ctx.strokeStyle = 'rgba(80, 220, 120, 0.9)'
      ctx.lineWidth = px(1.5)
      ctx.stroke()
    }
  }

  result.matches.forEach((match, index) => {
    const color = matchColor(index)
    const [a, b] = [match.a, match.b].map((ref) => result.cards[ref.card].symbols[ref.symbol])

    ctx.beginPath()
    ctx.moveTo(a.center.x, a.center.y)
    ctx.lineTo(b.center.x, b.center.y)
    ctx.setLineDash([px(8), px(8)])
    ctx.strokeStyle = color
    ctx.lineWidth = px(2)
    ctx.stroke()
    ctx.setLineDash([])

    for (const symbol of [a, b]) {
      ctx.beginPath()
      ctx.arc(symbol.center.x, symbol.center.y, symbol.radius * 1.15 + px(4), 0, Math.PI * 2)
      ctx.shadowColor = color
      ctx.shadowBlur = px(16)
      ctx.strokeStyle = color
      ctx.lineWidth = px(5)
      ctx.stroke()
      ctx.shadowBlur = 0
    }
  })
  ctx.restore()
}

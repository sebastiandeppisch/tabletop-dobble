<script lang="ts" setup>
import { computed, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import { useCamera } from './composables/useCamera'
import { useDetector } from './composables/useDetector'
import { drawScene, matchColor, sourceSize, type Drawable } from './render'
import type { DetectionResult } from './vision/types'

const DETECT_SIZE = 1280
const LOW_CONFIDENCE = 0.15

const detector = useDetector()
const camera = useCamera()

const canvas = ref<HTMLCanvasElement>()
const video = ref<HTMLVideoElement>()
const fileInput = ref<HTMLInputElement>()

const mode = ref<'idle' | 'camera' | 'photo'>('idle')
const showDetails = ref(false)
const still = shallowRef<ImageBitmap | null>(null)
const result = shallowRef<DetectionResult | null>(null)
const analyzing = ref(false)
const notice = ref<string | null>(null)

const grabCanvas = document.createElement('canvas')
const grabContext = grabCanvas.getContext('2d', { willReadFrequently: true })!
let frame = 0

const source = computed<Drawable | null>(() => still.value ?? (mode.value === 'camera' ? (video.value ?? null) : null))
const paused = computed(() => mode.value === 'camera' && still.value !== null)

const status = computed(() => {
  if (notice.value) return { text: notice.value, tone: 'ok' }
  if (detector.error.value) return { text: detector.error.value, tone: 'error' }
  if (camera.error.value) return { text: `Kamera: ${camera.error.value}`, tone: 'error' }
  if (!detector.ready.value) return { text: 'OpenCV wird geladen …', tone: 'muted' }
  if (mode.value === 'idle') return { text: 'Kamera starten oder Foto wählen', tone: 'muted' }
  if (!result.value) return { text: analyzing.value ? 'Analysiere …' : 'Suche Karten …', tone: 'muted' }

  const { cards, matches } = result.value
  if (cards.length < 2) return { text: cards.length ? 'Nur eine Karte erkannt' : 'Keine Karten erkannt', tone: 'muted' }
  if (!matches.length) return { text: 'Keine Symbole erkannt', tone: 'muted' }
  const confidence = Math.min(...matches.map((m) => m.confidence))
  const percent = Math.round(confidence * 100)
  return confidence < LOW_CONFIDENCE
    ? { text: `Unsicher (${percent} %) – Karten ruhig und flach halten`, tone: 'warn' }
    : { text: `Treffer! Sicherheit ${percent} %`, tone: 'ok' }
})

function grab(drawable: Drawable): ImageData {
  const { width, height } = sourceSize(drawable)
  const scale = Math.min(1, DETECT_SIZE / Math.max(width, height))
  grabCanvas.width = Math.round(width * scale)
  grabCanvas.height = Math.round(height * scale)
  grabContext.drawImage(drawable, 0, 0, grabCanvas.width, grabCanvas.height)
  return grabContext.getImageData(0, 0, grabCanvas.width, grabCanvas.height)
}

async function analyze(drawable: Drawable) {
  if (analyzing.value || !detector.ready.value || !sourceSize(drawable).width) return
  analyzing.value = true
  try {
    result.value = await detector.detect(grab(drawable))
  } catch (e) {
    detector.error.value = e instanceof Error ? e.message : String(e)
  } finally {
    analyzing.value = false
  }
}

function render() {
  frame = requestAnimationFrame(render)
  const el = canvas.value
  const drawable = source.value
  if (!el || !drawable) return

  const dpr = devicePixelRatio
  const width = Math.round(el.clientWidth * dpr)
  const height = Math.round(el.clientHeight * dpr)
  if (el.width !== width || el.height !== height) {
    el.width = width
    el.height = height
  }
  if (mode.value === 'camera' && !paused.value) analyze(drawable)
  drawScene(el.getContext('2d')!, drawable, result.value, showDetails.value)
}

async function startCamera() {
  if (!video.value) return
  still.value = null
  result.value = null
  mode.value = 'camera'
  if (!(await camera.start(video.value))) mode.value = 'idle'
}

async function togglePause() {
  if (!video.value) return
  if (still.value) {
    still.value = null
    return
  }
  still.value = await createImageBitmap(video.value)
  analyzing.value = false
  await analyze(still.value)
}

function notify(text: string) {
  notice.value = text
  setTimeout(() => notice.value === text && (notice.value = null), 2500)
}

// Draws synchronously so the frame matches the moment of the tap.
function frameBlob(drawable: Drawable): Promise<Blob> {
  const { width, height } = sourceSize(drawable)
  const out = document.createElement('canvas')
  out.width = width
  out.height = height
  out.getContext('2d')!.drawImage(drawable, 0, 0)
  return new Promise((resolve, reject) =>
    out.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Frame konnte nicht erzeugt werden'))), 'image/png'),
  )
}

async function copyFrame() {
  const drawable = source.value
  if (!drawable) return
  const blob = frameBlob(drawable)
  try {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
    notify('Frame kopiert')
    return
  } catch {
    // Clipboard images are not supported everywhere; fall back to share sheet or download.
  }

  const file = new File([await blob], `dobble-${new Date().toISOString().replace(/[:.]/g, '-')}.png`, { type: 'image/png' })
  try {
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file] })
      return
    }
  } catch {
    // Share was cancelled or blocked; download instead.
  }
  const link = document.createElement('a')
  link.href = URL.createObjectURL(file)
  link.download = file.name
  link.click()
  URL.revokeObjectURL(link.href)
  notify('Frame gespeichert')
}

function pickPhoto() {
  fileInput.value?.click()
}

async function onPhoto(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  camera.stop()
  result.value = null
  mode.value = 'photo'
  still.value = await createImageBitmap(file, { imageOrientation: 'from-image' })
  await analyze(still.value)
}

onMounted(render)
onBeforeUnmount(() => cancelAnimationFrame(frame))
</script>

<template>
  <main class="app">
    <header class="header">
      <h1>Dobble Matcher</h1>
      <div class="header-controls">
        <label class="toggle">
          <input v-model="showDetails" type="checkbox" />
          Details
        </label>
        <button class="small" :disabled="mode === 'idle'" title="Aktuellen Frame ohne Overlay kopieren" @click="copyFrame">
          Frame kopieren
        </button>
      </div>
    </header>

    <section class="stage">
      <canvas ref="canvas" class="canvas" :class="{ hidden: mode === 'idle' }" />
      <video ref="video" class="video" playsinline muted />

      <div v-if="mode === 'idle'" class="intro">
        <div class="logo" aria-hidden="true">
          <span class="disc disc-a" />
          <span class="disc disc-b" />
        </div>
        <p>Halte zwei Dobble-Karten in die Kamera oder lade ein Foto hoch – das gemeinsame Symbol wird markiert.</p>
        <div class="intro-actions">
          <button class="primary" :disabled="!detector.ready.value" @click="startCamera">Kamera starten</button>
          <button :disabled="!detector.ready.value" @click="pickPhoto">Foto wählen</button>
        </div>
      </div>
    </section>

    <footer class="footer">
      <div class="status" :class="status.tone">
        <span
          v-for="(match, i) in result?.matches ?? []"
          :key="i"
          class="swatch"
          :style="{ background: matchColor(i) }"
          :title="`Karte ${match.a.card + 1} ↔ Karte ${match.b.card + 1}`"
        />
        {{ status.text }}
        <small v-if="result && mode !== 'idle'">{{ Math.round(result.durationMs) }} ms</small>
      </div>
      <div v-if="mode !== 'idle'" class="actions">
        <button v-if="mode === 'camera'" class="primary" @click="togglePause">{{ paused ? 'Weiter' : 'Einfrieren' }}</button>
        <button v-else class="primary" @click="startCamera">Kamera</button>
        <button @click="pickPhoto">Foto</button>
      </div>
    </footer>

    <input ref="fileInput" type="file" accept="image/*" hidden @change="onPhoto" />
  </main>
</template>

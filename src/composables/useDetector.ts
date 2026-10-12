import { onBeforeUnmount, ref } from 'vue'
import type { DetectionResult, WorkerResponse } from '../vision/types'

interface Pending {
  resolve: (result: DetectionResult) => void
  reject: (error: Error) => void
}

export function useDetector() {
  const ready = ref(false)
  const busy = ref(false)
  const error = ref<string | null>(null)
  const pending = new Map<number, Pending>()
  let nextId = 0

  const worker = new Worker(new URL('../vision/worker.ts', import.meta.url), { type: 'module' })

  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (data.type === 'ready') {
      ready.value = true
      return
    }
    if (data.id === undefined) {
      error.value = data.type === 'error' ? data.message : null
      return
    }
    const request = pending.get(data.id)
    pending.delete(data.id)
    busy.value = pending.size > 0
    if (data.type === 'result') request?.resolve(data.result)
    else request?.reject(new Error(data.message))
  }

  worker.onerror = (event) => {
    error.value = event.message || 'OpenCV konnte nicht geladen werden'
  }

  function detect(image: ImageData): Promise<DetectionResult> {
    const id = nextId++
    busy.value = true
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject })
      worker.postMessage({ id, image: { data: image.data, width: image.width, height: image.height } }, [image.data.buffer])
    })
  }

  onBeforeUnmount(() => worker.terminate())

  return { ready, busy, error, detect }
}

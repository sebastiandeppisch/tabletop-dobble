import { detect } from './detector'
import { loadOpenCv } from './opencv'
import type { WorkerRequest, WorkerResponse } from './types'

const post = (message: WorkerResponse) => self.postMessage(message)

const ready = loadOpenCv().then((module) => {
  post({ type: 'ready' })
  return module
})

self.onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  const { cv } = await ready
  try {
    post({ type: 'result', id: data.id, result: detect(cv, data.image) })
  } catch (error) {
    post({ type: 'error', id: data.id, message: error instanceof Error ? error.message : String(error) })
  }
}

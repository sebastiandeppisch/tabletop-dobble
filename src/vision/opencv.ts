import cvModule from '@techstark/opencv-js'

export type OpenCv = typeof cvModule

// The module is a thenable, so it is wrapped to keep promise resolution from unwrapping it.
export async function loadOpenCv(): Promise<{ cv: OpenCv }> {
  const module = cvModule as OpenCv & { Mat?: unknown }
  if (module instanceof Promise) return { cv: await module }
  if (!module.Mat) await new Promise<void>((resolve) => (module.onRuntimeInitialized = resolve))
  return { cv: module }
}

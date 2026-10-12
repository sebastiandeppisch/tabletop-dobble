import { onBeforeUnmount, ref, shallowRef } from 'vue'

export function useCamera() {
  const stream = shallowRef<MediaStream | null>(null)
  const error = ref<string | null>(null)

  async function start(video: HTMLVideoElement): Promise<boolean> {
    error.value = null
    try {
      stream.value = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      })
      video.srcObject = stream.value
      await video.play()
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return false
    }
  }

  function stop() {
    stream.value?.getTracks().forEach((track) => track.stop())
    stream.value = null
  }

  onBeforeUnmount(stop)

  return { stream, error, start, stop }
}

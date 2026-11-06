/**
 * 从视频里取一帧（移植自上游 `lib/canvas/canvas-video-frame.ts`，MIT；自研重写 ✓）。
 *
 * 用途：画布上把一个视频卡的**首帧/末帧**取出来变成图片卡 ✓ ——
 * 这正是「首尾帧视频」工作流要的素材形态（`firstFrameImage` / `lastFrameImage`），
 * 比让用户自己截屏再上传顺手得多 ✓。
 *
 * ## 与源的两处差异
 *
 * 1. ⭐ **加超时** ✓：源只有 `loadedmetadata`/`seeked`/`error` 三个事件，
 *    服务端视频不响应时（无 CORS 头、编码不支持）会**一直等下去**，UI 上就是"点了没反应"✗。
 * 2. ⭐ 末尾 `video.removeAttribute('src') + load()` 保留 ✓（源同口径）：
 *    不这么清，Safari 会把解码器占住不释放，连开几个视频卡就卡顿 ✗。
 */

export type VideoFramePosition = 'first' | 'last' | 'current'

const DEFAULT_TIMEOUT_MS = 20_000

/**
 * 取帧 → PNG `Blob`（可直接 `uploadAPI.image` ✓）。
 *
 * @param source      视频 URL（同源 `/uploads/...` 最稳 ✓）
 * @param position    `first` / `last` / `current`
 * @param currentTime `current` 时用的秒数（`last` 会夹到 `duration - 0.001` ✓）
 */
export async function captureVideoFrame(
  source: string,
  position: VideoFramePosition = 'first',
  currentTime = 0,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Blob> {
  const video = document.createElement('video')
  video.crossOrigin = 'anonymous'
  video.muted = true
  video.playsInline = true
  video.preload = 'auto'

  const timer = { id: 0 as unknown as ReturnType<typeof setTimeout> }
  const timeout = new Promise<never>((_, reject) => {
    timer.id = setTimeout(() => reject(new Error('取帧超时（视频过大、编码不受支持或跨域被拒）')), timeoutMs)
  })

  try {
    const work = (async () => {
      const metadataLoaded = waitForVideo(video, 'loadedmetadata')
      video.src = source
      video.load()
      await metadataLoaded

      const endTime = Math.max(0, (Number.isFinite(video.duration) ? video.duration : 0) - 0.001)
      const time = position === 'first' ? 0 : position === 'last' ? endTime : Math.min(Math.max(0, currentTime), endTime)
      if (time > 0) {
        const seeked = waitForVideo(video, 'seeked')
        video.currentTime = time
        await seeked
      } else if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await waitForVideo(video, 'loadeddata')
      }

      if (!video.videoWidth || !video.videoHeight) throw new Error('视频没有可用画面（可能是音频文件）')

      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      if (!context) throw new Error('当前浏览器不支持 canvas 2d')
      context.drawImage(video, 0, 0)
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
      if (!blob) throw new Error('取帧失败（图片编码不出来）')
      return blob
    })()
    return await Promise.race([work, timeout])
  } finally {
    clearTimeout(timer.id)
    // ⚠️ 必须清解码器占用 ✓（不清会让后续视频卡变卡 ✗）
    video.removeAttribute('src')
    video.load()
  }
}

function waitForVideo(video: HTMLVideoElement, eventName: 'loadedmetadata' | 'loadeddata' | 'seeked'): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const finish = () => {
      video.removeEventListener(eventName, finish)
      video.removeEventListener('error', fail)
      resolve()
    }
    const fail = () => {
      video.removeEventListener(eventName, finish)
      video.removeEventListener('error', fail)
      reject(new Error('读取视频画面失败（跨域或编码不受支持）'))
    }
    video.addEventListener(eventName, finish)
    video.addEventListener('error', fail)
  })
}

/**
 * 素材文件的体积口径与上传前压缩（体积格式化移植自上游 `lib/image-utils.ts` 的 ``formatBytes``，MIT）。
 *
 * ## 为什么需要"上传前压缩"
 *
 * 后端 ``routers/upload.py`` 的图片上限是 **20MB**（照抄原 TS，不许改 ✗）。
 * 而手机/单反的一张 4K PNG 随手就 25–40MB ⇒ 用户拖进画布直接收到
 * 「File too large: 31.2MB. Max: 20MB」——**这是个死路**：他手上就这一张图 ✗。
 * 压缩是唯一能让这张图进得来的路 ✓，而且只在**超限时**发生（正常图零损耗 ✓）。
 *
 * ## 两个口径决定
 *
 * 1. **输出 WebP 而不是 JPEG** ✓：JPEG 没有 alpha ⇒ PNG 透明底会变成一块白 ✗；
 *    WebP 保透明且体积更小，后端白名单里本来就有 ``image/webp`` ✓（``ALLOWED_IMAGE_TYPES``）。
 * 2. **压缩失败就返回原文件** ✓：宁可让后端如实报「文件过大」，
 *    也不假装成功传一个坏图（静默降质比报错更难查 ✗）。
 */

/** 与后端 ``MAX_IMAGE_SIZE`` 一致 ✓（20MB）；这里只是**提示**用，真源在后端 */
export const MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024
/** 与后端 ``MAX_VIDEO_SIZE`` 一致 ✓ */
export const MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024

/** 压缩目标留 2MB 余量 ✓（multipart 边界/头部会让实际请求体比文件略大，卡死在上限上会被拒 ✗） */
const SAFE_TARGET_BYTES = 18 * 1024 * 1024
/** 压缩后长边上限 ✓（2560 够画布与后续生成用，再大只是白占带宽 ✓） */
const COMPRESS_MAX_EDGE = 2560
/** 逐档降质的质量表 ✓（先高质量试一次，不行再降两档 —— 一上来就 0.5 会糊得毫无必要 ✗） */
const QUALITY_STEPS = [0.92, 0.84, 0.76]

/**
 * 人类可读体积 ✓。
 *
 * ⚠️ 精度规则**刻意与源不同**：源是「≥10 就取整」（``12.5 MB`` 显示成 ``13 MB``），
 * 这里沿用本仓既有口径「≥100 才取整」⇒ ``12.5 MB`` 保住小数 ✓。
 * 源那条规则在本仓会让 models 页的体积**丢精度**（纯退化，没有收益 ✗）。
 * 空值/null 返回 ``''``（调用方自己决定要不要显示 ``-`` ✓，见 ``formatBytesOrDash``）。
 */
export function formatBytes(bytes?: number | null): string {
  if (!bytes || !Number.isFinite(bytes) || bytes <= 0) return ''
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let index = 0
  let value = bytes
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[index]}`
}

/** 列表/表格里用 ✓（没有体积就显示 ``-``，不留空单元格 ✗） */
export function formatBytesOrDash(bytes?: number | null): string {
  return formatBytes(bytes) || '-'
}

export interface CompressResult {
  file: File
  /** 是否真的压过 ✓（调用方据此提示用户，别让"传上去变小了"成为谜 ✗） */
  compressed: boolean
  originalBytes: number
  bytes: number
}

function loadImageElement(file: File): Promise<{ image: HTMLImageElement; revoke: () => void }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => resolve({ image, revoke: () => URL.revokeObjectURL(url) })
    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('图片解码失败（格式不受支持或文件损坏）'))
    }
    image.src = url
  })
}

/**
 * 超限图片压缩到可上传体积 ✓；不超限**原样返回** ✓（零损耗，别"顺手压一下"✗）。
 *
 * @param file        用户选的文件
 * @param targetBytes 目标体积（默认 18MB，比后端上限低 2MB ✓）
 */
export async function compressImageForUpload(
  file: File,
  targetBytes = SAFE_TARGET_BYTES,
): Promise<CompressResult> {
  const originalBytes = file.size
  const untouched: CompressResult = { file, compressed: false, originalBytes, bytes: originalBytes }
  if (originalBytes <= targetBytes) return untouched
  if (!file.type.startsWith('image/')) return untouched
  // GIF 压缩会丢动画 ✗ ⇒ 不碰（超限就让它如实报错，用户自己换图）
  if (file.type === 'image/gif') return untouched

  try {
    const { image, revoke } = await loadImageElement(file)
    try {
      for (const quality of QUALITY_STEPS) {
        const scale = Math.min(1, COMPRESS_MAX_EDGE / Math.max(image.naturalWidth, image.naturalHeight))
        const width = Math.max(1, Math.round(image.naturalWidth * scale))
        const height = Math.max(1, Math.round(image.naturalHeight * scale))
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d')
        if (!context) return untouched
        context.drawImage(image, 0, 0, width, height)
        const blob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob(resolve, 'image/webp', quality))
        if (!blob) continue
        if (blob.size <= targetBytes) {
          const name = file.name.replace(/\.[^.]+$/, '') || 'upload'
          return {
            file: new File([blob], `${name}.webp`, { type: 'image/webp' }),
            compressed: true,
            originalBytes,
            bytes: blob.size,
          }
        }
      }
    } finally {
      revoke()
    }
  } catch {
    // 解码不出来（HEIC 之类浏览器不支持）⇒ 交回原文件，让后端报它自己的错 ✓
    return untouched
  }
  // 压到最低质量仍超限 ⇒ 同样交回原文件（这时"报文件过大"是**正确结论** ✓）
  return untouched
}

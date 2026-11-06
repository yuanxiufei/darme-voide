/**
 * 画布节点尺寸：**默认规格**与**按素材比例自适应**。
 *
 * 移植自上游 `canvas-node-factory.ts` / `canvas-node-size.ts` / `constant/canvas.ts`
 * 的 `NODE_DEFAULT_SIZE`（MIT）；代码按本仓 `CanvasNode`（`w/h`）重写 ✓。
 *
 * ## 三处口径
 *
 * 1. ⭐ **默认尺寸是定值，不是"随便给个数"** ✓：文本/图片/配置都是 `340×240`，
 *    视频是 `420×236`（更宽，因为 16:9 是主流）、音频是 `420×180`（没有画面，
 *    但播放器控件条 + 地址行 + 按钮要放得下）、分组框 `760×480`。
 *    自己拍一个 `280×260` 会让**同一张图在画布上被压扁** ✗。
 * 2. ⭐ **新节点以"落点"为中心** ✓（`position = 落点 − 尺寸/2`）——
 *    以左上角对齐会让双击建卡时卡片**偏到右下**，手感是"点哪儿都不对" ✗。
 * 3. ⭐ **按比例适配是"装进方框"** ✓（`fitNodeSize`，最长边 ≤640）：
 *    竖图按宽定高、横图按高定宽；比例超出 `0.25–4` 的极端图**直接回默认**，
 *    否则一张 9:21 会缩成一根竖条、连标题都放不下 ✗。
 */
import type { CanvasNodeType } from '../composables/useCanvas'

/** 各类节点的默认尺寸（源 `NODE_DEFAULT_SIZE` ✓） */
export const DEFAULT_NODE_SIZE: Record<CanvasNodeType, { w: number; h: number }> = {
  text: { w: 340, h: 240 },
  image: { w: 340, h: 240 },
  video: { w: 420, h: 236 },
  audio: { w: 420, h: 180 },
  config: { w: 340, h: 240 },
  group: { w: 760, h: 480 },
}

/** 适配后的最长边上限 ✓（再大就把画布挤爆了 ✗） */
export const MEDIA_FIT_MAX = 640

/** 把素材真实像素**装进方框** ✓（只缩不放 ✓） */
export function fitNodeSize(width: number, height: number, maxWidth = MEDIA_FIT_MAX, maxHeight = MEDIA_FIT_MAX) {
  const w = Math.max(1, Number(width) || 1)
  const h = Math.max(1, Number(height) || 1)
  const scale = Math.min(1, maxWidth / w, maxHeight / h)
  return { w: Math.round(w * scale), h: Math.round(h * scale) }
}

/**
 * 按比例取节点尺寸 ✓：宽度对齐基准宽 / 高度对齐基准高，取更宽松的一边。
 * 比例超出 `0.25–4` 直接回默认 ✓（极端长条会把标题挤掉 ✗）。
 */
export function nodeSizeFromRatio(size: string, baseWidth: number, baseHeight: number) {
  const match = String(size || '').match(/^(\d+)(?:\s*[x:×]\s*)(\d+)/)
  if (!match) return null
  const width = Number(match[1])
  const height = Number(match[2])
  const ratio = width / Math.max(1, height)
  if (ratio < 0.25 || ratio > 4) return { w: baseWidth, h: baseHeight }
  return ratio >= baseWidth / baseHeight
    ? { w: baseWidth, h: Math.round(baseWidth / ratio) }
    : { w: Math.round(baseHeight * ratio), h: baseHeight }
}

<script setup lang="ts">
/**
 * 局部重绘（真蒙版）对话框（移植自上游 `canvas-node-mask-edit-dialog.tsx` 的交互语义，MIT；自研重写 ✓）。
 *
 * ## 为什么值得搬
 *
 * 本仓画布上的图此前只有「整张重来」和「四宫格切分」两种命运 ✗ ——
 * 而分镜图真正需要的常常是「只改脸/只改手/只改背景的某一块」✓。
 * 后端其实早已就位：`image-qwen21` 的 `inpaint` 变体（节点 3 `ImageToMask(red)` +
 * `InpaintCropImproved`/`StitchImproved`）**只缺一个人能画蒙版的地方** ✓。
 *
 * ## 三条口径
 *
 * 1. **刷的是原图像素坐标** ✓ 见 `utils/canvasMask.ts` 头注（显示坐标换窗口就漂 ✗）。
 * 2. **确认必须带提示词** ✓：inpaint 只给蒙版不给 prompt 时模型会"自由发挥"，
 *    结果是原区域被画成另一张脸 ✗（工作流节点 10 的 text 口是必填）。
 * 3. **一笔没刷不许确认** ✓：空蒙版跑下去等于整图重绘，白烧一次 20 秒装载 ✗。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  DEFAULT_BRUSH_SIZE,
  MAX_BRUSH_SIZE,
  MIN_BRUSH_SIZE,
  MASK_PREVIEW_COLOR,
  clampBrushSize,
  maskCanvas,
  maskCoverage,
  replayStrokes,
  toImagePoint,
  toPngBlob,
  type MaskPoint,
  type MaskStroke,
} from '../../utils/canvasMask'

const props = defineProps<{ src: string }>()
const emit = defineEmits<{
  (e: 'confirm', payload: { mask: Blob; prompt: string }): void
  (e: 'close'): void
}>()

const baseCanvas = ref<HTMLCanvasElement | null>(null)
const paintCanvas = ref<HTMLCanvasElement | null>(null)
const naturalSize = ref<{ width: number; height: number } | null>(null)
const strokes = ref<MaskStroke[]>([])
const mode = ref<'paint' | 'erase'>('paint')
const brush = ref(DEFAULT_BRUSH_SIZE)
const prompt = ref('')
const busy = ref(false)
const error = ref('')
const cursor = ref<{ x: number; y: number } | null>(null)

let drawing: MaskStroke | null = null

const filled = computed(() => strokes.value.length > 0)
const brushRadius = computed(() => clampBrushSize(brush.value) / 2)

/** 画笔在**显示尺寸**下的直径（滑杆是原图像素，缩放后得让人看见实际粗细 ✓） */
const brushOnScreen = computed(() => {
  const canvas = paintCanvas.value
  if (!canvas) return clampBrushSize(brush.value)
  const shown = canvas.getBoundingClientRect().width || canvas.width
  return Math.max(2, (clampBrushSize(brush.value) * shown) / Math.max(1, canvas.width))
})

function paintCtx() {
  return paintCanvas.value?.getContext('2d') ?? null
}

/** 底图只在加载完成时画一次 ✓（每帧重画 2048² 会让笔迹跟手明显变钝 ✗） */
function drawBase() {
  const canvas = baseCanvas.value
  const natural = naturalSize.value
  if (!canvas || !natural) return
  canvas.width = natural.width
  canvas.height = natural.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const image = new Image()
  image.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
  }
  image.src = props.src
  const paint = paintCanvas.value
  if (paint) {
    paint.width = natural.width
    paint.height = natural.height
  }
}

/** 撤销/清空走全量重放 ✓（笔迹数组很小，比存 ImageData 快照省 3 个数量级 ✓） */
function redrawPaint() {
  const canvas = paintCanvas.value
  const ctx = paintCtx()
  if (!canvas || !ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  replayStrokes(ctx, strokes.value, MASK_PREVIEW_COLOR, null)
}

watch(strokes, () => redrawPaint(), { deep: true })

function pointOf(ev: PointerEvent): MaskPoint | null {
  const canvas = paintCanvas.value
  const natural = naturalSize.value
  if (!canvas || !natural) return null
  const rect = canvas.getBoundingClientRect()
  return toImagePoint(
    { x: ev.clientX - rect.left, y: ev.clientY - rect.top },
    { width: rect.width, height: rect.height },
    natural,
  )
}

function startStroke(ev: PointerEvent) {
  const point = pointOf(ev)
  const ctx = paintCtx()
  if (!point || !ctx) return
  ev.preventDefault()
  const stroke: MaskStroke = { erase: mode.value === 'erase', size: clampBrushSize(brush.value), points: [point] }
  drawing = stroke
  strokes.value = [...strokes.value, stroke]
  // 落笔即画圆点：只 moveTo 不 lineTo 的单点笔迹在增量绘法下什么都不显示 ✗
  ctx.save()
  ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over'
  ctx.fillStyle = stroke.erase ? '#000' : MASK_PREVIEW_COLOR
  ctx.beginPath()
  ctx.arc(point.x, point.y, stroke.size / 2, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function moveStroke(ev: PointerEvent) {
  const point = pointOf(ev)
  const canvas = paintCanvas.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  cursor.value = { x: ev.clientX - rect.left, y: ev.clientY - rect.top }
  if (!drawing || !point) return
  const ctx = paintCtx()
  if (!ctx) return
  const previous = drawing.points[drawing.points.length - 1]
  drawing.points.push(point)
  ctx.save()
  ctx.globalCompositeOperation = drawing.erase ? 'destination-out' : 'source-over'
  ctx.strokeStyle = drawing.erase ? '#000' : MASK_PREVIEW_COLOR
  ctx.lineWidth = drawing.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(previous?.x ?? point.x, previous?.y ?? point.y)
  ctx.lineTo(point.x, point.y)
  ctx.stroke()
  ctx.restore()
}

function endStroke() {
  drawing = null
}

function undo() {
  strokes.value = strokes.value.slice(0, -1)
}

function clearAll() {
  strokes.value = []
}

async function confirm() {
  const natural = naturalSize.value
  if (!natural) return
  if (!prompt.value.trim()) {
    error.value = '请写要重绘成什么（只给蒙版不给提示词，模型会自由发挥 ✗）'
    return
  }
  if (!filled.value) {
    error.value = '先用画笔刷出要重绘的区域'
    return
  }
  busy.value = true
  error.value = ''
  try {
    const canvas = maskCanvas(natural, strokes.value)
    // 覆盖率在**导出后**算 ✓：涂得太少（<0.2%）几乎等于没刷，当场提醒而不是白跑一次
    const coverage = maskCoverage(canvas)
    if (coverage < 0.002) {
      error.value = '刷到的面积太小（不足 0.2%），建议放大画笔'
      busy.value = false
      return
    }
    const mask = await toPngBlob(canvas)
    emit('confirm', { mask, prompt: prompt.value.trim() })
  } catch (err: any) {
    error.value = `蒙版导出失败：${err?.message || err}`
  } finally {
    busy.value = false
  }
}

function onKey(ev: KeyboardEvent) {
  if (busy.value) return
  if (ev.key === 'Escape') emit('close')
  if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) void confirm()
  if ((ev.key === 'z' || ev.key === 'Z') && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault()
    undo()
  }
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="mask-overlay" @click.self="emit('close')">
    <div class="mask-panel">
      <div class="mask-tools">
        <button class="mask-tab" :class="{ active: mode === 'paint' }" @click="mode = 'paint'">画笔</button>
        <button class="mask-tab" :class="{ active: mode === 'erase' }" @click="mode = 'erase'">橡皮</button>
        <label class="mask-brush">
          粗细
          <input v-model.number="brush" type="range" :min="MIN_BRUSH_SIZE" :max="MAX_BRUSH_SIZE" step="1" />
          <span class="mask-num">{{ clampBrushSize(brush) }} px</span>
        </label>
        <button class="mask-mini" :disabled="!filled" @click="undo">撤销（Ctrl+Z）</button>
        <button class="mask-mini" :disabled="!filled" @click="clearAll">清空</button>
        <span class="mask-hint">蓝色 = 将被重绘</span>
      </div>

      <div class="mask-stage">
        <canvas ref="baseCanvas" class="mask-layer"></canvas>
        <canvas
          ref="paintCanvas"
          class="mask-layer mask-paint"
          @pointerdown="startStroke"
          @pointermove="moveStroke"
          @pointerup="endStroke"
          @pointerleave="endStroke"
          @pointerenter="moveStroke"
        ></canvas>
        <div
          v-if="cursor"
          class="mask-cursor"
          :style="{
            left: `${cursor.x}px`,
            top: `${cursor.y}px`,
            width: `${brushOnScreen}px`,
            height: `${brushOnScreen}px`,
          }"
        />
      </div>

      <div class="mask-prompt">
        <textarea
          v-model="prompt"
          rows="2"
          placeholder="要重绘成什么（例如：换成短发、加一条围巾、背景改成傍晚的街口）"
        ></textarea>
      </div>

      <div v-if="error" class="mask-error">{{ error }}</div>

      <div class="mask-actions">
        <span class="mask-size">
          {{ naturalSize ? `${naturalSize.width} × ${naturalSize.height}` : '加载中…' }}
          · 半径 {{ Math.round(brushRadius) }} px
        </span>
        <button class="mask-btn ghost" @click="emit('close')">取消（Esc）</button>
        <button class="mask-btn" :disabled="busy || !filled" @click="confirm">
          {{ busy ? '导出中…' : '建重绘卡（源图 + 蒙版）' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mask-overlay {
  position: fixed;
  inset: 0;
  z-index: 62;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.78);
}
.mask-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: min(1000px, 94vw);
  padding: 14px;
  border: 1px solid #2a2f3a;
  border-radius: 10px;
  background: #171a21;
  box-shadow: 0 18px 46px rgba(0, 0, 0, 0.5);
}
.mask-tools {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.mask-tab,
.mask-mini {
  padding: 3px 9px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #22262f;
  color: #d7dbe3;
  font-size: 12px;
  cursor: pointer;
}
.mask-tab.active {
  border-color: #2dd4bf;
  color: #2dd4bf;
}
.mask-mini:disabled {
  color: #58606e;
  cursor: default;
}
.mask-brush {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #8b93a2;
  font-size: 12px;
}
.mask-brush input {
  width: 120px;
}
.mask-num {
  min-width: 48px;
  color: #d7dbe3;
}
.mask-hint {
  margin-left: auto;
  color: #8b93a2;
  font-size: 12px;
}
.mask-stage {
  position: relative;
  display: inline-flex;
  align-self: center;
  touch-action: none;
  user-select: none;
}
.mask-layer {
  display: block;
  max-width: 100%;
  max-height: 60vh;
  border-radius: 6px;
  background: #0b0d12;
}
.mask-paint {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}
.mask-cursor {
  position: absolute;
  transform: translate(-50%, -50%);
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 50%;
  pointer-events: none;
}
.mask-prompt textarea {
  width: 100%;
  padding: 7px 9px;
  border: 1px solid #2f3541;
  border-radius: 6px;
  background: #0f1218;
  color: #e6e9ef;
  font-size: 12px;
  resize: vertical;
}
.mask-error {
  color: #f87171;
  font-size: 12px;
}
.mask-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mask-size {
  margin-right: auto;
  color: #8b93a2;
  font-size: 12px;
}
.mask-btn {
  padding: 6px 12px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #2dd4bf;
  color: #0b0d12;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.mask-btn:disabled {
  background: #22262f;
  color: #6b7280;
  cursor: default;
}
.mask-btn.ghost {
  background: #22262f;
  color: #d7dbe3;
  font-weight: 400;
}
</style>

<script setup lang="ts">
/**
 * 框选裁剪对话框（移植自上游 `canvas-node-crop-dialog.tsx` 的交互语义，MIT；自研重写 ✓）。
 *
 * ## 为什么值得搬
 *
 * 右键菜单里的「裁成方形」只能**居中切一刀** ✓ —— 而分镜图真正需要的常常是
 * 「把左下角那个人裁出来」「去掉右侧水印」这类**位置由人决定**的裁剪 ✗。
 * 这不是"多一个按钮"，是让画布上的图能当素材用 ✓。
 *
 * ## 三条口径
 *
 * 1. **矩形一律归一化 0–1** ✓：裁剪结果与卡片当前缩放/显示尺寸无关 ✓
 *    （用像素存会让同一张图在不同窗口大小下裁出不同区域 ✗）。
 * 2. **锁比例作用在拖动中** ✓：先拖再锁会把已选框弹到别处，手感极差 ✗。
 * 3. **太小的框不许确认** ✓：误点一下（2×3 像素）应当无效，而不是产出一张糊图 ✗。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { CropRect } from '../../utils/canvasImageOps'

const props = defineProps<{ src: string }>()
const emit = defineEmits<{
  (e: 'confirm', rect: CropRect): void
  (e: 'close'): void
}>()

/** 0 = 自由比例 ✓ */
const ratio = ref(0)
const box = ref<CropRect | null>(null)
const stage = ref<HTMLElement | null>(null)
const dragging = ref<{ x: number; y: number } | null>(null)
const naturalSize = ref<{ width: number; height: number } | null>(null)

const RATIOS = [
  { label: '自由', value: 0 },
  { label: '1:1', value: 1 },
  { label: '16:9', value: 16 / 9 },
  { label: '9:16', value: 9 / 16 },
]

function pointOf(ev: PointerEvent): { x: number; y: number } {
  const element = stage.value
  if (!element) return { x: 0, y: 0 }
  const rect = element.getBoundingClientRect()
  return {
    x: Math.min(1, Math.max(0, (ev.clientX - rect.left) / Math.max(1, rect.width))),
    y: Math.min(1, Math.max(0, (ev.clientY - rect.top) / Math.max(1, rect.height))),
  }
}

function startDrag(ev: PointerEvent) {
  ev.preventDefault()
  const point = pointOf(ev)
  dragging.value = point
  box.value = { x: point.x, y: point.y, width: 0, height: 0 }
}

function moveDrag(ev: PointerEvent) {
  const start = dragging.value
  if (!start) return
  const point = pointOf(ev)
  const next: CropRect = {
    x: Math.min(start.x, point.x),
    y: Math.min(start.y, point.y),
    width: Math.abs(point.x - start.x),
    height: Math.abs(point.y - start.y),
  }
  if (ratio.value > 0) {
    // 锁比例：以宽为准；右侧放不下就退成「按高定宽」✓（否则框会跑出画面 ✗）
    const height = next.width / ratio.value
    if (next.y + height <= 1) {
      next.height = height
    } else {
      next.height = 1 - next.y
      next.width = next.height * ratio.value
      if (next.x + next.width > 1) next.width = 1 - next.x
    }
  }
  box.value = next
}

function endDrag() {
  dragging.value = null
  // 松手时几乎没动 ⇒ 视为误点，清掉框（否则会留一个 0.001 的框等用户确认 ✗）
  const current = box.value
  if (current && (current.width < 0.01 || current.height < 0.01)) box.value = null
}

const boxStyle = computed(() => {
  const current = box.value
  if (!current) return { display: 'none' }
  return {
    left: `${current.x * 100}%`,
    top: `${current.y * 100}%`,
    width: `${current.width * 100}%`,
    height: `${current.height * 100}%`,
  }
})

/** 提示这次裁剪会产出多少像素 ✓（用户不用等上传后才知道尺寸） */
const sizeHint = computed(() => {
  const current = box.value
  const natural = naturalSize.value
  if (!current || !natural) return ''
  const width = Math.max(1, Math.round(current.width * natural.width))
  const height = Math.max(1, Math.round(current.height * natural.height))
  return `${width} × ${height}`
})

function confirm() {
  const current = box.value
  if (!current || current.width < 0.01 || current.height < 0.01) return
  emit('confirm', current)
}

function onKey(ev: KeyboardEvent) {
  if (ev.key === 'Escape') emit('close')
  if (ev.key === 'Enter') confirm()
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="crop-mask" @click.self="emit('close')">
    <div class="crop-panel">
      <div class="crop-tools">
        <button
          v-for="item in RATIOS"
          :key="item.label"
          class="crop-ratio"
          :class="{ active: ratio === item.value }"
          @click="ratio = item.value"
        >
          {{ item.label }}
        </button>
        <span class="crop-size">{{ sizeHint || '在图上拖出要保留的区域' }}</span>
      </div>

      <div
        ref="stage"
        class="crop-stage"
        @pointerdown="startDrag"
        @pointermove="moveDrag"
        @pointerup="endDrag"
        @pointerleave="endDrag"
      >
        <img
          :src="props.src"
          class="crop-image"
          alt=""
          draggable="false"
          @load="naturalSize = { width: ($event.target as HTMLImageElement).naturalWidth, height: ($event.target as HTMLImageElement).naturalHeight }"
        />
        <div v-if="box" class="crop-box" :style="boxStyle" />
      </div>

      <div class="crop-actions">
        <button class="crop-btn ghost" @click="emit('close')">取消（Esc）</button>
        <button class="crop-btn" :disabled="!box" @click="confirm">裁剪并带入画布</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.crop-mask {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.72);
}
.crop-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: min(920px, 92vw);
  padding: 14px;
  border: 1px solid #2a2f3a;
  border-radius: 10px;
  background: #171a21;
  box-shadow: 0 18px 46px rgba(0, 0, 0, 0.5);
}
.crop-tools {
  display: flex;
  align-items: center;
  gap: 8px;
}
.crop-ratio {
  padding: 3px 9px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #22262f;
  color: #d7dbe3;
  font-size: 12px;
  cursor: pointer;
}
.crop-ratio.active {
  border-color: #2dd4bf;
  color: #2dd4bf;
}
.crop-size {
  margin-left: auto;
  color: #8b93a2;
  font-size: 12px;
}
.crop-stage {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  touch-action: none;
  cursor: crosshair;
  user-select: none;
}
.crop-image {
  display: block;
  max-width: 100%;
  max-height: 68vh;
  border-radius: 6px;
  background: #0b0d12;
}
.crop-box {
  position: absolute;
  border: 1px solid #2dd4bf;
  background: rgba(45, 212, 191, 0.14);
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.45);
  pointer-events: none;
}
.crop-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.crop-btn {
  padding: 6px 12px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #2dd4bf;
  color: #0b0d12;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.crop-btn:disabled {
  background: #22262f;
  color: #6b7280;
  cursor: default;
}
.crop-btn.ghost {
  background: #22262f;
  color: #d7dbe3;
  font-weight: 400;
}
</style>

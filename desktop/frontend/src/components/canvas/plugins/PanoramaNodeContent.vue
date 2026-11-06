<script setup lang="ts">
/**
 * 全景节点（`panorama:viewer`）—— 等距柱状图贴成 **24 面 CSS 3D 圆柱**，横向拖拽转看 ✓。
 *
 * ## 为什么不是「拖一张长图左右平移」
 *
 * 那种做法看起来像全景，其实只是卷轴：左右两端不会接上，转到尽头就断 ✗。
 * 这里把图切成 24 段（每段 15°）贴到圆柱上，**首尾相接** —— 转 360° 回到原处 ✓。
 * 纯 CSS 实现（`rotateY + translateZ`），不引 WebGL、不加依赖。
 *
 * ## 如实交代边界
 *
 * 只做**横向** 360°。纵向（俯仰）不调 —— 真球面投影要顶点着色器，CSS 做不了；
 * 少一个自由度只是视野小，不会给出错画面 ✓。
 *
 * 几何都写成常数而不是"量出来的"：段宽 240px ⇒ 周长 24×240 ⇒ 半径
 * 24×240/(2π) ≈ 917px，观察者放在圆心（`perspective` = 半径）✓。
 */
import { computed, ref } from 'vue'
import { mediaUrl } from '~/utils/apiBase'

const props = defineProps<{ values: Record<string, string>; text: string; images?: string[] }>()
const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

/** 段数 24 ⇒ 每段 15°；段宽 240px ⇒ 半径 917px（`perspective` 必须等于半径，人就在圆心） */
const SEGMENTS = 24
const SEG_WIDTH = 240
const RADIUS = Math.round((SEGMENTS * SEG_WIDTH) / (2 * Math.PI))

const source = computed(() => props.values.url || props.images?.[0] || '')
const yaw = ref(0)
const dragging = ref(false)
let startX = 0
let startYaw = 0

/** 按第 `index` 段取整图的 1/24 横向切片（`bgWidth = 24×段宽` ⇒ 位移 = −i/(23) × 100%） */
function segmentStyle(index: number) {
  return {
    width: `${SEG_WIDTH}px`,
    transform: `rotateY(${(index * 360) / SEGMENTS}deg) translateZ(${RADIUS}px)`,
    backgroundImage: source.value ? `url("${mediaUrl(source.value)}")` : 'none',
    backgroundSize: `${SEGMENTS * 100}% 100%`,
    backgroundPositionX: `${(-index * 100) / (SEGMENTS - 1)}%`,
  }
}

function onDown(event: PointerEvent) {
  if (!source.value) return
  dragging.value = true
  startX = event.clientX
  startYaw = yaw.value
  ;(event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId)
}

function onMove(event: PointerEvent) {
  if (!dragging.value) return
  yaw.value = startYaw + (event.clientX - startX) * 0.35
}

function onUp() {
  dragging.value = false
}

function reset() {
  yaw.value = 0
}

function useUpstream() {
  const url = props.images?.[0]
  if (url) emit('patch', { url })
}
</script>

<template>
  <div class="pn-pano">
    <div
      v-if="source"
      class="pn-pano-stage"
      :class="{ 'is-dragging': dragging }"
      @pointerdown.stop="onDown"
      @pointermove="onMove"
      @pointerup="onUp"
      @pointercancel="onUp"
    >
      <div class="pn-pano-cyl" :style="{ transform: `rotateY(${yaw}deg)` }">
        <div v-for="index in SEGMENTS" :key="index" class="pn-pano-seg" :style="segmentStyle(index - 1)" />
      </div>
    </div>
    <div v-else class="pn-pano-empty">
      <p>拖一张等距柱状全景图到节点上，或让上游图片接到这里</p>
      <button v-if="props.images?.length" class="pn-pano-btn" @pointerdown.stop @click.stop="useUpstream">
        用上游那张
      </button>
    </div>
    <button v-if="source" class="pn-pano-reset" title="回到正面" @pointerdown.stop @click.stop="reset">
      正
    </button>
  </div>
</template>

<style scoped>
.pn-pano {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #0f1218;
}
.pn-pano-stage {
  width: 100%;
  height: 100%;
  /* 观察者站在圆柱圆心 ⇒ 透视距离 = 半径 ✓ */
  perspective: 917px;
  overflow: hidden;
  cursor: grab;
  touch-action: none;
}
.pn-pano-stage.is-dragging {
  cursor: grabbing;
}
.pn-pano-cyl {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 100%;
  transform-style: preserve-3d;
  transform-origin: center center;
}
.pn-pano-seg {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  /* 段宽 240 的一半：让每段的**中心**落在圆柱轴上（`translateZ` 以元素中心为准）✓ */
  margin-left: -120px;
  background-repeat: no-repeat;
  backface-visibility: hidden;
}
.pn-pano-empty {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 12px;
  font-size: 12px;
  color: #6b7280;
  text-align: center;
}
.pn-pano-empty p {
  margin: 0;
}
.pn-pano-btn,
.pn-pano-reset {
  padding: 3px 8px;
  font-size: 11px;
  color: #cbd5e1;
  cursor: pointer;
  background: #171c24;
  border: 1px solid #2a2f3a;
  border-radius: 4px;
}
.pn-pano-reset {
  position: absolute;
  top: 6px;
  right: 6px;
  opacity: 0;
  transition: opacity 0.15s;
}
.pn-pano:hover .pn-pano-reset {
  opacity: 1;
}
</style>

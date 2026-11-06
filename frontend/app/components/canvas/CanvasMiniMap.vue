<script setup lang="ts">
/**
 * 画布小地图：一眼看到「节点都在哪 + 我现在看的是哪一块」。
 *
 * 端口径来自上游 `canvas-mini-map.tsx` 的意图，实现按本仓 `CanvasNode` 重写：
 * 世界范围取**节点 ∪ 当前视口** ✓ —— 只算节点的话，画布空着时小地图会塌成一点，
 * 用户看不到自己在哪 ✗；只算视口则节点跑远后全部消失 ✗。
 */
import { computed } from 'vue'
import { nodeBounds } from '../../utils/canvasGeometry'
import type { CanvasNode, CanvasViewport } from '../../composables/useCanvas'

const props = withDefaults(
  defineProps<{
    nodes: CanvasNode[]
    viewport: CanvasViewport
    stage: { width: number; height: number }
    width?: number
  }>(),
  { width: 170 },
)

const emit = defineEmits<{ (e: 'jump', world: { x: number; y: number }): void }>()

const world = computed(() => {
  const s = props.viewport.scale || 1
  const view = {
    left: -props.viewport.x / s,
    top: -props.viewport.y / s,
    right: (-props.viewport.x + props.stage.width) / s,
    bottom: (-props.viewport.y + props.stage.height) / s,
  }
  if (!props.nodes.length) return view
  const b = nodeBounds(props.nodes)
  return {
    left: Math.min(b.left, view.left),
    top: Math.min(b.top, view.top),
    right: Math.max(b.right, view.right),
    bottom: Math.max(b.bottom, view.bottom),
  }
})

const scale = computed(() => {
  const w = Math.max(1, world.value.right - world.value.left)
  const h = Math.max(1, world.value.bottom - world.value.top)
  return Math.min(props.width / w, (props.width * 0.62) / h)
})

const panelHeight = computed(() => Math.max(64, Math.round((world.value.bottom - world.value.top) * scale.value)))

function box(x: number, y: number, w: number, h: number) {
  return {
    left: `${(x - world.value.left) * scale.value}px`,
    top: `${(y - world.value.top) * scale.value}px`,
    width: `${Math.max(2, w * scale.value)}px`,
    height: `${Math.max(2, h * scale.value)}px`,
  }
}

const viewBox = computed(() => {
  const s = props.viewport.scale || 1
  return box(-props.viewport.x / s, -props.viewport.y / s, props.stage.width / s, props.stage.height / s)
})

function onClick(ev: MouseEvent) {
  const rect = (ev.currentTarget as HTMLElement).getBoundingClientRect()
  emit('jump', {
    x: world.value.left + (ev.clientX - rect.left) / scale.value,
    y: world.value.top + (ev.clientY - rect.top) / scale.value,
  })
}
</script>

<template>
  <div class="mm" :style="{ width: `${width}px`, height: `${panelHeight}px` }" title="点击跳到该处" @mousedown.stop @click="onClick">
    <div v-for="n in nodes" :key="n.id" class="mm-node" :class="`is-${n.type}`" :style="box(n.x, n.y, n.w, n.h)" />
    <div class="mm-view" :style="viewBox" />
  </div>
</template>

<style scoped>
.mm {
  position: relative;
  background: rgba(15, 17, 21, 0.78);
  border: 1px solid #2a2f3a;
  border-radius: 6px;
  overflow: hidden;
  cursor: crosshair;
}
.mm-node {
  position: absolute;
  background: #3b4453;
  border-radius: 1px;
}
.mm-node.is-text {
  background: #4a5b7a;
}
.mm-node.is-image {
  background: #2f7d6f;
}
.mm-node.is-video {
  background: #7d5a2f;
}
.mm-node.is-config {
  background: #6b4a7a;
}
.mm-node.is-group {
  background: transparent;
  border: 1px dashed #4b5563;
}
.mm-view {
  position: absolute;
  border: 1px solid #2dd4bf;
  background: rgba(45, 212, 191, 0.1);
  pointer-events: none;
}
</style>

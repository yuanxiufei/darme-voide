<script setup lang="ts">
/**
 * 无限画布的渲染与交互层。
 *
 * 交互口径（与状态层 `useCanvas` 配套）：
 * - 滚轮 = **以光标为锚**缩放；中键 / 按住空格拖 = 平移；空白处左键拖 = 框选；
 * - 拖节点头 = 移动（多选时整组一起动）；拖节点右侧端口 = 连线；点连线 = 删除；
 * - Delete / Backspace = 删除选中（**在输入框里不触发** ✗ —— 否则改文字会删节点）。
 *
 * 世界层用 `transform: translate + scale` 一次变换 ✓，节点仍按世界坐标定位：
 * 这样命中测试只要「减平移、除缩放」一次，不必对每个节点单独换算 ✓。
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { findGroupDropTarget } from '../../utils/canvasGeometry'
import { collectReferenceInputs, resolvePromptText, type ReferenceInputs } from '../../utils/canvasReferences'
import { useCanvas } from '../../composables/useCanvas'
import type { GeneratedPayload, RunOverride } from '../../utils/canvasOps'
import CanvasNodeCard from './CanvasNodeCard.vue'

const canvas = useCanvas()
const { nodes, edges, viewport, selection } = canvas

const emit = defineEmits<{
  (e: 'contextmenu', payload: { x: number; y: number; nodeId: string | null }): void
  (e: 'error', message: string): void
  (e: 'notice', message: string): void
  (e: 'generated', payload: GeneratedPayload): void
}>()

const root = ref<HTMLElement | null>(null)
const spaceDown = ref(false)
const marquee = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const linkDraft = ref<{ from: string; x: number; y: number } | null>(null)

type Drag =
  | { kind: 'pan'; lastX: number; lastY: number }
  | { kind: 'move'; ids: string[]; lastX: number; lastY: number }
  | { kind: 'link'; from: string }
  | { kind: 'marquee'; startX: number; startY: number }
  | null

const drag = ref<Drag>(null)

/**
 * 每张卡的**上游素材**（连线 + 同组）✓ —— 生成时喂给工作流的参考图就是它 ✓。
 *
 * 一次算全表 ✓（模板里逐卡现算 = 节点数 × 连线数，拖拽时每帧都重算 ✗）。
 */
const refInputs = computed(() => {
  const map = new Map<string, ReferenceInputs>()
  for (const node of nodes.value) {
    map.set(node.id, collectReferenceInputs(node.id, nodes.value, edges.value))
  }
  return map
})

const worldStyle = computed(() => ({
  transform: `translate(${viewport.value.x}px, ${viewport.value.y}px) scale(${viewport.value.scale})`,
}))

/** 屏幕（相对容器）→ 世界 */
function toWorldFromEvent(e: PointerEvent | MouseEvent | WheelEvent) {
  const rect = root.value?.getBoundingClientRect()
  if (!rect) return { x: 0, y: 0 }
  return canvas.toWorld(e.clientX - rect.left, e.clientY - rect.top)
}

function nodeById(id: string) {
  return nodes.value.find((n) => n.id === id)
}

/** 节点连线锚点：出=右边中点，入=左边中点 */
function outPort(n: { x: number; y: number; w: number; h: number }) {
  return { x: n.x + n.w, y: n.y + n.h / 2 }
}
function inPort(n: { x: number; y: number; w: number; h: number }) {
  return { x: n.x, y: n.y + n.h / 2 }
}

const edgePaths = computed(() =>
  edges.value
    .map((e) => {
      const a = nodeById(e.from)
      const b = nodeById(e.to)
      if (!a || !b) return null
      const p1 = outPort(a)
      const p2 = inPort(b)
      const dx = Math.max(40, Math.abs(p2.x - p1.x) * 0.5)
      return { id: e.id, d: `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}` }
    })
    .filter(Boolean) as { id: string; d: string }[],
)

function hitNode(wx: number, wy: number) {
  // 后加的在上 ⇒ 从后往前找，命中上层 ✓
  for (let i = nodes.value.length - 1; i >= 0; i -= 1) {
    const n = nodes.value[i]
    if (wx >= n.x && wx <= n.x + n.w && wy >= n.y && wy <= n.y + n.h) return n
  }
  return null
}

function onBackgroundDown(e: PointerEvent) {
  if (e.button === 1 || spaceDown.value) {
    drag.value = { kind: 'pan', lastX: e.clientX, lastY: e.clientY }
  } else if (e.button === 0) {
    const w = toWorldFromEvent(e)
    drag.value = { kind: 'marquee', startX: w.x, startY: w.y }
    marquee.value = { x1: w.x, y1: w.y, x2: w.x, y2: w.y }
    if (!e.shiftKey) canvas.setSelection([])
  }
  bindDrag()
}

function onNodeMoveStart(id: string, e: PointerEvent) {
  const ids = selection.value.includes(id) ? selection.value : [id]
  if (!selection.value.includes(id)) canvas.setSelection(ids)
  drag.value = { kind: 'move', ids, lastX: e.clientX, lastY: e.clientY }
  bindDrag()
}

function onNodeLinkStart(id: string, e: PointerEvent) {
  const w = toWorldFromEvent(e)
  drag.value = { kind: 'link', from: id }
  linkDraft.value = { from: id, x: w.x, y: w.y }
  bindDrag()
}

function onMove(e: PointerEvent) {
  const d = drag.value
  if (!d) return
  if (d.kind === 'pan') {
    canvas.panBy(e.clientX - d.lastX, e.clientY - d.lastY)
    d.lastX = e.clientX
    d.lastY = e.clientY
  } else if (d.kind === 'move') {
    const dx = (e.clientX - d.lastX) / viewport.value.scale
    const dy = (e.clientY - d.lastY) / viewport.value.scale
    canvas.moveNodes(d.ids, dx, dy)
    d.lastX = e.clientX
    d.lastY = e.clientY
  } else if (d.kind === 'link') {
    const w = toWorldFromEvent(e)
    linkDraft.value = { from: d.from, x: w.x, y: w.y }
  } else if (d.kind === 'marquee') {
    const w = toWorldFromEvent(e)
    marquee.value = { x1: d.startX, y1: d.startY, x2: w.x, y2: w.y }
  }
}

function onUp(e: PointerEvent) {
  const d = drag.value
  if (d?.kind === 'link') {
    const w = toWorldFromEvent(e)
    const target = hitNode(w.x, w.y)
    if (target && target.id !== d.from) canvas.connect(d.from, target.id)
    linkDraft.value = null
  } else if (d?.kind === 'marquee' && marquee.value) {
    const m = marquee.value
    const x1 = Math.min(m.x1, m.x2)
    const x2 = Math.max(m.x1, m.x2)
    const y1 = Math.min(m.y1, m.y2)
    const y2 = Math.max(m.y1, m.y2)
    // 相交即选中（不要求完全包住 ✓ —— 框选要的是「扫到就选中」）
    const inside = nodes.value
      .filter((n) => n.x < x2 && n.x + n.w > x1 && n.y < y2 && n.y + n.h > y1)
      .map((n) => n.id)
    canvas.setSelection(inside)
    marquee.value = null
  } else if (d?.kind === 'move') {
    // 拖动结束：落进分组就收进去（**整体平移** ✓，成员相对位置不变）
    const group = findGroupDropTarget(d.ids, nodes.value)
    if (group) canvas.snapIntoGroup(d.ids, group)
  }
  drag.value = null
  unbindDrag()
}

function bindDrag() {
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', onUp)
}
function unbindDrag() {
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('pointerup', onUp)
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const rect = root.value?.getBoundingClientRect()
  if (!rect) return
  canvas.zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top)
}

function onKeyDown(e: KeyboardEvent) {
  if (e.key === ' ') spaceDown.value = true
  const el = e.target as HTMLElement | null
  const typing = el && /^(INPUT|TEXTAREA)$/.test(el.tagName)
  if ((e.key === 'Delete' || e.key === 'Backspace') && !typing && selection.value.length) {
    e.preventDefault()
    canvas.removeNodes(selection.value)
  }
}
function onKeyUp(e: KeyboardEvent) {
  if (e.key === ' ') spaceDown.value = false
}

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
  unbindDrag()
})

/** 右键：先把命中项纳入选择，再把**屏幕坐标**交出去（菜单项由页面决定 ✓） */
function onContextMenu(e: MouseEvent) {
  e.preventDefault()
  const w = toWorldFromEvent(e)
  const hit = hitNode(w.x, w.y)
  if (hit && !selection.value.includes(hit.id)) canvas.setSelection([hit.id])
  emit('contextmenu', { x: e.clientX, y: e.clientY, nodeId: hit?.id ?? null })
}

/** 双击空白：在该处加一个文本节点（画布最快的用法 ✓） */
function onDoubleClick(e: MouseEvent) {
  if ((e.target as HTMLElement)?.closest('.cn-card')) return
  canvas.addNode('text', toWorldFromEvent(e))
}

/*
 * 卡片实例登记册 —— 只为「对话面板代点生成」一件事存在 ✓。
 * 卡片自己握着档位/参数/引擎任务状态，页面拿不到；用回调 ref 收一份
 * 比把整套提交逻辑提升到页面（等于重写 runEngine）安全得多。
 */
const cardRefs = new Map<string, { run?: (override?: RunOverride) => unknown }>()

function setCardRef(id: string, el: unknown) {
  if (el) cardRefs.set(id, el as { run?: (override?: RunOverride) => unknown })
  else cardRefs.delete(id)
}

/**
 * op 上的 `prompt` 是**引用串**（`@[node:text-x]`，后端 `_generation_flow` 就这么回），
 * 到卡片手上必须已经是正文 —— 解析要整份节点表，所以收在这一层做 ✓，
 * 别指望每张卡自己去翻 nodes（卡只拿得到自己的上游 ✓）。
 */
function resolveOverride(override?: RunOverride): RunOverride | undefined {
  if (!override) return undefined
  const raw = (override.prompt || '').trim()
  const prompt = raw ? resolvePromptText(raw, nodes.value).text : ''
  return { mode: override.mode, prompt: prompt || undefined }
}

/**
 * 对话里「建卡 + 立刻生成」时，卡是**同一批 op 新建的**，此刻还没挂载
 * （Vue 渲染是异步的）⇒ 直接返回 false 会把这个主用例判成「没提交」✗。
 * 所以分两种：已挂载 ⇒ 当场跑并如实回报；还没挂载但节点确实存在 ⇒ 等挂载后跑。
 *
 * ⭐ 但「等」**不能拿 `true` 顶账** ✗✗：那样闸没过（档位类别不符、缺首帧…）时，
 * 面板已经写下「已提交生成 1」，而实际一条请求都没发 —— 用户看到的回执是假的。
 * 所以没挂载这条路**返回 promise**，跑完再给真答案（`applyCanvasOps` 会 await 它 ✓）。
 */
function runNode(id: string, override?: RunOverride): boolean | Promise<boolean> {
  const resolved = resolveOverride(override)
  const card = cardRefs.get(id)
  if (typeof card?.run === 'function') return verdictOf(card.run(resolved))
  if (!nodes.value.some((n) => n.id === id)) return false
  // 还没挂载：等它挂上再跑（override 要一起带过去，否则模型这轮写的提示词会丢 ✗）
  return scheduleRun(id, resolved)
}

/** 卡片自己也可能给 promise（档位清单还没到，它同样要等一次）⇒ 统一折成「真值或 promise」✓ */
function verdictOf(value: unknown): boolean | Promise<boolean> {
  const thenable = value as Promise<unknown> | null
  if (thenable && typeof thenable.then === 'function') return Promise.resolve(thenable).then((ok) => ok === true)
  return value === true
}

/** 一直没挂上（超过 8 次重试 ≈ 240ms）⇒ 如实返回 false，别谎报已提交 ✗ */
function scheduleRun(id: string, override?: RunOverride, tries = 8): Promise<boolean> {
  return nextTick().then(() => {
    const card = cardRefs.get(id)
    if (typeof card?.run === 'function') return Promise.resolve(verdictOf(card.run(override)))
    if (tries > 0) {
      return new Promise((resolve) => setTimeout(resolve, 30))
        .then(() => scheduleRun(id, override, tries - 1))
    }
    return false
  })
}

defineExpose({ runNode })
</script>

<template>
  <div
    ref="root"
    class="cb-root"
    :class="{ 'is-pan': spaceDown }"
    @pointerdown="onBackgroundDown"
    @wheel="onWheel"
    @dblclick="onDoubleClick"
    @contextmenu="onContextMenu"
  >
    <div class="cb-world" :style="worldStyle">
      <svg class="cb-edges">
        <path v-for="p in edgePaths" :key="p.id" :d="p.d" class="cb-edge" />
        <path v-for="p in edgePaths" :key="`hit-${p.id}`" :d="p.d" class="cb-edge-hit" @click.stop="canvas.removeEdge(p.id)" />
      </svg>

      <div
        v-if="marquee"
        class="cb-marquee"
        :style="{
          left: Math.min(marquee.x1, marquee.x2) + 'px',
          top: Math.min(marquee.y1, marquee.y2) + 'px',
          width: Math.abs(marquee.x2 - marquee.x1) + 'px',
          height: Math.abs(marquee.y2 - marquee.y1) + 'px',
        }"
      />

      <svg v-if="linkDraft" class="cb-edges">
        <path
          :d="`M ${(nodeById(linkDraft.from)?.x || 0) + (nodeById(linkDraft.from)?.w || 0)} ${(nodeById(linkDraft.from)?.y || 0) + (nodeById(linkDraft.from)?.h || 0) / 2} L ${linkDraft.x} ${linkDraft.y}`"
          class="cb-edge-draft"
        />
      </svg>

      <CanvasNodeCard
        v-for="n in nodes"
        :key="n.id"
        :ref="(el) => setCardRef(n.id, el)"
        :node="n"
        :selected="selection.includes(n.id)"
        @move-start="onNodeMoveStart(n.id, $event)"
        @link-start="onNodeLinkStart(n.id, $event)"
        @update="canvas.updateNode(n.id, $event)"
        @error="emit('error', $event)"
        @notice="emit('notice', $event)"
        :references="refInputs.get(n.id)"
        @generated="emit('generated', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
.cb-root {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background-color: #0f1115;
  background-image: radial-gradient(circle, #2a2f3a 1px, transparent 1px);
  background-size: 24px 24px;
  cursor: default;
  touch-action: none;
}
.cb-root.is-pan {
  cursor: grab;
}
.cb-world {
  position: absolute;
  top: 0;
  left: 0;
  transform-origin: 0 0;
  width: 0;
  height: 0;
}
.cb-edges {
  position: absolute;
  overflow: visible;
  width: 1px;
  height: 1px;
  pointer-events: none;
}
.cb-edge {
  fill: none;
  stroke: #4b5563;
  stroke-width: 2;
}
.cb-edge-hit {
  fill: none;
  stroke: transparent;
  stroke-width: 12;
  pointer-events: stroke;
  cursor: pointer;
}
.cb-edge-hit:hover + .cb-edge,
.cb-edge-hit:hover {
  stroke: #ef5350;
}
.cb-edge-draft {
  fill: none;
  stroke: #2dd4bf;
  stroke-width: 2;
  stroke-dasharray: 6 4;
}
.cb-marquee {
  position: absolute;
  border: 1px solid #2dd4bf;
  background: rgba(45, 212, 191, 0.12);
  pointer-events: none;
}
</style>

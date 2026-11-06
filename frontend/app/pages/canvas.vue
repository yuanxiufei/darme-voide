<script setup lang="ts">
/**
 * 无限画布页：文本 / 图片 / 视频 / 音频 / 配置 / 分组节点 + 连线 + 框选，本地自动保存。
 *
 * 与项目其它页面的分工：这里是**创作草稿台**（摆素材、写提示词、连关系），
 * 不是生成链路本身 ✓ —— 生成仍走 图片 / 视频 / 分镜 那些页面，画布只负责把思路摆清楚。
 *
 * 画布内容存在**浏览器本地**（localStorage），不进数据库：
 * 草稿是个人即时的东西，落库要带用户/剧集维度，那是另一条设计 ✓（先不混进来 ✗）。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { Download, Maximize2, Plus, Sparkles, Trash2, Upload } from 'lucide-vue-next'
import CanvasBoard from '../components/canvas/CanvasBoard.vue'
import CanvasContextMenu from '../components/canvas/CanvasContextMenu.vue'
import CanvasImageCompare from '../components/canvas/CanvasImageCompare.vue'
import CanvasMaskDialog from '../components/canvas/CanvasMaskDialog.vue'
import CanvasMiniMap from '../components/canvas/CanvasMiniMap.vue'
import CanvasOutpaintDialog from '../components/canvas/CanvasOutpaintDialog.vue'
import CanvasZoomControls from '../components/canvas/CanvasZoomControls.vue'
import {
  NODE_TYPE_LABEL,
  useCanvas,
  type CanvasAnyNodeType,
  type CanvasNode,
  type CanvasNodeType,
} from '../composables/useCanvas'
import { PLUGIN_NODES } from '../utils/canvasPlugins'
import { canGroupSelectedNodes, canUngroupSelectedNodes } from '../utils/canvasGeometry'
import { cropImage, rotateImage, splitImage, upscaleImage } from '../utils/canvasImageOps'
import type { Paddings } from '../utils/canvasOutpaint'
import { captureVideoFrame } from '../utils/videoFrame'
import { describeReferences } from '../utils/canvasReferences'
import { runCanvasAgent } from '../utils/canvasAgent'
import { uploadAPI } from '../composables/useApi'
/**
 * 节点 `url` 是**后端给的根相对路径**（`/static/...`）。凡是要喂给 `<img>/<video>` 或被
 * canvas 加载的（裁剪/切分/取帧、对比视图、重绘蒙版底图）都必须过 `mediaUrl` ✓；
 * **送后端**当参考图的保持根相对 ✗（后端只认自己的路径，绝对地址会多一层解析）。
 * 桌面壳 origin 是 `tauri.localhost` ⇒ 漏掉就整片 404，而 dev 被 Vite proxy 掩盖。
 */
import { mediaUrl } from '../utils/apiBase'

const {
  nodes,
  edges,
  viewport,
  selection,
  addNode,
  updateNode,
  removeNodes,
  connect,
  resetView,
  clearAll,
  exportDoc,
  importDoc,
  toWorld,
  zoomAt,
  groupSelected,
  ungroupSelected,
  focusNode,
} = useCanvas()

const fileInput = ref<HTMLInputElement | null>(null)
const stageEl = ref<HTMLElement | null>(null)
const stageSize = ref({ width: 900, height: 600 })
const message = ref('')
const menu = ref<{ x: number; y: number; nodeId: string | null } | null>(null)
const comparePair = ref<{ left?: string; right?: string } | null>(null)
/** 局部重绘 / 扩图的来源卡（对话框关掉即清空 ✓，避免"改了图还按老图算" ✗） */
const maskTarget = ref<{ nodeId: string; src: string } | null>(null)
const outpaintTarget = ref<{ nodeId: string; src: string } | null>(null)

const TYPES: CanvasNodeType[] = ['text', 'image', 'video', 'audio', 'config', 'group']

/** 新节点落在给定世界坐标（默认当前视口中心 ✓ —— 缩放/平移后原点常常看不见） */
function add(type: CanvasAnyNodeType, world?: { x: number; y: number }) {
  // ⚠️ 这里**不许再减偏移**：`addNode` 已经按「落点为中心」摆位，
  // 页面再减一次 (140,100) 会让卡片永远偏到左上、点哪儿都不对 ✗
  const point = world ?? toWorld(stageSize.value.width / 2, stageSize.value.height / 2)
  addNode(type, point)
}

// ---- 画布 AI ----
// 三段分工见 `utils/canvasAgent.ts` 抬头。这里只做编排与如实交代：
// 模型回话 ≠ 画布真做到的，所以「落地结果」必须单独算出来给用户看 ✓
const aiText = ref('')
const aiBusy = ref(false)
const aiNote = ref('')

async function askAI() {
  if (aiBusy.value || !aiText.value.trim()) return
  aiBusy.value = true
  aiNote.value = ''
  try {
    const outcome = await runCanvasAgent(aiText.value)
    const parts: string[] = []
    if (outcome.applied) parts.push(`已落 ${outcome.applied} 条`)
    if (outcome.pendingRun.length) {
      // `run_generation` 没有回调可调 —— 画布把开跑留在卡片上，如实告诉用户点哪 ✓
      parts.push(`${outcome.pendingRun.length} 张配置卡已选中，点它上面的「运行」即开跑`)
    }
    if (outcome.errors.length) parts.push(`未编译：${outcome.errors[0]}`)
    if (outcome.rejected.length) parts.push(`未落地：${outcome.rejected[0]}`)
    aiNote.value = outcome.text || parts.join('；')
    if (parts.length) flash(parts.join('；'))
    aiText.value = ''
  } catch (error: any) {
    aiNote.value = ''
    flash(`AI 失败：${error?.message || error}`)
  } finally {
    aiBusy.value = false
  }
}

function flash(text: string) {
  message.value = text
  // 引用摘要这类长句 1.8s 根本读不完 ✓（按字数给时间，但别超过 6s 留着手感 ✗）
  const ttl = Math.min(6000, Math.max(1800, text.length * 90))
  setTimeout(() => {
    message.value = ''
  }, ttl)
}

/** 菜单是用 **clientX/clientY** 定位的（fixed）⇒ 换算世界坐标要先减画布矩形 ✓ */
function worldAtMenu() {
  const rect = stageEl.value?.getBoundingClientRect()
  if (!menu.value || !rect) return undefined
  return toWorld(menu.value.x - rect.left, menu.value.y - rect.top)
}

const selectedImageNodes = computed(() =>
  nodes.value.filter((node) => selection.value.includes(node.id) && node.type === 'image' && node.url),
)
const imageSelection = computed(() => selectedImageNodes.value.map((node) => node.url as string))
const videoSelection = computed(() =>
  nodes.value.filter((node) => selection.value.includes(node.id) && node.type === 'video' && node.url),
)

const menuItems = computed(() => [
  { key: 'text', label: '加文本卡' },
  { key: 'image', label: '加图片卡' },
  { key: 'video', label: '加视频卡' },
  { key: 'audio', label: '加音频卡' },
  { key: 'config', label: '加配置卡' },
  { key: 'group-frame', label: '加分组框' },
  { key: 'img-rotate', label: '图片旋转 90°', disabled: !imageSelection.value.length },
  { key: 'img-square', label: '裁成方形（居中）', disabled: !imageSelection.value.length },
  { key: 'img-2k', label: '放大到 2K（长边 2048）', disabled: !imageSelection.value.length },
  { key: 'img-split', label: '切分 2×2（分镜切图）', disabled: imageSelection.value.length !== 1 },
  { key: 'img-inpaint', label: '局部重绘（画蒙版）', disabled: imageSelection.value.length !== 1 },
  { key: 'img-outpaint', label: '扩图（补四边画布）', disabled: imageSelection.value.length !== 1 },
  { key: 'vid-first', label: '取首帧为图片卡', disabled: videoSelection.value.length !== 1 },
  { key: 'vid-last', label: '取末帧为图片卡', disabled: videoSelection.value.length !== 1 },
  { key: 'refs', label: '查看引用（连线 / 同组）', disabled: selection.value.length !== 1 },
  { key: 'group', label: '成组（选 ≥2 个）', disabled: !canGroupSelectedNodes(selection.value, nodes.value) },
  { key: 'ungroup', label: '解组', disabled: !canUngroupSelectedNodes(selection.value, nodes.value) },
  { key: 'focus', label: '聚焦到该节点', disabled: selection.value.length !== 1 },
  { key: 'compare', label: 'A/B 对比（选 2 张图）', disabled: imageSelection.value.length !== 2 },
  { key: 'delete', label: '删除选中', danger: true, disabled: !selection.value.length },
  { key: 'reset', label: '复位视图' },
])

/** 画布只存 URL ✓：算出来的图先上传拿 `/uploads/xxx.png` 再落卡（dataURL 会撑爆 localStorage ✗） */
async function uploadImageBlob(blob: Blob, name: string) {
  const file = new File([blob], name, { type: blob.type || 'image/png' })
  const { url } = await uploadAPI.image(file)
  return url
}

/** 在原卡右侧放一张新图片卡 ✓（叠在原图上等于"点了没反应" ✗） */
function addImageNear(node: CanvasNode | undefined, url: string) {
  const anchor = node
    ? { x: node.x + node.w + 40, y: node.y }
    : worldAtMenu() ?? toWorld(stageSize.value.width / 2, stageSize.value.height / 2)
  const id = addNode('image', anchor)
  updateNode(id, { url })
  return id
}

/** 图片算子：旋转 / 裁方形 / 放大 / 切分（结果一律**新开卡片** ✓，不覆盖原图 ✗） */
async function runImageOp(key: 'img-rotate' | 'img-square' | 'img-2k' | 'img-split') {
  const targets = selectedImageNodes.value
  if (!targets.length) return
  const done: string[] = []
  try {
    for (const node of targets) {
      //: 这几个算子都在**浏览器**里跑（canvas 重采样）⇒ 要能真加载到像素 ⇒ 过 mediaUrl ✓
      const source = mediaUrl(node.url as string)
      if (key === 'img-split') {
        const pieces = await splitImage(source, { rows: 2, columns: 2 })
        for (const piece of pieces) {
          const url = await uploadImageBlob(piece.blob, `split-${piece.row}-${piece.column}.png`)
          // 摆到原卡下方，按行列成网格 ✓（散开会让用户分不清哪块是哪个位置 ✗）
          const id = addNode('image', {
            x: node.x + piece.column * 320,
            y: node.y + node.h + 24 + piece.row * 240,
          })
          updateNode(id, { url })
        }
        done.push(`${pieces.length} 块`)
        continue
      }
      const result = key === 'img-rotate'
        ? await rotateImage(source, 90)
        : key === 'img-square'
          ? await cropImage(source)
          : await upscaleImage(source, 2048)
      addImageNear(node, await uploadImageBlob(result.blob, `${key}.png`))
      done.push(`${node.title} ${result.width}×${result.height}`)
    }
    flash(`已生成：${done.join('，')}`)
  } catch (err: any) {
    flash(`处理失败：${err?.message || err}`)
  }
}

/** 视频卡取帧 → 图片卡（首尾帧视频工作流要的素材就是这么来的 ✓） */
async function takeVideoFrame(position: 'first' | 'last') {
  const node = videoSelection.value[0]
  if (!node?.url) return
  try {
    //: 取帧是浏览器里建 `<video>` 抽 canvas ⇒ 同样要能真加载 ✓
    const blob = await captureVideoFrame(mediaUrl(node.url), position)
    addImageNear(node, await uploadImageBlob(blob, `frame-${position}.png`))
    flash(position === 'first' ? '已取首帧为图片卡' : '已取末帧为图片卡')
  } catch (err: any) {
    flash(`取帧失败：${err?.message || err}`)
  }
}

/**
 * 局部重绘 / 扩图：**对话框不自己发请求** ✓
 *
 * 它只回答"人怎么画蒙版 / 要补多少"，落成一张**真会跑的配置卡**（源图与蒙版按顺序连线 ⇒
 * bindings 的 `imageIndex` 0/1 自动正确 ✓），提交仍走卡片上那条唯一通道 ——
 * 进度、产物回卡、取消都是现成的 ✓（另开一条提交路径必然和它漂开 ✗）。
 */
function openInpaint() {
  const node = selectedImageNodes.value[0]
  if (!node?.url) return
  maskTarget.value = { nodeId: node.id, src: mediaUrl(node.url) }
}

function openOutpaint() {
  const node = selectedImageNodes.value[0]
  if (!node?.url) return
  outpaintTarget.value = { nodeId: node.id, src: mediaUrl(node.url) }
}

/** 建一张 comfyui 配置卡：工作流 / 变体 / 提示词全部预置好 ✓ */
function spawnConfigCard(
  source: CanvasNode,
  spec: { title: string; variant: string; prompt: string; values?: Record<string, string> },
) {
  const id = addNode('config', { x: source.x + source.w + 60, y: source.y })
  updateNode(id, {
    title: spec.title,
    runMode: 'comfyui',
    workflowId: 'image-qwen21',
    variant: spec.variant,
    text: spec.prompt,
    values: spec.values,
  })
  return id
}

async function confirmInpaint(payload: { mask: Blob; prompt: string }) {
  const target = maskTarget.value
  maskTarget.value = null
  const source = nodes.value.find((node) => node.id === target?.nodeId)
  if (!source) return
  try {
    const maskUrl = await uploadImageBlob(payload.mask, 'mask.png')
    // 蒙版卡摆在源图正下方 ✓（一眼看出"白的那块就是我要改的地方"）
    const maskId = addNode('image', { x: source.x, y: source.y + source.h + 32 })
    updateNode(maskId, { url: maskUrl, title: '蒙版（白 = 重绘）' })
    const configId = spawnConfigCard(source, { title: '局部重绘', variant: 'inpaint', prompt: payload.prompt })
    // ⚠️ 连线顺序 = `imageIndex` 顺序：源图必须**先**连（bindings 里 source=0、mask=1）✓
    connect(source.id, configId)
    connect(maskId, configId)
    flash('已建重绘卡：源图在前、蒙版在后，参数确认后点「生成」')
  } catch (err: any) {
    flash(`重绘建单失败：${err?.message || err}`)
  }
}

function confirmOutpaint(payload: {
  paddings: Paddings
  width: number
  height: number
  feathering: number
  megapixels: number
  prompt: string
}) {
  const target = outpaintTarget.value
  outpaintTarget.value = null
  const source = nodes.value.find((node) => node.id === target?.nodeId)
  if (!source) return
  const configId = spawnConfigCard(source, {
    title: `扩图 ${payload.width}×${payload.height}`,
    variant: 'outpaint',
    prompt: payload.prompt,
    // 这八格没在注册表参数表里露面 ⇒ 卡片表单上不显示，
    // 但 `buildPayload` 会把 values 原样带下去 ✓（数值在对话框里已经算准，见 utils/canvasOutpaint.ts）
    values: {
      left: String(payload.paddings.left),
      top: String(payload.paddings.top),
      right: String(payload.paddings.right),
      bottom: String(payload.paddings.bottom),
      feathering: String(payload.feathering),
      megapixels: String(payload.megapixels),
      outpaintWidth: String(payload.width),
      outpaintHeight: String(payload.height),
    },
  })
  connect(source.id, configId)
  flash(`已建扩图卡 ${payload.width}×${payload.height}，确认参数后点「生成」`)
}

/**
 * 生成完成 ⇒ **新开一张卡**带进画布 ✓。
 *
 * 不覆盖原配置卡：那张卡上还挂着工作流与参数，是"下次再跑"的入口 ✓；
 * 也不覆盖选中的参考图（用户拿它当输入，覆盖掉就找不回来了 ✗）。
 */
function onGenerated(payload: { kind: 'image' | 'video' | 'audio'; url: string; runId: number | null }) {
  const center = toWorld(stageSize.value.width / 2, stageSize.value.height / 2)
  const id = addNode(payload.kind, { x: center.x + 120, y: center.y - 80 })
  updateNode(id, {
    url: payload.url,
    title: payload.kind === 'video' ? '生成视频' : payload.kind === 'audio' ? '生成音频' : '生成图片',
  })
  flash(`已生成并带入画布${payload.runId ? `（run #${payload.runId}）` : ''}`)
}

function showReferences() {
  const id = selection.value[0]
  if (!id) return
  const node = nodes.value.find((item) => item.id === id)
  const lines = describeReferences(id, nodes.value, edges.value)
  flash(`${node?.title || '该卡'}：${lines.join('；')}`)
}

function onMenuSelect(key: string) {
  const at = worldAtMenu()
  if (key === 'text' || key === 'image' || key === 'video' || key === 'audio'
      || key === 'config' || key === 'group-frame') {
    add((key === 'group-frame' ? 'group' : key) as CanvasNodeType, at)
  } else if (key === 'img-rotate' || key === 'img-square' || key === 'img-2k' || key === 'img-split') {
    void runImageOp(key)
  } else if (key === 'img-inpaint') {
    openInpaint()
  } else if (key === 'img-outpaint') {
    openOutpaint()
  } else if (key === 'vid-first' || key === 'vid-last') {
    void takeVideoFrame(key === 'vid-first' ? 'first' : 'last')
  } else if (key === 'refs') {
    showReferences()
  } else if (key === 'group') {
    groupSelected()
  } else if (key === 'ungroup') {
    ungroupSelected()
  } else if (key === 'focus') {
    const id = selection.value[0] ?? menu.value?.nodeId
    if (id) focusNode(id, stageSize.value)
  } else if (key === 'compare') {
    //: 对比视图是两张 `<img>` 并排 ⇒ 同样要能真加载 ✓（`undefined` 保持原语义：那一侧留空）
    const [leftUrl, rightUrl] = imageSelection.value
    comparePair.value = {
      left: leftUrl ? mediaUrl(leftUrl) : undefined,
      right: rightUrl ? mediaUrl(rightUrl) : undefined,
    }
  } else if (key === 'delete') {
    removeNodes(selection.value)
  } else if (key === 'reset') {
    resetView()
  }
  menu.value = null
}

/** 点小地图跳转：把该世界点摆到画布中心 ✓ */
function jumpTo(world: { x: number; y: number }) {
  viewport.value = {
    scale: viewport.value.scale,
    x: stageSize.value.width / 2 - world.x * viewport.value.scale,
    y: stageSize.value.height / 2 - world.y * viewport.value.scale,
  }
}

/** 按钮缩放以**画布中心**为锚 ✓（用左上角会越点越偏 ✗） */
function zoomBy(factor: number) {
  zoomAt(factor, stageSize.value.width / 2, stageSize.value.height / 2)
}

function doExport() {
  const blob = new Blob([exportDoc()], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `canvas-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
  flash('已导出画布 JSON')
}

async function onFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    importDoc(await file.text())
    flash('已导入画布')
  } catch (err: any) {
    flash(`导入失败：${err?.message || err}`)
  } finally {
    input.value = ''
  }
}

let observer: ResizeObserver | null = null
onMounted(() => {
  const el = stageEl.value
  if (!el) return
  const apply = () => {
    stageSize.value = { width: el.clientWidth, height: el.clientHeight }
  }
  apply()
  observer = new ResizeObserver(apply)
  observer.observe(el)
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <div class="cv-page">
    <div class="cv-bar">
      <div class="cv-add">
        <span class="cv-label">添加</span>
        <button v-for="t in TYPES" :key="t" class="btn btn-ghost btn-sm" @click="add(t)">
          <Plus :size="14" />{{ NODE_TYPE_LABEL[t] }}
        </button>
        <!-- 插件节点 ✓（清单来自 `utils/canvasPlugins.ts`）：不加这一排，插件就只能靠导入画布才出现 ✗ -->
        <button
          v-for="p in PLUGIN_NODES"
          :key="p.type"
          class="btn btn-ghost btn-sm"
          :title="`${p.title}（${p.type}）`"
          @click="add(p.type)"
        >
          <span class="cv-add-icon">{{ p.icon }}</span>{{ p.title }}
        </button>
        <button class="btn btn-ghost btn-sm" :disabled="!canGroupSelectedNodes(selection, nodes)" @click="groupSelected()">成组</button>
        <button class="btn btn-ghost btn-sm" :disabled="!canUngroupSelectedNodes(selection, nodes)" @click="ungroupSelected()">解组</button>
      </div>

      <div class="cv-ai">
        <input
          v-model="aiText"
          class="cv-ai-input"
          placeholder="让 AI 摆画布：在这张图右边生成一张类似的…"
          :disabled="aiBusy"
          @keydown.enter="askAI"
        />
        <button class="btn btn-ghost btn-sm" :disabled="aiBusy || !aiText.trim()" @click="askAI">
          <Sparkles :size="14" />{{ aiBusy ? '思考中…' : 'AI' }}
        </button>
        <span v-if="aiNote" class="cv-ai-note">{{ aiNote }}</span>
      </div>

      <div class="cv-right">
        <span v-if="message" class="cv-msg">{{ message }}</span>
        <span class="cv-count">{{ nodes.length }} 节点 · {{ edges.length }} 连线</span>
        <button class="btn btn-ghost btn-sm" :disabled="!selection.length" @click="removeNodes(selection)">
          <Trash2 :size="14" />删除选中
        </button>
        <button class="btn btn-ghost btn-sm" @click="resetView()"><Maximize2 :size="14" />复位</button>
        <button class="btn btn-ghost btn-sm" @click="doExport"><Download :size="14" />导出</button>
        <button class="btn btn-ghost btn-sm" @click="fileInput?.click()"><Upload :size="14" />导入</button>
        <button class="btn btn-ghost btn-sm" @click="clearAll()">清空</button>
        <input ref="fileInput" type="file" accept="application/json" class="cv-file" @change="onFile" />
      </div>
    </div>

    <div ref="stageEl" class="cv-stage">
      <CanvasBoard
        @contextmenu="menu = $event"
        @error="flash"
        @notice="flash"
        @generated="onGenerated"
      />
      <p class="cv-hint">
        滚轮缩放 · 空格或中键拖动平移 · 空白处拖拽框选 · 拖卡片头部移动（拖进分组会收纳）· 拖右侧圆点连线 · 点连线删除 · 右键打开菜单 · 双击空白建文本卡
      </p>
      <div class="cv-overlay-map">
        <CanvasMiniMap :nodes="nodes" :viewport="viewport" :stage="stageSize" @jump="jumpTo" />
      </div>
      <div class="cv-overlay-zoom">
        <CanvasZoomControls :scale="viewport.scale" @zoom="zoomBy" @reset="resetView" />
      </div>
    </div>

    <CanvasContextMenu v-if="menu" :x="menu.x" :y="menu.y" :items="menuItems" @select="onMenuSelect" @close="menu = null" />
    <CanvasImageCompare v-if="comparePair" :left="comparePair.left" :right="comparePair.right" @close="comparePair = null" />
    <CanvasMaskDialog v-if="maskTarget" :src="maskTarget.src" @confirm="confirmInpaint" @close="maskTarget = null" />
    <CanvasOutpaintDialog
      v-if="outpaintTarget"
      :src="outpaintTarget.src"
      @confirm="confirmOutpaint"
      @close="outpaintTarget = null"
    />
  </div>
</template>

<style scoped>
.cv-page {
  display: flex;
  flex-direction: column;
  /* 父级 .content 已是 flex:1 的 flex 列 ⇒ 这里跟着 flex 即可，别再硬算 100vh − 顶栏 ✗
     （顶栏高度一改就错位）；min-height:0 是让画布区能真的收缩 ✓ */
  flex: 1;
  min-height: 0;
  gap: 10px;
  padding: 12px;
}
.cv-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 8px 12px;
  background: #171a21;
  border: 1px solid #2a2f3a;
  border-radius: 8px;
}
.cv-add,
.cv-right {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.cv-ai {
  display: flex;
  align-items: center;
  gap: 6px;
  /* 吃掉中间的剩余宽度，但留得住最小可用宽度；两侧块换行时它独占一行 ✓ */
  flex: 1 1 260px;
  min-width: 200px;
}
.cv-ai-input {
  flex: 1;
  min-width: 120px;
  padding: 5px 9px;
  font-size: 12px;
  color: #e6e9ef;
  background: #0f1218;
  border: 1px solid #2a2f3a;
  border-radius: 6px;
  outline: none;
}
.cv-ai-input:focus {
  border-color: #2dd4bf;
}
.cv-ai-input:disabled {
  opacity: 0.6;
}
.cv-ai-note {
  max-width: 320px;
  overflow: hidden;
  font-size: 12px;
  color: #2dd4bf;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cv-label {
  font-size: 12px;
  color: #9aa4b2;
}
/* 插件按钮的图标是 emoji 字符串（源的 ReactNode 在 Vue 侧吃不下 ✗）⇒ 只负责占位定宽 */
.cv-add-icon {
  display: inline-flex;
  width: 14px;
  justify-content: center;
  font-size: 13px;
  line-height: 1;
}
.cv-count,
.cv-msg {
  font-size: 12px;
  color: #9aa4b2;
}
.cv-msg {
  color: #2dd4bf;
}
.cv-file {
  display: none;
}
.cv-stage {
  position: relative;
  flex: 1;
  min-height: 0;
  border: 1px solid #2a2f3a;
  border-radius: 8px;
  overflow: hidden;
}
.cv-hint {
  position: absolute;
  left: 10px;
  bottom: 8px;
  margin: 0;
  font-size: 11px;
  color: #6b7280;
  background: rgba(15, 17, 21, 0.75);
  padding: 3px 8px;
  border-radius: 4px;
  pointer-events: none;
  max-width: calc(100% - 220px);
}
.cv-overlay-map {
  position: absolute;
  right: 10px;
  bottom: 44px;
}
.cv-overlay-zoom {
  position: absolute;
  right: 10px;
  bottom: 10px;
}

/* ==========================================================================
   手机档：工具条本来就是 flex-wrap，这里只让 AI 输入独占整行、把左下角
   说明文字挪到左上，避免它被「小地图 + 缩放」两个浮层挤成一条 80px 的缝。
   ========================================================================== */
@media (max-width: 767px) {
  .cv-page { padding: 8px; gap: 8px; }
  .cv-bar { gap: 8px; padding: 8px 10px; }
  .cv-ai { flex: 1 1 100%; min-width: 0; }
  .cv-ai-note { max-width: 100%; }
  .cv-hint {
    left: 10px; right: auto;
    top: 8px; bottom: auto;
    max-width: calc(100% - 24px);
  }
}

/* 触屏：工具条上的 btn-sm 原高 26px，抬到 34px 才好点（缩放手柄在
   CanvasZoomControls.vue 里自己那档，scoped 样式跨不过组件边界） */
@media (pointer: coarse) {
  .cv-bar .btn-sm, .cv-bar .btn { min-height: var(--tap-min); }
}
</style>

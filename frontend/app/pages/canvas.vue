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
import {
  ChevronDown,
  Download,
  HelpCircle,
  Maximize2,
  MessageSquare,
  MoreHorizontal,
  Plus,
  Trash2,
  Upload,
} from 'lucide-vue-next'
import CanvasBoard from '../components/canvas/CanvasBoard.vue'
import CanvasChatPanel from '../components/canvas/CanvasChatPanel.vue'
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
import type { RunAnswer, RunOverride } from '../utils/canvasOps'
import type { GeneratedPayload } from '../utils/canvasOps'
import { DEFAULT_NODE_SIZE } from '../utils/canvasNodeSize'
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
/** 画板句柄：只为了把「代点生成」转给卡片（卡片自己握着档位与引擎任务） */
const boardRef = ref<InstanceType<typeof CanvasBoard> | null>(null)
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

// ---- 画布 AI：入口在右侧对话面板 ----
// 原来这块是一个单行 input：宽度被挤到 120px，回话只显示**最后一条**、3.6 秒后消失、
// 落地后画布上没有任何「哪张卡动了」的标记 ✗。对话式操作要的是历史、定位与撤销，
// 这三件事都搬进 `components/canvas/CanvasChatPanel.vue` 了 ✓；
// 页面这边只留：开合状态、把可视区尺寸递进去（「跳到改动」要 `focusNode`）、移动端遮罩。
const CHAT_KEY = 'darme.canvas.chat'
const chatOpen = ref(true)

function readChatPref() {
  try {
    const saved = localStorage.getItem(CHAT_KEY)
    if (saved === '0') chatOpen.value = false
    else if (saved === '1') chatOpen.value = true
  } catch {
    // 隐私模式下 localStorage 会抛 —— 不记偏好而已，不该让画布打不开 ✗
  }
}

function setChatOpen(value: boolean) {
  chatOpen.value = value
  // 开助手就把工具条浮层收掉：窄屏抽屉一展开，底下的「添加 / 更多」菜单会跟着露在抽屉上方 ✗
  if (value) panel.value = null
  try {
    localStorage.setItem(CHAT_KEY, value ? '1' : '0')
  } catch {
    // 同上
  }
}

/**
 * 对话里说「生成」⇒ 代你点一下那张配置卡上的「生成」。
 *
 * 走的是卡片自己那条 `runEngine` 通路（档位就绪、首帧、缺件都在那儿拦），
 * 这里只负责转交，不复制一份提交逻辑 ✓。
 * 返回 false ⇒ 面板如实写「待运行」而不是谎报「已提交」✗。
 *
 * ⚠️ 原样转交那句 promise ✗：同一批 op 刚建的卡还没挂载，画布板只能等挂上再跑，
 * 它会回一个 promise —— 这里若写 `=== true` 就把 promise 判成 false（该跑的没跑），
 * 反过来乐观返回 true 则会把「被拒」写成「已提交」✗✗。两种都不许。
 *
 * 同理**不许在这里把 `{ ok, reason }` 折成布尔** ✓：那句话（「zimage 是图片档，
 * 跑不了视频」）就是回执要写的内容，折掉它，面板又只能报「看卡片上的缺项提示」✗。
 */
function runNodeFromChat(nodeId: string, override?: RunOverride): RunAnswer | Promise<RunAnswer> {
  return boardRef.value?.runNode(nodeId, override) ?? { ok: false, reason: '画布还没就绪，这次没有提交' }
}

/** 工具条浮层：添加 / 更多 / 快捷键 —— 同时只开一个 ✓ */
const panel = ref<'add' | 'more' | 'help' | null>(null)

function togglePanel(name: 'add' | 'more' | 'help') {
  panel.value = panel.value === name ? null : name
}

function closePanels() {
  panel.value = null
}

/**
 * 窄屏判定：≤1023 时对话面板是**底部抽屉**（见 CanvasChatPanel 的媒体查询），
 * 这时才需要遮罩 ✗ 桌面右栏常驻不该挡住画布点击。
 */
const narrow = ref(false)
let mediaQuery: MediaQueryList | null = null

function onMediaChange(event: MediaQueryListEvent) {
  narrow.value = event.matches
}

/** Ctrl/⌘ + J 开合面板；Esc 先关浮层（顺序不能反：有浮层时 Esc 不该顺手把面板也收掉 ✗） */
function onKeydown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'j') {
    event.preventDefault()
    setChatOpen(!chatOpen.value)
    return
  }
  if (event.key === 'Escape' && panel.value) {
    event.preventDefault()
    closePanels()
  }
}

// ---- 清空：二次确认 + 8 秒内可撤销 ----
// 旧代码 `@click="clearAll()"` 一键清空并立刻 persist ⇒ 本地那份画布当场没了 ✗。
// 全页只有这一个不可逆操作，保护成本只是一次点击，没有理由省掉 ✓。
const clearing = ref(false)
const clearUndo = ref<{ doc: string; timer: ReturnType<typeof setTimeout> } | null>(null)

function askClear() {
  panel.value = null
  clearing.value = true
}

function doClear() {
  clearing.value = false
  const doc = exportDoc()
  clearAll()
  if (clearUndo.value) clearTimeout(clearUndo.value.timer)
  const timer = setTimeout(() => {
    clearUndo.value = null
  }, 8000)
  clearUndo.value = { doc, timer }
}

function undoClear() {
  if (!clearUndo.value) return
  importDoc(clearUndo.value.doc)
  clearTimeout(clearUndo.value.timer)
  clearUndo.value = null
  flash('已恢复清空前的画布')
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
/**
 * 成品落在哪 ✓
 *
 * 有源卡（对话里点名那张 / 工具栏里点的那张）⇒ 贴它**右侧、与它同高**：
 * 「这张图是哪张卡跑出来的」一眼看得出来 ✓；那块地被占了就往下挪 ——
 * 叠在一张卡上，用户会以为「跑失败了、什么都没出」✗。
 * 没有源卡（工具栏直接跑工作流）⇒ 仍落视口中心，保持原来手感 ✓。
 */
function artifactSpot(sourceId: string | undefined, kind: 'image' | 'video' | 'audio') {
  const size = DEFAULT_NODE_SIZE[kind]
  const source = sourceId ? nodes.value.find((node) => node.id === sourceId) : undefined
  if (!source) {
    const center = toWorld(stageSize.value.width / 2, stageSize.value.height / 2)
    return { x: center.x + 120, y: center.y - 80 }
  }
  const gap = 56
  // `addNode` 以落点为**中心** ⇒ 这里算的也是中心点 ✓
  let spot = { x: source.x + source.w + gap + size.w / 2, y: source.y + size.h / 2 }
  const free = (point: { x: number; y: number }) => {
    const left = point.x - size.w / 2
    const top = point.y - size.h / 2
    return !nodes.value.some((node) => node.type !== 'group'
      && left < node.x + node.w + 24 && node.x - 24 < left + size.w
      && top < node.y + node.h + 24 && node.y - 24 < top + size.h)
  }
  // 只往下试 12 格：再挤就说明这一列真满了，宁可轻微重叠也别飘到看不见的地方 ✗
  for (let step = 0; step < 12 && !free(spot); step++) spot = { x: spot.x, y: spot.y + size.h + 36 }
  return spot
}

function onGenerated(payload: GeneratedPayload) {
  const id = addNode(payload.kind, artifactSpot(payload.sourceId, payload.kind))
  updateNode(id, {
    url: payload.url,
    title: payload.kind === 'video' ? '生成视频' : payload.kind === 'audio' ? '生成音频' : '生成图片',
  })
  /*
   * 连线不是装饰 ✓：下一轮对话 agent 读到的画布快照里，这条边就是
   * 「这张成品出自那张卡」的唯一凭据 —— 不连，它会以为这是一张来路不明的图 ✗。
   * `connect` 自带三不许（自连/连分组/配置卡互连）校验，被拒也只是不连线，不throw ✓。
   */
  if (payload.sourceId && nodes.value.some((node) => node.id === payload.sourceId)) {
    connect(payload.sourceId, id)
  }
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
  readChatPref()
  window.addEventListener('keydown', onKeydown)
  mediaQuery = window.matchMedia('(max-width: 1023px)')
  narrow.value = mediaQuery.matches
  mediaQuery.addEventListener('change', onMediaChange)

  const el = stageEl.value
  if (!el) return
  const apply = () => {
    stageSize.value = { width: el.clientWidth, height: el.clientHeight }
  }
  apply()
  observer = new ResizeObserver(apply)
  observer.observe(el)
})
onBeforeUnmount(() => {
  observer?.disconnect()
  window.removeEventListener('keydown', onKeydown)
  mediaQuery?.removeEventListener('change', onMediaChange)
  mediaQuery = null
  if (clearUndo.value) clearTimeout(clearUndo.value.timer)
})
</script>

<template>
  <div class="cv-page">
    <!-- 工具条：原来 16 个按钮平铺、换行后挤成三段，每个按钮都在抢注意力 ✗
         现在只留「添加 / 删除 / 复位 / 更多 / ? / 助手」六个入口，成组解组归到添加面板里 ✓ -->
    <div class="cv-bar">
      <div class="cv-tools">
        <button
          class="btn btn-sm cv-trigger"
          :class="{ 'is-on': panel === 'add' }"
          :aria-expanded="panel === 'add'"
          @click.stop="togglePanel('add')"
        >
          <Plus :size="14" />添加<ChevronDown :size="12" />
        </button>

        <div v-if="panel === 'add'" class="cv-pop cv-pop-wide" @click.stop>
          <p class="cv-pop-title">新建卡片</p>
          <div class="cv-pop-grid">
            <button v-for="t in TYPES" :key="t" class="cv-pop-item" @click="add(t); closePanels()">
              {{ NODE_TYPE_LABEL[t] }}
            </button>
          </div>
          <!-- 插件节点 ✓（清单来自 `utils/canvasPlugins.ts`）：不给入口就只能靠导入画布才出现 ✗ -->
          <p class="cv-pop-title">插件节点</p>
          <div class="cv-pop-grid">
            <button
              v-for="p in PLUGIN_NODES"
              :key="p.type"
              class="cv-pop-item"
              :title="`${p.title}（${p.type}）`"
              @click="add(p.type); closePanels()"
            >
              <span class="cv-add-icon">{{ p.icon }}</span>{{ p.title }}
            </button>
          </div>
          <div class="cv-pop-sep" />
          <button
            class="cv-pop-item cv-pop-row"
            :disabled="!canGroupSelectedNodes(selection, nodes)"
            @click="groupSelected(); closePanels()"
          >
            成组（选 ≥2 个）
          </button>
          <button
            class="cv-pop-item cv-pop-row"
            :disabled="!canUngroupSelectedNodes(selection, nodes)"
            @click="ungroupSelected(); closePanels()"
          >
            解组
          </button>
        </div>
      </div>

      <div class="cv-right">
        <span v-if="message" class="cv-msg">{{ message }}</span>
        <span class="cv-count">{{ nodes.length }} 节点 · {{ edges.length }} 连线</span>

        <button class="btn btn-ghost btn-sm" :disabled="!selection.length" @click="removeNodes(selection)">
          <Trash2 :size="14" />删除
        </button>
        <button class="btn btn-ghost btn-sm" title="复位视图" @click="resetView()">
          <Maximize2 :size="14" /><span class="cv-btn-text">复位</span>
        </button>

        <div class="cv-tools">
          <button
            class="btn btn-ghost btn-sm"
            :class="{ 'is-on': panel === 'more' }"
            :aria-expanded="panel === 'more'"
            @click.stop="togglePanel('more')"
          >
            <MoreHorizontal :size="14" />更多<ChevronDown :size="12" />
          </button>
          <div v-if="panel === 'more'" class="cv-pop cv-pop-end" @click.stop>
            <button class="cv-pop-item cv-pop-row" @click="doExport(); closePanels()">
              <Download :size="14" />导出画布 JSON
            </button>
            <button class="cv-pop-item cv-pop-row" @click="closePanels(); fileInput?.click()">
              <Upload :size="14" />导入画布 JSON
            </button>
            <div class="cv-pop-sep" />
            <button class="cv-pop-item cv-pop-row is-danger" @click="askClear()">
              <Trash2 :size="14" />清空画布
            </button>
          </div>
        </div>

        <div class="cv-tools">
          <button
            class="btn btn-ghost btn-icon"
            title="快捷键"
            :aria-expanded="panel === 'help'"
            @click.stop="togglePanel('help')"
          >
            <HelpCircle :size="16" />
          </button>
          <div v-if="panel === 'help'" class="cv-pop cv-pop-end cv-pop-help" @click.stop>
            <p class="cv-pop-title">画布操作</p>
            <dl>
              <div><dt>滚轮</dt><dd>以指针为中心缩放</dd></div>
              <div><dt>空格 / 中键拖动</dt><dd>平移画布</dd></div>
              <div><dt>空白处拖拽</dt><dd>框选多张卡</dd></div>
              <div><dt>双击空白</dt><dd>新建文本卡</dd></div>
              <div><dt>拖卡片头部</dt><dd>移动卡片（拖进分组即收纳）</dd></div>
              <div><dt>拖右侧圆点</dt><dd>连线；点连线可删除</dd></div>
              <div><dt>右键</dt><dd>卡片/空白菜单（图片算子、对比、成组…）</dd></div>
              <div><dt>Ctrl / ⌘ + J</dt><dd>开合右侧对话面板</dd></div>
            </dl>
          </div>
        </div>

        <button
          class="btn btn-sm cv-ai-toggle"
          :class="{ 'is-on': chatOpen }"
          :aria-pressed="chatOpen"
          title="画布助手：一句话改画布（Ctrl / ⌘ + J）"
          @click="setChatOpen(!chatOpen)"
        >
          <MessageSquare :size="14" /><span class="cv-btn-text">{{ chatOpen ? '收起助手' : '助手' }}</span>
        </button>

        <input ref="fileInput" type="file" accept="application/json" class="cv-file" @change="onFile" />
      </div>
    </div>

    <!-- 浮层点外部关闭：垫在弹出菜单之下、工具条之上，不遮住画布交互太久（点一下就退） -->
    <div v-if="panel" class="cv-scrim" @click="closePanels()" />

    <div class="cv-body">
      <div ref="stageEl" class="cv-stage">
        <CanvasBoard
          ref="boardRef"
          @contextmenu="menu = $event"
          @error="flash"
          @notice="flash"
          @generated="onGenerated"
        />
        <!-- 两套说明：宽屏是鼠标全套手势；窄屏没有「右键 / 空格 / 框选」，
             把桌面那套原文搬过去只会盖住三行画布（实测太吵 ✗） -->
        <p class="cv-hint cv-hint-wide">
          滚轮缩放 · 空格或中键拖动平移 · 空白处拖拽框选 · 拖卡片头部移动（拖进分组会收纳）· 拖右侧圆点连线 · 点连线删除 · 右键打开菜单 · 双击空白建文本卡
        </p>
        <p class="cv-hint cv-hint-narrow">单指拖动平移 · 双指缩放 · 双击空白建文本卡</p>
        <div class="cv-overlay-map">
          <CanvasMiniMap :nodes="nodes" :viewport="viewport" :stage="stageSize" @jump="jumpTo" />
        </div>
        <div class="cv-overlay-zoom">
          <CanvasZoomControls :scale="viewport.scale" @zoom="zoomBy" @reset="resetView" />
        </div>
      </div>

      <!-- ≤1023 时面板变成底部抽屉 ⇒ 需要一层遮罩把画布挡住，点空白处收起 ✓ -->
      <div v-if="chatOpen && narrow" class="cv-sheet-scrim" @click="setChatOpen(false)" />

      <CanvasChatPanel
        :open="chatOpen"
        :stage="stageSize"
        :generate="runNodeFromChat"
        @update:open="setChatOpen"
      />
    </div>

    <!-- 清空：二次确认 + 8 秒撤销（全页唯一的不可逆操作） -->
    <div v-if="clearing" class="cv-confirm" role="alertdialog" aria-live="assertive">
      <span>清空后这张画布本地就没了，确定？</span>
      <button class="btn btn-sm" @click="clearing = false">取消</button>
      <button class="btn btn-sm cv-danger" @click="doClear">确定清空</button>
    </div>
    <div v-else-if="clearUndo" class="cv-confirm">
      <span>画布已清空</span>
      <button class="btn btn-sm cv-undo-btn" @click="undoClear">撤销</button>
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
/*
 * 配色一律走全局令牌：画布舞台本身是暗色网格（那是「工作台」），工具条与对话面板
 * 属于**应用外壳**，跟站内其它页面同色系，才不会像贴上去的一排黑砖。
 */
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
  position: relative;
  z-index: 30;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px 8px;
  background: var(--panel-bg, #fff);
  border: 1px solid var(--border, #dbe4f0);
  border-radius: var(--radius-lg, 14px);
  box-shadow: var(--shadow-xs, 0 1px 2px rgba(50, 74, 114, 0.05));
}
.cv-tools {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.cv-right {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
  margin-left: auto;
}
.cv-trigger.is-on,
.cv-tools .btn.is-on {
  color: var(--accent-text, #0f766e);
  background: var(--accent-bg, rgba(13, 148, 136, 0.1));
  border-color: var(--accent-border, #b8eee6);
}
.cv-count,
.cv-msg {
  font-size: 12px;
  color: var(--text-3, #8a94a6);
  font-variant-numeric: tabular-nums;
}
.cv-msg {
  max-width: 42ch;
  overflow: hidden;
  color: var(--accent-text, #0f766e);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cv-file {
  display: none;
}

/* ---- 工具条浮层 ---- */
.cv-pop {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 2;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 220px;
  max-width: min(320px, calc(100vw - 24px));
  padding: 8px;
  background: var(--bg-0, #fff);
  border: 1px solid var(--border, #dbe4f0);
  border-radius: var(--radius, 8px);
  box-shadow: var(--shadow-lg, 0 14px 30px rgba(50, 74, 114, 0.13));
}
.cv-pop-end {
  left: auto;
  right: 0;
  min-width: 200px;
}
.cv-pop-wide {
  min-width: 280px;
}
.cv-pop-title {
  margin: 2px 2px 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-3, #8a94a6);
  letter-spacing: 0.04em;
}
.cv-pop-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px;
}
.cv-pop-item {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 7px 9px;
  font-size: 12.5px;
  font-family: var(--font-body, inherit);
  color: var(--text-1, #2c3850);
  text-align: left;
  background: var(--bg-1, #f8fbff);
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.cv-pop-item:hover:not(:disabled) {
  color: var(--accent-text, #0f766e);
  background: var(--accent-bg, rgba(13, 148, 136, 0.1));
  border-color: var(--accent-border, #b8eee6);
}
.cv-pop-item:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.cv-pop-row {
  justify-content: flex-start;
}
.cv-pop-item.is-danger {
  color: var(--error, #b91c1c);
}
.cv-pop-item.is-danger:hover {
  color: var(--error, #b91c1c);
  background: var(--error-bg, #fdecec);
  border-color: var(--error-border, #f3c6c6);
}
.cv-pop-sep {
  height: 1px;
  margin: 4px 2px;
  background: var(--border, #dbe4f0);
}
/* 快捷键表是「手势｜说明」两列，200px 宽会把说明压成竖条（实测挤 ✗） */
.cv-pop-help {
  min-width: 300px;
}
.cv-pop-help dl {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
}
.cv-pop-help dl > div {
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 8px;
}
.cv-pop-help dt {
  color: var(--text-1, #2c3850);
  font-weight: 600;
}
.cv-pop-help dd {
  margin: 0;
  color: var(--text-2, #60718a);
}
/* 插件按钮的图标是 emoji 字符串（源的 ReactNode 在 Vue 侧吃不下 ✗）⇒ 只负责占位定宽 */
.cv-add-icon {
  display: inline-flex;
  width: 14px;
  justify-content: center;
  font-size: 13px;
  line-height: 1;
}
/* 浮层点外部关闭：垫在工具条之下（bar 是 z-index:30）、页面之上，点一下即退 ✓ */
.cv-scrim {
  position: fixed;
  inset: 0;
  z-index: 20;
}
.cv-ai-toggle.is-on {
  color: var(--accent-text, #0f766e);
  background: var(--accent-bg, rgba(13, 148, 136, 0.1));
  border-color: var(--accent-border, #b8eee6);
}

/* ---- 画布 + 右侧对话面板 ---- */
.cv-body {
  display: flex;
  flex: 1;
  gap: 10px;
  min-height: 0;
}
.cv-stage {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  border: 1px solid #2a2f3a;
  border-radius: 10px;
  overflow: hidden;
}
.cv-hint {
  position: absolute;
  left: 10px;
  /* 右边界吃死、不用 max-width：小地图是 102px + 10 右边距，用百分比算宽度
     会在 1440 下算到 816px 把字压到小地图底下（实测压字了 ✗） */
  right: 240px;
  bottom: 8px;
  margin: 0;
  padding: 3px 8px;
  font-size: 11px;
  color: #6b7280;
  background: rgba(15, 17, 21, 0.75);
  border-radius: 4px;
  pointer-events: none;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cv-hint-narrow {
  display: none;
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
.cv-sheet-scrim {
  position: fixed;
  inset: 0;
  z-index: 130;
  background: rgba(15, 23, 42, 0.38);
}

/* ---- 清空确认 / 撤销 ---- */
.cv-confirm {
  position: fixed;
  left: 50%;
  bottom: 24px;
  z-index: 220;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px 8px 14px;
  font-size: 12.5px;
  color: var(--text-1, #2c3850);
  background: var(--bg-0, #fff);
  border: 1px solid var(--border-strong, #bcc9d9);
  border-radius: 10px;
  box-shadow: var(--shadow-lg, 0 14px 30px rgba(50, 74, 114, 0.13));
  transform: translateX(-50%);
  max-width: calc(100vw - 24px);
}
.cv-danger {
  color: #fff;
  background: var(--error, #b91c1c);
  border-color: var(--error, #b91c1c);
}
.cv-danger:hover {
  color: #fff;
  background: #a11717;
  border-color: #a11717;
}
.cv-undo-btn {
  color: var(--accent-text, #0f766e);
  border-color: var(--accent-border, #b8eee6);
}

/* ==========================================================================
   ≤1023：面板转底部抽屉（样式在 CanvasChatPanel.vue 里），页面这边只收间距。
   ========================================================================== */
@media (max-width: 1023px) {
  .cv-body { gap: 0; }
}

/* ==========================================================================
   ≤767：工具条只留图标 + 关键文字；左下角说明文字挪到左上，避免被
   「小地图 + 缩放」两个浮层挤成一条 80px 的缝。
   ========================================================================== */
@media (max-width: 767px) {
  .cv-page { padding: 8px; gap: 8px; }
  .cv-bar { gap: 6px; padding: 6px; }
  .cv-btn-text { display: none; }
  .cv-count { display: none; }
  .cv-msg {
    max-width: 100%;
    flex: 1 1 100%;
    order: 2;
  }
  .cv-pop-wide { min-width: 0; width: calc(100vw - 24px); }
  .cv-hint {
    left: 10px;
    right: auto;
    top: 8px;
    bottom: auto;
    max-width: calc(100% - 24px);
  }
  .cv-hint-wide { display: none; }
  .cv-hint-narrow { display: block; }
}

/* 触屏：btn-sm 原高 26px，抬到 34px 才好点（缩放手柄在 CanvasZoomControls.vue
   自己那档，scoped 样式跨不过组件边界） */
@media (pointer: coarse) {
  .cv-bar .btn,
  .cv-pop-item {
    min-height: var(--tap-min, 34px);
  }
}
</style>

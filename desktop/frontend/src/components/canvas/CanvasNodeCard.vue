<script setup lang="ts">
/**
 * 画布上的一张节点卡。
 *
 * 两处交互边界（写错就会「拖不动」或「拖一下就选中文字」✗）：
 * 1. **只有头部是拖拽把手** ✓ —— 正文里的 textarea / input 要能正常选词、打字，
 *    所以它们的 pointerdown **不能**冒泡成节点拖动 ✗；
 * 2. **右侧端口只管发起连线** ✓，同样要 stop，否则会同时触发节点移动 ✗。
 */
import { computed, onMounted, ref, watch } from 'vue'
import { nodeTypeLabel, type CanvasNode } from '../../composables/useCanvas'
import { isPluginNodeType } from '../../utils/canvasPlugins'
import { fitNodeSize } from '../../utils/canvasNodeSize'
import PluginNodeBody from './plugins/PluginNodeBody.vue'
import { uploadAPI } from '../../composables/useApi'
/**
 * 媒体地址出口 ✓✗。节点 `url` 是**后端给的根相对路径**（`/static/engine/xxx.wav`）或用户手填的，
 * 原样塞进 `<img>/<video>/<audio>`，在 Web 端同源没事；桌面壳的页面 origin 是
 * `tauri.localhost` ⇒ 打包后整片 404（dev 被 Vite 的 `/static` proxy 掩盖，看不出来 ✗）。
 * 两端 `apiBase.ts` 都导出同名函数（Web 端同源原样返回 ✓），所以这份组件才能在两个工程里逐字相同。
 */
import { mediaUrl } from '../../utils/apiBase'
import { compressImageForUpload, formatBytes } from '../../utils/mediaFile'
import { cropImage, type CropRect } from '../../utils/canvasImageOps'
import { useComfyRun, type ComfyWorkflow } from '../../composables/useComfyRun'
import { useEngineRun, type EngineStage } from '../../composables/useEngineRun'
import CanvasCropDialog from './CanvasCropDialog.vue'

const props = defineProps<{
  node: CanvasNode
  selected: boolean
  /** 上游素材（连线 / 同组 ✓，由画布统一算好传进来 ✓）—— 生成时当参考图 ✓ */
  references?: { images: string[]; text?: string }
}>()

const emit = defineEmits<{
  (e: 'move-start', ev: PointerEvent): void
  (e: 'link-start', ev: PointerEvent): void
  (e: 'update', patch: Partial<CanvasNode>): void
  (e: 'error', message: string): void
  /** 非错误提示 ✓（如"已压缩后再上传"）—— 混进 error 通道会让人以为传失败了 ✗ */
  (e: 'notice', message: string): void
  /** 生成完成 ⇒ 由画布**新开一张卡**带进来 ✓（原地覆盖会把用户手上的参考图冲掉 ✗） */
  (e: 'generated', payload: { kind: 'image' | 'video' | 'audio'; url: string; runId: number | null }): void
}>()

// ─────────────────── 插件节点（`<pluginId>:<name>`）───────────────────
/**
 * 内置五类之外的类型一律交给 `PluginNodeBody` ✓。
 *
 * 判据是**形态**（带冒号）而不是「在不在清单里」：清单里查不到的类型也要把卡片画出来、
 * 并报「未安装的插件」—— 按「查不到就不渲染」写，用户导入一份装了别家插件的画布时，
 * 那些节点会**凭空消失**（数据还在，但看起来像被吞了 ✗）。
 */
const isPlugin = computed(() => isPluginNodeType(props.node.type))

/** 插件改值 ⇒ 并进 `values` ✓（插件键一律落这里；写平铺字段会与真字段撞名 ✗） */
function onPluginPatch(patch: Record<string, string>) {
  emit('update', { values: { ...(props.node.values || {}), ...patch } })
}

// ─────────────────── 生成（config 卡 = 一次运行 ✓） ───────────────────
// 两条通道 ✗✗（同构但后端不同 ✓）：
//   * engine  = 本仓自研引擎（**默认** ✓ 不依赖任何外部服务 ✓）；
//   * comfyui = 外部 ComfyUI 工作流（注册表 `workflowId` + 8765 上游 ✓）。
const run = useComfyRun()
const workflows = run.workflows
const runState = run.stateOf(props.node.id)

const engine = useEngineRun()
const engineStages = engine.stages
const engineStagesError = engine.stagesError
const engineState = engine.stateOf(props.node.id)

/** 比例口径 ✓（就是请求里的 `ratio` 字符串 ✓ —— 引擎会按档位吸附到合法宽高 ✓） */
const RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

/** 运行方式 ✓ —— 老档没有 `runMode` 时按「有没有绑工作流」推断 ✓（存量数据不迁移 ✗） */
const runMode = computed<'engine' | 'comfyui'>(
  () => props.node.runMode || (props.node.workflowId ? 'comfyui' : 'engine'),
)

/** 当前档位 ✓ —— 没选就取**第一个件齐的**（空表 ⇒ undefined，界面会提示去查清单） */
const currentStage = computed<EngineStage | undefined>(() => {
  const list = engineStages.value
  if (!list.length) return undefined
  return list.find((item) => item.stage === props.node.engineStage)
    || list.find((item) => item.ready)
    || list[0]
})

/** 档位建议 ✓：装得下 / 装不下会走联合或 CPU（**不要说"能跑"** ✗ —— 慢 10 倍也是能跑 ✓） */
function stageHint(stage: EngineStage): string {
  const parts: string[] = []
  if (!stage.ready) parts.push(`缺 ${stage.missing.length} 件`)
  else if (stage.fits === false) parts.push('装不下（会走联合/CPU，慢）')
  if (stage.requiredGiB) parts.push(`${stage.requiredGiB.toFixed(1)} GiB`)
  return parts.length ? `（${parts.join(' · ')}）` : ''
}

function switchMode(mode: 'engine' | 'comfyui') {
  emit('update', { runMode: mode })
  if (mode === 'engine') void engine.loadStages()
}

const currentWorkflow = computed<ComfyWorkflow | undefined>(
  () => workflows.value.find((item) => item.id === props.node.workflowId),
)

interface FormField {
  key: string
  label: string
  kind: 'number' | 'text' | 'select'
  options?: unknown[]
  placeholder?: string
}

/**
 * 表单字段 = 注册表声明的 ``parameters`` ＋ **按类型补的通用块** ✓。
 *
 * 为什么还要补：注册表只为少数工作流声明了 ``seed/sampler/scheduler``，
 * 而 ``bindings`` 里实际吃 ``size.width`` / ``quality.steps`` 的一大片 ——
 * 只渲染声明的字段，用户就**选不了尺寸和步数**（出图永远是默认档 ✗）。
 * 已经声明过的键不重复（``new Set`` 去重 ✓）。
 */
const fields = computed<FormField[]>(() => {
  const declared: FormField[] = (currentWorkflow.value?.parameters || []).map((param) => ({
    key: param.key,
    label: param.label || param.key,
    kind: param.type === 'select' ? 'select' : param.type === 'number' ? 'number' : 'text',
    options: param.options,
    placeholder: param.description,
  }))
  const seen = new Set(declared.map((item) => item.key))
  const common: FormField[] = currentWorkflow.value?.type === 'video'
    ? [
        { key: 'seconds', label: '时长(秒)', kind: 'number', placeholder: '留空用工作流默认' },
        { key: 'width', label: '宽', kind: 'number' },
        { key: 'height', label: '高', kind: 'number' },
      ]
    : [
        { key: 'width', label: '宽', kind: 'number' },
        { key: 'height', label: '高', kind: 'number' },
        { key: 'steps', label: '步数', kind: 'number' },
        { key: 'cfg', label: 'CFG', kind: 'number' },
      ]
  return [...declared, ...common.filter((item) => !seen.has(item.key))]
})

const referenceCount = computed(() => props.references?.images?.length || 0)

const comfyRunning = computed(
  () => runState.value.status === 'queued' || runState.value.status === 'running',
)
const engineRunning = computed(
  () => engineState.value.status === 'queued' || engineState.value.status === 'running',
)
const running = computed(() => (runMode.value === 'engine' ? engineRunning.value : comfyRunning.value))

const progressText = computed(() => {
  if (runMode.value === 'engine') {
    const state = engineState.value
    if (state.status === 'failed') return state.error || '生成失败'
    if (state.status === 'cancelled') return '已取消'
    if (state.status === 'succeeded') {
      return `完成${state.elapsedMs ? ` · ${Math.round(state.elapsedMs / 1000)}s` : ''}（已带入画布）`
    }
    const phase = engine.phaseLabel(state.phase)
    if (state.total > 0) {
      return `${phase} · 第 ${state.step}/${state.total} 步 · ${Math.round(state.percent)}%`
    }
    if (phase) return `${phase} · ${Math.round(state.percent)}%`
    // ⚠️ 装载权重那段**没有步进** ✓（几十秒）⇒ 如实说"在装载" ✗ 别画 0% 让人以为卡死 ✓
    return state.status === 'queued' ? '排队中（等前一个任务跑完）' : '运行中（装载权重 / 编码）'
  }
  const state = runState.value
  if (state.status === 'failed') return state.error || '运行失败'
  if (state.status === 'cancelled') return '已取消'
  if (state.status === 'succeeded') {
    return `完成${state.elapsedMs ? ` · ${Math.round(state.elapsedMs / 1000)}s` : ''}（已带入画布）`
  }
  // ⚠️ 没有逐帧进度时**如实说** ✓：画一个永远 0% 的环比什么都不画更糟 ✗
  if (!state.progressAvailable) {
    return state.status === 'queued' ? '排队中（本机无逐帧进度）' : '运行中（本机无逐帧进度）'
  }
  return state.total
    ? `第 ${state.step}/${state.total} 步 · ${Math.round(state.percent)}%`
    : `${Math.round(state.percent)}%`
})

/** 进度条百分比 ✓（两条通道各读各的 ✓） */
const progressPercent = computed(
  () => (runMode.value === 'engine' ? engineState.value.percent : runState.value.percent),
)

function valueOf(key: string): string {
  const raw = props.node.values?.[key]
  if (raw !== undefined && raw !== '') return raw
  const declared = currentWorkflow.value?.parameters?.find((item) => item.key === key)
  return declared?.default === undefined || declared.default === null ? '' : String(declared.default)
}

function patchValue(key: string, value: string) {
  if (key === 'variant') {
    emit('update', { variant: value || undefined })
    return
  }
  emit('update', { values: { ...(props.node.values || {}), [key]: value } })
}

/** 换工作流 ⇒ **清空旧参数** ✓（两套工作流的键不通用，留着会静默沿用上一个的 seed/steps ✗） */
function onWorkflowChange(id: string) {
  emit('update', { workflowId: id || undefined, variant: undefined, values: {} })
}

function numeric(raw: Record<string, string>, key: string): number | undefined {
  const text = (raw[key] ?? '').trim()
  if (!text) return undefined
  const value = Number(text)
  return Number.isFinite(value) ? value : undefined
}

/** 表单 → ``POST /comfyui/runs`` 的注册表形态 ✓（后端负责注入，前端**不拼图** ✗） */
function buildPayload() {
  const raw = props.node.values || {}
  const width = numeric(raw, 'width')
  const height = numeric(raw, 'height')
  const steps = numeric(raw, 'steps')
  const cfg = numeric(raw, 'cfg')
  const values: Record<string, unknown> = {}
  for (const [key, text] of Object.entries(raw)) {
    if (text === '' || key === 'width' || key === 'height' || key === 'steps' || key === 'cfg') continue
    // 纯数字串转 number ✓：送字符串会让上游把它当文件名/表达式 ✗
    values[key] = /^-?\d+(\.\d+)?$/.test(text) ? Number(text) : text
  }
  if (width !== undefined) values.width = width
  if (height !== undefined) values.height = height
  if (steps !== undefined) values.steps = steps
  if (cfg !== undefined) values.cfg = cfg
  return {
    workflowId: props.node.workflowId,
    variant: props.node.variant || undefined,
    // ⚠️ 提示词走**顶层 prompt** ✓：bindings 里 prompt 是特例（没有 from 路径）
    prompt: props.node.text || '',
    size: width && height ? { width, height } : undefined,
    quality: steps !== undefined || cfg !== undefined ? { steps, cfg } : undefined,
    count: numeric(raw, 'count') ?? 1,
    seconds: numeric(raw, 'seconds'),
    // 上游素材 → 工作流输入 ✓（只有图片有落点：`from: "image"` 才会建 LoadImage ✓）
    referenceImageUrls: referenceCount.value ? props.references?.images : undefined,
    values,
  }
}

async function runWorkflow() {
  if (!props.node.workflowId) return
  try {
    await run.start(props.node.id, buildPayload())
  } catch (err: any) {
    emit('error', `提交失败：${err?.message || err}`)
  }
}

async function cancelRun() {
  try {
    await run.cancel(props.node.id)
  } catch (err: any) {
    emit('error', `取消失败：${err?.message || err}`)
  }
}

/** 只校验不提交 ✓（缺节点包/悬空连线现在就知道，不用等队列跑完才发现 ✗） */
async function checkWorkflow() {
  if (!props.node.workflowId) return
  try {
    const result: any = await run.validate(buildPayload())
    if (!result?.checked) {
      emit('notice', `未能校验：${result?.reason || '上游校验服务不可达'}`)
      return
    }
    if (result.ok) {
      emit('notice', `校验通过（${result.nodes ?? '?'} 个节点）`)
      return
    }
    const problems = (result.problems || []).slice(0, 3).join('；')
    emit('error', `校验未通过：${problems || '未知原因'}`)
  } catch (err: any) {
    emit('error', `校验失败：${err?.message || err}`)
  }
}

// ─────────────────── 引擎通道（`/engine/generate` ✓） ───────────────────
/**
 * ⭐ **按档写死的三件事收在一张表里** ✓✗。
 *
 * `video`（出 mp4 ⇒ 回卡要建成**视频**节点）、`firstFrame`（I2V 必须给首帧）、
 * `fps`（时长 ⇒ 帧数）本来就是同一件事的三个侧面 ✓ —— 拆成三个正则/字典各写一遍，
 * 新增一档必然漏改其中一个 ✓✗（漏 `video` ⇒ 视频被当图片卡建进画布；
 * 漏 `firstFrame` ⇒ 提交了才在装载阶段被拒；漏 `fps` ⇒ 帧数提示写错 ✓✗）。
 *
 * ⚠️ `fps` **只用于展示量级** ✗：真正的帧吸附（Wan2.2 是 `4k+1`、H3 是它自己的网格 ✓）
 * 在后端 `geometry.snap_frames` 里 ✓ —— 前端不重算（重算＝第二份真源 ✗）。
 */
interface StageTraits {
  /** 出 mp4 ✓ */
  video: boolean
  /** **必须**给首帧 ✓（I2V）—— ⚠️ h3 是文生视频 ✗，给它塞 firstFrame 是多余的 ✓ */
  firstFrame: boolean
  /** 展示用帧率 ✓（`0` ⇒ 不是视频档） */
  fps: number
  /**
   * 出**音频** ✓（`music3` 是唯一一档 ✓）。
   *
   * ⚠️ 它**不是** `video` 的别名 ✗：H3 是「视频＋音频双流」✓，产物仍是 mp4 ⇒ `video: true,
   * audio: false` ✓；宁可多一个字段也不要让「有声音」被读成「出 wav」✗（那会把 mp4 塞进
   * `<audio>`，播放器一片静默 ✗）。漏填这一格 ⇒ 歌跑完了按图片卡建进画布 ✓✗。
   */
  audio: boolean
}
const STAGE_TRAITS: Record<string, StageTraits> = {
  wan22: { video: true, firstFrame: true, fps: 16, audio: false },
  seedvr2: { video: true, firstFrame: true, fps: 8, audio: false },
  h3: { video: true, firstFrame: false, fps: 24, audio: false },
  music3: { video: false, firstFrame: false, fps: 0, audio: true },
}

const traits = computed<StageTraits | null>(
  () => STAGE_TRAITS[currentStage.value?.stage || ''] || null,
)
const isVideoStage = computed(() => Boolean(traits.value?.video))
const needsFirstFrame = computed(() => Boolean(traits.value?.firstFrame))
const isAudioStage = computed(() => Boolean(traits.value?.audio))
/** 要时长那一格 ✓（视频与音频都要，但**只有它们要** ✓ —— 图档给 seconds 是噪音 ✗） */
const needsSeconds = computed(() => isVideoStage.value || isAudioStage.value)

/** 时长 → 帧数的**量级**提示 ✓（实测踩过：留空 5 秒 = 81 帧，一跑半小时 ✓✗） */
const frameHint = computed(() => {
  const fps = traits.value?.fps || 0
  if (!fps) {
    // 音频档没有帧率 ✓，但**同一个 5 秒默认对它照样成立** ✓✗（后端 `_fill_music3_plan` 留空按
    // 5 秒算 → 一句都唱不完 ✓）；不说这句，用户会以为留空＝整首一直唱下去 ✓
    if (!isAudioStage.value) return ''
    return numeric(props.node.values || {}, 'seconds') ? '' : '留空就是 5 秒（够一句，不够一首）'
  }
  // 填 0 / 负数时别显示"0 帧"（后端会吸附到最小帧数 ✓，这里只防止把提示写歪 ✗）
  const seconds = Math.max(0.25, numeric(props.node.values || {}, 'seconds') ?? 5)
  const frames = Math.round(seconds * fps)
  const tail = seconds === 5 ? '（留空就是 5 秒）' : ''
  // ⚠️ 「装不下卡」时才提这一句 ✓：装得下还劝人缩短时长就是瞎紧张 ✗
  const heavy = currentStage.value?.fits === false
    ? '；这档装不下 24 GB 卡 ⇒ 走联合模式（部分层在 CPU），帧数越多越慢（注意力随帧数平方涨），建议先 1 秒验通路'
    : ''
  return `${seconds}s × ${fps}fps ≈ ${frames} 帧${tail}${heavy}`
})

/**
 * 连线图在这张卡里**到底干什么** ✓（三条通道各不相同 ✓）。
 *
 * ⚠️ 必须写出来 ✗✗：默认（文生图四档）**完全不读**连线图 ✓ —— 不说明的话，
 * 「连了一张图 → 点生成 → 出了一张无关的新图」就是必然结果 ✓✗，而且没有任何报错 ✓
 * （后端那条 `_warn_unused_references` 是兜底 ✓，界面这一句才是**点之前**就看得见的 ✓）。
 */
const referenceHint = computed<{ text: string; warn: boolean }>(() => {
  if (runMode.value !== 'engine') return { text: '', warn: false }
  if (needsFirstFrame.value) return { text: '第 1 张当首帧（I2V）', warn: false }
  if (isVideoStage.value) return { text: '全部当参考图（H3 ref2va）', warn: false }
  const denoise = numeric(props.node.values || {}, 'denoise')
  if (typeof denoise === 'number' && denoise > 0 && denoise < 1) {
    return { text: `第 1 张当起点图（denoise ${denoise}）`, warn: false }
  }
  return { text: '本档不吃参考图 ⇒ 会被忽略（要按图改请填起点图 denoise）', warn: true }
})

/**
 * 表单 → ``POST /engine/generate`` ✓。
 *
 * ⚠️ **不传权重路径** ✗✗ —— 后端按档位从清单装配 ✓（`assembly_for_stage` ✓），
 * 缺件会进回执 `autoNote` ✓。前端拼路径 ＝ 把清单解析抄第二遍 ✓✗。
 * ⚠️ 也不传 `width/height` ✗：引擎的尺寸口径是 `ratio` + `megapixels`（**按档吸附** ✓），
 * 硬给像素会被计划层再吸一次 ✓，两边不一致时用户看到的数对不上 ✗。
 */
function buildEnginePayload() {
  const raw = props.node.values || {}
  const payload: Record<string, unknown> = {
    stage: currentStage.value?.stage,
    prompt: props.node.text || '',
    negative: raw.negative || undefined,
    ratio: raw.ratio || undefined,
    megapixels: numeric(raw, 'megapixels'),
    steps: numeric(raw, 'steps'),
    seed: numeric(raw, 'seed'),
    // 引导强度 ✓：1 ＝ 不引导（Krea2 官方口径 ✓），>1 ＝ 每步多算一次负条件 ✓（慢但更贴提示词）
    cfg: numeric(raw, 'cfg'),
  }
  // ── 音频档（music3 ✓）────────────────────────────────────────────────
  // ⭐ 两格必带 ✓✗：`seconds` 是总时长（引擎按它算潜帧 ✓）；`lyrics` 是**歌词正文** ✓ ——
  //    歌词**不能**并进 prompt ✗：prompt 是曲风/编曲描述，后端 `_fill_music3_plan` 把两者
  //    分开送（语义塔 + DiT 条件 ✓），混进 prompt 只会让「曲风」多出一段正文 ✓✗。
  if (isAudioStage.value) {
    payload.seconds = numeric(raw, 'seconds')
    if (raw.lyrics) payload.lyrics = String(raw.lyrics)
    return payload
  }
  if (isVideoStage.value) {
    payload.seconds = numeric(raw, 'seconds')
    // ⭐ H3 双流**必须**带音频潜帧取整口径 ✓✗（2026-11-02 真跑抓的 ✓）：
    //    H3 形态没有单流前向 ✓ ⇒ 这格不给 ⇒ 后端装载 20 秒之后采样才炸 ✓✗
    //    （报错还读起来像「主干坏了」✗）。round / ceil / floor 只差「秒数→音频潜帧」的取整 ✓，
    //    界面取中性值 round ✓（后端纪律「不许默认」防的是**后端静默猜** ✓；这里看得见、可改 ✗）。
    if (currentStage.value?.stage === 'h3' && !raw.audioLatentMode) {
      payload.audioLatentMode = 'round'
    } else if (raw.audioLatentMode) {
      payload.audioLatentMode = String(raw.audioLatentMode)
    }
    // ⭐ 同一条纪律的第二格 ✓✗：双流 init 要把像素帧换算成视频潜帧 ✓，压缩比不给 ⇒ init 才炸 ✓。
    //    H3 的视频 VAE 是 **4 倍时间压缩** ✓（后端 `pipeline`/`h3_edit` 全线都是 4 ✓）。
    if (currentStage.value?.stage === 'h3' && !raw.temporalCompression) {
      payload.temporalCompression = 4
    }
    // Wan2.2 / SEEDVR2 是 I2V ⇒ **首帧是必填** ✓（后端缺了会当场拒 ✓，这里先把手上的第一张送过去 ✓）
    if (needsFirstFrame.value) {
      const first = props.references?.images?.[0]
      if (first) payload.firstFrame = first
    } else if (referenceCount.value) {
      // ⭐ H3 双流是**全仓唯一真吃参考图**的档 ✓（走 `ref2va`、一张一个 `<|image_pad|>` 槽 ✓）——
      //    之前这里直接 `return` ✗✗ ⇒ 画布上连了图也不送 ✓，`<Picture 1>` 永远是空的 ✓，
      //    而用户以为「连上了就是在用它」 ✓✗。
      payload.referenceFrames = props.references?.images
    }
    return payload
  }

  // ── 起点图（图生图 / 尾帧精修 ✓）────────────────────────────────────────
  // ⚠️ 与 `referenceFrames` **不是一回事** ✗✗：那份是「参考什么」（本档根本不读 ✓），
  //    这份是「从哪一格起跑」✓（VAE 编码后按 σ0 掺噪声 ✓）。给 `denoise<1` 却不给起点图
  //    ⇒ 后端**当场拒** ✓（只砍 σ 不换起点 = 与参考图无关的灰图 ✓✗）—— 所以这里成对送 ✓。
  const denoise = numeric(raw, 'denoise')
  if (typeof denoise === 'number' && denoise < 1 && denoise > 0) {
    const start = props.references?.images?.[0]
    if (start) {
      payload.initImage = start
      payload.denoise = denoise
    }
  } else if (referenceCount.value) {
    payload.referenceFrames = props.references?.images
  }
  return payload
}

async function runEngine() {
  const stage = currentStage.value
  if (!stage) {
    emit('error', '还没有档位清单：后端 `/engine/stages` 没返回任何档位')
    return
  }
  if (!stage.ready) {
    // 缺件就别提交 ✓：跑起来也会在装载阶段失败 ✓，那几十秒白等 ✗
    const detail = stage.missing.slice(0, 2).join('；')
    emit('error', `${stage.label} 缺 ${stage.missing.length} 件：${detail || '看 /engine/stages'}`)
    return
  }
  // 只有 I2V 档才逼首帧 ✓（h3 是文生视频 ✗ —— 在这里拦它等于让用户以为必须连图 ✓✗）
  if (needsFirstFrame.value && !referenceCount.value) {
    emit('error', `${stage.label} 需要首帧：先连一张图片卡过来`)
    return
  }
  // 起点图那一格成对使用 ✓：`denoise<1` 却没图 ⇒ 后端会拒（只砍 σ 不换起点＝与参考图无关的灰图 ✗）
  // 音频档没有"起点图"这回事（音频潜帧不吃 initImage ✓）⇒ 这一格不适用 ✗
  const startDenoise = numeric(props.node.values || {}, 'denoise')
  if (!isVideoStage.value && !isAudioStage.value && typeof startDenoise === 'number'
      && startDenoise < 1 && startDenoise > 0 && !referenceCount.value) {
    emit('error', `起点图 denoise=${startDenoise} 需要一张起点图：先连一张图片卡过来（或把 denoise 清空）`)
    return
  }
  try {
    await engine.start(props.node.id, buildEnginePayload())
  } catch (err: any) {
    emit('error', `提交失败：${err?.message || err}`)
  }
}

async function cancelEngine() {
  try {
    await engine.cancel(props.node.id)
  } catch (err: any) {
    emit('error', `取消失败：${err?.message || err}`)
  }
}

// 引擎跑完 ⇒ 产物回卡 ✓（按 taskId 只带一次 ✓；`primaryUrl` 为空时**如实报** ✗ 别静默 ✓）
const engineHandedOff = ref<string | null>(null)
watch(
  () => engineState.value.status,
  (status) => {
    const state = engineState.value
    if (!state.taskId) return
    // ⚠️ 刷新后恢复出来的"已完成"**不许再回卡** ✗✗：产物在刷新前就加进画布并持久化了 ✓，
    //    再 emit 一次会凭空多一张一样的卡 ✓✗（视频档尤其容易，因为中途刷新最可能 ✓）
    if (state.restored) {
      engineHandedOff.value = state.taskId
      return
    }
    if (status !== 'succeeded' || engineHandedOff.value === state.taskId) return
    if (!state.primaryUrl) {
      engineHandedOff.value = state.taskId
      emit('error', `跑完了但没有产物地址（看 primaryPath：${state.primaryPath || '空'}）`)
      return
    }
    engineHandedOff.value = state.taskId
    // 产物类型**先看后缀**（看得见的事实 ✓）再看档位特征 ✓；两处都漏 ⇒ wav 被当图片卡建进来 ✗
    const kind = /\.(mp3|wav|flac|m4a|ogg)$/i.test(state.primaryUrl) || isAudioStage.value
      ? 'audio'
      : /\.(mp4|webm|mov)$/i.test(state.primaryUrl) || isVideoStage.value ? 'video' : 'image'
    emit('generated', { kind, url: state.primaryUrl, runId: null })
  },
)

// 跑完自动把产物带进画布 ✓（只带一次 —— 每次 apply 都 emit 会连开好几张一样的卡 ✗）
const handedOff = ref<number | null>(null)
watch(
  () => runState.value.status,
  (status) => {
    const state = runState.value
    if (status !== 'succeeded' || !state.runId || handedOff.value === state.runId) return
    const url = state.localPath || state.primaryUrl
    if (!url) return
    handedOff.value = state.runId
    emit('generated', {
      kind: currentWorkflow.value?.type === 'video' ? 'video' : 'image',
      url,
      runId: state.runId,
    })
  },
)

onMounted(() => {
  if (props.node.type !== 'config') return
  // 只拉**当前模式**那份清单 ✓（画布上 20 张卡 × 两份 = 40 次请求 ✗；另一份切换时再拉 ✓）
  if (runMode.value === 'engine') void engine.loadStages()
  else void run.loadWorkflows()
  // 刷新过的画布：把这张卡上一个还没跑完的任务**接回来** ✓（否则引擎在跑、界面却回到"未生成" ✗）
  void engine.restore(props.node.id)
})

// 切换运行方式时补拉对应清单 ✓（模块级缓存 ⇒ 只会真拉一次 ✓）
watch(runMode, (mode) => {
  if (mode === 'engine') void engine.loadStages()
  else void run.loadWorkflows()
})

const fileInput = ref<HTMLInputElement | null>(null)
/** 素材卡（图片/视频/音频）的 `accept` 与地址占位 ✓ —— 模板里套三层三元读不出来 ✓✗ */
const mediaAccept = computed(() => (
  props.node.type === 'video' ? 'video/*' : props.node.type === 'audio' ? 'audio/*' : 'image/*'
))
const mediaUrlHint = computed(() => (
  props.node.type === 'video'
    ? '视频地址 /uploads/xxx.mp4'
    : props.node.type === 'audio'
      ? '音频地址 /uploads/xxx.wav'
      : '图片地址 /uploads/xxx.png'
))
const busy = ref(false)
const cropOpen = ref(false)

function pickFile() {
  fileInput.value?.click()
}

/**
 * 素材真正**落到后端**再写卡 ✓（`/uploads/xxx.png`）。
 * ⚠️ 不许把 `URL.createObjectURL(file)` 写进 `url`：那是**刷新即失效**的临时句柄，
 * 画布又恰好把 url 存进 localStorage ⇒ 下次打开全是坏图 ✗（这类"当时好好的"最难查 ✗）。
 */
async function uploadFile(file: File) {
  busy.value = true
  try {
    // ⭐ 超过后端上限（20MB，照抄原 TS 不许改 ✗）的图**先压到 18MB** ✓：
    //    不压的话用户手上那张 4K 图永远传不上来（死路 ✗）；不超限时零损耗直传 ✓
    // ⚠️ 视频与音频**不压** ✗：`compressImageForUpload` 走 canvas 重编码，喂它一个 wav
    //    只会得到一张空的 png（还一脸成功 ✓✗）。
    const isVideo = props.node.type === 'video'
    const isAudio = props.node.type === 'audio'
    const prepared = isVideo || isAudio ? null : await compressImageForUpload(file)
    if (prepared?.compressed) {
      emit('notice', `图片 ${formatBytes(prepared.originalBytes)} 超出上传上限，已压到 ${formatBytes(prepared.bytes)}`)
    }
    const target = prepared?.file ?? file
    const { url } = isVideo
      ? await uploadAPI.video(target)
      : isAudio
        ? await uploadAPI.audio(target)
        : await uploadAPI.image(target)
    emit('update', { url })
  } catch (err: any) {
    emit('error', `上传失败：${err?.message || err}`)
  } finally {
    busy.value = false
  }
}

/**
 * 框选裁剪 ⇒ **新开一张卡** ✓（与其它图片算子同口径：原图留着当参考 ✓）。
 *
 * 结果仍是"先上传拿 URL 再落卡"：画布只存 URL，dataURL 写进 localStorage 会爆 ✗。
 */
async function applyCrop(rect: CropRect) {
  const source = props.node.url
  if (!source) return
  cropOpen.value = false
  busy.value = true
  try {
    //: 裁剪是在**浏览器里**用 canvas 做的（不是送后端）⇒ 这份地址要能被 `<img>` 加载 ✓，
    //: 所以必须过 mediaUrl；而 `props.node.url` 本身（送后端的参考图）保持根相对 ✓
    const result = await cropImage(mediaUrl(source), rect)
    const file = new File([result.blob], 'crop.png', { type: result.blob.type || 'image/png' })
    const { url } = await uploadAPI.image(file)
    emit('generated', { kind: 'image', url, runId: null })
    emit('notice', `已裁剪 ${result.width}×${result.height}，已带入画布`)
  } catch (err: any) {
    emit('error', `裁剪失败：${err?.message || err}`)
  } finally {
    busy.value = false
  }
}

async function onPick(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file) await uploadFile(file)
}

/** 拖文件进卡片 ✓（比"点开文件框再翻目录"快得多） */
async function onDrop(ev: DragEvent) {
  ev.stopPropagation()
  const file = ev.dataTransfer?.files?.[0]
  if (!file) return
  const [expected, noun] = props.node.type === 'video'
    ? ['video/', '视频']
    : props.node.type === 'audio'
      ? ['audio/', '音频']
      : ['image/', '图片']
  if (!file.type.startsWith(expected)) {
    emit('error', `这张卡只收 ${noun}文件`)
    return
  }
  await uploadFile(file)
}

function onHeadDown(ev: PointerEvent) {
  if (ev.button !== 0) return
  ev.stopPropagation()
  emit('move-start', ev)
}
function onPortDown(ev: PointerEvent) {
  ev.stopPropagation()
  ev.preventDefault()
  emit('link-start', ev)
}

/**
 * 素材加载完 ⇒ 卡片按**真实比例**重排 ✓（源 `fitNodeSize` 口径：最长边 ≤640、只缩不放）。
 * 差 2px 以内不动 ✓ —— 否则每次重渲染都写一次状态，会把持久化抖爆 ✗。
 */
function onMediaLoad(ev: Event) {
  const el = ev.target as HTMLImageElement | HTMLVideoElement
  const width = (el as HTMLImageElement).naturalWidth || (el as HTMLVideoElement).videoWidth
  const height = (el as HTMLImageElement).naturalHeight || (el as HTMLVideoElement).videoHeight
  if (!width || !height) return
  const fitted = fitNodeSize(width, height)
  if (Math.abs(fitted.w - props.node.w) < 2 && Math.abs(fitted.h - props.node.h) < 2) return
  emit('update', { w: fitted.w, h: fitted.h })
}
</script>

<template>
  <div
    class="cn-card"
    :class="[isPlugin ? 'is-plugin' : `is-${props.node.type}`, { 'is-selected': props.selected }]"
    :style="{ left: props.node.x + 'px', top: props.node.y + 'px', width: props.node.w + 'px', minHeight: props.node.h + 'px' }"
    @pointerdown.stop
  >
    <header class="cn-head" @pointerdown="onHeadDown">
      <span class="cn-title">{{ props.node.title }}</span>
      <span class="cn-type">{{ nodeTypeLabel(props.node.type) }}</span>
    </header>

    <!-- 拖放挂在整个卡片正文上 ✓：已有图时也能直接拖新的覆盖（只在空卡上挂＝换个图要先把图删掉 ✗） -->
    <div class="cn-body" @dragover.prevent.stop @drop.prevent="onDrop">
      <textarea
        v-if="props.node.type === 'text'"
        class="cn-text"
        :value="props.node.text || ''"
        placeholder="写提示词 / 台词 / 备注…"
        @pointerdown.stop
        @input="emit('update', { text: ($event.target as HTMLTextAreaElement).value })"
      />

      <template v-else-if="props.node.type === 'image' || props.node.type === 'video' || props.node.type === 'audio'">
        <img
          v-if="props.node.type === 'image' && props.node.url"
          :src="mediaUrl(props.node.url)"
          class="cn-media"
          alt=""
          @load="onMediaLoad"
        />
        <video
          v-else-if="props.node.type === 'video' && props.node.url"
          :src="mediaUrl(props.node.url)"
          class="cn-media"
          controls
          muted
          @loadedmetadata="onMediaLoad"
        />
        <!-- 音频 ✓：`preload="metadata"` 只先取时长，不整段下载 ✓（一首歌几十 MB ✗） -->
        <audio
          v-else-if="props.node.type === 'audio' && props.node.url"
          :src="mediaUrl(props.node.url)"
          class="cn-audio"
          controls
          preload="metadata"
        />
        <div v-else class="cn-drop" @dragover.prevent.stop @drop.prevent="onDrop">
          拖文件到这里，或点下方「选文件」
        </div>
        <input
          class="cn-url"
          :value="props.node.url || ''"
          :placeholder="mediaUrlHint"
          @pointerdown.stop
          @input="emit('update', { url: ($event.target as HTMLInputElement).value })"
        />
        <div class="cn-actions">
          <button class="cn-upload" :disabled="busy" @pointerdown.stop @click.stop="pickFile">
            {{ busy ? '上传中…' : '选文件' }}
          </button>
          <!-- 框选裁剪 ✓：位置由人定（"一刀居中"裁不出左下角那个人 ✗） -->
          <button
            v-if="props.node.type === 'image' && props.node.url"
            class="cn-upload"
            :disabled="busy"
            @pointerdown.stop
            @click.stop="cropOpen = true"
          >
            裁剪…
          </button>
        </div>
        <input
          ref="fileInput"
          class="cn-file"
          type="file"
          :accept="mediaAccept"
          @change="onPick"
        />
      </template>

      <template v-else-if="props.node.type === 'config'">
        <!-- 运行方式 ✓（默认「本机引擎」✓ —— 不依赖 8765/ComfyUI ✓；工作流通道作为可选加成 ✓） -->
        <div class="cn-modes">
          <button
            class="cn-mode"
            :class="{ on: runMode === 'engine' }"
            @pointerdown.stop
            @click.stop="switchMode('engine')"
          >
            本机引擎
          </button>
          <button
            class="cn-mode"
            :class="{ on: runMode === 'comfyui' }"
            @pointerdown.stop
            @click.stop="switchMode('comfyui')"
          >
            ComfyUI
          </button>
        </div>

        <!-- 引擎档位 ✓（标签里带"缺几件 / 装不下" ⇒ 别让人点下去才发现 ✗） -->
        <select
          v-if="runMode === 'engine'"
          class="cn-select"
          :value="currentStage?.stage || ''"
          @pointerdown.stop
          @change="emit('update', { engineStage: ($event.target as HTMLSelectElement).value || undefined })"
        >
          <option v-if="!engineStages.length" value="">（没有可用档位）</option>
          <option v-for="item in engineStages" :key="item.stage" :value="item.stage">
            {{ item.label }}{{ stageHint(item) }}
          </option>
        </select>
        <p v-if="runMode === 'engine' && engineStagesError" class="cn-hint err">{{ engineStagesError }}</p>
        <!-- 缺件要**说到件名** ✓：只说"缺 2 件"用户不知道该补什么 ✓✗（下拉里也只放了个数 ✓） -->
        <p v-else-if="runMode === 'engine' && currentStage && !currentStage.ready" class="cn-hint err">
          缺 {{ currentStage.missing.length }} 件：{{ currentStage.missing.slice(0, 3).join('；')
          }}{{ currentStage.missing.length > 3 ? ' …' : '' }}
        </p>
        <p v-else-if="runMode === 'engine' && currentStage?.note" class="cn-hint">{{ currentStage.note }}</p>

        <select
          v-else
          class="cn-select"
          :value="props.node.workflowId || ''"
          @pointerdown.stop
          @change="onWorkflowChange(($event.target as HTMLSelectElement).value)"
        >
          <option value="">选择工作流…</option>
          <option v-for="item in workflows" :key="item.id" :value="item.id">
            {{ item.name }}（{{ item.type === 'video' ? '视频' : '图片' }}）
          </option>
        </select>

        <select
          v-if="runMode === 'comfyui' && currentWorkflow?.variants?.length"
          class="cn-select"
          :value="props.node.variant || ''"
          @pointerdown.stop
          @change="patchValue('variant', ($event.target as HTMLSelectElement).value)"
        >
          <option value="">主形态</option>
          <option v-for="item in currentWorkflow.variants" :key="item.id" :value="item.id">
            {{ item.label }}
          </option>
        </select>

        <textarea
          class="cn-text cn-prompt"
          :value="props.node.text || ''"
          placeholder="提示词（这张卡跑的就是它）"
          @pointerdown.stop
          @input="emit('update', { text: ($event.target as HTMLTextAreaElement).value })"
        />

        <!-- 引擎参数 ✓（**不填像素** ✗：引擎按 `ratio`+百万像素算，再按档吸附 ✓） -->
        <div v-if="runMode === 'engine'" class="cn-fields">
          <label v-if="!isAudioStage" class="cn-field">
            <span class="cn-field-label">比例</span>
            <select
              class="cn-input"
              :value="valueOf('ratio')"
              @pointerdown.stop
              @change="patchValue('ratio', ($event.target as HTMLSelectElement).value)"
            >
              <option value="">按档默认</option>
              <option v-for="item in RATIOS" :key="item" :value="item">{{ item }}</option>
            </select>
          </label>
          <label v-if="!isAudioStage" class="cn-field">
            <span class="cn-field-label">百万像素</span>
            <input
              class="cn-input"
              type="number"
              step="0.05"
              :value="valueOf('megapixels')"
              placeholder="按档默认"
              @pointerdown.stop
              @input="patchValue('megapixels', ($event.target as HTMLInputElement).value)"
            />
          </label>
          <label class="cn-field">
            <span class="cn-field-label">步数</span>
            <input
              class="cn-input"
              type="number"
              :value="valueOf('steps')"
              placeholder="按档默认"
              @pointerdown.stop
              @input="patchValue('steps', ($event.target as HTMLInputElement).value)"
            />
          </label>
          <label class="cn-field">
            <span class="cn-field-label">种子</span>
            <input
              class="cn-input"
              type="number"
              :value="valueOf('seed')"
              placeholder="随机"
              @pointerdown.stop
              @input="patchValue('seed', ($event.target as HTMLInputElement).value)"
            />
          </label>
          <label v-if="needsSeconds" class="cn-field">
            <span class="cn-field-label">时长(秒)</span>
            <input
              class="cn-input"
              type="number"
              step="0.5"
              min="0.5"
              :value="valueOf('seconds')"
              placeholder="5"
              @pointerdown.stop
              @input="patchValue('seconds', ($event.target as HTMLInputElement).value)"
            />
          </label>
          <!-- 起点图比例 ✓（图生图 / 尾帧精修 ✓）：<1 ⇒ 从连线那张图起跑、σ 梯子截尾 ✓
               ⚠️ 与「参考图」**不是一回事** ✗：参考图是条件（本档不吃 ✓），这是**起点** ✓ -->
          <label v-if="!isVideoStage && !isAudioStage" class="cn-field">
            <span class="cn-field-label">起点图 denoise</span>
            <input
              class="cn-input"
              type="number"
              step="0.05"
              min="0.05"
              max="1"
              :value="valueOf('denoise')"
              placeholder="1 = 不用"
              @pointerdown.stop
              @input="patchValue('denoise', ($event.target as HTMLInputElement).value)"
            />
          </label>
          <!-- 引导强度 ✓：填 1 ⇒ 引擎**短路不引导** ✓（Krea2 官方就是 1 ✓），填 >1 才多算一次负条件 ✓ -->
          <label class="cn-field">
            <span class="cn-field-label">引导</span>
            <input
              class="cn-input"
              type="number"
              step="0.5"
              :value="valueOf('cfg')"
              placeholder="1 = 不引导"
              @pointerdown.stop
              @input="patchValue('cfg', ($event.target as HTMLInputElement).value)"
            />
          </label>
        </div>

        <!-- 时长 ⇒ 帧数的量级 ✓（放在网格**外** ✗：`cn-fields` 是 grid，塞个 <p> 会顶掉一格 ✓✗） -->
        <p v-if="frameHint" class="cn-hint">{{ frameHint }}</p>

        <!-- 歌词 ✓（只有音频档有这一格 ✗：别的档喂进去等于多一段没人读的正文 ✓✗）。
             段落标签用方括号 ✓（后端 `te_music3.normalize_lyrics` 会统一小写 ✓）、
             ` ^ ` 当换行 ✓；留空 ⇒ 纯器乐 ✓（参考侧只喂 caption 支路 ✓）。 -->
        <textarea
          v-if="runMode === 'engine' && isAudioStage"
          class="cn-text cn-lyrics"
          :value="valueOf('lyrics')"
          placeholder="歌词正文（可留空 ⇒ 纯器乐）；段落标签如 [Verse] / [Chorus]，^ 表示换行"
          @pointerdown.stop
          @input="patchValue('lyrics', ($event.target as HTMLTextAreaElement).value)"
        />

        <!-- 负面提示词 ✓（引擎四档都吃 `negative` ✓；留空 ⇒ 用该档清单里的默认负向 ✓） -->
        <textarea
          v-if="runMode === 'engine'"
          class="cn-text cn-negative"
          :value="valueOf('negative')"
          placeholder="负面提示词（可留空）"
          @pointerdown.stop
          @input="patchValue('negative', ($event.target as HTMLTextAreaElement).value)"
        />

        <div v-else-if="fields.length" class="cn-fields">
          <label v-for="field in fields" :key="field.key" class="cn-field">
            <span class="cn-field-label">{{ field.label }}</span>
            <select
              v-if="field.kind === 'select'"
              class="cn-input"
              :value="valueOf(field.key)"
              @pointerdown.stop
              @change="patchValue(field.key, ($event.target as HTMLSelectElement).value)"
            >
              <option v-for="option in field.options || []" :key="String(option)" :value="String(option)">
                {{ option }}
              </option>
            </select>
            <input
              v-else
              class="cn-input"
              :type="field.kind === 'number' ? 'number' : 'text'"
              :value="valueOf(field.key)"
              :placeholder="field.placeholder || ''"
              @pointerdown.stop
              @input="patchValue(field.key, ($event.target as HTMLInputElement).value)"
            />
          </label>
        </div>

        <div class="cn-run-row">
          <button
            class="cn-run"
            :disabled="(runMode === 'engine' ? !currentStage : !props.node.workflowId) || running"
            @pointerdown.stop
            @click.stop="runMode === 'engine' ? runEngine() : runWorkflow()"
          >
            {{ running ? `生成中 ${Math.round(progressPercent)}%` : '生成' }}
          </button>
          <button
            v-if="running"
            class="cn-run ghost"
            @pointerdown.stop
            @click.stop="runMode === 'engine' ? cancelEngine() : cancelRun()"
          >
            取消
          </button>
          <button
            v-else-if="runMode === 'comfyui'"
            class="cn-run ghost"
            :disabled="!props.node.workflowId"
            @pointerdown.stop
            @click.stop="checkWorkflow"
          >
            检查
          </button>
        </div>

        <!-- 让用户看见"连线真的生效了" ✓：不显示的话，连了三张图却出图不带它们，只能靠猜 ✗ -->
        <p v-if="referenceCount" class="cn-refs">
          参考图 {{ referenceCount }} 张（来自连线 / 同组）
          <span v-if="referenceHint.text" :class="{ warn: referenceHint.warn }">
            · {{ referenceHint.text }}
          </span>
        </p>

        <!-- 采样预览 ✓：跑了 40 秒还没图时，这张小图是唯一的"真的在工作"的证据 -->
        <div v-if="runMode === 'comfyui' && runState.preview" class="cn-preview">
          <img :src="runState.preview" alt="" />
          <span class="cn-preview-tag">采样预览</span>
        </div>

        <div
          v-if="(runMode === 'engine' ? engineState.status : runState.status) !== 'idle'"
          class="cn-progress"
        >
          <div class="cn-progress-bar">
            <i :style="{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }" />
          </div>
          <span class="cn-progress-text">{{ progressText }}</span>
          <span v-if="runMode === 'engine' && engineState.synthetic" class="cn-progress-text warn">
            干跑产物（参考件未训练，别当成品）
          </span>
        </div>

        <!-- 这次**实际**挑到哪些件 ✓（后端 `autoNote` ✓）：出图不对时第一件事就是核这个 ✓
             —— 比如"跑的是 2.1 不是 2512"「VAE 拿成了 qwen_image_vae」这类 ✓ -->
        <p v-if="runMode === 'engine' && engineState.note" class="cn-hint">{{ engineState.note }}</p>

        <!-- 引擎对这次请求的**说明** ✓（`plan.warnings` ✓）—— 全是"不报错但结果和你以为的不同"：
             尺寸被吸附成多少 / 参考图**被忽略** / 24fps 换成了 16 / denoise<1 实跑步数比名义少 ✓。
             后端刻意不把它们当 error（确实没失败 ✓）⇒ 这里不显示，用户就永远不知道 ✓✗。
             只放前两条（再多就是噪音 ✗），全量在 `title` 里悬停可见 ✓ -->
        <p
          v-if="runMode === 'engine' && engineState.warnings.length"
          class="cn-hint cn-warn-list"
          :title="engineState.warnings.join('\n')"
        >
          ⚠️ {{ engineState.warnings[0] }}<template v-if="engineState.warnings.length > 1">
            <br />⚠️ {{ engineState.warnings[1] }}<template v-if="engineState.warnings.length > 2">（+{{ engineState.warnings.length - 2 }} 条，悬停看全部）</template>
          </template>
        </p>
      </template>

      <!-- 插件节点 ✓（`<pluginId>:<name>`）：正文交给插件自己的组件 ✓。
           三种状态（正常 / 登记了但缺组件 / 未安装）都由 `PluginNodeBody` 画出来 ✓，
           一律不许留白 ✗ —— 留白等于让人以为卡片坏了 -->
      <PluginNodeBody
        v-else-if="isPlugin"
        :node="props.node"
        :text="props.references?.text || ''"
        :images="props.references?.images"
        @patch="onPluginPatch"
      />

      <div v-else class="cn-group">把节点拖进来归组（框选后可整体移动）</div>
    </div>

    <!-- ⚠️ 必须 Teleport 到 body：画布世界层有 `transform`，`position: fixed` 的子元素
         会以**它**为包含块 ⇒ 对话框跟着缩放变小、位置也偏 ✗（这类 bug 只在缩放后显形） -->
    <Teleport to="body">
      <!-- 对话框里是 `<img>` ⇒ 同样要过 mediaUrl ✓（`applyCrop` 送后端的仍是**根相对**的
           `props.node.url` ✓ —— 后端只认自己的路径，给它绝对地址会多一层解析 ✗） -->
      <CanvasCropDialog
        v-if="cropOpen && props.node.url"
        :src="mediaUrl(props.node.url)"
        @pointerdown.stop
        @close="cropOpen = false"
        @confirm="applyCrop"
      />
    </Teleport>

    <span class="cn-port" title="拖到另一张卡发起连线" @pointerdown="onPortDown" />
  </div>
</template>

<style scoped>
.cn-card {
  position: absolute;
  display: flex;
  flex-direction: column;
  background: #171a21;
  border: 1px solid #2a2f3a;
  border-radius: 8px;
  color: #e6e8ee;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
  user-select: none;
}
.cn-card.is-selected {
  border-color: #2dd4bf;
  box-shadow: 0 0 0 1px #2dd4bf, 0 6px 18px rgba(0, 0, 0, 0.4);
}
.cn-card.is-group {
  background: rgba(23, 26, 33, 0.55);
  border-style: dashed;
}
.cn-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border-bottom: 1px solid #2a2f3a;
  cursor: grab;
  font-size: 13px;
}
.cn-head:active {
  cursor: grabbing;
}
.cn-title {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cn-type {
  flex: none;
  font-size: 11px;
  color: #9aa4b2;
  background: #22262f;
  border-radius: 4px;
  padding: 1px 6px;
}
.cn-body {
  flex: 1;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}
.cn-text {
  flex: 1;
  width: 100%;
  resize: none;
  background: #10131a;
  border: 1px solid #2a2f3a;
  border-radius: 6px;
  color: #e6e8ee;
  font: inherit;
  font-size: 12px;
  padding: 6px;
  user-select: text;
}
.cn-url {
  width: 100%;
  background: #10131a;
  border: 1px solid #2a2f3a;
  border-radius: 6px;
  color: #e6e8ee;
  font-size: 12px;
  padding: 5px 6px;
  user-select: text;
}
.cn-media {
  width: 100%;
  max-height: 150px;
  object-fit: contain;
  border-radius: 6px;
  background: #0b0d12;
}
/* 音频 ✓：`<audio>` 的自带控件是**替换元素**，不吃 `object-fit`/`max-height` ✓（写上去也不出错，
   只是白写 ✗）；真正要的是固定高度 —— 不给的话 flex 列里的播放器会被压成一条细缝 ✗ */
.cn-audio {
  width: 100%;
  height: 54px;
  border-radius: 6px;
  background: #0b0d12;
}
.cn-drop {
  flex: 1;
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
  border: 1px dashed #38404e;
  border-radius: 6px;
  color: #6b7280;
  font-size: 12px;
  text-align: center;
}
.cn-actions {
  display: flex;
  align-self: flex-start;
  gap: 6px;
}
.cn-upload {
  align-self: flex-start;
  padding: 4px 10px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #22262f;
  color: #d7dbe3;
  font-size: 12px;
  cursor: pointer;
}
.cn-upload:hover:not(:disabled) {
  background: #2b313d;
}
.cn-upload:disabled {
  color: #6b7280;
  cursor: default;
}
.cn-file {
  display: none;
}
.cn-select,
.cn-input {
  width: 100%;
  padding: 4px 6px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #12151b;
  color: #d7dbe3;
  font-size: 12px;
}
.cn-prompt {
  min-height: 54px;
}
.cn-fields {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.cn-field {
  display: flex;
  align-items: center;
  gap: 6px;
}
.cn-field-label {
  flex: none;
  width: 58px;
  color: #8b93a2;
  font-size: 11px;
}
.cn-run-row {
  display: flex;
  gap: 6px;
}
.cn-run {
  flex: 1;
  padding: 5px 8px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #2dd4bf;
  color: #0b0d12;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.cn-run:disabled {
  background: #22262f;
  color: #6b7280;
  cursor: default;
}
.cn-run.ghost {
  flex: 0 0 auto;
  background: #22262f;
  color: #d7dbe3;
  font-weight: 400;
}
.cn-preview {
  position: relative;
  border-radius: 6px;
  overflow: hidden;
}
.cn-preview img {
  display: block;
  width: 100%;
}
.cn-preview-tag {
  position: absolute;
  left: 6px;
  bottom: 6px;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.6);
  color: #9aa3b2;
  font-size: 10px;
}
.cn-progress {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.cn-progress-bar {
  height: 4px;
  border-radius: 999px;
  background: #22262f;
  overflow: hidden;
}
.cn-progress-bar i {
  display: block;
  height: 100%;
  background: #2dd4bf;
  transition: width 0.3s ease;
}
.cn-progress-text {
  color: #8b93a2;
  font-size: 11px;
}
.cn-refs {
  margin: 0;
  color: #6b7280;
  font-size: 11px;
}
.cn-refs .warn {
  color: #fbbf24;
}
.cn-warn-list {
  color: #fbbf24;
  opacity: 0.9;
}
.cn-modes {
  display: flex;
  gap: 4px;
}
.cn-mode {
  flex: 1;
  padding: 3px 6px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #16191f;
  color: #8b93a2;
  font-size: 11px;
  cursor: pointer;
}
.cn-mode.on {
  border-color: #2dd4bf;
  color: #2dd4bf;
}
.cn-hint {
  margin: 0;
  color: #6b7280;
  font-size: 11px;
  line-height: 1.4;
}
.cn-negative {
  min-height: 34px;
  font-size: 11px;
  opacity: 0.85;
}
/* 歌词 ✓：比负面那格高（是一首歌的正文 ✓），但**到 96px 就滚** ✗ —— 再高会把生成按钮顶出卡外 */
.cn-lyrics {
  min-height: 96px;
  max-height: 160px;
  overflow-y: auto;
  font-size: 11px;
  line-height: 1.5;
}
.cn-hint.err {
  color: #f87171;
}
.cn-progress-text.warn {
  color: #fbbf24;
}
.cn-group {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  font-size: 12px;
}
.cn-port {
  position: absolute;
  right: -7px;
  top: 50%;
  width: 12px;
  height: 12px;
  margin-top: -6px;
  border-radius: 50%;
  background: #2dd4bf;
  border: 2px solid #0f1115;
  cursor: crosshair;
}
.cn-text:focus,
.cn-url:focus {
  outline: 1px solid #2dd4bf;
}
</style>

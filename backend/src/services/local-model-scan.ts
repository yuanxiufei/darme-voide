/**
 * 本地模型扫描服务 — 扫描本机任意目录下的模型文件，自动识别类型与运行时。
 *
 * 设计目标：让 Drama Studio 能「发现」电脑上任意位置的模型，而不再依赖
 * configs/models.json 静态清单。识别结果供注册到 ai_service_configs 后由
 * 现有 adapter（provider + baseUrl 切换）调用。
 *
 * 识别策略：纯启发式（目录名 + 文件名 + 扩展名），零依赖，按优先级命中。
 * 优先级规则与 configs/models.json 的 category/runtime/kind 语义对齐。
 */
import fs from 'fs'
import path from 'path'
import { config } from '../config.js'

// ===== 类型定义 =====

/** 模型大类（对齐 ai_service_configs.serviceType） */
export type ModelKind = 'text' | 'image' | 'video' | 'audio' | 'unknown'

/** 模型运行时（对齐 models.json 的 runtime 字段，扩展了本地推理引擎） */
export type ModelRuntime = 'comfyui' | 'ollama' | 'local-sd' | 'h3' | 'cosyvoice' | 'unknown'

/** 识别置信度 */
export type Confidence = 'high' | 'medium' | 'low'

/** 单个扫描到的模型 */
export interface ScannedModel {
  /** 绝对路径 */
  path: string
  /** 文件名（含扩展名） */
  filename: string
  /** 所在目录名（用于展示其分类，如 checkpoints / diffusion_models） */
  dirname: string
  /** 文件大小（字节） */
  sizeBytes: number
  /** 人类可读大小 */
  sizeHuman: string
  /** 扩展名（小写，含点） */
  ext: string
  /** 识别的大类 */
  kind: ModelKind
  /** 识别的运行时 */
  runtime: ModelRuntime
  /** 置信度 */
  confidence: Confidence
  /** standalone=可独立调用；component=需在 ComfyUI 工作流中引用 */
  role: 'standalone' | 'component'
  /** 命中的规则 id 列表（可追溯为什么这样判） */
  matchedBy: string[]
  /** 推荐的注册配置（供阶段 2 注册到 ai_service_configs 参考） */
  suggested: SuggestedConfig | null
}

/** 推荐的服务配置 */
export interface SuggestedConfig {
  serviceType: 'text' | 'image' | 'video' | 'audio'
  provider: string
  baseUrl: string
  model: string
  runtime: ModelRuntime
  role: 'standalone' | 'component'
  /** 是否可被项目直接调用（false=只读扫描来源，如 ComfyUI，不生成可调用注册） */
  callable?: boolean
  /** 说明/注意事项 */
  note: string
}

export interface ScanOptions {
  /** 扫描根目录（绝对路径）。为空时使用默认候选目录 */
  roots?: string[]
  /** 最大递归深度，默认 5 */
  maxDepth?: number
  /** 最大扫描文件数（含被跳过的非模型文件），默认 8000 */
  maxFiles?: number
  /** 仅返回这些大类 */
  kinds?: ModelKind[]
}

export interface ScanResult {
  /** 实际使用的根目录 */
  roots: string[]
  /** 扫描结果 */
  models: ScannedModel[]
  /** 命中模型总数 */
  total: number
  /** 是否因达到上限被截断 */
  truncated: boolean
  /** 耗时（毫秒） */
  elapsedMs: number
  /** 各类别计数 */
  byKind: Record<ModelKind, number>
}

// ===== 常量 =====

/** 模型权重扩展名（大小写不敏感） */
const MODEL_EXTS = new Set([
  '.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx', '.sft', '.engine',
])

/**
 * 递归时跳过的目录名 —— 仅跳过「非模型」目录。
 * 注意：clip / vae / controlnet / upscale_models / embeddings / photomaker /
 * style_models / ipadapter 等目录下都是模型权重，**不能跳过**，需被扫描识别。
 */
const SKIP_DIRS = new Set([
  'node_modules', '.git', 'venv', '.venv', '__pycache__', '.cache', '.huggingface',
  'output', 'temp', 'input', 'assets', 'metadata', 'samples', 'custom_nodes',
  '.codebuddy', 'python', 'lib', 'site-packages',
])

/**
 * 系统级目录（磁盘/全盘扫描时跳过，避免误入 Windows 系统目录导致
 * 大量 EACCES、杀软拦截、OneDrive 占位文件与无穷递归）。
 * 注意：这里用「目录名小写」匹配，模型几乎不可能放在这些目录下。
 */
const SYSTEM_DIRS = new Set([
  'windows', 'program files', 'program files (x86)', 'programdata',
  '$recycle.bin', 'system volume information', 'recovery', 'perflogs',
  'msocache', 'documents and settings', 'windows.old', 'appdata',
  'onedrive', 'onedrivetemp', 'intel', 'amd', 'nvidia', 'drivers',
])

/** ComfyUI 安装位置探测候选（与 scripts/model_manager.py 保持一致） */
const COMFYUI_CANDIDATES = [
  'D:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI',
  'D:/Comfy-Desktop/ComfyUI-Shared',
  'D:/code/ComfyUI/ComfyUI',
  'D:/code/ComfyUI',
  'D:/ComfyUI/ComfyUI',
  'D:/ComfyUI',
  'C:/ComfyUI',
]

// ===== 识别规则引擎 =====

interface Rule {
  id: string
  kind: ModelKind
  runtime: ModelRuntime
  confidence: Confidence
  /** standalone=可独立调用（如 LLM、主扩散模型、H3 DiT）；component=需在 ComfyUI 工作流中引用 */
  role: 'standalone' | 'component'
  /** 命中则返回 true。p=全路径小写, dir=目录名小写, name=文件名小写, ext=扩展名小写 */
  test: (p: string, dir: string, name: string, ext: string) => boolean
}

/**
 * 规则按优先级从高到低排列，先命中先返回。
 * 设计原则：**不跳过任何模型**。VAE/CLIP/ControlNet/Upscale/Embeddings 等
 * 「组件」也纳入扫描并归类为 image/video，仅 role 标为 component。
 *
 * 顺序关键点：
 *  - H3 的 text_encoders（qwen3vl / qwen_3_4b）属视频，须排在 text LLM 之前；
 *  - Wan/Hunyuan GGUF 须排在通用 GGUF LLM 之前；
 *  - 组件目录（vae/text_encoders/clip…）须排在「目录=LLM」与「diffusion 主权重」之前，
 *    避免 clip/controlnet 被 llm-dir 或主权重规则误吞。
 */
const RULES: Rule[] = [
  // ===== 视频 =====
  // 1. H3 视频 DiT（minimax_h3 系列，可独立驱动生成）
  // 注意排除 qwen 开头的文本编码器（qwen3vl_32b_minimax_h3 是 TE 而非 DiT）
  {
    id: 'h3-dit', kind: 'video', runtime: 'h3', confidence: 'high', role: 'standalone',
    test: (_p, _d, name) => /minimax.?h3/.test(name) && !/vae/.test(name) && !/qwen/.test(name) && !/te\d*$/.test(name),
  },
  // 2. H3 视频 VAE（组件）
  {
    id: 'h3-vae', kind: 'video', runtime: 'h3', confidence: 'high', role: 'component',
    test: (_p, _d, name) => /minimax.?h3/.test(name) && /vae/.test(name),
  },
  // 3. H3 文本编码器（qwen3vl 32b / qwen_3_4b TE，组件）
  {
    id: 'h3-text-encoder', kind: 'video', runtime: 'h3', confidence: 'high', role: 'component',
    test: (_p, dir, name) =>
      (dir === 'text_encoders' && /qwen|minimax/.test(name)) || /qwen3vl/.test(name),
  },
  // 4. Wan / Hunyuan 视频 GGUF（可灵类，可独立驱动）
  {
    id: 'wan-video-gguf', kind: 'video', runtime: 'comfyui', confidence: 'high', role: 'standalone',
    test: (_p, _d, name, ext) => ext === '.gguf' && /wan|hunyuan|kling/.test(name),
  },
  // 5. Wan / Hunyuan 视频 safetensors 主权重
  {
    id: 'wan-video-safetensors', kind: 'video', runtime: 'comfyui', confidence: 'high', role: 'standalone',
    test: (_p, dir, name) =>
      /wan|hunyuan|kling/.test(name) && /diffusion_models|unet/.test(dir),
  },

  // ===== 音频 =====
  // 6. 语音合成（CosyVoice / GPT-SoVITS / VITS 等）
  {
    id: 'tts-service', kind: 'audio', runtime: 'cosyvoice', confidence: 'high', role: 'standalone',
    test: (p, dir, name) => /cosyvoice|cosy|tts|speech|vits|gpt.?sovits|bert.?vits|fish.?speech|chat.?tts|xtts/.test(`${dir} ${name} ${p}`),
  },

  // ===== 文本 =====
  // 7. 独立 LLM GGUF（量化文本模型，可被 Ollama 导入）
  {
    id: 'gguf-llm', kind: 'text', runtime: 'ollama', confidence: 'high', role: 'standalone',
    test: (_p, _d, name, ext) => ext === '.gguf' && !/wan|hunyuan|kling|flux/.test(name),
  },
  // 8. 目录「LLM」或知名 LLM 名（ComfyUI LLM_party 挂载）
  {
    id: 'llm-dir', kind: 'text', runtime: 'comfyui', confidence: 'medium', role: 'standalone',
    test: (p, dir, name) =>
      /\/llm\//.test(p) ||
      dir === 'llm' ||
      /llm|qwen3-?4b|qwen\d+\.?\d*|llama|mistral|phi-?|deepseek|gemma|glm|baichuan|chatglm|yi-?|internlm/.test(`${dir} ${name}`),
  },

  // ===== 图像：主权重 =====
  // 9. diffusion 主权重（checkpoints / diffusion_models / unet / loras）
  {
    id: 'diffusion-weights', kind: 'image', runtime: 'comfyui', confidence: 'high', role: 'standalone',
    test: (_p, dir, name) =>
      /checkpoint|diffusion_models|unet|loras|dora/.test(dir) && !/minimax.?h3/.test(name),
  },

  // ===== 图像：组件 =====
  // 10. ComfyUI 图像生成组件（VAE/CLIP/ControlNet/Upscale/Embeddings…）
  {
    id: 'image-component', kind: 'image', runtime: 'comfyui', confidence: 'high', role: 'component',
    test: (_p, dir) =>
      /vae|clip|clip_vision|text_encoders|controlnet|upscale_models|embeddings|photomaker|insightface|ipadapter|style_models|blip|onnx|flux|t5|hypernetworks/.test(dir),
  },

  // 11. 知名图像模型文件名
  {
    id: 'image-by-name', kind: 'image', runtime: 'comfyui', confidence: 'medium', role: 'standalone',
    test: (_p, dir, name) =>
      /flux|sd.?xl|sdxl|sd15|sd_?1\.?5|z-?image|dreamshaper|realistic.?vision|juggernaut|animagine|pony|illustrious|noobai|majicmix|chilloutmix/.test(`${dir} ${name}`),
  },

  // ===== 兜底 =====
  // 12. 其余 safetensors/ckpt/pt/pth/bin 默认判为图像权重
  {
    id: 'default-weights', kind: 'image', runtime: 'comfyui', confidence: 'low', role: 'standalone',
    test: (_p, _d, name, ext) =>
      ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.onnx', '.sft', '.engine'].includes(ext),
  },
]

/** 按规则命中结果 */
interface RuleHit {
  kind: ModelKind
  runtime: ModelRuntime
  confidence: Confidence
  role: 'standalone' | 'component'
  matchedBy: string[]
}

function classify(p: string): RuleHit {
  const dir = path.basename(path.dirname(p)).toLowerCase()
  const name = path.basename(p).toLowerCase()
  const ext = path.extname(name)
  const full = p.toLowerCase()
  for (const rule of RULES) {
    if (rule.test(full, dir, name, ext)) {
      return {
        kind: rule.kind,
        runtime: rule.runtime,
        confidence: rule.confidence,
        role: rule.role,
        matchedBy: [rule.id],
      }
    }
  }
  return { kind: 'unknown', runtime: 'unknown', confidence: 'low', role: 'standalone', matchedBy: [] }
}

// ===== 推荐配置映射 =====

const RUNTIME_BASE_URLS: Record<ModelRuntime, string> = {
  // ComfyUI 仅作为「只读的可选扫描来源」——只探测/展示其模型列表，
  // 不作为可调用推理后端，故不提供 baseUrl（不生成可调用注册建议）。
  comfyui: '',
  ollama: 'http://localhost:11434',
  'local-sd': 'http://localhost:7860',
  h3: 'http://localhost:8765',
  cosyvoice: 'http://localhost:9880',
  unknown: '',
}

function buildSuggestion(hit: RuleHit, name: string, ext: string): SuggestedConfig | null {
  const model = name.replace(/\.[^.]+$/, '')
  const role = hit.role
  const isComponent = role === 'component'

  // ComfyUI 至多作为「只读的可选扫描来源」：只探测/展示其模型列表，
  // 不生成可调用注册建议（不写入 ai_service_configs，也不作为推理后端）。
  if (hit.runtime === 'comfyui') {
    return {
      serviceType: hit.kind === 'unknown' ? 'image' : hit.kind,
      provider: '',
      baseUrl: '',
      model,
      runtime: 'comfyui',
      role,
      callable: false,
      note: 'ComfyUI 仅作只读扫描来源，不直接调用；如需使用请部署为项目自有服务（H3/local-sd/Ollama/CosyVoice）',
    }
  }

  switch (hit.kind) {
    case 'text':
      // 此处只可能是 ollama（comfyui 已在上方短路）
      return {
        serviceType: 'text', provider: 'openai',
        baseUrl: RUNTIME_BASE_URLS.ollama, model: ext === '.gguf' ? model : name,
        runtime: 'ollama', role, callable: true,
        note: '需先 `ollama create <name> -f Modelfile` 导入该 GGUF',
      }
    case 'image':
      // 当前 image 规则均为 comfyui（已在上方短路）；未来命中 local-sd 规则时走到这里
      return {
        serviceType: 'image', provider: 'local-sd',
        baseUrl: RUNTIME_BASE_URLS['local-sd'], model,
        runtime: 'local-sd', role, callable: true,
        note: '本地 SD WebUI 服务，可直接调用',
      }
    case 'video':
      if (isComponent) {
        return {
          serviceType: 'video', provider: 'minimax',
          baseUrl: RUNTIME_BASE_URLS.h3, model: 'hailuo-02',
          runtime: 'h3', role, callable: true,
          note: 'H3 组件（VAE/文本编码器），由 H3 服务 checkpoint_map 引用，不单独注册',
        }
      }
      return {
        serviceType: 'video', provider: 'minimax',
        baseUrl: RUNTIME_BASE_URLS.h3, model: 'hailuo-02',
        runtime: 'h3', role, callable: true,
        note: '本地 H3 服务，通过 checkpoint_map 路由到该权重',
      }
    case 'audio':
      return {
        serviceType: 'audio', provider: 'cosyvoice',
        baseUrl: RUNTIME_BASE_URLS.cosyvoice, model,
        runtime: 'cosyvoice', role, callable: true,
        note: 'CosyVoice 本地服务',
      }
    default:
      return null
  }
}

// ===== 路径解析（CLI/环境变量 > configs/model-paths.json > 探测） =====

function readPathsConfig(): Record<string, any> {
  try {
    const p = path.join(config.projectRoot, 'configs', 'model-paths.json')
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf-8'))
  } catch { /* 忽略解析失败 */ }
  return {}
}

function detectComfyUI(): string {
  for (const cand of COMFYUI_CANDIDATES) {
    if (cand && fs.existsSync(path.join(cand, 'models'))) return cand
  }
  return ''
}

/** 解析「默认扫描根目录」集合：环境变量 > model-paths.json > 探测候选 */
export function getDefaultRoots(): string[] {
  const cfg = readPathsConfig()
  const roots = new Set<string>()

  const comfyuiRoot = process.env.COMFYUI_PATH || cfg.comfyui_root || detectComfyUI()
  const comfyuiModelsDir = comfyuiRoot ? path.join(comfyuiRoot, 'models') : ''
  const modelsDir = process.env.MODELS_DIR || cfg.models_dir || comfyuiModelsDir
  const servicesDir = process.env.LOCAL_SERVICES_DIR || cfg.local_services_dir || path.join(config.projectRoot, 'local_services')

  // 模型存储目录与 ComfyUI 默认模型目录并存扫描（设置自定义存储目录后，ComfyUI 目录仍会被扫）
  for (const r of [modelsDir, comfyuiModelsDir, servicesDir, comfyuiRoot]) {
    if (r && fs.existsSync(r)) roots.add(path.resolve(r))
  }

  // 用户自定义的额外扫描目录（configs/model-paths.json extra_roots）
  for (const r of getExtraRoots()) {
    if (r && fs.existsSync(r)) roots.add(path.resolve(r))
  }

  // 若完全没有可扫描目录，回退到磁盘常见模型目录，避免空结果
  if (roots.size === 0) {
    for (const cand of COMFYUI_CANDIDATES) {
      if (fs.existsSync(cand)) { roots.add(cand); break }
    }
  }
  return Array.from(roots)
}

/** 读取用户自定义的额外扫描目录（configs/model-paths.json extra_roots） */
export function getExtraRoots(): string[] {
  const cfg = readPathsConfig()
  const extra = cfg.extra_roots
  if (Array.isArray(extra)) return extra.filter((x) => typeof x === 'string' && x.trim())
  return []
}

/** 保存用户自定义的额外扫描目录，返回去重后的规范路径列表 */
export function saveExtraRoots(roots: string[]): string[] {
  const p = path.join(config.projectRoot, 'configs', 'model-paths.json')
  const cfg = readPathsConfig()
  const cleaned = Array.from(new Set(
    (roots || []).map((r) => String(r).trim()).filter(Boolean).map((r) => path.resolve(r)),
  ))
  cfg.extra_roots = cleaned
  fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + '\n', 'utf-8')
  return cleaned
}

/** 读取完整模型路径配置（comfyui_root / models_dir / nodes_dir / local_services_dir / extra_roots） */
export function getModelPaths(): Record<string, any> {
  const cfg = readPathsConfig()
  const comfyuiRoot = process.env.COMFYUI_PATH || cfg.comfyui_root || detectComfyUI()
  return {
    comfyui_root: comfyuiRoot || '',
    models_dir: process.env.MODELS_DIR || cfg.models_dir || '',
    nodes_dir: cfg.nodes_dir || '',
    local_services_dir: cfg.local_services_dir || '',
    extra_roots: getExtraRoots(),
  }
}

/** 保存模型路径配置（合并写 model-paths.json），仅更新传入的字段，空字符串表示清空 */
export function saveModelPaths(patch: Record<string, any>): Record<string, any> {
  const p = path.join(config.projectRoot, 'configs', 'model-paths.json')
  const cfg = readPathsConfig()
  const allowed = ['comfyui_root', 'models_dir', 'nodes_dir', 'local_services_dir']
  for (const k of allowed) {
    if (typeof patch[k] !== 'string') continue
    const v = patch[k].trim()
    if (v) cfg[k] = v
    else delete cfg[k]
  }
  fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + '\n', 'utf-8')
  return cfg
}

// ===== 扫描实现 =====

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = bytes
  let i = -1
  do { v /= 1024; i++ } while (v >= 1024 && i < units.length - 1)
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}

/**
 * 扫描本地模型。roots 为空时使用默认候选目录。
 * 单个根目录递归扫描，带深度/数量上限，跳过无关目录。
 */
export function scanLocalModels(opts: ScanOptions = {}): ScanResult {
  const start = Date.now()
  const maxDepth = opts.maxDepth ?? 5
  const maxFiles = opts.maxFiles ?? 8000
  const kindsFilter = opts.kinds && opts.kinds.length ? new Set(opts.kinds) : null

  const roots = (opts.roots && opts.roots.length ? opts.roots : getDefaultRoots())
    .map((r) => path.resolve(r))
    .filter((r) => fs.existsSync(r))

  const models: ScannedModel[] = []
  let fileCount = 0
  let truncated = false

  const seen = new Set<string>()
  // 已访问目录（realpath 去重，防 junction/符号链接死循环与重复扫描）
  const visitedDirs = new Set<string>()
  for (const root of roots) {
    try { visitedDirs.add(fs.realpathSync(root)) } catch { /* 忽略 */ }
  }

  function walk(dir: string, depth: number): void {
    if (truncated || depth > maxDepth) return
    let entries: fs.Dirent[]
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch { return }

    for (const ent of entries) {
      if (truncated || fileCount >= maxFiles) { truncated = true; return }
      const full = path.join(dir, ent.name)

      if (ent.isDirectory()) {
        const low = ent.name.toLowerCase()
        if (SKIP_DIRS.has(low) || SYSTEM_DIRS.has(low)) continue
        if (ent.isSymbolicLink()) continue // 不跟随符号链接/junction
        let real = full
        try { real = fs.realpathSync(full) } catch { continue }
        if (visitedDirs.has(real)) continue
        visitedDirs.add(real)
        walk(full, depth + 1)
        continue
      }

      fileCount++
      const ext = path.extname(ent.name).toLowerCase()
      if (!MODEL_EXTS.has(ext)) continue

      // 去重（同一文件通过不同根目录重复出现）
      const key = full.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)

      let sizeBytes = 0
      try { sizeBytes = fs.statSync(full).size } catch { /* 忽略 */ }

      const hit = classify(full)
      if (kindsFilter && !kindsFilter.has(hit.kind)) continue

      models.push({
        path: full,
        filename: ent.name,
        dirname: path.basename(dir),
        sizeBytes,
        sizeHuman: humanSize(sizeBytes),
        ext,
        kind: hit.kind,
        runtime: hit.runtime,
        confidence: hit.confidence,
        role: hit.role,
        matchedBy: hit.matchedBy,
        suggested: buildSuggestion(hit, ent.name, ext),
      })
    }
  }

  for (const root of roots) walk(root, 0)

  const byKind: Record<ModelKind, number> = { text: 0, image: 0, video: 0, audio: 0, unknown: 0 }
  for (const m of models) byKind[m.kind]++

  // 排序：video > image > text > audio > unknown，同类按大小降序
  const order: Record<ModelKind, number> = { video: 0, image: 1, text: 2, audio: 3, unknown: 4 }
  models.sort((a, b) =>
    order[a.kind] - order[b.kind] || b.sizeBytes - a.sizeBytes)

  return {
    roots,
    models,
    total: models.length,
    truncated,
    elapsedMs: Date.now() - start,
    byKind,
  }
}

/** 枚举当前机器可用盘符（Windows 探测 A-Z） */
export function listDrives(): { letter: string; root: string }[] {
  const drives: { letter: string; root: string }[] = []
  for (let c = 65; c <= 90; c++) {
    const letter = String.fromCharCode(c)
    const root = `${letter}:\\`
    try {
      if (fs.existsSync(root)) drives.push({ letter, root })
    } catch { /* 忽略无介质/权限盘符 */ }
  }
  return drives
}

/** 异步扫描进度快照 */
export interface ScanProgress {
  scannedFiles: number
  foundModels: number
  currentDir: string
  done: boolean
  cancelled: boolean
  error?: string
  result?: ScanResult
}

/** 异步扫描选项（在 ScanOptions 基础上增加进度与取消回调） */
export interface AsyncScanOptions extends ScanOptions {
  onProgress?: (p: ScanProgress) => void
  shouldCancel?: () => boolean
}

/** 扫描被取消时抛出的错误 */
export class ScanCancelledError extends Error {
  constructor() {
    super('scan cancelled')
    this.name = 'ScanCancelledError'
  }
}

/**
 * 异步版本地模型扫描：使用 fs.promises 递归，扫描期间不阻塞事件循环，
 * 支持进度回调与取消（shouldCancel 返回 true 时中止）。适合磁盘/全盘级大范围扫描。
 */
export async function scanLocalModelsAsync(opts: AsyncScanOptions = {}): Promise<ScanResult> {
  const start = Date.now()
  const maxDepth = opts.maxDepth ?? 8
  const maxFiles = opts.maxFiles ?? 50000
  const kindsFilter = opts.kinds && opts.kinds.length ? new Set(opts.kinds) : null
  const onProgress = opts.onProgress
  const shouldCancel = opts.shouldCancel

  const roots = (opts.roots && opts.roots.length ? opts.roots : getDefaultRoots())
    .map((r) => path.resolve(r))
    .filter((r) => fs.existsSync(r))

  const models: ScannedModel[] = []
  let fileCount = 0
  let truncated = false
  let cancelled = false

  const seen = new Set<string>()
  const visitedDirs = new Set<string>()
  for (const root of roots) {
    try { visitedDirs.add(fs.realpathSync(root)) } catch { /* 忽略 */ }
  }

  const emit = (currentDir: string) => {
    onProgress?.({
      scannedFiles: fileCount,
      foundModels: models.length,
      currentDir,
      done: false,
      cancelled,
    })
  }

  async function walk(dir: string, depth: number): Promise<void> {
    if (truncated || cancelled || depth > maxDepth) return
    if (shouldCancel?.()) { cancelled = true; return }
    let entries: fs.Dirent[]
    try {
      entries = await fs.promises.readdir(dir, { withFileTypes: true })
    } catch { return }

    for (const ent of entries) {
      if (truncated || cancelled || fileCount >= maxFiles) { truncated = true; return }
      if (shouldCancel?.()) { cancelled = true; return }
      const full = path.join(dir, ent.name)

      if (ent.isDirectory()) {
        const low = ent.name.toLowerCase()
        if (SKIP_DIRS.has(low) || SYSTEM_DIRS.has(low)) continue
        if (ent.isSymbolicLink()) continue
        let real = full
        try { real = fs.realpathSync(full) } catch { continue }
        if (visitedDirs.has(real)) continue
        visitedDirs.add(real)
        await walk(full, depth + 1)
        continue
      }

      fileCount++
      if (fileCount % 1000 === 0) emit(dir)
      const ext = path.extname(ent.name).toLowerCase()
      if (!MODEL_EXTS.has(ext)) continue

      const key = full.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)

      let sizeBytes = 0
      try { sizeBytes = (await fs.promises.stat(full)).size } catch { /* 忽略 */ }

      const hit = classify(full)
      if (kindsFilter && !kindsFilter.has(hit.kind)) continue

      models.push({
        path: full,
        filename: ent.name,
        dirname: path.basename(dir),
        sizeBytes,
        sizeHuman: humanSize(sizeBytes),
        ext,
        kind: hit.kind,
        runtime: hit.runtime,
        confidence: hit.confidence,
        role: hit.role,
        matchedBy: hit.matchedBy,
        suggested: buildSuggestion(hit, ent.name, ext),
      })
    }
  }

  for (const root of roots) {
    if (cancelled || truncated) break
    await walk(root, 0)
  }

  if (cancelled) throw new ScanCancelledError()

  const byKind: Record<ModelKind, number> = { text: 0, image: 0, video: 0, audio: 0, unknown: 0 }
  for (const m of models) byKind[m.kind]++

  const order: Record<ModelKind, number> = { video: 0, image: 1, text: 2, audio: 3, unknown: 4 }
  models.sort((a, b) =>
    order[a.kind] - order[b.kind] || b.sizeBytes - a.sizeBytes)

  return {
    roots,
    models,
    total: models.length,
    truncated,
    elapsedMs: Date.now() - start,
    byKind,
  }
}

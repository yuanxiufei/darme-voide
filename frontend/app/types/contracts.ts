/**
 * **前端侧**共享契约类型（HTTP 传输层请求/响应 DTO）
 *
 * ⚠️ 2026-09-15 从 `backend/src/shared/contracts.ts` 迁到这里：那个位置属于 **TS 旧后端**
 * （`backend/` 已完成使命、待删），留在那儿会让**前端构建失败**（`~contracts` 别名指过去）。
 * 现在本文件属于**前端**，通过 nuxt alias `~contracts` 被 `app/composables/useApi.ts` 引用。
 *
 * 字段**权威在后端**（Python：`backend-py/app/` 的模型与响应层）—— 本文件是前端侧镜像，
 * 后端改字段时**必须同步这里**（这条不变量和「画风词表两处同步」同类）。
 *
 * 规则：
 * 1. 本文件只允许 export type / interface，禁止任何运行时逻辑与第三方依赖
 *    （保证前端可安全 `import type` 而不引入后端依赖树）。
 * 2. HTTP 传输字段统一 **snake_case**（对齐后端 toSnakeCase 输出与 DB 列名）。
 * 3. 新增 DTO 时保持最小可用，不照抄 DB 全列——只声明确实被消费的字段。
 */
export {}

/** 剧集时代背景（dramas.era_background 落库 JSON 的规范结构） */
export interface EraBackground {
  /** 时代标签（中文，简短），如：古代仙侠 / 民国谍战 / 现代都市·赛博朋克 */
  era: string
  /** 中文概述：世界观 / 地域 / 年代 / 社会风貌 / 常见场景 */
  summary: string
  /** 英文画面指令：注入角色/场景等文生图 prompt，保证全剧时代感一致 */
  imageHint: string
}

/** PUT /dramas/:id 更新请求体（部分字段，未列出即不更新） */
export interface DramaUpdateBody {
  title?: string
  description?: string | null
  genre?: string | null
  style?: string | null
  status?: string | null
  tags?: string[]
  metadata?: string | null
  /** 时代背景 JSON 文本（EraBackground 序列化）；空串 '' 表示清空 */
  era_background?: string | null
}

/** GET /dramas/:id 返回的剧集详情 DTO（snake_case） */
export interface DramaDetailDTO {
  id: number
  title: string
  description: string | null
  genre: string | null
  style: string | null
  style_id: string | null
  /** 时代背景原始 JSON 文本，消费方用 JSON.parse 后得到 EraBackground；未提炼为 null */
  era_background: string | null
  total_episodes: number | null
  total_duration: number | null
  status: string
  thumbnail: string | null
  tags: string[]
  metadata: string | null
  created_at: string
  updated_at: string
  deleted_at: string | null
  // 嵌套集合暂未契约化（toSnakeCaseArray 产物，字段随表新增会漂移）。
  // 出于务实先用 any[] 兜底，等子 DTO（Episode/Character/Scene）契约化时逐个收紧。
  episodes: any[]
  characters: any[]
  scenes: any[]
}

/** 剧集生产进度统计（GET /dramas 列表项附带的聚合计数） */
export interface DramaProgress {
  total_episodes: number
  scripted_episodes: number
  storyboarded_episodes: number
  storyboards: number
  images: number
  videos: number
  tts: number
}

/** GET /dramas 列表项：详情 DTO 超集 + 进度统计 */
export interface DramaListItemDTO extends DramaDetailDTO {
  progress: DramaProgress
}

/** GET /dramas 列表响应体 */
export interface DramaListResponse {
  items: DramaListItemDTO[]
  pagination: { page: number; page_size: number; total: number; total_pages: number }
}

/**
 * 技能列表项（GET /skills ✓）
 *
 * ⚠️ 2026-09-21 补：此前前端这几处都是 `ref([])` ✗ ⇒ 推出 **`never[]`** ✗
 * （元素是 `never` ⇒ 每个属性访问都报 TS2339 ✓）。打开类型检查后一次冒出来几百条 ✓✗
 * —— 这类"不报错但全是 never"的形状**只有真跑类型检查才看得见** ✗。
 * 字段按 **界面实际消费** 声明 ✓（本文件规则 3 ✓）。
 */
export interface SkillVO {
  id: string
  name: string
  description?: string | null
  category?: string | null
  source?: string | null
  /** 外部技能库来源标签（取 `library.yaml` 的 label）；非 vendor 为空 */
  sourceLabel?: string | null
  /** 注入体量（字符数，与后端预算闸同口径；缺失按 0 计） */
  charCount?: number
  /** 声明依赖、但本项目（宿主）未提供的工具（后端 `missingTools` 直出） */
  missingTools?: string[]
  /** 正文引用但**不会被注入**的参考文件数（加载器只读 `SKILL.md`） */
  referenceCount?: number
  /** 项目自带 Skill：受删除保护（停用要去 Agent 配置里取消勾选） */
  protected?: boolean
  /** 绑定了该 Skill 的 Agent 类型列表 */
  boundAgents?: string[]
  /** 适用阶段（如 `script` / `storyboard`） */
  phases?: string[]
}

/** Agent 的 Skill 绑定项（`agent_configs.skills` JSON 内的元素；前端 UI 也用它） */
export interface SkillBinding {
  /** 指向 SkillVO.id */
  id: string
  /** 只有 enabled 的会真正注入 */
  enabled: boolean
  /** 注入顺序（越小越前；保存时按列表顺序重写） */
  priority: number
}

/** Agent 配置项（GET /agent-configs ✓） */
export interface AgentConfigVO {
  id: number
  /** 后端 Agent 类型键（如 `script_rewriter`） */
  agent_type: string
  name?: string | null
  model?: string | null
  temperature?: number | null
  max_tokens?: number | null
  system_prompt?: string | null
  /** Skill 绑定 JSON **文本**（消费方 JSON.parse 成 `SkillBinding[]`） */
  skills?: string | null
}

/** AI 服务配置项（GET /ai-configs ✓） */
export interface AiServiceConfigVO {
  id: number
  /** `text` / `image` / `video` / `audio` … */
  service_type: string
  name: string
  provider?: string | null
  model?: string | null
  api_key?: string | null
  base_url?: string | null
  is_active?: boolean
  /** 本地服务（Ollama 等）：界面用它把「在线/本地」两栏分开 */
  is_local?: boolean
  /** 同类型内的优先级（越小越前） */
  priority?: number
  negative_prompt?: string | null
  /** 视频双权重映射（fl2va = 首尾帧、ref2va = 参考图） */
  checkpoint_map?: { fl2va?: string | null; ref2va?: string | null } | null
}

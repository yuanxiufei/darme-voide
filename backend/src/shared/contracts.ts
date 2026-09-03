/**
 * 前后端共享契约（single source of truth）
 *
 * 用途：HTTP 传输层的请求/响应 DTO 唯一定义处。
 * 后端路由/服务 import type 校验自己的实现；前端 useApi.ts 通过
 * nuxt alias `~contracts` 引用同一份类型，消灭字段双写与漂移。
 *
 * 规则：
 * 1. 本文件只允许 export type / interface，禁止任何运行时逻辑与第三方依赖，
 *    保证前后端可安全地 `import type` 而不引入对方依赖树。
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

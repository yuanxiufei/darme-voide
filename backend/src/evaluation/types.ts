/**
 * 评测闭环类型定义
 *
 * 对齐 PenguinHarness 的 benchmark-design 思想：
 * - statement（公开）：喂给 Agent 的输入（剧本 + 角色 + 场景）
 * - rubric（私密）：金标准答案 + 确定性评分规则，优化器不得读取
 */

/** storyboard_breaker 输出的一组分镜字段（对应 save_storyboards 的入参） */
export interface StoryboardShot {
  shot_number?: number
  title?: string
  shot_type?: string
  angle?: string
  movement?: string
  location?: string
  time?: string
  action?: string
  dialogue?: string
  description?: string
  result?: string
  atmosphere?: string
  image_prompt?: string
  video_prompt?: string
  bgm_prompt?: string
  sound_effect?: string
  duration?: number
  scene_id?: number | null
  character_ids?: number[]
}

/** extractor 输出的角色（对应 save_dedup_characters 的入参） */
export interface ExtractedCharacter {
  name?: string
  role?: string
  description?: string
  appearance?: string
  personality?: string
}

/** extractor 输出的场景（对应 save_dedup_scenes 的入参） */
export interface ExtractedScene {
  location?: string
  time?: string
  prompt?: string
}

export interface StoryboardRubric {
  /** 镜头数量合理区间 */
  minShots: number
  maxShots: number
  /** 每个镜头必须完整填写的字段 */
  requiredFields: string[]
  /** video_prompt 必须包含的标记 */
  videoPromptTags: string[]
  /** duration 合法区间（秒） */
  durationRange: [number, number]
  /** title 合法长度区间（字符数） */
  titleLengthRange: [number, number]
}

export interface ExtractorRubric {
  /** 金标准角色名（剧本中真实出现、应被提取） */
  goldenCharacters: string[]
  /** 金标准场景（地点+时间段） */
  goldenScenes: Array<{ location: string; time: string }>
  /** 外貌描述最短长度（低于视为不完整） */
  minAppearanceLength: number
  /** 场景 prompt 最短长度 */
  minPromptLength: number
}

/** script_rewriter 输出的格式化剧本（save_script 的 content 入参） */
export interface ScriptRewriterRubric {
  /** 最少场景数（场景头 `## S编号 | ... | ...` 的数量下限） */
  minScenes: number
  /** 禁止出现的镜头语言关键词（景别/运镜等属于分镜步骤，不应出现在剧本） */
  forbiddenCameraWords: string[]
}

/** voice_assigner 输出的音色分配（assign_voice 的入参） */
export interface VoiceAssignerRubric {
  /** 合法音色 ID 集合（来自 list_voices 返回的音色库） */
  legalVoiceIds: string[]
  /** 是否要求分配时说明理由 */
  requireReason: boolean
}

export interface StoryboardCase {
  id: string
  kind: 'storyboard'
  statement: {
    script: string
    characters: Array<{ name: string; role: string; appearance: string }>
    scenes: Array<{ location: string; time: string; prompt: string }>
  }
  rubric: StoryboardRubric
}

export interface ExtractorCase {
  id: string
  kind: 'extractor'
  statement: {
    script: string
  }
  rubric: ExtractorRubric
}

export interface ScriptRewriterCase {
  id: string
  kind: 'script_rewriter'
  statement: {
    /** 原始内容（一句话梗概或小说片段，需改写为格式化剧本） */
    content: string
  }
  rubric: ScriptRewriterRubric
}

export interface VoiceAssignerCase {
  id: string
  kind: 'voice_assigner'
  statement: {
    /** 剧本（用于理解角色性格） */
    script: string
    /** 待分配音色的角色（seed 阶段写入 characters 表） */
    characters: Array<{ name: string; role: string; personality: string }>
  }
  rubric: VoiceAssignerRubric
}

export type BenchmarkCase = StoryboardCase | ExtractorCase | ScriptRewriterCase | VoiceAssignerCase

export interface ScoreDimension {
  name: string
  score: number
  max: number
  detail: string
}

export interface ScoreReport {
  caseId: string
  kind: string
  dimensions: ScoreDimension[]
  /** 归一化总分 0-100 */
  total: number
  /** 实际评测使用的模型（供 runtime 一致性校验：候选与 Reference 必须同模型，否则分数不可比） */
  runtimeModel?: string
}

/** 优化器历史记录中的单次迭代 */
export interface OptimizationIteration {
  version: number
  score: number
  accepted: boolean
  prompt: string
  dimensions: ScoreDimension[]
}

export interface OptimizationHistory {
  agentType: string
  caseId: string
  reference: { score: number; dimensions: ScoreDimension[]; prompt: string }
  iterations: OptimizationIteration[]
  best: { version: number; score: number; prompt: string }
  /** 自动落库结果（autoPersist 开启且 best 严格高于 Reference 时非空；未开启或未达标为 null） */
  persisted?: { version: number; score: number; name: string } | null
}

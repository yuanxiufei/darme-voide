/**
 * 逐镜路由决策（Shot Routing）——对齐参考项目 H3-Codex-Drama「逐镜路由」机制
 *
 * 每个分镜在提交视频生成前，显式决策采用哪条生成路线（T2V / I2V / FL2VA / R2V），
 * 并记录决策结果与原因（写入 storyboards.route / route_reason，以及 video_generations 快照）。
 *
 * 路由类型（对齐 H3-Codex-Drama 的 text-to-video / frame-to-video / reference-to-video）：
 * - text_to_video        (T2V)：纯文本提示词生成，无任何参考帧
 * - first_frame_to_video (I2V)：以首帧图为起帧的图生视频（本系统默认路线）
 * - first_last_frame     (FL2VA)：首帧 + 尾帧连接动画化（锁定结束画面）
 * - reference_to_video   (R2V)：多参考图（角色立绘/场景图/关键帧）+ 可选参考音频（H3 Ref2VA）
 * - keyframe_to_video    (I2V-K)：以中段关键帧图为主要参考的图生视频
 * - video_editor         (Editor)：基于已有视频的精确修改（本系统暂不自动触发，保留类型）
 * - blocked              资产门禁阻断（无首帧且标记 needs_regeneration），不提交生成
 */
import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { now } from '../utils/response.js'
import { logTaskProgress } from '../utils/task-logger.js'

export const SHOT_ROUTES = [
  'text_to_video',
  'first_frame_to_video',
  'first_last_frame',
  'reference_to_video',
  'keyframe_to_video',
  'video_editor',
  'blocked',
] as const

export type ShotRoute = (typeof SHOT_ROUTES)[number]

export interface RouteDecision {
  /** 决策选中的路由 */
  route: ShotRoute
  /** 决策原因（人类可读，写入 route_reason 供可复现账本追溯） */
  reason: string
  /** 决策对应的 reference_mode（映射到生成请求参数） */
  referenceMode: 'none' | 'single' | 'multiple'
}

/** 对话/多人/争吵类场景关键词（R2V 多参考图适用） */
const DIALOGUE_PATTERN = /dialogue|meeting|argument|conversation|multi|multi-person|多人|对话|争吵|会议/i

export interface RouteDecisionInput {
  /** 分镜记录（route/route_reason 将被回写） */
  storyboardId: number
  sceneType?: string | null
  firstFrameImage?: string | null
  lastFrameImage?: string | null
  keyframeImage?: string | null
  /** 资产门禁是否阻断（无首帧且 needs_regeneration） */
  blocked?: boolean
  /** 视频提供商（小写）；决定是否支持多参考图 */
  provider: string
  /** 提供商是否支持多参考图 */
  canMultiRef: boolean
  /** 已收集的角色/场景参考图 */
  referenceImages: string[]
  /** H3 Ref2VA 参考音频 */
  referenceAudioUrls: string[]
  /** 同场景顺接时上一镜尾帧（非空表示首帧衔接场景延续） */
  prevTail?: string | null
}

/**
 * 逐镜路由决策：按优先级返回 { route, reason, referenceMode }，并回写分镜的 route 字段。
 *
 * 优先级（高 → 低）：
 * 1. blocked：资产门禁阻断
 * 2. text_to_video：无首帧（纯文本生成）
 * 3. first_last_frame：有尾帧目标（FL2VA 锁定结束画面）
 * 4. reference_to_video：对话/多人场景 + 多参考提供商 + 有参考图，或 H3 有参考音频（Ref2VA）
 * 5. keyframe_to_video：有中段关键帧图
 * 6. first_frame_to_video：默认（首帧图生视频）
 */
export function decideShotRoute(input: RouteDecisionInput): RouteDecision {
  const {
    storyboardId,
    sceneType,
    firstFrameImage,
    lastFrameImage,
    keyframeImage,
    blocked,
    provider,
    canMultiRef,
    referenceImages,
    referenceAudioUrls,
  } = input

  // 1. 资产门禁阻断
  if (blocked) {
    const decision: RouteDecision = {
      route: 'blocked',
      reason: 'first frame asset needs_regeneration，视频生成被资产门禁阻断',
      referenceMode: 'none',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  // 2. 无首帧：只能纯文本生成（T2V）
  if (!firstFrameImage) {
    const decision: RouteDecision = {
      route: 'text_to_video',
      reason: '分镜无首帧图，采用纯文本提示词生成（T2V）',
      referenceMode: 'none',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  // 3. 有尾帧目标：首尾帧连接（FL2VA，锁定结束画面）
  if (lastFrameImage) {
    const decision: RouteDecision = {
      route: 'first_last_frame',
      reason: `分镜配置了尾帧目标，采用首尾帧连接动画化（FL2VA）以锁定结束画面`,
      referenceMode: 'single',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  const isDialogue = DIALOGUE_PATTERN.test(sceneType || '')

  // 4. R2V：对话/多人场景 + 多参考提供商 + 有参考图
  if (canMultiRef && referenceImages.length > 0 && isDialogue) {
    const decision: RouteDecision = {
      route: 'reference_to_video',
      reason: `对话/多人场景（${sceneType || 'dialogue'}）且提供商 ${provider} 支持多参考图，采用 R2V（角色+场景参考图 ${referenceImages.length} 张）`,
      referenceMode: 'multiple',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  // 4b. H3 Ref2VA：有参考音频（出场角色声线样本）
  if (provider === 'minimax' && referenceAudioUrls.length > 0) {
    const decision: RouteDecision = {
      route: 'reference_to_video',
      reason: `提供商 minimax 带参考音频（Ref2VA，${referenceAudioUrls.length} 条声线样本），采用 R2V 音视频联合生成`,
      referenceMode: referenceImages.length ? 'multiple' : 'single',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  // 5. 关键帧参考
  if (keyframeImage) {
    const decision: RouteDecision = {
      route: 'keyframe_to_video',
      reason: '分镜配置了中段关键帧图，以其为主要参考锁定动作/道具/机位中间态',
      referenceMode: canMultiRef && referenceImages.length ? 'multiple' : 'single',
    }
    persistRoute(storyboardId, decision)
    return decision
  }

  // 6. 默认：首帧图生视频
  const decision: RouteDecision = {
    route: 'first_frame_to_video',
    reason: `默认路线：以首帧图起帧的图生视频（I2V）${prevTailHint(input)}`,
    referenceMode: 'single',
  }
  persistRoute(storyboardId, decision)
  return decision
}

/** 顺接提示（决策原因附加信息） */
function prevTailHint(input: RouteDecisionInput): string {
  return input.prevTail ? '；同场景顺接，以上一镜尾帧衔接起帧' : ''
}

/** 回写分镜 route / route_reason（幂等，供可复现账本与前端展示） */
function persistRoute(storyboardId: number, decision: RouteDecision): void {
  try {
    db.update(schema.storyboards)
      .set({ route: decision.route, routeReason: decision.reason, updatedAt: now() })
      .where(eq(schema.storyboards.id, storyboardId))
      .run()
    logTaskProgress('ShotRouter', 'route-decided', { storyboardId, route: decision.route })
  } catch (err: any) {
    // 决策记录失败不阻断生成（尽力而为）
    logTaskProgress('ShotRouter', 'route-persist-failed', { storyboardId, error: err.message })
  }
}

/** 批量计算某集所有分镜的路由（供启动/展示使用；不覆盖已有 route 以外的逻辑） */
export function recomputeEpisodeRoutes(episodeId: number): void {
  const sbs = db.select().from(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).all()
  for (const sb of sbs) {
    decideShotRoute({
      storyboardId: sb.id,
      sceneType: sb.sceneType,
      firstFrameImage: sb.firstFrameImage,
      lastFrameImage: sb.lastFrameImage,
      keyframeImage: sb.keyframeImage,
      provider: 'default',
      canMultiRef: false,
      referenceImages: [],
      referenceAudioUrls: [],
    })
  }
}

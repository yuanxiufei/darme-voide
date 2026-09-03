/**
 * 统一视觉 Prompt 工具
 * 
 * 解决全项目 prompt 风格碎片化问题：
 * - 角色图片：cinematic illustration style
 * - 场景图片：高质量场景, 电影感
 * - 网格图片：cinematic lighting, high quality
 * - 视频 prompt：无固定风格
 * 
 * 所有生成路径统一使用本模块的风格预设和 prompt 构建器
 */

import { db, schema } from '../db/index.js'
import { and, eq, inArray, isNull } from 'drizzle-orm'
import { getCameraMovementComposition } from './camera-movement-guides.js'
import { resolveVisualTerm } from './visual-graph.js'

// ============================================================
// 风格预设常量（全域一致）
// ============================================================

/**
 * 主视觉风格标签 - 所有角色/场景/分镜图像生成共用
 */
export const VISUAL_STYLE_MASTER = 'cinematic illustration style, consistent art style, soft cinematic lighting, high quality, no text, no watermark'

/**
 * 角色立绘专用风格标签
 */
export const VISUAL_STYLE_CHARACTER = 'stylized character design, front-facing character portrait, clean white background, illustration style, not photorealistic'

/**
 * 场景背景专用风格标签
 */
export const VISUAL_STYLE_SCENE = 'highly detailed cinematic environment, atmospheric lighting, movie quality composition, depth of field'

/**
 * 视频生成风格引导前缀
 *
 * 对齐参考项目 Mx-Shell 质感层规范：
 * - 真实镜头锚点（camera lens profile）：35mm/50mm 电影镜头质感、浅景深
 * - 现实瑕疵锚点（imperfection anchors）：克制真实、不堆特效
 * - 克制结尾（restrained ending）：不追求爆炸/胜利姿势
 * - 声音策略（sound policy）：画面内不包含背景音乐（后期统一配乐）
 */
export const VISUAL_STYLE_VIDEO =
  'cinematic motion, smooth camera movement, consistent character design, lighting continuity, ' +
  'real lens feel with natural depth of field (35mm/50mm cinematic lens), subtle film grain, ' +
  'grounded realistic texture with small natural imperfections, restrained ending (no explosion, no victory pose, no text overlay)'

/**
 * UI 屏幕留白规则（对齐参考项目 ui_plate 资产约束）：
 * 视频画面中的屏幕/UI 元素（手机/新闻/短信/监控/地图/电脑/文件/时间码）只保留留白，
 * 禁止生成可读文字/图标，所有文字一律后期叠加，防止 AI 生成乱码文字。
 */
export const UI_OVERLAY_RULE =
  'screens and displays (phone/computer/news/SMS/surveillance/map/clock) show only blank or neutral panels: no readable text, no legible characters, no app icons with text; all on-screen text is added in post-production'

/** 屏幕/UI 元素关键词（用于检测分镜描述是否含屏幕画面） */
const SCREEN_ELEMENT_PATTERN =
  /手机|短信|屏幕|显示器|监控|地图|电脑|平板|文件|时间码|新闻|聊天|打字|发消息|phone|screen|display|monitor|surveillance|map|computer|tablet|text message|news|sms|chat/i

/** 检测文本中是否包含屏幕/UI 元素 */
export function containsScreenElement(text: string | null | undefined): boolean {
  if (!text) return false
  return SCREEN_ELEMENT_PATTERN.test(text)
}

/** 屏幕留白图类别名（props 物品库用） */
export const UI_PLATE_CATEGORY = '屏幕留白'

/** 构建屏幕留白图（ui_plate）prompt：供后期叠加文字 */
export function buildUiPlateImagePrompt(plate: {
  type?: string | null
  context?: string | null
}): string {
  const type = plate.type || 'phone screen'
  const parts: string[] = []
  parts.push(`blank ${type} plate, no text, no readable characters, no icons with text`)
  if (plate.context) parts.push(`context: ${plate.context}`)
  parts.push('flat neutral screen surface ready for post-production text overlay, empty UI layout')
  return `${parts.join(', ')}, ${VISUAL_STYLE_SCENE}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 通用预设类型定义（Preset Framework Types）
// ============================================================

/** Variation Card 的单个 Shot 配置 */
export interface PresetVariationCardShot {
  shotIndex: number
  themeFamily: string
  compositionPattern: string
  spaceType: string
  foregroundFrame: string
  mainFocalPoint: string
  thematicClue: string
  activity: string
  cameraPosition: string
  windDirection: string
  lightStructure: string
  characterLayout: string
  livingElement: string | null
}

/** Variation Card（5 镜配置） */
export interface PresetVariationCard {
  themeFamily: string
  shots: PresetVariationCardShot[]
}

// ============================================================
// 统一负面提示词（Negative Prompt）体系
// 每一条正向提示词都配套对应的反向提示词，明确排除不需要的内容/风格/元素。
// 按生成类型拆分为针对性预设，各调用方在构建正向 prompt 时成对引用。
// ============================================================

/**
 * 通用负面基础（所有生成类型共用）
 * 排除低质、常见瑕疵与文字/水印等通用干扰项
 */
export const NEGATIVE_BASE = 'text, watermark, signature, logo, subtitles, low quality, blurry, out of focus, pixelated, jpeg artifacts, oversaturated, distorted, deformed, disfigured, bad anatomy, extra limbs, extra fingers, mutated hands, bad proportions, duplicated elements'

/**
 * 角色立绘负面提示词
 * 排除写实照片质感（角色走 stylized 插画风）、杂乱/多人背景、角色重复与裁切
 */
export const CHARACTER_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, photorealistic, realistic photo, 3d render, cluttered background, busy background, multiple characters, multiple people, duplicated character, inconsistent character, cropped head, cut off face`

// ============================================================
// 画风统一（dramas.style → 统一英文画风描述）
//
// 根因：角色立绘此前不读 dramas.style，且 customPrompt 会完全绕过
// 统一风格标签，导致同一剧内不同角色时而动漫、时而真人。
// 现由本映射把「剧集视觉风格」注入所有角色图 prompt 与负面提示词，
// 与分镜/场景/视频的 dramaStyle 注入对齐，保证全链路画风一致。
// ============================================================

/** 戏剧视觉风格 → 英文画风描述（对齐前端创建/编辑剧集时的 6 选） */
export const DRAMA_ART_STYLE_MAP: Record<string, string> = {
  realistic: 'realistic cinematic character design, natural skin texture, film quality lighting',
  anime: '2D anime illustration style, clean line art, cel shading, vibrant colors',
  ghibli: 'ghibli-inspired hand-drawn animation style, soft watercolor background, gentle rounded character design',
  cinematic: 'cinematic film still style, dramatic moody lighting, shallow depth of field',
  comic: 'comic book illustration style, bold ink line work, halftone shading, dynamic comic character design',
  watercolor: 'watercolor painting style, soft brush strokes, translucent washes, delicate pastel palette',
}

/** 各画风对应的对立风格负面词（防止模型混用动漫/真人） */
const DRAMA_ART_NEGATIVE_MAP: Record<string, string> = {
  realistic: 'anime style, cartoon, illustration, cel shading, line art, 3d render',
  anime: 'photorealistic, realistic photo, live action, 3d render',
  ghibli: 'photorealistic, realistic photo, live action, 3d render',
  cinematic: 'anime style, cartoon, illustration, flat coloring',
  comic: 'photorealistic, realistic photo, live action',
  watercolor: 'photorealistic, realistic photo, live action, harsh outline',
}

/**
 * 构建角色图统一画风后缀（追加在 prompt 末尾）。
 * - 未指定 dramaStyle：维持默认 stylized illustration（not photorealistic）
 * - 指定 dramaStyle：优先使用映射画风（anime/realistic/ghibli...），同剧强制一致
 */
export function buildCharacterArtStyleSuffix(dramaStyle?: string | null): string {
  const art = (dramaStyle && DRAMA_ART_STYLE_MAP[dramaStyle]) || ''
  if (!art) {
    return `, ${VISUAL_STYLE_CHARACTER}, ${VISUAL_STYLE_MASTER}`
  }
  return `, ${art}, cinematic illustration style, consistent art style, soft cinematic lighting, high quality, no text, no watermark`
}

/**
 * 装备/服装/武器/首饰图的画风尾（无人物语义）。
 * DRAMA_ART_STYLE_MAP 是为角色像准备的（含 character design / skin texture），
 * 若直接用于「纯物品」三视图会诱导模型生成人物或穿着人体的服装，故单独定义。
 */
const EQUIP_ART_STYLE_MAP: Record<string, string> = {
  realistic: 'professional product photography, studio softbox lighting, clean neutral background, sharp fabric and metal texture detail',
  anime: '2D anime prop illustration style, clean line art, cel shading, vibrant colors',
  ghibli: 'ghibli-inspired hand-drawn art style, soft colors, gentle shading',
  cinematic: 'cinematic concept art style, dramatic moody lighting, shallow depth of field',
  comic: 'comic book illustration style, bold ink line work, halftone shading',
  watercolor: 'watercolor painting style, soft brush strokes, translucent washes, delicate pastel palette',
}

/** 装备图统一画风/无人物后缀：追加在 equip prompt 末尾 */
export function buildEquipArtStyleSuffix(dramaStyle?: string | null): string {
  const art = (dramaStyle && EQUIP_ART_STYLE_MAP[dramaStyle]) || ''
  const base = ', isolated product shot, single object, empty scene, no people, no person, no model, no mannequin body, no body parts, no skin'
  if (!art) return `${base}, stylized concept art, high quality, no text, no watermark`
  return `${base}, ${art}, high quality, no text, no watermark`
}

/**
 * 构建角色图默认负面提示词（按画风排除对立风格）。
 * 未指定 dramaStyle 或映射缺失时回退 CHARACTER_IMAGE_NEGATIVE，保持向后兼容。
 */
export function buildCharacterNegativePrompt(dramaStyle?: string | null): string {
  const artNegative = (dramaStyle && DRAMA_ART_NEGATIVE_MAP[dramaStyle]) || ''
  if (!artNegative) return CHARACTER_IMAGE_NEGATIVE
  return `${NEGATIVE_BASE}, ${artNegative}, cluttered background, busy background, multiple characters, multiple people, duplicated character, inconsistent character, cropped head, cut off face`
}

/**
 * 场景背景负面提示词
 * 排除人物角色（场景图只保留环境）、特写人脸与空洞扁平构图
 */
export const SCENE_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, people, person, human figure, characters, portrait, close-up face, face, flat composition, empty boring layout`

/**
 * 分镜图片负面提示词
 * 排除角色不一致、多余人物与时代/现代元素穿帮
 */
export const STORYBOARD_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, inconsistent character, character mismatch, extra characters, wrong number of people, anachronism, modern objects, modern clothing`

/**
 * 视频负面提示词（分镜视频 / 预设视频 / 手动视频共用）
 * 排除运动瑕疵、闪烁、形变、角色漂移与镜头抖动
 */
export const VIDEO_NEGATIVE = `${NEGATIVE_BASE}, motion blur, jittery, flickering, flicker, warping, morphing, melting, distorted face, inconsistent character, character drift, frame inconsistency, jump cuts, camera shake, static image, frozen frame`

/** 预设图片负面提示词（兼容旧引用，基于统一负面基础） */
export const PRESET_IMAGE_NEGATIVE = `${NEGATIVE_BASE}, mutated body parts`

/** 预设视频负面提示词 */
export const PRESET_VIDEO_NEGATIVE = VIDEO_NEGATIVE

/**
 * 根据 Shot 配置构建预设图片生成 prompt
 */
export function buildPresetImagePrompt(shot: PresetVariationCardShot): string {
  const parts = [
    shot.themeFamily,
    shot.compositionPattern,
    shot.spaceType,
    shot.foregroundFrame,
    shot.mainFocalPoint,
    shot.thematicClue,
    shot.activity,
    shot.characterLayout,
    shot.lightStructure,
    shot.windDirection,
    shot.cameraPosition,
  ]
  if (shot.livingElement) parts.push(shot.livingElement)
  parts.push(VISUAL_STYLE_MASTER)
  return parts.join(', ')
}

/**
 * 根据 Shot 配置 + 运镜方式构建预设视频生成 prompt
 */
export function buildPresetVideoPrompt(shot: PresetVariationCardShot, cameraMove: string): string {
  const parts = [
    shot.mainFocalPoint,
    shot.activity,
    shot.compositionPattern,
    shot.spaceType,
    shot.characterLayout,
    cameraMove,
    shot.lightStructure,
    shot.windDirection,
  ]
  parts.push(VISUAL_STYLE_VIDEO, VISUAL_STYLE_MASTER)
  return parts.join(', ')
}

// ============================================================
// 角色视觉 Prompt 构建
// ============================================================

/** 解析角色核心特征标签（JSON 数组字符串 → 字符串数组） */
function parseCoreFeatures(coreFeatures?: string | null): string[] {
  if (!coreFeatures) return []
  try {
    const arr = JSON.parse(coreFeatures)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is string => typeof x === 'string' && !!x.trim()).map(x => x.trim())
  } catch {
    return []
  }
}

/** 解析角色多套服装变体（JSON 数组字符串 → 字符串数组） */
function parseCostumes(costumes?: string | null): string[] {
  if (!costumes) return []
  try {
    const arr = JSON.parse(costumes)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is string => typeof x === 'string' && !!x.trim()).map(x => x.trim())
  } catch {
    return []
  }
}

/** 归一化装备/首饰文本：JSON 数组 → 逗号拼接；纯文本 → 去引号原样返回 */
function formatEquip(text?: string | null): string {
  if (!text) return ''
  const arr = parseCoreFeatures(text)
  if (arr.length) return arr.join(', ')
  return text.replace(/[\[\]"']/g, '').replace(/\s+/g, ' ').trim()
}

/** 解析角色多变体立绘（JSON 数组字符串 → {name, imageUrl}[]） */
function parseVariations(variations?: string | null): Array<{ name: string; imageUrl: string | null }> {
  if (!variations) return []
  try {
    const arr = JSON.parse(variations)
    if (!Array.isArray(arr)) return []
    return arr.filter((x): x is { name: string; imageUrl: string | null } =>
      x && typeof x === 'object' && typeof x.name === 'string')
  } catch {
    return []
  }
}

/**
 * 构建角色「服装/武器/首饰」正向子句。
 * 供自定义 prompt 场景强制注入：即使走了用户手写 prompt，
 * 智能拆分出的三个视觉字段也始终进入生图 prompt，避免信息被丢弃。
 */
export function buildCharacterVisualsClause(char: {
  clothing?: string | null
  costume?: string | null
  costumes?: string | null
  weapons?: string | null
  accessories?: string | null
}): string {
  const parts: string[] = []
  // 服装：优先本次选中的 costume，其次单套 clothing，最后回退多套 costumes 首套
  const costume = char.costume || char.clothing || parseCostumes(char.costumes)[0]
  if (costume) parts.push(`wearing ${costume}`)
  const weapons = formatEquip(char.weapons)
  if (weapons) parts.push(`armed with ${weapons}`)
  const accessories = formatEquip(char.accessories)
  if (accessories) parts.push(`wearing accessories: ${accessories}`)
  return parts.join(', ')
}

/** 三视图组合排版：一张横向长图内含正面/侧面/背面三个视角，同一个人物三次出现 */
export const THREE_VIEW_COMBINED_LAYOUT =
  'full body three-view turnaround, one single wide horizontal image with the same character shown three times side by side: front view on the left, side view in the middle, back view on the right, identical outfit and appearance across all three views, consistent character, head to toe'

/** 三视图画布尺寸：超宽横向比例（≈2.29:1），保证三个全身视角横向排开不挤压 */
export const THREE_VIEW_SIZE = '2048x896'

/**
 * 三视图负面提示词：在画风负面基础上排除分屏拼接/文字/边框/多余人，
 * 保证三个视角自然排列在同一张横向长图内。
 */
export function buildThreeViewNegative(dramaStyle?: string | null): string {
  const base = buildCharacterNegativePrompt(dramaStyle)
  return `${base}, split image, split panel, grid layout, frame border, panel divider, side-by-side separate images, multiple different characters, different people, clone, extra person, text, watermark, logo, label`
}

/**
 * 构建角色图片生成 prompt
 * @param name 角色名
 * @param appearance 外貌描述
 * @param description 补充描述
 * @param personality 性格特征（可选，用于表情氛围）
 * @returns 完整的角色图片 prompt
 */
export function buildCharacterImagePrompt(char: {
  name: string
  appearance?: string | null
  description?: string | null
  personality?: string | null
  coreFeatures?: string | null
  clothing?: string | null
  costume?: string | null
  costumes?: string | null
  weapons?: string | null
  accessories?: string | null
  /** 剧集视觉风格（dramas.style）；注入统一画风，保证同剧角色一致 */
  dramaStyle?: string | null
}): string {
  const parts: string[] = [char.name]
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(`core features: ${core.join(', ')}`)
  if (char.appearance) parts.push(char.appearance)
  const visuals = buildCharacterVisualsClause(char)
  if (visuals) parts.push(visuals)
  if (char.description && char.description !== char.appearance) parts.push(char.description)

  // 性格特征影响表情氛围
  if (char.personality) {
    parts.push(`${char.personality} expression and mannerisms`)
  }

  // 统一画风收口：dramaStyle 存在时按剧集视觉风格注入，否则维持默认插画风
  return `${parts.join(', ')}${buildCharacterArtStyleSuffix(char.dramaStyle)}`
}

/**
 * 构建角色外观描述文本（用于分镜/视频 prompt 中引用）
 * 提取角色核心视觉特征，不含风格标签
 */
export function buildCharacterAppearanceText(char: {
  name: string
  appearance?: string | null
  description?: string | null
  coreFeatures?: string | null
  clothing?: string | null
  costume?: string | null
  costumes?: string | null
  weapons?: string | null
  accessories?: string | null
}): string {
  const parts: string[] = [char.name]
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(core.join(', '))
  if (char.appearance) parts.push(char.appearance)
  const costume = char.costume || char.clothing || parseCostumes(char.costumes)[0]
  if (costume) parts.push(`wearing ${costume}`)
  const weapons = formatEquip(char.weapons)
  if (weapons) parts.push(`armed with ${weapons}`)
  const accessories = formatEquip(char.accessories)
  if (accessories) parts.push(`accessories: ${accessories}`)
  if (char.description && char.description !== char.appearance) parts.push(char.description)
  return parts.join(': ')
}

/**
 * 构建角色「装备/服饰特写图」prompt（服装/武器/首饰独立设定图）
 *
 * 设计原则：三类对象是「不同的东西」，提示词必须聚焦各自对象本体——
 * - 服装：完整衣物的剪裁/面料/配色，无人物
 * - 武器：武器的整体造型/金属质感/雕刻细节，无手持
 * - 首饰：单件首饰的近景/宝石/金属工艺，无佩戴
 * 不再把整段角色外貌、核心特征、角色名灌入（那是角色立绘的职责），
 * 否则服装/武器/首饰提示词会与立绘雷同、彼此雷同。
 *
 * @param type clothing/weapon/accessory
 */
export function buildEquipImagePrompt(
  type: 'clothing' | 'weapon' | 'accessory',
  char: {
    name: string
    appearance?: string | null
    coreFeatures?: string | null
    clothing?: string | null
    weapons?: string | null
    accessories?: string | null
    /** 剧集视觉风格（dramas.style）；注入统一画风 */
    dramaStyle?: string | null
  },
): string {
  const parts: string[] = []
  if (type === 'clothing') {
    const c = formatEquip(char.clothing)
    parts.push(c ? `detailed costume three-view design of ${c}` : 'detailed costume three-view design')
    parts.push('one wide horizontal image with the same garment shown three times side by side: front view, back view, side view')
    parts.push('ghost mannequin, silhouette, fabric texture and color scheme, no person wearing it, no body')
  } else if (type === 'weapon') {
    const w = formatEquip(char.weapons)
    parts.push(w ? `detailed weapon three-view concept art of ${w}` : 'detailed weapon three-view concept art')
    parts.push('one wide horizontal image with the same weapon shown three times side by side: front view, side view, top view')
    parts.push('metal texture, engravings and material details, isolated on clean background, no hand, no character holding it')
  } else {
    const a = formatEquip(char.accessories)
    parts.push(a ? `detailed jewelry three-view design of ${a}` : 'detailed jewelry three-view design')
    parts.push('one wide horizontal image with the same piece of jewelry shown three times side by side: front view, side view, top view')
    parts.push('gemstone and metalwork details, isolated on clean background, no person wearing it')
  }
  parts.push('object focused concept sheet, clean studio lighting, consistent item across all three views, no split panel borders, no grid frames, no text, no labels, no watermark')
  // 画风收口：装备图使用「无人物」画风尾，绝不引入 character design / skin 等人物语义词
  return `${parts.join(', ')}${buildEquipArtStyleSuffix(char.dramaStyle)}`
}

/**
 * 装备三视图负面提示词：在画风负面基础上排除人物/手持/不同物体/边框文字，
 * 保证服装/武器/首饰同一物品的三个视角自然排列在同一张横向长图内。
 * 注意：不排除 side-by-side / 三视图布局，否则模型会把三视角画成单个物体。
 */
export function buildEquipNegative(dramaStyle?: string | null): string {
  const base = buildCharacterNegativePrompt(dramaStyle)
  return `${base}, person, human figure, character, hands, body parts, multiple different items, different objects, duplicate inconsistent object, frame border, grid lines, panel divider, section divider, text, watermark, logo, label, extra floating parts, cropped object, cut off object`
}

// ============================================================
// 单件高清道具图（区别于三视图设定图：单件纯物品图，无人物，可直接进道具库/分镜）
// ============================================================

/** 单件道具图尺寸：方形高清，便于道具库卡片与分镜引用 */
export const ITEM_IMAGE_SIZE = '1024x1024'

/**
 * 构建单件高清道具图 prompt（服装/武器/首饰各自独立单品图）。
 * 与 buildEquipImagePrompt（三视角并排设定图）不同：单品居中展示、纯物品、无人物，
 * 用于入库与分镜画面中的物品质感呈现。
 */
export function buildItemImagePrompt(
  type: 'clothing' | 'weapon' | 'accessory',
  char: {
    clothing?: string | null
    weapons?: string | null
    accessories?: string | null
    /** 剧集视觉风格（dramas.style）；注入统一画风 */
    dramaStyle?: string | null
  },
): string {
  const parts: string[] = []
  if (type === 'clothing') {
    const c = formatEquip(char.clothing)
    parts.push(c ? `hero product shot of ${c}` : 'hero product shot of a single costume')
    parts.push('one garment displayed alone and centered, full view, fabric texture, color and tailoring details, studio product lighting, clean soft gradient background, no person, no mannequin, no body')
  } else if (type === 'weapon') {
    const w = formatEquip(char.weapons)
    parts.push(w ? `hero product shot of ${w}` : 'hero product shot of a single weapon')
    parts.push('one weapon displayed alone and centered, full view, metal texture, engravings and material details, studio product lighting, clean soft gradient background, no hand, no character holding it')
  } else {
    const a = formatEquip(char.accessories)
    parts.push(a ? `hero product shot of ${a}` : 'hero product shot of a single piece of jewelry')
    parts.push('one piece of jewelry displayed alone and centered, full view, gemstone and metalwork details, studio product lighting, clean soft gradient background, no person wearing it')
  }
  parts.push('single object only, high-detail prop asset, isolated, no text, no labels, no watermark, no logo, not a three-view, not side by side')
  // 画风收口：与装备三视图同一「无人物」画风尾
  return `${parts.join(', ')}${buildEquipArtStyleSuffix(char.dramaStyle)}`
}

/** 单件道具图负面提示词：排除人物/手持/三视角拼接/多物品/文字（单品入库基准） */
export function buildItemNegative(dramaStyle?: string | null): string {
  const base = buildCharacterNegativePrompt(dramaStyle)
  return `${base}, person, human figure, character, hands, body parts, mannequin, ghost mannequin, side by side, three views, multi view, turnaround, multiple items, different objects, duplicate, frame border, grid lines, panel divider, section divider, text, watermark, logo, label, cropped object, cut off object`
}

// ============================================================
// 角色表情头像特写组（表情演出：分镜/表情参考）
// ============================================================

/**
 * 角色表情预设（9 个默认表情）：表情演出用头像特写组。
 * 后端在生成/回填/校验时与前端共用同一份常量，保证 key 对齐。
 */
export const EXPRESSION_PRESETS: Array<{ key: string; label: string; en: string }> = [
  { key: 'smile', label: '微笑', en: 'gentle warm smile, softly curved lips, kind happy eyes' },
  { key: 'laugh', label: '大笑', en: 'hearty open laugh, bright joyful eyes, wide laughing mouth' },
  { key: 'mischievous', label: '俏皮', en: 'playful mischievous grin, slightly raised eyebrow, cheerful teasing look' },
  { key: 'angry', label: '愤怒', en: 'fierce angry glare, furrowed brows, tight frowning mouth' },
  { key: 'sad', label: '悲伤', en: 'sad melancholy face, downcast eyes, drooping mouth corners' },
  { key: 'surprised', label: '惊讶', en: 'surprised wide eyes, raised eyebrows, slightly open mouth' },
  { key: 'tearful', label: '泪目', en: 'teary glistening eyes about to cry, reddened rims, sorrowful trembling lips' },
  { key: 'serious', label: '严肃', en: 'serious stern neutral face, calm unreadable gaze, pressed lips' },
  { key: 'sobbing', label: '大哭', en: 'crying hard, eyes squeezed shut, tears streaming down cheeks, open wailing mouth' },
]

/** 按 key 查找表情预设；未知 key 返回空（调用方需自行兜底） */
export function findExpressionPreset(key: string): { key: string; label: string; en: string } | undefined {
  return EXPRESSION_PRESETS.find(p => p.key === key)
}

/**
 * 构建单个表情头像特写 prompt（头肩特写 + 表情 + 服装一致性）。
 *
 * 设计原则：表情图用于「表情演出/参考」，构图必须单一表情、单一人物、
 * 头肩取景，避免生成整身/多人/表情拼接；同时保留角色核心五官与服装，
 * 保证与立绘/三视图是同一个人。
 */
export function buildExpressionImagePrompt(
  char: {
    name: string
    appearance?: string | null
    coreFeatures?: string | null
    description?: string | null
    clothing?: string | null
    costumes?: string | null
  },
  expressionKey: string,
  dramaStyle?: string | null,
): string {
  const preset = findExpressionPreset(expressionKey)
  const parts: string[] = []
  parts.push(`close-up portrait headshot of ${char.name}`)
  const core = parseCoreFeatures(char.coreFeatures)
  if (core.length) parts.push(`core features: ${core.join(', ')}`)
  if (char.appearance) parts.push(char.appearance)
  const costume = char.clothing || parseCostumes(char.costumes)[0]
  if (costume) parts.push(`wearing ${costume}`)
  parts.push(`facial expression: ${preset?.en || 'neutral calm expression'}`)
  parts.push('looking directly at viewer, head and shoulders framing, single face only')
  // 画风收口：与角色立绘保持一致（同剧统一）
  return `${parts.join(', ')}${buildCharacterArtStyleSuffix(dramaStyle)}`
}

/**
 * 表情头像特写负面提示词：在画风负面基础上排除多人/多表情/拼接/整身，
 * 保证一张图只有单个头肩、单一表情。
 */
export function buildExpressionNegative(dramaStyle?: string | null): string {
  const base = buildCharacterNegativePrompt(dramaStyle)
  return `${base}, multiple faces, multiple expressions, two or more different expressions, extra person, duplicate character, group shot, collage, split image, grid layout, side-by-side panels, body below shoulders, full body shot, whole figure, text, watermark, logo, label`
}

// ============================================================
// 物品视觉 Prompt 构建（连续性状态机 v3 物品库）
// ============================================================

/** 构建物品（道具/信物/线索/法器）设定图 prompt */
export function buildPropImagePrompt(prop: {
  name: string
  category?: string | null
  description?: string | null
  appearance?: string | null
  sizeHint?: string | null
  color?: string | null
  holder?: string | null
}): string {
  const parts: string[] = []
  parts.push(`detailed prop design sheet of ${prop.name}（${prop.category || '道具'}）`)
  if (prop.appearance) parts.push(`appearance: ${prop.appearance}`)
  if (prop.color) parts.push(`color: ${prop.color}`)
  if (prop.sizeHint) parts.push(`size: ${prop.sizeHint}`)
  if (prop.description) parts.push(`description: ${prop.description}`)
  if (prop.holder) parts.push(`used/hold by: ${prop.holder}`)
  parts.push('single object centered, clean background, faithful to the drama art style')
  return `${parts.join(', ')}, ${VISUAL_STYLE_SCENE}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 场景视觉 Prompt 构建
// ============================================================

/**
 * 构建场景图片生成 prompt
 * @param location 地点名称
 * @param time 时间（白天/夜晚等）
 * @param prompt 场景描述
 * @param dramaStyle 戏剧风格标签（来自 dramas.style）
 */
export function buildSceneImagePrompt(options: {
  location: string
  time?: string
  prompt?: string
  dramaStyle?: string
}): string {
  const parts: string[] = []
  if (options.prompt) parts.push(options.prompt)
  else {
    parts.push(options.location)
    if (options.time) parts.push(`${options.time} lighting and atmosphere`)
  }
  if (options.dramaStyle) parts.push(`${options.dramaStyle} visual style`)

  return `${parts.join(', ')}, ${VISUAL_STYLE_SCENE}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 分镜视觉 Prompt 构建
// ============================================================

/**
 * 构建分镜叙事图片 prompt（融合角色+场景+镜头描述）
 * 
 * 核心修复：之前分镜图片生成只传 description，没有角色外观和场景信息，
 * 导致角色在不同分镜中视觉不一致
 */
export function buildStoryboardImagePrompt(options: {
  description?: string | null         // 分镜叙事描述
  storyboardDescription?: string | null  // 分镜视觉描述
  characterDescription?: string | null   // 角色描述
  sceneDescription?: string | null       // 场景描述
  location?: string | null               // 地点
  shotType?: string | null               // 景别
  cameraAngle?: string | null            // 机位
  dramaStyle?: string                    // 戏剧风格
}): string {
  const parts: string[] = []

  // 1. 场景环境
  if (options.sceneDescription) {
    parts.push(`Scene: ${options.sceneDescription}`)
  } else if (options.location) {
    parts.push(`Location: ${options.location}`)
  }

  // 2. 角色外观
  if (options.characterDescription) {
    parts.push(`Characters: ${options.characterDescription}`)
  }

  // 3. 镜头描述（景别/机位经视觉图谱翻译为英文电影术语，避免中文混入英文 prompt）
  const cinematicHints: string[] = []
  const shotEn = resolveVisualTerm(options.shotType)
  if (options.shotType) cinematicHints.push(shotEn || options.shotType)
  const angleEn = resolveVisualTerm(options.cameraAngle)
  if (options.cameraAngle) cinematicHints.push(angleEn || options.cameraAngle)
  if (cinematicHints.length) parts.push(`Camera: ${cinematicHints.join(', ')}`)

  // 4. 叙事内容
  if (options.storyboardDescription) {
    parts.push(options.storyboardDescription)
  } else if (options.description) {
    parts.push(options.description)
  }

  // 5. 风格
  if (options.dramaStyle) parts.push(`${options.dramaStyle} visual style`)

  // 底线
  if (parts.length === 0) return `cinematic shot, ${VISUAL_STYLE_MASTER}`

  return `${parts.join('. ')}, ${VISUAL_STYLE_MASTER}`
}

/**
 * 构建分镜视频生成 prompt（融合角色+场景+叙事+动作）
 * 
 * 核心修复：之前视频生成只用 storyboardDescription，完全没有角色外观上下文，
 * 导致视频中角色形象突变
 */
export function buildStoryboardVideoPrompt(options: {
  description?: string | null
  storyboardDescription?: string | null
  characterAppearances?: string[]          // 各角色外观描述列表
  scenePrompt?: string | null              // 场景视觉描述
  action?: string | null                   // 动作描述
  movement?: string | null                 // 运镜
  dramaStyle?: string                      // 戏剧风格
  backgroundAudio?: string | null          // 场景声/环境音描述（H3 原生 [background_audio] 标记）
}): string {
  const parts: string[] = []

  // 1. 角色一致性 - 最关键的部分
  if (options.characterAppearances?.length) {
    parts.push(`Characters (maintain strict visual consistency): ${options.characterAppearances.join('; ')}`)
  }

  // 2. 场景
  if (options.scenePrompt) {
    parts.push(`Setting: ${options.scenePrompt}`)
  }

  // 3. 叙事
  if (options.storyboardDescription) {
    parts.push(options.storyboardDescription)
  } else if (options.description) {
    parts.push(options.description)
  }

  // 4. 动作与运镜（运镜经视觉图谱翻译为英文电影术语）
  if (options.action) parts.push(`Action: ${options.action}`)
  if (options.movement) {
    const moveEn = resolveVisualTerm(options.movement)
    parts.push(`Camera movement: ${moveEn || options.movement}`)
  }

  // 5. 风格
  if (options.dramaStyle) parts.push(`${options.dramaStyle} cinematic style`)

  // 6. 声音策略（对齐 Mx-Shell sound policy：画面内不混音，音乐后期叠加）
  const soundPolicy = 'Sound: production audio only, no background music in frame (music is mixed in post-production)'

  // 7. UI 屏幕留白规则（ui_plate 约束）：分镜含屏幕/UI 元素时强制只留白、文字后期叠加
  const hasScreen = [options.description, options.storyboardDescription, options.action]
    .some(containsScreenElement)
  const uiRule = hasScreen ? `, ${UI_OVERLAY_RULE}` : ''

  // 8. H3 原生场景声标记 [background_audio]（对齐参考项目 minimax-h3-comfyui 的 prompt 语法）：
  //    H3 会在生成视频时同步合成场景环境音；该标记放在 prompt 末尾即可生效。
  const bgAudio = options.backgroundAudio?.trim()
    ? ` [background_audio] ${options.backgroundAudio.trim()}`
    : ''

  return `${parts.join('. ')}, ${soundPolicy}.${uiRule}${bgAudio} ${VISUAL_STYLE_VIDEO}, ${VISUAL_STYLE_MASTER}`
}

// ============================================================
// 数据库查询辅助
// ============================================================

/**
 * 根据分镜ID获取关联的角色外表描述列表
 */
export function getStoryboardCharacterAppearances(storyboardId: number): string[] {
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()

  if (!spChars.length) return []

  // 镜头级服装变体：每个角色可指定本镜头穿的造型
  const costumeByCharId = new Map<number, string>()
  for (const link of spChars) {
    if (link.costume) costumeByCharId.set(link.characterId, link.costume)
  }

  const characterIds = spChars.map(link => link.characterId)
  // 使用 inArray 替代内存过滤
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters)
        .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
        .all()
    : []

  return characters.map(char =>
    buildCharacterAppearanceText({
      name: char.name,
      appearance: char.appearance,
      description: char.description,
      coreFeatures: char.coreFeatures,
      clothing: char.clothing,
      costume: costumeByCharId.get(char.id),
      costumes: char.costumes,
    })
  )
}

/**
 * 根据分镜ID获取关联的场景描述
 */
export function getStoryboardSceneDescription(storyboardId: number): string | null {
  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId))
    .all()

  if (!sb?.sceneId) return null

  const [scene] = db.select().from(schema.scenes)
    .where(eq(schema.scenes.id, sb.sceneId))
    .all()

  if (!scene) return null

  return buildSceneImagePrompt({
    location: scene.location,
    time: scene.time,
    prompt: scene.prompt,
  })
}

/**
 * 根据角色ID列表获取所有角色的图片URL（用于视频/图片 reference_images）
 */
export function getCharacterImageUrls(characterIds: number[]): string[] {
  if (!characterIds.length) return []
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()
    .filter(char => char.imageUrl)

  return characters.map(char => char.imageUrl!)
}

/**
 * 根据分镜ID获取关联角色的图片URL
 */
export function getStoryboardCharacterImageUrls(storyboardId: number): string[] {
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()
  if (!spChars.length) return []

  // 镜头级服装变体：若角色指定了变体且有对应立绘，用变体图；否则主图
  const costumeByCharId = new Map<number, string>()
  for (const link of spChars) {
    if (link.costume) costumeByCharId.set(link.characterId, link.costume)
  }

  const characterIds = spChars.map(link => link.characterId)
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()

  const urls: string[] = []
  for (const char of characters) {
    const costume = costumeByCharId.get(char.id)
    if (costume) {
      const variation = parseVariations(char.variations).find(v => v.name === costume)
      if (variation?.imageUrl) {
        urls.push(variation.imageUrl)
        continue
      }
    }
    if (char.imageUrl) urls.push(char.imageUrl)
  }
  return urls
}

/**
 * 根据分镜ID获取参考图（角色立绘 + 场景图），用于分镜图片/视频生成保持人物与场景一致
 */
export function getStoryboardReferenceImages(storyboardId: number): string[] {
  const urls = getStoryboardCharacterImageUrls(storyboardId)

  const [sb] = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.id, storyboardId))
    .all()
  if (sb?.sceneId) {
    const [scene] = db.select().from(schema.scenes)
      .where(eq(schema.scenes.id, sb.sceneId))
      .all()
    if (scene?.imageUrl) urls.push(scene.imageUrl)
  }

  // 物品参考图（连续性状态机 v3 物品库）：本镜关联物品的设定图
  const propLinks = db.select().from(schema.storyboardProps)
    .where(eq(schema.storyboardProps.storyboardId, storyboardId))
    .all()
  if (propLinks.length) {
    const propIds = propLinks.map((l) => l.propId)
    const props = db.select().from(schema.propTemplates)
      .where(and(inArray(schema.propTemplates.id, propIds), isNull(schema.propTemplates.deletedAt)))
      .all()
    for (const p of props) {
      if (p.imageUrl) urls.push(p.imageUrl)
    }
  }

  return urls
}

/**
 * 将媒体相对路径转成可被本地/远程服务访问的绝对 URL（本地 H3 服务需能拉取参考音频）
 */
function toPublicMediaUrl(p: string): string {
  if (/^(https?:|data:)/.test(p)) return p
  const base = (process.env.PUBLIC_BASE_URL || 'http://localhost:5789').replace(/\/+$/, '')
  return `${base}/${p.replace(/^\/+/, '')}`
}

/**
 * 收集分镜出场角色的声线样本音频 URL（H3 Ref2VA 参考音频 / reference conditioning，最多 3 条）
 */
export function getStoryboardReferenceAudioUrls(storyboardId: number): string[] {
  const links = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()
  if (!links.length) return []

  const characterIds = links.map(l => l.characterId)
  const characters = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
    .all()

  return characters
    .map(c => c.voiceSampleUrl)
    .filter((u): u is string => !!u && !!u.trim())
    .map(toPublicMediaUrl)
    .slice(0, 3)
}

// ============================================================
// 对话-角色一致性验证
// ============================================================

/**
 * 验证分镜 dialogue 中的角色名是否与分镜关联的 character_ids 匹配
 * @param dialogue 对话文本
 * @param storyboardId 分镜ID
 * @returns { mismatches: 不匹配的角色名列表, matchCount: 匹配的角色名数量 }
 */
export function validateDialogueCharacterConsistency(
  dialogue: string,
  storyboardId: number,
): { mismatches: string[]; matchCount: number; allMatch: boolean } {
  if (!dialogue) return { mismatches: [], matchCount: 0, allMatch: true }

  // 提取对话中的角色名
  const regex = /([^\n:：]{1,10}?)[:：]([^:：\n]{8,})/g
  const speakerNames = new Set<string>()
  let match
  while ((match = regex.exec(dialogue)) !== null) {
    speakerNames.add(match[1].trim())
  }

  // 如果正则没匹配到，尝试用 parseDialogueForTTS 逻辑
  if (speakerNames.size === 0) {
    const parsed = parseDialogueForTTSLocal(dialogue)
    if (parsed.speaker && !parsed.ignorable) {
      speakerNames.add(parsed.speaker)
    }
  }

  if (speakerNames.size === 0) return { mismatches: [], matchCount: 0, allMatch: true }

  // 获取分镜关联的角色（使用 WHERE 查询替代全表扫描）
  const spChars = db.select().from(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .all()

  const characterIds = spChars.map(link => link.characterId)
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters)
        .where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt)))
        .all()
    : []

  const charNames = new Set(characters.map(c => c.name))

  const mismatches: string[] = []
  let matchCount = 0

  for (const name of speakerNames) {
    // 旁白/画外音不校验
    if (/^(旁白|画外音|narrator)$/i.test(name)) continue
    if (charNames.has(name)) {
      matchCount++
    } else {
      mismatches.push(name)
    }
  }

  return { mismatches, matchCount, allMatch: mismatches.length === 0 }
}

/**
 * parseDialogueForTTS 本地副本，避免循环依赖
 */
function parseDialogueForTTSLocal(dialogue: string): { ignorable: boolean; speaker: string; pureText: string } {
  const text = (dialogue || '').trim()
  if (!text) return { ignorable: true, speaker: '', pureText: '' }

  const labelMatch = text.match(/^(.*?)[:：]\s*(.+)/)
  if (labelMatch) {
    return {
      ignorable: false,
      speaker: labelMatch[1].trim(),
      pureText: labelMatch[2].trim(),
    }
  }
  return { ignorable: false, speaker: '', pureText: text }
}

// ============================================================
// 视频提示词标签剥离
// ============================================================

/**
 * 剥离视频提示词中的结构化标记标签，只保留标签内的自然语言内容。
 *
 * 背景：分镜 agent 生成的 video_prompt 使用 <location>/<role>/<voice>/<n>
 * 作为结构化 DSL，供程序解析（分段、角色绑定、场景提取）使用。
 * 但视频扩散模型不识别这些 XML 标签，原样保留只会成为文本噪声，
 * 占用注意力并可能干扰语义理解。因此在发送给视频生成模型前应剥离。
 *
 * 规则：
 * - <location>…</location> / <role>…</role> / <voice>…</voice> → 去掉开闭标签，保留中间内容
 * - <n>（时间段分隔符）→ 换成换行，让每个时间段独立成段
 * - 清理多余空白
 */
export function stripVideoPromptTags(prompt: string): string {
  if (!prompt) return prompt
  return prompt
    .replace(/<\/?(?:location|role|voice)>/gi, '')
    .replace(/<n\s*\/?>/gi, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\s*\n\s*/g, '\n')
    .trim()
}

// ============================================================
// 宫格图 Prompt 构建（唯一入口）
//
// 宫格图 prompt 曾有三处重复实现（grid 路由内嵌模板、grid-prompt-tools、
// storyboard-tools），现统一收敛到本模块。grid 路由与 grid_prompt_generator
// agent 工具均调用 buildGridPrompt / buildGridCellPrompts，避免逻辑分叉。
// ============================================================

export type GridReferenceAsset = {
  path: string
  label: string
  kind: 'scene' | 'character' | 'storyboard'
  imageLabel: string
  sceneId?: number
  characterId?: number
  storyboardId?: number
}

function safeParseJsonArray(value: any): string[] {
  if (!value) return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed.filter(Boolean) : []
  } catch {
    return []
  }
}

function posLabel(i: number, rows: number, cols: number) {
  const r = Math.floor(i / cols), c = i % cols
  return `row ${r + 1} col ${c + 1}`
}

function cellLabel(i: number, rows: number, cols: number) {
  return `格${i + 1}（${posLabel(i, rows, cols)}）`
}

/** 获取每个分镜关联的角色 ID 列表（用于注入角色外观与参考图） */
function getStoryboardCharacterIds(storyboardIds: number[]) {
  if (!storyboardIds.length) return new Map<number, number[]>()
  const links = db.select().from(schema.storyboardCharacters).all()
    .filter((link) => storyboardIds.includes(link.storyboardId))
  const map = new Map<number, number[]>()
  for (const link of links) {
    const arr = map.get(link.storyboardId) || []
    arr.push(link.characterId)
    map.set(link.storyboardId, arr)
  }
  return map
}

/** 构建角色 ID → 外观描述文本的映射，用于将角色视觉特征注入分镜描述 */
function buildCharacterAppearanceMap(storyboardCharacterIds: Map<number, number[]>) {
  const allCharIds = new Set<number>()
  for (const ids of storyboardCharacterIds.values()) ids.forEach(id => allCharIds.add(id))
  if (!allCharIds.size) return new Map<number, string>()

  const chars = db.select().from(schema.characters)
    .where(and(inArray(schema.characters.id, [...allCharIds]), isNull(schema.characters.deletedAt)))
    .all()

  const map = new Map<number, string>()
  for (const char of chars) {
    map.set(char.id, buildCharacterAppearanceText(char))
  }
  return map
}

/** 构建分镜关联的参考图提示（角色立绘 + 场景图 + 已有分镜图） */
function buildStoryboardReferenceHints(
  sb: any,
  referenceAssets: GridReferenceAsset[],
  storyboardCharacterIds: Map<number, number[]>,
) {
  const hints: string[] = []
  const charIds = storyboardCharacterIds.get(sb.id) || []

  for (const asset of referenceAssets) {
    if (asset.kind === 'scene' && sb.sceneId && asset.sceneId === sb.sceneId) {
      hints.push(`${asset.imageLabel}（${asset.label}）`)
    }
    if (asset.kind === 'character') {
      if (asset.characterId && charIds.includes(asset.characterId)) {
        hints.push(`${asset.imageLabel}（${asset.label}）`)
      }
    }
    if (asset.kind === 'storyboard' && asset.storyboardId === sb.id) {
      hints.push(`${asset.imageLabel}（${asset.label}）`)
    }
  }

  return [...new Set(hints)].slice(0, 4)
}

/** 构建增强的分镜 cell 描述：注入角色外观 + 参考图 + 分镜自身描述 */
function buildEnrichedCellDescription(
  sb: any,
  index: number,
  storyboardCharacterIds: Map<number, number[]>,
  charAppearanceMap: Map<number, string>,
  referenceAssets: GridReferenceAsset[],
): string {
  const parts: string[] = []

  const charIds = storyboardCharacterIds.get(sb.id) || []
  const appearances: string[] = []
  for (const cid of charIds) {
    const appearance = charAppearanceMap.get(cid)
    if (appearance) appearances.push(appearance)
  }
  if (appearances.length) {
    parts.push(`Characters: ${appearances.join('; ')}`)
  }

  const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
  if (refs.length) {
    parts.push(`参考${refs.join('、')}`)
  }

  const desc = sb.imagePrompt || sb.description || sb.title || `shot ${index + 1}`
  parts.push(desc)

  return parts.join('. ')
}

/** 收集宫格图涉及的参考图（首尾帧/镜头图/参考图 + 场景图 + 角色立绘，上限 6 张） */
export function collectGridReferenceAssets(storyboards: any[]): GridReferenceAsset[] {
  const storyboardIds = storyboards.map((sb) => sb.id)
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboardIds)
  const sceneIds = [...new Set(storyboards.map((sb) => sb.sceneId).filter(Boolean))]
  const characterIds = [...new Set([...storyboardCharacterIds.values()].flat().filter(Boolean))]

  const scenes = sceneIds.length > 0
    ? db.select().from(schema.scenes).where(inArray(schema.scenes.id, sceneIds)).all()
    : []
  const characters = characterIds.length > 0
    ? db.select().from(schema.characters).where(and(inArray(schema.characters.id, characterIds), isNull(schema.characters.deletedAt))).all()
    : []

  const assets: GridReferenceAsset[] = []
  const seen = new Set<string>()
  const pushAsset = (
    path: string | null | undefined,
    label: string,
    kind: 'scene' | 'character' | 'storyboard',
    extra: { sceneId?: number; characterId?: number; storyboardId?: number } = {},
  ) => {
    if (!path || seen.has(path) || assets.length >= 6) return
    seen.add(path)
    assets.push({ path, label, kind, ...extra } as GridReferenceAsset)
  }

  for (const sb of storyboards) {
    pushAsset(sb.firstFrameImage, `镜头${sb.storyboardNumber}首帧`, 'storyboard', { storyboardId: sb.id })
    pushAsset(sb.lastFrameImage, `镜头${sb.storyboardNumber}尾帧`, 'storyboard', { storyboardId: sb.id })
    pushAsset(sb.composedImage, `镜头${sb.storyboardNumber}镜头图`, 'storyboard', { storyboardId: sb.id })
    for (const ref of safeParseJsonArray(sb.referenceImages)) {
      pushAsset(ref, `镜头${sb.storyboardNumber}参考图`, 'storyboard', { storyboardId: sb.id })
    }
  }
  for (const scene of scenes) {
    pushAsset(scene.imageUrl, `${scene.location}${scene.time ? `（${scene.time}）` : ''}场景`, 'scene', { sceneId: scene.id })
  }
  for (const char of characters) {
    pushAsset(char.imageUrl, `${char.name}角色`, 'character', { characterId: char.id })
  }

  return assets.map((asset, index) => ({
    ...asset,
    imageIndex: index + 1,
    imageLabel: `图片${index + 1}`,
  }))
}

export function buildReferenceLegend(referenceAssets: Array<{ imageLabel: string; label: string }>) {
  if (!referenceAssets.length) return ''
  return referenceAssets.map((asset) => `${asset.imageLabel}=${asset.label}`).join('；')
}

const GRID_ANGLES = [
  'wide establishing shot', 'medium shot character focus',
  'close-up detail', 'dramatic low angle', 'over-the-shoulder view',
  'bird eye view', 'side profile', 'atmospheric detail',
  'extreme close-up', 'dutch angle', 'silhouette shot',
  'depth of field focus', 'symmetrical composition', 'leading lines',
  'negative space', 'high angle looking down', 'ground level',
  'panoramic wide', 'intimate two-shot', 'reflection shot',
  'shadow play', 'backlit silhouette', 'macro detail',
  'split lighting', 'rim light portrait',
]

/**
 * 构建宫格图整体 prompt（三种模式：first_frame / first_last / multi_ref）
 * 统一注入：风格、参考图映射、角色外观、运镜构图。
 */
export function buildGridPrompt(
  mode: string,
  storyboards: any[],
  rows: number,
  cols: number,
  dramaStyle: string,
  referenceAssets: GridReferenceAsset[],
): string {
  const style = dramaStyle || 'cinematic'
  const legend = buildReferenceLegend(referenceAssets)
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboards.map((sb) => sb.id))
  const charAppearanceMap = buildCharacterAppearanceMap(storyboardCharacterIds)

  if (mode === 'first_frame') {
    const cells = storyboards.map((sb, i) => {
      const desc = buildEnrichedCellDescription(sb, i, storyboardCharacterIds, charAppearanceMap, referenceAssets)
      return `${cellLabel(i, rows, cols)}: ${desc}`
    })
    return [
      `${rows}x${cols} grid layout, consistent art style, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      '当画面涉及角色或场景时，优先使用对应的图片编号来约束一致性。',
      ...cells,
      'high quality, cinematic lighting, no text, no watermark',
    ].filter(Boolean).join('\n')
  }

  if (mode === 'first_last') {
    const totalCells = rows * cols
    const cells = Array.from({ length: totalCells }, (_, i) => {
      const sb = storyboards[i % storyboards.length]
      const desc = sb.imagePrompt || sb.description || sb.title || `shot ${i + 1}`
      const action = sb.action || sb.movement || ''
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      const isFirst = i % 2 === 0
      const composition = getCameraMovementComposition(sb.movement || '', isFirst ? 'start' : 'end')
      const comp = composition ? `, ${composition}` : ''
      const frameHint = isFirst
        ? 'opening moment'
        : `${action ? `${action}, ` : ''}closing moment, subtle motion change`
      return `${cellLabel(i, rows, cols)}: ${refs.length ? `参考${refs.join('、')}，` : ''}${desc}, ${frameHint}${comp}`
    })
    return [
      `${rows}x${cols} grid layout, consistent art style, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      'first/last frame visual rhythm, alternating opening and closing beats across the grid,',
      ...cells,
      'continuous motion implied between left and right, high quality, no text',
    ].filter(Boolean).join('\n')
  }

  if (mode === 'multi_ref') {
    const sb = storyboards[0]
    const desc = sb.imagePrompt || sb.description || sb.title || 'scene'
    const totalCells = rows * cols
    const cells = Array.from({ length: totalCells }, (_, i) => {
      return `${cellLabel(i, rows, cols)}: ${legend ? `参考${legend}，` : ''}${desc}, ${GRID_ANGLES[i % GRID_ANGLES.length]}`
    })
    return [
      `${rows}x${cols} grid layout, same scene different angles and compositions, ${style},`,
      legend ? `参考图映射：${legend}` : '',
      `main scene: ${desc},`,
      ...cells,
      'consistent lighting and color palette, high quality, no text',
    ].filter(Boolean).join('\n')
  }

  return `${rows}x${cols} grid, ${style}, storyboard frames, high quality`
}

/**
 * 构建宫格图逐格 prompt（与 buildGridPrompt 同源，供 split 回写/逐格生成使用）
 */
export function buildGridCellPrompts(
  mode: string,
  storyboards: any[],
  rows: number,
  cols: number,
  referenceAssets: GridReferenceAsset[],
) {
  if (!storyboards.length) return []
  const storyboardCharacterIds = getStoryboardCharacterIds(storyboards.map((sb) => sb.id))

  if (mode === 'multi_ref') {
    const sb = storyboards[0]
    const desc = sb.imagePrompt || sb.description || sb.title || 'scene'
    return Array.from({ length: rows * cols }, (_, i) => {
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      return {
        shot_number: sb.storyboardNumber,
        frame_type: 'reference',
        prompt: `${cellLabel(i, rows, cols)}: ${refs.length ? `参考${refs.join('、')}，` : ''}${desc}, ${GRID_ANGLES[i % GRID_ANGLES.length]}`,
      }
    })
  }

  if (mode === 'first_last') {
    return Array.from({ length: rows * cols }, (_, i) => {
      const sb = storyboards[i % storyboards.length]
      const desc = sb.imagePrompt || sb.description || sb.title || `shot ${sb.storyboardNumber || ''}`
      const motion = sb.action || sb.movement || ''
      const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
      const isFirst = i % 2 === 0
      const composition = getCameraMovementComposition(sb.movement || '', isFirst ? 'start' : 'end')
      const comp = composition ? `, ${composition}` : ''
      return {
        shot_number: sb.storyboardNumber,
        frame_type: isFirst ? 'first_frame' : 'last_frame',
        prompt: isFirst
          ? `${cellLabel(i, rows, cols)}，首帧：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}${comp}`
          : `${cellLabel(i, rows, cols)}，尾帧：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${motion ? `, ${motion}` : ''}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}${comp}`,
      }
    })
  }

  return storyboards.slice(0, rows * cols).map((sb, index) => {
    const desc = sb.imagePrompt || sb.description || sb.title || `shot ${sb.storyboardNumber || ''}`
    const refs = buildStoryboardReferenceHints(sb, referenceAssets, storyboardCharacterIds)
    const composition = getCameraMovementComposition(sb.movement || '', 'start')
    const comp = composition ? `, ${composition}` : ''
    return {
      shot_number: sb.storyboardNumber,
      frame_type: 'first_frame',
      prompt: `${cellLabel(index, rows, cols)}：${refs.length ? `参考${refs.join('、')}，` : ''}${desc}${sb.location ? `, ${sb.location}` : ''}${sb.shotType ? `, ${sb.shotType}` : ''}, opening scene${comp}`,
    }
  })
}


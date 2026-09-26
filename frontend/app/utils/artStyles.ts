/**
 * 画风选项（前端单一数据源）
 *
 * ⚠️ 必须与后端 `backend-py/app/services/prompt_utils.py` 的 ART_STYLE_CATALOG 保持一致
 * （溯源：TS 时代在 `backend/src/shared/prompt-utils.ts`，已随 Node 后端删除）：
 * value 是落库值（dramas.style / characters.style / app_settings.art_style）。
 * 新增画风时后端还要同步补 DRAMA_ART_STYLE_MAP / EQUIP_ART_STYLE_MAP 的英文画风词，
 * 否则会出现「前端能选、模型不认」——画风后缀为空，退化成默认插画风。
 *
 * 英文画风词一律由后端 buildCharacterArtStyleSuffix / buildEquipArtStyleSuffix 收口，
 * 前端不要自行拼接风格词，避免同一剧内画风漂移。
 */

export interface ArtStyleOption {
  /** 落库值 */
  value: string
  /** 完整中文名（下拉、详情页） */
  label: string
  /** 紧凑短名（列表角标、工具条切换） */
  shortLabel: string
  /** 一句话视觉说明（设置页副标题） */
  desc: string
  /** 分组，用于设置页分组展示 */
  group: string
}

export const ART_STYLE_OPTIONS: ArtStyleOption[] = [
  // 实拍质感：靠镜头规格/光位/胶片调色锚定真实感
  { value: 'realistic', label: '写实电影', shortLabel: '写实', desc: '真人质感、电影级镜头语言与胶片调色', group: '实拍质感' },
  { value: 'cinematic', label: '电影感', shortLabel: '电影感', desc: '商业电影帧、强氛围与色彩分级', group: '实拍质感' },
  { value: 'noir', label: '黑色电影', shortLabel: '黑色', desc: '黑白高反差、硬光影、1940s 侦探片质感', group: '实拍质感' },
  // 绘画插画：靠渲染方式锚定（cel shading / 水墨 / 晕染）
  { value: 'anime', label: '日式动漫', shortLabel: '动漫', desc: '赛璐璐上色、鲜明线条、动画 key visual', group: '绘画插画' },
  { value: 'ghibli', label: '吉卜力', shortLabel: '吉卜力', desc: '手绘质感、温暖配色、治愈系', group: '绘画插画' },
  { value: 'ink-wash', label: '国风水墨', shortLabel: '水墨', desc: '水墨晕染、留白意境、宣纸质感', group: '绘画插画' },
  { value: 'watercolor', label: '水彩', shortLabel: '水彩', desc: '水彩晕染、柔和过渡、纸面纹理', group: '绘画插画' },
  // 三维与漫画
  { value: 'comic', label: '美漫漫画', shortLabel: '漫画', desc: '美式漫画、粗线条、网点阴影', group: '三维与漫画' },
  { value: 'cyberpunk', label: '赛博朋克', shortLabel: '赛博', desc: '霓虹雨夜、高饱和洋红青、全息光斑', group: '三维与漫画' },
  { value: 'pixar3d', label: '三维动画', shortLabel: '3D', desc: '皮克斯式三维渲染、圆润造型、柔和全局光', group: '三维与漫画' },
]

/** 分组展示顺序（显式声明，不依赖数组插入顺序） */
export const ART_STYLE_GROUPS: string[] = ['实拍质感', '绘画插画', '三维与漫画']

/** value → 完整中文名 */
export const ART_STYLE_LABELS: Record<string, string> = Object.fromEntries(
  ART_STYLE_OPTIONS.map(o => [o.value, o.label]),
)

/** value → 紧凑短名（列表角标、切换按钮） */
export const ART_STYLE_SHORT_LABELS: Record<string, string> = Object.fromEntries(
  ART_STYLE_OPTIONS.map(o => [o.value, o.shortLabel]),
)

/** 安全取完整中文名（历史数据里的未知 key 原样回显，不显示空白） */
export function artStyleLabel(value?: string | null): string {
  if (!value) return ''
  return ART_STYLE_LABELS[value] || value
}

/** 安全取紧凑短名 */
export function artStyleShortLabel(value?: string | null): string {
  if (!value) return ''
  return ART_STYLE_SHORT_LABELS[value] || value
}

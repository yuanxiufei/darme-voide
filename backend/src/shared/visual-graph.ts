/**
 * 视觉图谱（Visual Graph）——对齐参考项目「视觉图谱：景别/构图/运镜/灯光，由 Adviser 图谱驱动提示词」
 *
 * 四类图谱（知识节点）：
 * 1. shot_size   景别图谱：大远景/远景/全景/中景/中近景/近景/特写/大特写
 * 2. composition 构图图谱：三分法/对称/引导线/框架/中心/负空间/对角线/前景遮挡/纵深
 * 3. movement    运镜图谱：固定/推/拉/摇/移/跟/升降/环绕/手持/主观/慢动作/滑动变焦
 * 4. lighting    灯光图谱：自然光/硬光/柔光/逆光/侧光/顶光/低光/暖光/冷光/黄昏/夜景
 *
 * 提供两类能力：
 * - resolveVisualTerm()：把分镜里的中文景别/机位/运镜/灯光映射为英文电影术语，
 *   供 image/video prompt 构造（避免中文术语直接混入英文 prompt 导致生图偏差）。
 * - buildVisualGraphGuidance()：按剧风格生成一段「视觉图谱引导」，注入 storyboard_breaker
 *   instructions，让拆镜时主动使用图谱（景别递进 / 构图节奏 / 运镜动机 / 灯光氛围）。
 */

export type VisualCategory = 'shot_size' | 'composition' | 'movement' | 'lighting'

export interface VisualGraphNode {
  /** 中文别名（用于匹配分镜字段值，数据库实测为中文） */
  zh: string[]
  /** 英文电影术语（用于 prompt 构造） */
  en: string
  /** 使用场景提示（拆镜引导用） */
  usage?: string
  /** 建议搭配（如景别搭配构图/运镜） */
  pairsWith?: string[]
}

export const VISUAL_GRAPH: Record<VisualCategory, VisualGraphNode[]> = {
  shot_size: [
    { zh: ['大远景', '极远景', '超远景'], en: 'extreme wide shot', usage: '建立空间关系，交代环境与人物位置，适合开场/转场', pairsWith: ['establishing', 'symmetrical'] },
    { zh: ['远景', '远景镜头'], en: 'wide shot', usage: '人物全身入画，展示动作与环境的互动', pairsWith: ['leading lines'] },
    { zh: ['全景', '全景镜头'], en: 'full shot', usage: '人物从头到脚完整入画，交代动作起点', pairsWith: ['rule of thirds'] },
    { zh: ['中景', '中景镜头'], en: 'medium shot', usage: '膝盖以上入画，对话场景主力景别，兼顾动作与表情', pairsWith: ['rule of thirds'] },
    { zh: ['中近景'], en: 'medium close-up', usage: '腰部以上入画，对话微表情与手势', pairsWith: ['rule of thirds'] },
    { zh: ['近景', '近景镜头'], en: 'close-up', usage: '肩部以上入画，聚焦表情与情绪', pairsWith: ['centered', 'negative space'] },
    { zh: ['特写', '特写镜头'], en: 'big close-up', usage: '面部/局部细节，强调关键信息（眼神/手部/道具）', pairsWith: ['centered'] },
    { zh: ['大特写', '极特写'], en: 'extreme close-up', usage: '单一细节充满画面（瞳孔/戒指/裂缝），高张力时刻', pairsWith: ['centered'] },
  ],
  composition: [
    { zh: ['三分法', '三分之一'], en: 'rule of thirds', usage: '主体置于画面三分线上，通用且最稳的构图' },
    { zh: ['对称构图', '居中对称'], en: 'symmetrical composition', usage: '庄严/仪式感/压迫感，适合权力场面与正面冲突' },
    { zh: ['引导线', '引导线构图', '透视引导'], en: 'leading lines', usage: '用道路/栏杆/视线引导观众目光到主体' },
    { zh: ['框架构图', '框中框'], en: 'frame within frame', usage: '门框/窗户/栏杆框住主体，增加纵深与窥视感' },
    { zh: ['中心构图', '居中'], en: 'centered composition', usage: '主体居中，直给与注目感，适合特写' },
    { zh: ['负空间', '留白'], en: 'negative space', usage: '大面积空白衬托孤独/渺小/呼吸感' },
    { zh: ['对角线构图', '斜线构图'], en: 'diagonal composition', usage: '动势与不稳定感，适合追逐/冲突' },
    { zh: ['前景遮挡', '前景虚化'], en: 'foreground obstruction', usage: '前景物体虚化遮挡，营造窥视与空间层次' },
    { zh: ['纵深构图', '景深分层'], en: 'deep staging', usage: '前景/主体/背景三层景深，交代空间关系' },
  ],
  movement: [
    { zh: ['固定', '固定镜头', '静止', '静态', '固定机位'], en: 'static locked-off shot', usage: '机位稳定，适合对白与情绪静观', pairsWith: ['centered'] },
    { zh: ['推镜', '推近', '推进', '推镜头', '前推'], en: 'dolly in', usage: '镜头向主体推进，聚焦注意力/施加压力', pairsWith: ['close-up'] },
    { zh: ['拉镜', '拉远', '拉出', '拉镜头', '后拉'], en: 'dolly out', usage: '镜头远离主体，揭示环境/孤独感', pairsWith: ['wide shot'] },
    { zh: ['摇镜', '摇', '横摇', '摇拍', '摇镜头'], en: 'pan', usage: '机位不动镜头水平转动，扫描环境或追踪横向运动' },
    { zh: ['横移', '平移', '横向移动', '左移', '右移', '轨道横移'], en: 'lateral tracking', usage: '镜头与主体平行移动，营造旁观感或展示空间连续' },
    { zh: ['跟拍', '跟镜', '跟随', '跟随镜头', '跟踪'], en: 'tracking shot', usage: '镜头跟随主体移动，身临其境' },
    { zh: ['升降', '升镜', '降镜', '升降镜头'], en: 'crane shot', usage: '镜头垂直升降，揭示空间尺度或权力关系' },
    { zh: ['环绕', '环绕拍摄', '环拍', '360环绕', '旋转'], en: 'orbiting shot', usage: '镜头环绕主体，360度审视，揭示全貌' },
    { zh: ['手持', '手持跟拍', '手持镜头', '手持摄影'], en: 'handheld', usage: '画面轻微晃动，纪实感/慌乱感' },
    { zh: ['主观视角', '主观镜头', '第一人称'], en: 'point-of-view shot', usage: '角色第一人称视野，代入感强' },
    { zh: ['慢动作', '升格', '慢镜', '慢镜头'], en: 'slow motion', usage: '升格拍摄，放大情绪/细节/关键瞬间' },
    { zh: ['滑动变焦', '希区柯克变焦'], en: 'dolly zoom', usage: '主体不动背景压缩/拉伸，眩晕与不安感' },
    { zh: ['斜角', '荷兰角', '倾斜', '斜拍'], en: 'dutch angle', usage: '地平线倾斜，不安/失衡/精神错乱感' },
  ],
  lighting: [
    { zh: ['自然光', '自然光线', '日光'], en: 'natural light', usage: '真实自然光，纪录片感/日常感' },
    { zh: ['硬光', '强光'], en: 'hard light', usage: '强对比硬阴影，戏剧感/悬疑感' },
    { zh: ['柔光', '柔光箱', '柔和光'], en: 'soft light', usage: '均匀柔和，商业感/温情画面' },
    { zh: ['逆光', '背光', '轮廓光', '逆光剪影'], en: 'backlight / rim light', usage: '主体轮廓发光，神圣/神秘/剪影', pairsWith: ['silhouette'] },
    { zh: ['侧光', '伦勃朗光', '侧逆光'], en: 'side lighting', usage: '明暗各半，塑造立体感与人物深度', pairsWith: ['portrait'] },
    { zh: ['顶光', '顶部光'], en: 'top light', usage: '头顶直射，压抑/审讯感/悬疑' },
    { zh: ['低光', '暗调', '低调光', '低照度'], en: 'low-key lighting', usage: '大面积阴影，黑色电影/紧张氛围' },
    { zh: ['暖光', '暖色调', '暖色光'], en: 'warm lighting', usage: '橙黄色温，怀旧/温馨/暧昧' },
    { zh: ['冷光', '冷色调', '冷色光'], en: 'cool lighting', usage: '蓝青色温，疏离/科技感/夜色' },
    { zh: ['黄昏', '金色时刻', '黄金时刻'], en: 'golden hour light', usage: '日落前暖橙光线，浪漫/告别/高光时刻' },
    { zh: ['夜景', '夜晚光线'], en: 'night scene lighting', usage: '夜色照明，城市霓虹或月光，静谧/危险' },
  ],
}

/** 扁平化英文术语表：中文 → 英文（供 resolveVisualTerm 快速查表） */
const EN_BY_ZH = new Map<string, { en: string; category: VisualCategory }>()
for (const category of Object.keys(VISUAL_GRAPH) as VisualCategory[]) {
  for (const node of VISUAL_GRAPH[category]) {
    for (const zh of node.zh) {
      EN_BY_ZH.set(zh, { en: node.en, category })
    }
  }
}

interface VisualMatch {
  en: string
  category: VisualCategory
}

/**
 * 匹配中文视觉术语 → 英文电影术语（含类别）。
 * 匹配策略（与 camera-movement-guides 一致）：
 * 1. 精确匹配优先；
 * 2. 包含匹配兜底（如「快速横移」包含「横移」）。
 * @returns { en, category }；未命中返回 null（调用方保留原值）
 */
function matchVisualTerm(zhText: string | null | undefined): VisualMatch | null {
  const t = (zhText || '').trim()
  if (!t) return null

  // 1. 精确匹配（含别名）
  if (EN_BY_ZH.has(t)) return EN_BY_ZH.get(t)!

  // 2. 包含匹配：值包含某术语且术语长度 > 1
  for (const [zh, entry] of EN_BY_ZH) {
    if (zh.length > 1 && t.includes(zh)) return entry
  }
  return null
}

/** 把中文视觉术语映射为英文电影术语（不考虑类别，适合单字段翻译）。未命中返回 null。 */
export function resolveVisualTerm(zhText: string | null | undefined): string | null {
  return matchVisualTerm(zhText)?.en ?? null
}

/**
 * 按类别批量翻译：输入多段中文（景别/机位/运镜/灯光），逐段翻译，
 * 仅当命中术语属于目标类别时输出英文，否则保留原文。
 * 返回过滤掉空串后的英文/原文混合数组。
 */
export function resolveVisualTerms(category: VisualCategory, values: Array<string | null | undefined>): string[] {
  const out: string[] = []
  for (const v of values) {
    if (!v) continue
    const m = matchVisualTerm(v)
    if (m && m.category === category) {
      out.push(m.en)
    } else {
      out.push(v)
    }
  }
  return out
}

/** 获取某类图谱的全部英文术语列表（供 prompt 枚举） */
export function listVisualTerms(category: VisualCategory): string[] {
  return VISUAL_GRAPH[category].map((n) => n.en)
}

/**
 * 获取图谱某个类别的全部节点（供前端展示图谱库 / 生成图谱文档）。
 */
export function getVisualGraph(category?: VisualCategory): Record<VisualCategory, VisualGraphNode[]> {
  if (category) return { [category]: VISUAL_GRAPH[category] } as Record<VisualCategory, VisualGraphNode[]>
  return VISUAL_GRAPH
}

/**
 * 按剧风格生成「视觉图谱引导」文本（注入 storyboard_breaker instructions）。
 * 依据风格选择强调的图谱节点，并给出景别递进/构图节奏/运镜动机/灯光氛围的通用创作指导。
 */
export function buildVisualGraphGuidance(genre?: string | null, style?: string | null): string {
  const g = (genre || '').toLowerCase()
  const s = (style || '').toLowerCase()

  // 风格特征词 → 推荐图谱节点
  const sizeHints: string[] = []
  const compHints: string[] = []
  const moveHints: string[] = []
  const lightHints: string[] = []

  if (/(悬疑|惊悚|犯罪|推理|暗黑|黑帮)/.test(g + s)) {
    sizeHints.push('近景/特写', '中近景')
    compHints.push('框架构图')
    lightHints.push('低光', '硬光', '侧光', '顶光')
    moveHints.push('固定', '缓慢推镜', '手持')
  }
  if (/(爱情|甜宠|温情|治愈|家庭)/.test(g + s)) {
    sizeHints.push('中景', '中近景', '近景')
    compHints.push('三分法', '留白/负空间')
    lightHints.push('柔光', '暖光', '黄昏/金色时刻')
    moveHints.push('固定', '缓慢拉镜')
  }
  if (/(玄幻|武侠|古装|仙侠|奇幻|神话)/.test(g + s)) {
    sizeHints.push('全景', '远景')
    compHints.push('对称构图', '引导线', '纵深构图')
    lightHints.push('逆光', '冷光', '黄昏/金色时刻')
    moveHints.push('环绕', '升降', '慢动作')
  }
  if (/(都市|职场|现实|商战|行业)/.test(g + s)) {
    sizeHints.push('中景', '近景', '全景')
    compHints.push('三分法', '引导线', '框架构图')
    lightHints.push('自然光', '侧光', '硬光')
    moveHints.push('固定', '跟拍', '横移')
  }
  if (/(科幻|末世|末日|未来|赛博)/.test(g + s)) {
    sizeHints.push('大远景', '特写')
    compHints.push('对称构图', '引导线', '负空间')
    lightHints.push('冷光', '低光', '硬光')
    moveHints.push('环绕', '升降', '滑动变焦')
  }
  if (/(动作|警匪|打斗|复仇|战斗)/.test(g + s)) {
    sizeHints.push('中景', '全景', '特写')
    compHints.push('对角线构图', '前景遮挡')
    lightHints.push('硬光', '低光', '冷光')
    moveHints.push('手持', '快速横移', '跟拍', '慢动作')
  }

  const parts: string[] = []
  parts.push(`【视觉图谱引导】根据本剧类型${genre ? `（${genre}）` : ''}${style ? ` / 风格（${style}）` : ''}，拆镜时主动使用以下视觉图谱：`)
  parts.push('')
  parts.push('1. 景别递进：')
  parts.push('   - 开场/转场优先远景或全景建立空间；对话主力中景/中近景；情绪高潮推进到近景/特写；')
  parts.push('   - 相邻镜头避免连续同景别（如需同景别必须换机位或换构图，否则画面跳剪感强）；')
  if (sizeHints.length) parts.push(`   - 本剧推荐景别：${sizeHints.join(' / ')}。`)
  parts.push('')
  parts.push('2. 构图法则（每个镜头至少明确一种）：')
  parts.push('   - 三分法/对称/引导线/框架/中心/负空间/对角线/前景遮挡/纵深，按剧情意图选择；')
  parts.push('   - 构图意图要写进 image_prompt 与 first_frame_prompt/last_frame_prompt 的构图描述。')
  if (compHints.length) parts.push(`   - 本剧推荐构图：${compHints.join(' / ')}。`)
  parts.push('')
  parts.push('3. 运镜动机（每个镜头运镜必须有叙事理由）：')
  parts.push('   - 推镜=聚焦/施压，拉镜=揭示/孤独，摇镜=扫描/追踪，跟拍=身临其境，环绕=审视，手持=慌乱纪实；')
  parts.push('   - movement 字段写中文运镜名（如「缓慢推镜」「360环绕」），不要写英文。')
  if (moveHints.length) parts.push(`   - 本剧推荐运镜：${moveHints.join(' / ')}。`)
  parts.push('')
  parts.push('4. 灯光氛围（atmosphere 必须包含光线描述，体现「主光方向+光质+色温」）：')
  parts.push('   - 自然光/硬光/柔光/逆光/侧光/顶光/低光/暖光/冷光/黄昏/夜景，按情绪选择；')
  parts.push('   - 逆光配剪影制造神秘，侧光塑造立体，低光营造紧张，暖光传递温情。')
  if (lightHints.length) parts.push(`   - 本剧推荐灯光：${lightHints.join(' / ')}。`)
  parts.push('')
  parts.push('5. 一致性与安全区：以上图谱约束连同安全区/质感层/Mx-Shell 规则一并写入各镜 prompt，保持全剧视觉风格统一。')

  return parts.join('\n')
}

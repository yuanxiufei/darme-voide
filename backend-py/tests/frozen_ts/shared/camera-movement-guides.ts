/**
 * 运镜 → 首帧/尾帧构图指导（纯中文）
 *
 * 背景：voide 分镜的 movement（运镜）字段此前只被原样拼成 `Camera movement: xxx`，
 * 首帧/尾帧关键帧的 prompt 几乎完全相同，导致首尾关键帧与运镜方向脱节
 * （如「左摇」却首帧主体已在左侧）。本表把运镜翻译成首帧/尾帧的具体构图短语，
 * 让图生视频的首尾关键帧与运镜方向连贯。
 *
 * 语言设计：数据库实测 movement 字段 100% 为中文（固定/跟拍/推镜/横移/摇镜/慢动作/
 * 快速横移…），故匹配 key 仅用中文，无需英文 key（避免过度设计）。
 *
 * 排序至关重要：带方向的「具体」运镜在前，通用运镜在后。
 * 匹配规则（getCameraMovementComposition）：
 * 1. 精确匹配优先（含别名），保证「跟拍」命中通用跟拍、而非被「斜线跟拍」捕获；
 * 2. 包含匹配兜底（movement 值包含 key，如「快速横移」包含「横移」），
 *    且 key 长度 > 1（排除「摇/推/拉」等单字，避免误命中）。
 */

export interface CameraMovementGuide {
  /** 中文运镜名（用于匹配 movement 值） */
  zh: string
  /** 额外中文别名 */
  zhAliases?: string[]
  /** 首帧构图指导 */
  start: string
  /** 尾帧构图指导 */
  end: string
}

export const CAMERA_MOVEMENT_GUIDES: CameraMovementGuide[] = [
  { zh: '360环绕', zhAliases: ['360度环绕', '环绕360', '360环绕拍摄', '360度环绕拍摄'], start: '主体居中，镜头位于360度环绕起点', end: '主体居中，镜头完成360度整圈环绕' },
  { zh: '斜线跟拍', zhAliases: ['对角跟拍', '斜向跟拍', '斜线移拍'], start: '主体位于斜线运动路径，镜头沿斜线跟随', end: '保持斜线透视，空间动态推进' },
  { zh: '平行跟拍', zhAliases: ['平行移拍', '平行跟镜'], start: '主体与镜头平行并进', end: '保持平行关系，主体穿越空间' },
  { zh: '左移', zhAliases: ['向左移', '横移左', '左横移'], start: '主体位于画面右侧，左侧预留运动空间', end: '主体移至画面左侧，完成从右到左的横移' },
  { zh: '右移', zhAliases: ['向右移', '横移右', '右横移'], start: '主体位于画面左侧，右侧预留运动空间', end: '主体移至画面右侧，完成从左到右的横移' },
  { zh: '横移', zhAliases: ['横向移动', '平移', '横向平移'], start: '主体偏置一侧，预留横向移动空间', end: '主体完成横向平移，画面横向延展' },
  { zh: '左摇', zhAliases: ['向左摇', '左摇镜', '摇左'], start: '画面聚焦右侧区域，预留向左摇镜空间', end: '画面展现左侧区域，完成向左摇镜' },
  { zh: '右摇', zhAliases: ['向右摇', '右摇镜', '摇右'], start: '画面聚焦左侧区域，预留向右摇镜空间', end: '画面展现右侧区域，完成向右摇镜' },
  { zh: '升镜', zhAliases: ['上升', '升起', '升', '上升镜头'], start: '低位视角，主体贴近地面', end: '升至高位，主体上升展现向上运动' },
  { zh: '降镜', zhAliases: ['下降', '降下', '降', '下降镜头'], start: '高位视角，主体位于画面上方', end: '降至低位，主体下降展现向下运动' },
  { zh: '上摇', zhAliases: ['向上摇', '仰摇', '抬镜'], start: '镜头平视或略向下，捕捉主体下方', end: '镜头向上摇，展现上方高度' },
  { zh: '下摇', zhAliases: ['向下摇', '俯摇', '压镜'], start: '镜头平视或略向上，捕捉主体上方', end: '镜头向下摇，展现下方元素与地面' },
  { zh: '推镜', zhAliases: ['推近', '推进', '推', '推镜头'], start: '全景开场，完整场景入画，主体在画面中较小', end: '紧贴主体的特写，细节充满画面' },
  { zh: '拉镜', zhAliases: ['拉远', '拉出', '拉', '拉镜头'], start: '主体特写，突出细节', end: '拉远至全景，展现周围环境' },
  { zh: '滑动变焦', zhAliases: ['希区柯克变焦', '希区柯克', '滑动变焦镜头'], start: '眩晕效果前的平衡构图', end: '透视畸变，前景与背景关系被改变' },
  { zh: '推轨', zhAliases: ['轨道', '轨道推进', '轨道镜头'], start: '初始构图，主体处于特定距离', end: '透视变化，展现纵深与空间' },
  { zh: '过肩', zhAliases: ['过肩拍', '过肩镜头', '越肩', '越肩镜头'], start: '前景角色的肩部框定主体', end: '保持过肩构图，焦点或角度微调' },
  { zh: '鸟瞰', zhAliases: ['俯视全景', '鸟瞰视角', '俯拍全景', '顶视'], start: '正上方垂直俯视，展现整体布局', end: '保持垂直俯视，空间排布已变化' },
  { zh: '仰拍', zhAliases: ['仰视', '仰角', '仰角镜头'], start: '低机位仰视，强调高度与气势', end: '保持仰角，主体高大形成戏剧张力' },
  { zh: '俯拍', zhAliases: ['俯视', '俯角', '俯角镜头'], start: '高机位俯视，形成鸟瞰式全貌', end: '保持俯角，强调空间尺度与布局' },
  { zh: '主观视角', zhAliases: ['主观镜头', '主观', '第一人称'], start: '角色第一人称主观视角', end: '保持主观视角，展现角色所见' },
  { zh: '手持', zhAliases: ['手持跟拍', '手持镜头', '手持摄影'], start: '手持镜头动态构图，自然动感', end: '保持手持质感，随动自然重新构图' },
  { zh: '慢动作', zhAliases: ['升格', '慢镜', '慢镜头'], start: '动作起始，慢动作序列开端', end: '动作推进，突出优雅细节' },
  { zh: '斜角', zhAliases: ['荷兰角', '倾斜', '斜拍', '倾斜镜头'], start: '地平线倾斜，荷兰角构图，制造不安感', end: '保持荷兰角，强化迷失与张力' },
  { zh: '旋转', zhAliases: ['旋转镜', '转圈', '旋转镜头'], start: '主体入画，镜头开始旋转', end: '主体朝向随镜头旋转而变化' },
  { zh: '环绕', zhAliases: ['环绕拍摄', '环绕镜头', '环拍', '环绕跟拍'], start: '主体居中，镜头位于环绕路径起始角度', end: '主体仍居中，镜头转至对侧新角度' },
  { zh: '跟拍', zhAliases: ['跟镜', '跟随', '跟', '跟随镜头'], start: '主体入画，前后与侧方预留跟拍空间', end: '主体随镜头穿越空间，保持视觉关系' },
  { zh: '固定', zhAliases: ['固定镜头', '静止', '静态', '固定机位'], start: '固定机位，画面稳定', end: '机位不变，仅主体在画面内运动' },
  { zh: '摇镜', zhAliases: ['摇', '横摇', '摇拍', '摇镜头'], start: '主体偏置一侧，预留摇镜空间', end: '主体重新定位，完成摇镜运动' },
]

/**
 * 根据运镜类型返回首帧/尾帧构图指导。
 *
 * @param movement  运镜字段值（中文，如「左摇」「快速横移」）
 * @param frameType 'start'（首帧）| 'end'（尾帧）
 * @returns 匹配到的构图短语；未匹配时返回 null（调用方自行决定是否跳过）
 */
export function getCameraMovementComposition(
  movement: string | null | undefined,
  frameType: 'start' | 'end',
): string | null {
  const m = (movement || '').trim()
  if (!m) return null

  const pick = (g: CameraMovementGuide) => (frameType === 'start' ? g.start : g.end)

  // 1) 精确匹配优先（含别名）：保证「跟拍」命中通用跟拍、而非被「斜线跟拍」捕获
  for (const guide of CAMERA_MOVEMENT_GUIDES) {
    const keys = [guide.zh, ...(guide.zhAliases || [])]
    if (keys.includes(m)) return pick(guide)
  }

  // 2) 包含匹配兜底：仅当 movement 值包含 key（如「快速横移」包含「横移」），
  //    且 key 长度 > 1（排除「摇/推/拉」等单字，避免误命中）
  for (const guide of CAMERA_MOVEMENT_GUIDES) {
    const keys = [guide.zh, ...(guide.zhAliases || [])]
    for (const k of keys) {
      if (k.length > 1 && m.includes(k)) return pick(guide)
    }
  }

  return null
}

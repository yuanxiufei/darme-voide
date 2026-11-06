/**
 * 分镜行的**纯取值**工具：只看传进来的那个对象，不读任何页面状态。
 *
 * 为什么单独成文件：这些判定在 `pages/drama/[id]/episode/[episodeNumber].vue` 里
 * 被模板反复调用（一集几十个镜头 × 每个镜头十几处），但它们其实与那个页面无关——
 * 换个页面或抽成组件时就得整段再抄一遍。
 *
 * 收进来的判据只有一条：**不碰 ref、不碰 props、不碰 store**，同样的输入永远同样的输出。
 * 带状态的一律留在页面（例：`getStoryboardCharacterNames` 要读 characters 列表，
 * `getPreviousStoryboard` 要读 sbs 序列）。
 *
 * 另一个前提：后端字段有 snake_case 与 camelCase 两套（历史原因），取值一律两个都认——
 * 少认一套的后果是「明明有视频却显示待生成」，且不报错。
 */

// ===== 帧与视频 =====

export function getFirstFrame(s: any) { return s?.first_frame_image || s?.firstFrameImage || null }
export function getLastFrame(s: any) { return s?.last_frame_image || s?.lastFrameImage || null }
export function getKeyframe(s: any) { return s?.keyframe_image || s?.keyframeImage || null }

/**
 * 镜头封面取图优先级：合成图 → 首帧 → 尾帧。
 * 合成图排最前是有意的——它已经是「这个镜头最后长什么样」，拿首帧当封面会让人
 * 以为镜头没做完。
 */
export function getStoryboardCover(s: any) {
  return s?.composed_image || s?.composedImage || getFirstFrame(s) || getLastFrame(s) || null
}

export function getVideoUrl(s: any) { return s?.video_url || s?.videoUrl || null }
export function getComposedVideoUrl(s: any) { return s?.composed_video_url || s?.composedVideoUrl || null }
export function hasImg(s: any) { return !!getStoryboardCover(s) }
export function hasVid(s: any) { return !!getVideoUrl(s) }
export function hasComposed(s: any) { return !!getComposedVideoUrl(s) }

// ===== 台词与配音 =====

/** 这些「说话人」其实是环境音标注，不是角色，不该生成配音 */
export const IGNORE_TTS_SPEAKERS = /^(环境音|环境声|音效|效果音|sfx|sound ?effect|bgm|背景音|背景音乐|ambient)$/i
/** 这些「台词」等于「没有台词」，各种写法都见过 */
export const IGNORE_TTS_TEXT = /^(无|无对白|无台词|无旁白|无需配音|无需对白|none|null|n\/a|na|环境音|环境声|音效|效果音|纯音效|纯环境音|只有环境音|仅环境音|背景音|背景音乐|bgm|sfx|ambient)$/i

/**
 * 把一段多角色对话拆成独立对话行。
 *
 * 只认「角色名：较长台词」：冒号后必须 >= 8 字且不含冒号。这个门槛是踩出来的——
 * 放宽到任意冒号会把「时间：傍晚」这类场景描述也当成台词，于是镜头被判定
 * 「需要配音」并真的去合成一段念「傍晚」的音频。
 */
export function getDialogueLines(sb: any): { speaker: string; text: string }[] {
  const raw = sb?.dialogue?.trim() || ''
  if (!raw) return []
  const lines: { speaker: string; text: string }[] = []
  const regex = /([^\n:：]{1,10}?)[:：]([^:：\n]{8,})/g
  let match

  while ((match = regex.exec(raw)) !== null) {
    const speaker = match[1].replace(/[（(].+?[)）]/g, '').trim()
    const text = match[2].trim()
    if (!speaker || speaker.length > 10 || IGNORE_TTS_SPEAKERS.test(speaker)) continue
    if (IGNORE_TTS_TEXT.test(text)) continue
    lines.push({ speaker, text })
  }

  return lines
}

/** 第一个冒号之前的部分＝说话人（拆不出多行时用），括号里的情绪标注一并去掉 */
export function getDialogueSpeakerRaw(sb: any) {
  const dialogue = sb?.dialogue?.trim() || ''
  const match = dialogue.match(/^(.+?)[:：]/)
  return match ? match[1].replace(/[（(].+?[)）]/g, '').trim() : ''
}

/** 第一个冒号之后的全部＝台词正文 */
export function getDialogueText(sb: any) {
  const dialogue = sb?.dialogue?.trim() || ''
  return dialogue ? dialogue.replace(/^.+?[:：]\s*/, '').trim() : ''
}

/** 这个镜头到底需不需要配音（要不要给「批量配音」算进待办） */
export function isTTSIgnorable(sb: any) {
  const lines = getDialogueLines(sb)
  const raw = sb?.dialogue?.trim() || ''
  if (!raw) return true
  // 有有效对话行 → 不忽略
  if (lines.length > 0) return false
  // 拆分不出多角色对话行，但有原文 → fallback 到单人对话口径
  const speaker = getDialogueSpeakerRaw(sb)
  const text = getDialogueText(sb)
  if (speaker && IGNORE_TTS_SPEAKERS.test(speaker)) return true
  if (!text) return true
  if (IGNORE_TTS_TEXT.test(text)) return true
  return false
}

export function hasDialogue(sb: any) { return !isTTSIgnorable(sb) }
export function hasTTS(sb: any) { return !!(sb?.tts_audio_url || sb?.ttsAudioUrl) }
export function getTTSUrl(sb: any) { return sb?.tts_audio_url || sb?.ttsAudioUrl || '' }

// ===== 角色/音色标签 =====

/** 角色类型标签（旁白/主角/反派/配角）——音色库快速筛选的元数据 */
export const ROLE_TAGS = [
  { label: '旁白', cls: 'role-tag-narrator' },
  { label: '主角', cls: 'role-tag-protagonist' },
  { label: '反派', cls: 'role-tag-villain' },
  { label: '配角', cls: 'role-tag-supporting' },
]

export function roleTagCls(label: string) {
  if (label === '龙套') return 'role-tag-extra'
  return ROLE_TAGS.find(t => t.label === label)?.cls || ''
}

/** 依据音色名 + 描述关键词推断该音色适合的角色类型（一个音色可打多个标签） */
export function inferRoleTags(name: string, desc: string[] | string = []) {
  const text = `${name || ''} ${Array.isArray(desc) ? desc.join(' ') : (desc || '')}`.toLowerCase()
  const tags: string[] = []
  if (/旁白|讲述|解说|叙述|播音|磁性|沉稳|醇厚|浑厚|大气|成熟男/.test(text)) tags.push('旁白')
  if (/反派|低沉|冷酷|霸道|阴冷|沙哑|狠辣|凶狠|邪恶|强势|腹黑/.test(text)) tags.push('反派')
  if (/主角|男主|女主|青年|少年|少女|阳光|甜美|温柔|清亮|青春|活力|甜润|元气|朝气/.test(text)) tags.push('主角')
  if (/配角|大叔|大爷|大妈|奶奶|爷爷|老年|萝莉|正太|孩童|小孩|逗趣|憨厚|邻居/.test(text)) tags.push('配角')
  return tags.slice(0, 3)
}

/** 依据角色姓名 + 角色定位推断角色类型，用于角色卡片展示对照标签 */
export function inferCharRoleTag(c: any) {
  const text = `${c?.name || ''} ${c?.role || ''}`.toLowerCase()
  if (text.includes('龙套') || text.includes('路人') || text.includes('群演') || text.includes('extra')) return '龙套'
  const rt = c?.role_type || c?.roleType || ''
  if (['主角', '反派', '配角', '旁白', '龙套'].includes(rt)) return rt
  if (text.includes('旁白') || text.includes('narrator') || text.includes('画外音')) return '旁白'
  if (text.includes('反派') || text.includes('villain') || text.includes('antagonist')) return '反派'
  if (text.includes('主角') || text.includes('男主') || text.includes('女主') || text.includes('protagonist') || text.includes('hero')) return '主角'
  if (text.includes('配角') || text.includes('supporting') || text.includes('support')) return '配角'
  return ''
}

/** 依据音色名/描述猜性别——只用于展示，猜不出就老实说「中性」 */
export function inferVoiceGender(name: string, desc: string[] | string = []) {
  const text = `${name} ${Array.isArray(desc) ? desc.join(' ') : ''}`
  if (/[男|青年|大爷|学长|boy|man|male]/i.test(text)) return '男声'
  if (/[女|少女|御姐|奶奶|girl|woman|female]/i.test(text)) return '女声'
  return '中性'
}

// ===== AI 配置显示名 =====

/**
 * AI 配置的下拉显示名：`名称 · 模型 (provider)`。
 *
 * `model` 字段后端存的是 JSON 字符串（可能 `["qwen-image"]`、也可能是纯字符串），
 * 解析失败就退回原值——这里绝不能抛错，它只负责显示。
 *
 * `empty` 给不同页面留口：剧集页要「未配置」提醒没选，剧集列表页宁可空白。
 */
export function configLabel(config: any, empty = '未配置') {
  if (!config) return empty
  let modelName = ''
  try {
    const m = JSON.parse(config.model || '[]')
    modelName = Array.isArray(m) ? (m[0] || '') : (m || '')
  } catch {
    modelName = config.model || ''
  }
  return modelName ? `${config.name} · ${modelName} (${config.provider})` : `${config.name} (${config.provider})`
}

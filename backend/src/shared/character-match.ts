/**
 * 台词说话人 → 角色匹配（跨集一致性）：
 * 对白可能使用简称/昵称/带称谓称呼（如「阿晚」「晚晚」「林小姐」），
 * 与角色全名（如「林晚」）做轻量别名归一，保证跨集对白落到同一角色/音色。
 * 仅在「全剧角色中唯一命中」时采用，避免误配。
 */

/** 常见称谓后缀/前缀，用于说话人别名归一 */
const SPEAKER_TITLE_SUFFIX = /(?:小姐|夫人|太太|先生|公子|老爷|少爷|姑娘|师傅|师父|前辈|老师|阿姨|哥哥|姐姐|妹妹|弟弟|妈妈|爸爸|母亲|父亲|奶奶|爷爷|祖母|祖父)$/
const SPEAKER_TITLE_PREFIX = /^(?:阿|小|老)/

export function stripSpeakerAffixes(name: string): string {
  let out = name.trim()
  let prev = ''
  while (prev !== out && out.length > 0) {
    prev = out
    out = out.replace(SPEAKER_TITLE_SUFFIX, '').replace(SPEAKER_TITLE_PREFIX, '')
  }
  return out
}

export type SpeakerNameChar = { id: number; name: string | null }

/**
 * 全剧唯一命中才返回；无法唯一确定时返回 null（调用方按 not_found 处理）。
 * 1) 全名精确
 * 2) 说话人去称谓词缀后与全名精确（如「阿晚」「林小姐」→「林晚」）
 * 3) 双向包含（如「晚晚」「晚儿」⊂「林晚」，须长度>=2 且全剧唯一）
 */
export function matchCharacterBySpeakerName<T extends SpeakerNameChar>(
  chars: T[],
  speaker: string,
): T | null {
  const raw = (speaker || '').trim()
  if (!raw) return null

  const exact = chars.filter(c => (c.name || '').trim() === raw)
  if (exact.length === 1) return exact[0]

  const norm = stripSpeakerAffixes(raw)
  if (norm && norm !== raw) {
    const candidates = chars.filter((c) => {
      const n = (c.name || '').trim()
      if (!n || n === raw) return false
      if (n === norm || stripSpeakerAffixes(n) === norm) return true
      const a = norm.length >= 2 && n.includes(norm)
      const b = n.length >= 2 && norm.includes(n)
      return a || b
    })
    if (candidates.length === 1) return candidates[0]
  }
  return null
}

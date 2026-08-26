/**
 * 确定性评分器（纯函数，不依赖 LLM / DB）
 *
 * 把"分镜拆得好不好""角色提取全不全"变成可计算的分数。
 * 所有维度都来自 Agent 提示词里明确要求的规则，优化提示词时这些维度会随之变化，
 * 因此分数能客观反映提示词质量。
 */
import type {
  StoryboardShot,
  ExtractedCharacter,
  ExtractedScene,
  StoryboardRubric,
  ExtractorRubric,
  ScriptRewriterRubric,
  VoiceAssignerRubric,
  ScoreReport,
  ScoreDimension,
} from './types.js'

const round = (n: number) => Math.round(n * 10) / 10

/** 判断字段是否"填写了有意义的值" */
function isFilled(v: unknown): boolean {
  if (v == null) return false
  if (typeof v === 'string') return v.trim().length > 0
  if (typeof v === 'number') return Number.isFinite(v)
  if (Array.isArray(v)) return v.length > 0
  return true
}

/**
 * 分镜评分。
 * @param shots        save_storyboards 提交的分镜数组
 * @param rubric       评分规则
 * @param legal        seed 时产生的合法 ID 集合（用于校验绑定是否凭空捏造）
 */
export function scoreStoryboards(
  shots: StoryboardShot[],
  rubric: StoryboardRubric,
  legal: { characterIds: Set<number>; sceneIds: Set<number> },
): ScoreReport {
  const dims: ScoreDimension[] = []
  const n = shots.length

  // 1. 镜头数量（10 分）
  {
    const inRange = n >= rubric.minShots && n <= rubric.maxShots
    const score = n === 0 ? 0 : inRange ? 10 : 5
    dims.push({
      name: '镜头数量合理',
      score,
      max: 10,
      detail: `${n} 个镜头（期望 ${rubric.minShots}~${rubric.maxShots} 之间）`,
    })
  }

  // 2. 核心字段完整率（30 分）
  {
    const perShot = shots.map(s => {
      const filled = rubric.requiredFields.filter(f => isFilled((s as Record<string, unknown>)[f]))
      return filled.length / rubric.requiredFields.length
    })
    const avg = n ? perShot.reduce((a, b) => a + b, 0) / n : 0
    const missing = rubric.requiredFields.filter(f => shots.some(s => !isFilled((s as Record<string, unknown>)[f])))
    dims.push({
      name: '核心字段完整率',
      score: round(avg * 30),
      max: 30,
      detail: n ? `平均覆盖率 ${(avg * 100).toFixed(0)}%${missing.length ? `，常缺字段：${missing.join('/')}` : ''}` : '无分镜',
    })
  }

  // 3. video_prompt 标记完整（20 分）
  {
    const perShot = shots.map(s => {
      const vp = s.video_prompt || ''
      const present = rubric.videoPromptTags.filter(t => vp.includes(t))
      return present.length / rubric.videoPromptTags.length
    })
    const avg = n ? perShot.reduce((a, b) => a + b, 0) / n : 0
    dims.push({
      name: 'video_prompt 标记',
      score: round(avg * 20),
      max: 20,
      detail: `标记 ${rubric.videoPromptTags.join(' ')} 平均覆盖 ${(avg * 100).toFixed(0)}%`,
    })
  }

  // 4. duration 范围（10 分）
  {
    const ok = shots.filter(s => {
      const d = s.duration
      return d != null && d >= rubric.durationRange[0] && d <= rubric.durationRange[1]
    }).length
    dims.push({
      name: 'duration 范围',
      score: n ? round((ok / n) * 10) : 0,
      max: 10,
      detail: `${ok}/${n} 个镜头在 ${rubric.durationRange[0]}~${rubric.durationRange[1]} 秒`,
    })
  }

  // 5. title 长度（10 分）
  {
    const ok = shots.filter(s => {
      const t = (s.title || '').trim()
      return t.length >= rubric.titleLengthRange[0] && t.length <= rubric.titleLengthRange[1]
    }).length
    dims.push({
      name: 'title 长度',
      score: n ? round((ok / n) * 10) : 0,
      max: 10,
      detail: `${ok}/${n} 个镜头标题在 ${rubric.titleLengthRange[0]}~${rubric.titleLengthRange[1]} 字`,
    })
  }

  // 6. 角色/场景绑定合法（20 分）—— 防止凭空捏造 ID
  {
    const okCount = shots.filter(s => {
      const sceneOk = s.scene_id == null || legal.sceneIds.has(s.scene_id)
      const charsOk = !s.character_ids?.length || s.character_ids.every(id => legal.characterIds.has(id))
      return sceneOk && charsOk
    }).length
    dims.push({
      name: '角色/场景绑定合法',
      score: n ? round((okCount / n) * 20) : 0,
      max: 20,
      detail: `${okCount}/${n} 个镜头 scene_id/character_ids 均来自上下文合法集合`,
    })
  }

  const total = round(dims.reduce((a, d) => a + d.score, 0))
  return { caseId: '', kind: 'storyboard', dimensions: dims, total }
}

/**
 * 提取评分。
 * @param characters   save_dedup_characters 提交的角色（已合并多次调用）
 * @param scenes       save_dedup_scenes 提交的场景
 */
export function scoreExtraction(
  characters: ExtractedCharacter[],
  scenes: ExtractedScene[],
  rubric: ExtractorRubric,
): ScoreReport {
  const dims: ScoreDimension[] = []

  const goldenChars = rubric.goldenCharacters
  const extractedNames = new Set(characters.map(c => (c.name || '').trim()).filter(Boolean))
  const extractedLocs = new Set(scenes.map(s => (s.location || '').trim()).filter(Boolean))
  const goldenLocs = rubric.goldenScenes.map(s => s.location)

  // 1. 角色召回率（35 分）
  {
    const recalled = goldenChars.filter(g => extractedNames.has(g)).length
    const rate = goldenChars.length ? recalled / goldenChars.length : 0
    dims.push({
      name: '角色召回率',
      score: round(rate * 35),
      max: 35,
      detail: `${recalled}/${goldenChars.length}（漏提：${goldenChars.filter(g => !extractedNames.has(g)).join('、') || '无'}）`,
    })
  }

  // 2. 角色精确率（25 分）—— 防止多提/幻觉角色
  {
    const precise = [...extractedNames].filter(n => goldenChars.includes(n)).length
    const rate = extractedNames.size ? precise / extractedNames.size : 0
    dims.push({
      name: '角色精确率',
      score: round(rate * 25),
      max: 25,
      detail: `${precise}/${extractedNames.size}（多提：${[...extractedNames].filter(n => !goldenChars.includes(n)).join('、') || '无'}）`,
    })
  }

  // 3. 场景召回率（15 分）
  {
    const recalled = goldenLocs.filter(g => extractedLocs.has(g)).length
    const rate = goldenLocs.length ? recalled / goldenLocs.length : 0
    dims.push({
      name: '场景召回率',
      score: round(rate * 15),
      max: 15,
      detail: `${recalled}/${goldenLocs.length}（漏提：${goldenLocs.filter(g => !extractedLocs.has(g)).join('、') || '无'}）`,
    })
  }

  // 4. 场景精确率（10 分）
  {
    const precise = [...extractedLocs].filter(l => goldenLocs.includes(l)).length
    const rate = extractedLocs.size ? precise / extractedLocs.size : 0
    dims.push({
      name: '场景精确率',
      score: round(rate * 10),
      max: 10,
      detail: `${precise}/${extractedLocs.size}（多提：${[...extractedLocs].filter(l => !goldenLocs.includes(l)).join('、') || '无'}）`,
    })
  }

  // 5. 外貌描述完整（15 分）
  {
    const withAppearance = characters.filter(c => (c.appearance || '').trim().length > 0)
    const ok = withAppearance.filter(c => (c.appearance || '').trim().length >= rubric.minAppearanceLength).length
    const rate = characters.length ? ok / characters.length : 0
    dims.push({
      name: '外貌描述完整',
      score: round(rate * 15),
      max: 15,
      detail: `${ok}/${characters.length} 个角色外貌描述 ≥ ${rubric.minAppearanceLength} 字`,
    })
  }

  const total = round(dims.reduce((a, d) => a + d.score, 0))
  return { caseId: '', kind: 'extractor', dimensions: dims, total }
}

/**
 * 剧本改写评分。
 * @param content  save_script 提交的格式化剧本
 */
export function scoreScriptRewrite(content: string, rubric: ScriptRewriterRubric): ScoreReport {
  const dims: ScoreDimension[] = []
  const text = content || ''

  // 场景头正则：`## S01 | 内景 · 地点 | 时间段`（编号 + 内外景·地点 + 时间段三段）
  const sceneHeaderRe = /##\s*S(\d+)\s*\|\s*[^|\n]*\|\s*[^|\n]*/g
  const headers = text.match(sceneHeaderRe) || []
  const sceneNumbers = headers
    .map(h => parseInt(h.match(/S(\d+)/i)?.[1] || '', 10))
    .filter(n => Number.isFinite(n))

  // 1. 剧本已保存（20 分）
  {
    const len = text.trim().length
    const score = len === 0 ? 0 : len >= 100 ? 20 : 10
    dims.push({ name: '剧本已保存', score, max: 20, detail: `剧本长度 ${len} 字` })
  }

  // 2. 场景头格式规范（25 分）
  {
    const rate = headers.length >= rubric.minScenes ? 1 : headers.length / Math.max(rubric.minScenes, 1)
    dims.push({
      name: '场景头格式规范',
      score: round(rate * 25),
      max: 25,
      detail: `${headers.length} 个场景头（期望 ≥ ${rubric.minScenes}）`,
    })
  }

  // 3. 场景编号连续（15 分）
  {
    const sorted = [...sceneNumbers].sort((a, b) => a - b)
    let consecutive = 0
    for (let i = 0; i < sorted.length; i++) if (sorted[i] === i + 1) consecutive++
    const rate = sorted.length ? consecutive / sorted.length : 0
    dims.push({
      name: '场景编号连续',
      score: round(rate * 15),
      max: 15,
      detail: `${consecutive}/${sorted.length} 个场景编号从 S01 连续递增`,
    })
  }

  // 4. 对白格式（20 分）：`角色名：（动作）台词` 行
  {
    const dialogueRe = /^[^\n#|]{1,12}[：:]\s*(?:（[^）]*）)?[^\n]/gm
    const dialogueLines = text.match(dialogueRe) || []
    dims.push({
      name: '对白格式',
      score: dialogueLines.length > 0 ? 20 : 0,
      max: 20,
      detail: `${dialogueLines.length} 行对白`,
    })
  }

  // 5. 无镜头语言（20 分）：景别/运镜等属于分镜步骤，不应出现在剧本
  {
    const hits = rubric.forbiddenCameraWords.filter(w => text.includes(w))
    dims.push({
      name: '无镜头语言',
      score: hits.length === 0 ? 20 : 0,
      max: 20,
      detail: hits.length ? `出现违禁词：${hits.join('/')}` : '无景别/运镜等镜头语言',
    })
  }

  const total = round(dims.reduce((a, d) => a + d.score, 0))
  return { caseId: '', kind: 'script_rewriter', dimensions: dims, total }
}

/**
 * 音色分配评分。
 * @param assignments  assign_voice 提交的分配（character_id + voice_id + reason）
 * @param legal        seed 时产生的合法角色 ID 集合
 */
export function scoreVoiceAssignment(
  assignments: Array<{ character_id?: number; voice_id?: string; reason?: string }>,
  rubric: VoiceAssignerRubric,
  legal: { characterIds: Set<number> },
): ScoreReport {
  const dims: ScoreDimension[] = []
  const n = legal.characterIds.size
  const assignedIds = new Set(assignments.map(a => a.character_id).filter((id): id is number => id != null))

  // 1. 角色覆盖（40 分）：每个待分配角色都拿到音色
  {
    const covered = [...legal.characterIds].filter(id => assignedIds.has(id)).length
    const rate = n ? covered / n : 0
    dims.push({ name: '角色覆盖', score: round(rate * 40), max: 40, detail: `${covered}/${n} 个角色已分配音色` })
  }

  // 2. 音色 ID 合法（30 分）：voice_id 必须来自 list_voices 的音色库
  {
    const legalSet = new Set(rubric.legalVoiceIds)
    const ok = assignments.filter(a => a.voice_id && legalSet.has(a.voice_id)).length
    const totalAssign = assignments.length
    const rate = totalAssign ? ok / totalAssign : 0
    dims.push({
      name: '音色 ID 合法',
      score: round(rate * 30),
      max: 30,
      detail: `${ok}/${totalAssign} 条分配的 voice_id 在音色库中`,
    })
  }

  // 3. 分配理由说明（15 分）
  {
    if (!rubric.requireReason) {
      dims.push({ name: '分配理由说明', score: 15, max: 15, detail: '不要求理由（跳过）' })
    } else {
      const withReason = assignments.filter(a => (a.reason || '').trim().length > 0).length
      const rate = assignments.length ? withReason / assignments.length : 0
      dims.push({
        name: '分配理由说明',
        score: round(rate * 15),
        max: 15,
        detail: `${withReason}/${assignments.length} 条分配附理由`,
      })
    }
  }

  // 4. 角色 ID 合法无重复（15 分）
  {
    const inLegal = assignments.filter(a => a.character_id != null && legal.characterIds.has(a.character_id)).length
    const dupCount = assignments.length - assignedIds.size
    const rate = assignments.length ? (inLegal - dupCount) / assignments.length : 0
    const clamped = Math.max(0, Math.min(1, rate))
    dims.push({
      name: '角色 ID 合法无重复',
      score: round(clamped * 15),
      max: 15,
      detail: `${inLegal}/${assignments.length} 条 character_id 合法，重复 ${dupCount} 条`,
    })
  }

  const total = round(dims.reduce((a, d) => a + d.score, 0))
  return { caseId: '', kind: 'voice_assigner', dimensions: dims, total }
}

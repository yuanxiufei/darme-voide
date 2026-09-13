/**
 * 临时探针（用完即删）：对照实验 —— 精简 save_storyboards 的字段数是否能提高落库率。
 * 用法：PROBE_RUNS=n 控制跑几次。判据只看 toolCalls 里有没有 save*、以及落库条数。
 */
import { writeFileSync, appendFileSync } from 'node:fs'
import { eq } from 'drizzle-orm'
import { db } from './src/db/index.js'
import * as schema from './src/db/schema.js'
import { now } from './src/utils/response.js'
import { runAgentWithRetry } from './src/agents/index.js'

const LABEL = process.env.PROBE_LABEL || 'run'
const RUNS = Number(process.env.PROBE_RUNS || 1)
const out: string[] = []
const log = (...a: unknown[]) => out.push(a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' '))

const SCRIPT = `## S01 | 外景 · 城中村小巷 · 夜

（暴雨如注，霓虹灯牌在积水里碎成一片红绿。）

林晚：（举着破伞，喘着气）别追了……我真的不知道东西在哪。
陈默：（逼近一步，压低声音）你哥把它交给你了。我知道。
林晚：（后退，背抵上卷帘门）……你疯了。

## S02 | 内景 · 24小时便利店 · 夜

（冷白灯管下，林晚把湿透的帆布包放在收银台上，手在抖。）

店员：（困倦）要热咖啡吗？
林晚：（摇头，突然盯着玻璃门外的雨）……刚才有人进来过吗？
陈默：（隔着雨帘）我不想要那把伞。我想要你哥留下的坐标。`

async function runOnce(idx: number) {
  const ts = now()
  const drama = db.insert(schema.dramas).values({ title: `[probe] schema-${idx}`, status: 'draft', createdAt: ts, updatedAt: ts }).run()
  const dramaId = Number(drama.lastInsertRowid)
  const ep = db.insert(schema.episodes)
    .values({ dramaId, episodeNumber: 1, title: '探针集', scriptContent: SCRIPT, createdAt: ts, updatedAt: ts }).run()
  const episodeId = Number(ep.lastInsertRowid)
  for (const c of [
    { name: '林晚', role: '女主角', appearance: '短发，黑色风衣，旧帆布包', personality: '外冷内热' },
    { name: '陈默', role: '男主角', appearance: '高个，灰色夹克，左手腕有旧疤', personality: '沉默寡言' },
    { name: '店员', role: '配角', appearance: '便利店制服，倦容', personality: '事不关己' },
  ]) {
    const r = db.insert(schema.characters)
      .values({ dramaId, name: c.name, role: c.role, appearance: c.appearance, personality: c.personality, createdAt: ts, updatedAt: ts }).run()
    db.insert(schema.episodeCharacters).values({ episodeId, characterId: Number(r.lastInsertRowid), createdAt: ts }).run()
  }
  for (const s of [
    { location: '城中村小巷', time: '夜', prompt: '暴雨夜的城中村窄巷，两侧卷帘门与霓虹招牌，地面积水反射红绿灯光' },
    { location: '24小时便利店', time: '夜', prompt: '便利店冷白灯管照明，货架整齐，玻璃门上映着雨幕' },
  ]) {
    const r = db.insert(schema.scenes)
      .values({ dramaId, episodeId, location: s.location, time: s.time, prompt: s.prompt, createdAt: ts, updatedAt: ts }).run()
    db.insert(schema.episodeScenes).values({ episodeId, sceneId: Number(r.lastInsertRowid), createdAt: ts }).run()
  }

  let status = 'unknown'
  let timer: NodeJS.Timeout | undefined
  try {
    const t0 = Date.now()
    const ret: any = await Promise.race([
      runAgentWithRetry('storyboard_breaker', episodeId, dramaId, '请对当前集剧本进行分镜拆解并保存。', { maxSteps: 10 }),
      new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('TIMEOUT_180s')), 180000) }),
    ])
    if (timer) clearTimeout(timer)
    const names: string[] = (ret?.toolCalls || []).map((c: any) => String(c?.toolName || '?'))
    const sbs = db.select().from(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).all()
    status = sbs.length > 0 ? 'OK(' + sbs.length + ')' : 'BROKEN'
    log(`  [${LABEL}#${idx}] ${((Date.now() - t0) / 1000).toFixed(1)}s tools=[${names.join(',')}] out=${ret?.usage?.outputTokens} reply=${String(ret?.text || '').length} saved=${sbs.length} => ${status}`)
  } catch (e: any) {
    if (timer) clearTimeout(timer)
    log(`  [${LABEL}#${idx}] ERROR ${e?.message || String(e)}`)
  }

  try {
    const sbs = db.select().from(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).all()
    for (const s of sbs) db.delete(schema.storyboardCharacters).where(eq(schema.storyboardCharacters.storyboardId, (s as any).id)).run()
    db.delete(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).run()
    db.delete(schema.episodeScenes).where(eq(schema.episodeScenes.episodeId, episodeId)).run()
    db.delete(schema.episodeCharacters).where(eq(schema.episodeCharacters.episodeId, episodeId)).run()
    db.delete(schema.scenes).where(eq(schema.scenes.dramaId, dramaId)).run()
    db.delete(schema.characters).where(eq(schema.characters.dramaId, dramaId)).run()
    db.delete(schema.episodes).where(eq(schema.episodes.id, episodeId)).run()
    db.delete(schema.dramas).where(eq(schema.dramas.id, dramaId)).run()
  } catch (e: any) {
    log('  cleanup FAILED: ' + (e?.message || String(e)))
  }
}

for (let i = 1; i <= RUNS; i++) await runOnce(i)
appendFileSync('../tmp/schema-probe.txt', out.join('\n') + '\n', 'utf8')
console.log(out.join('\n'))

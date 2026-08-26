import Database from 'better-sqlite3'

const BASE = 'http://localhost:5789/api/v1'
const DB_PATH = 'd:/code/voides/voide-darme/data/drama.db'
const db = new Database(DB_PATH, { timeout: 30000 })

let pass = 0, fail = 0
function assert(name, cond, extra = '') {
  if (cond) pass++; else fail++
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  [' + extra + ']' : ''}`)
}

async function req(method, path, body) {
  const opts = { method, headers: {} }
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body) }
  const res = await fetch(BASE + path, opts)
  const text = await res.text()
  let json; try { json = JSON.parse(text) } catch { json = text }
  return { status: res.status, json }
}

let dramaId, charId, epId

try {
  // 1. 创建 drama（自动建 1 集 episode）
  const d = await req('POST', '/dramas', { title: 'SMOKE-SD-TEST', total_episodes: 1 })
  dramaId = d.json?.data?.id
  assert('创建测试 drama', dramaId != null, `dramaId=${dramaId}`)

  const detail = await req('GET', `/dramas/${dramaId}`)
  epId = detail.json?.data?.episodes?.[0]?.id
  assert('自动创建 episode', epId != null, `epId=${epId}`)

  // 2. 创建测试角色
  await req('PUT', `/dramas/${dramaId}/characters`, { characters: [{ name: 'SMOKE-VISIBLE-CHAR', role: '配角' }] })
  const detail2 = await req('GET', `/dramas/${dramaId}`)
  const ch = (detail2.json?.data?.characters || []).find(c => c.name === 'SMOKE-VISIBLE-CHAR')
  charId = ch?.id
  assert('创建测试角色并可见', charId != null, `charId=${charId}`)

  // 3. 软删前 GET /characters/:id 可见
  const before = await req('GET', `/characters/${charId}`)
  assert('软删前 GET /characters/:id 可见', before.json?.code === 200 && before.json?.data?.name === 'SMOKE-VISIBLE-CHAR')

  // 4. 软删
  const del = await req('DELETE', `/characters/${charId}`)
  assert('软删成功', del.json?.code === 200, `code=${del.json?.code}`)

  // 5. 软删后各端点验证
  const after = await req('GET', `/characters/${charId}`)
  assert('软删后 GET /characters/:id 不可见(404)', after.json?.code === 404, `code=${after.json?.code} msg=${after.json?.message}`)

  const putAfter = await req('PUT', `/characters/${charId}`, { name: 'SHOULD-NOT-UPDATE' })
  assert('软删后 PUT /characters/:id 读回为空(不复活)', putAfter.json?.data == null, `data=${JSON.stringify(putAfter.json?.data)}`)

  const detailAfter = await req('GET', `/dramas/${dramaId}`)
  const charsAfter = detailAfter.json?.data?.characters || []
  assert('软删后 GET /dramas/:id 不含该角色', !charsAfter.some(c => c.id === charId), `charCount=${charsAfter.length}`)

  const listAfter = await req('GET', '/dramas?page=1&page_size=200')
  const dInList = (listAfter.json?.data?.items || []).find(x => x.id === dramaId)
  const listChars = dInList?.characters || []
  assert('软删后 GET /dramas 列表不含该角色', dInList != null && !listChars.some(c => c.id === charId), `listCharCount=${listChars.length}`)

  const ps = await req('GET', `/episodes/${epId}/pipeline-status`)
  const extCount = ps.json?.data?.steps?.extract_characters?.count
  assert('软删后 pipeline-status 角色统计为 0', extCount === 0, `extract_characters.count=${extCount}`)
} catch (e) {
  console.error('EXCEPTION:', e)
  fail++
} finally {
  if (dramaId) {
    db.prepare('DELETE FROM characters WHERE drama_id = ?').run(dramaId)
    db.prepare('DELETE FROM episodes WHERE drama_id = ?').run(dramaId)
    db.prepare('DELETE FROM dramas WHERE id = ?').run(dramaId)
    console.log(`[cleanup] 已物理清理 dramaId=${dramaId} 的测试数据`)
  }
  db.close()
  console.log(`\nRESULT: ${pass} pass, ${fail} fail`)
  process.exit(fail === 0 ? 0 : 1)
}

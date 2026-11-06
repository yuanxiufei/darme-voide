<template>
  <span class="pf-shell">
    <button
      class="btn pf-trigger"
      :class="{ 'is-blocked': shownBlockers > 0 }"
      :disabled="!episodeId"
      :title="'开跑前体检：连续性 / 资产验收门（只读）'"
      @click="run"
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 11l3 3 8-8" />
        <path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9" />
      </svg>
      体检
      <span v-if="shownBlockers > 0" class="pf-badge">{{ shownBlockers }}</span>
    </button>

    <div v-if="open" class="pf-overlay" @click.self="close">
      <div class="pf-panel" role="dialog" aria-modal="true" aria-label="开跑前体检">
        <header class="pf-head">
          <div class="pf-head-copy">
            <h3>开跑前体检</h3>
            <p class="pf-sub">只读：连续性 / 资产验收门，不改任何数据</p>
          </div>
          <button class="pf-x" aria-label="关闭" @click="close">×</button>
        </header>

        <div v-if="loading" class="pf-state">正在体检…（后端按集取数，镜头多时约 1–2 秒）</div>
        <div v-else-if="err" class="pf-state pf-err">{{ err }}</div>

        <div v-else-if="report" class="pf-body">
          <div class="pf-summary" :class="report.ready ? 'ok' : 'bad'">
            <b>{{ report.ready ? '可以开跑' : '还不能开跑' }}</b>
            <span>阻断 {{ (report.blockers || []).length }} · 提示 {{ (report.warnings || []).length }}</span>
          </div>

          <div class="pf-counts">
            <span v-for="c in sourceCounts" :key="c.k">{{ c.k }} <b>{{ c.v }}</b></span>
          </div>

          <p v-if="coverageBlocked" class="pf-note">
            连续性状态一条都没有 ⇒ 道具时间线 / 越轴 / 线索揭示 / 转场动机全都判不了。
            这不是「通过」，先把状态补上再体检。
          </p>

          <section v-if="contGroups.length" class="pf-sec">
            <h4>连续性 · 按类别聚合<span>{{ contGroups.length }} 类 / {{ contTotal }} 条</span></h4>
            <div v-for="g in contGroups" :key="g.key" class="pf-group">
              <button class="pf-group-head" @click="toggle(g.key)">
                <span class="pf-caret" :class="{ open: opened[g.key] }">▸</span>
                <span class="pf-group-name">{{ g.label }}</span>
                <span class="pf-group-n" :class="g.severity">{{ g.items.length }}</span>
              </button>
              <ul v-if="opened[g.key]" class="pf-list">
                <li v-for="(m, i) in g.items.slice(0, 60)" :key="i">{{ m }}</li>
                <li v-if="g.items.length > 60" class="pf-more">还有 {{ g.items.length - 60 }} 条…</li>
              </ul>
            </div>
          </section>

          <section v-if="otherBlockers.length" class="pf-sec">
            <h4>其他阻断<span>{{ otherBlockers.length }} 条</span></h4>
            <ul class="pf-list is-open">
              <li v-for="(b, i) in otherBlockers" :key="i">
                <em v-if="stageLabel(b.stage)" class="pf-tag">{{ stageLabel(b.stage) }}</em>{{ b.message }}
              </li>
            </ul>
          </section>

          <section v-if="assetGate && (assetProblems.length || assetWarns.length)" class="pf-sec">
            <h4>资产验收门<span>{{ assetProblems.length }} 阻断 / {{ assetWarns.length }} 提示</span></h4>
            <ul v-if="assetProblems.length" class="pf-list is-open">
              <li v-for="(p, i) in assetProblems.slice(0, 40)" :key="i" class="pf-bad">{{ text(p) }}</li>
            </ul>
            <ul v-if="assetWarns.length" class="pf-list is-open pf-dim">
              <li v-for="(w, i) in assetWarns.slice(0, 40)" :key="i">{{ text(w) }}</li>
              <li v-if="assetWarns.length > 40" class="pf-more">还有 {{ assetWarns.length - 40 }} 条…</li>
            </ul>
          </section>

          <section v-if="nextActions.length" class="pf-sec">
            <h4>下一步</h4>
            <ol class="pf-next">
              <li v-for="(a, i) in nextActions" :key="i">{{ a }}</li>
            </ol>
          </section>
        </div>

        <footer class="pf-foot">
          <button class="btn" :disabled="loading" @click="run">重新体检</button>
          <button class="btn btn-primary" @click="close">关闭</button>
        </footer>
      </div>
    </div>
  </span>
</template>

<script setup lang="ts">
import { ref, computed, onBeforeUnmount } from 'vue'
// 后端 `/production/preflight` 只要 episodeId 就自己取数组装（plan/manifest 形状前端不必知道）
import { productionAPI } from '~/composables/useApi'

const props = defineProps<{ episodeId: number | null | undefined }>()

const open = ref(false)
const loading = ref(false)
const err = ref('')
const report = ref<any>(null)
const shownBlockers = ref(0)
const opened = ref<Record<string, boolean>>({})

function esc(ev: KeyboardEvent) {
  if (ev.key === 'Escape' && open.value) close()
}

async function run() {
  const id = Number(props.episodeId || 0)
  if (!id) return
  open.value = true
  if (loading.value) return
  loading.value = true
  err.value = ''
  window.addEventListener('keydown', esc)
  try {
    report.value = await productionAPI.preflightForEpisode(id)
    shownBlockers.value = (report.value?.blockers || []).length
  } catch (e: any) {
    err.value = e?.message || '体检失败'
  } finally {
    loading.value = false
  }
}

function close() {
  open.value = false
  window.removeEventListener('keydown', esc)
}

onBeforeUnmount(() => window.removeEventListener('keydown', esc))

function toggle(key: string) {
  opened.value = { ...opened.value, [key]: !opened.value[key] }
}

const CODE_LABEL: Record<string, string> = {
  'table-field': '资产卡字段不全（场景缺左/右/后/前景/轴线）',
  'unknown-location': '分镜写了未登记的地点',
  'unknown-character': '分镜写了未登记的角色',
  'action-missing': '镜头没有动作',
  'action-inert': '动作没做到五件事（可删或可并）',
  'action-actor': '动作没有主体',
  'action-actor-absent': '动作主体不在本镜',
  'action-actor-unregistered': '动作主体未登记',
  'action-place-mismatch': '动作地点与本镜不符',
  'transition-missing': '镜头没有转场动机',
  'transition-motive': '转场动机不在八类内',
  'prop-state': '道具状态缺字段',
  'prop-missing': '道具未登记',
  'prop-teleport': '道具凭空出现或消失',
  'clue-early': '线索提前暴露',
  'clue-order': '线索揭示顺序颠倒',
  'clue-declared-absent': '线索未声明在哪一镜露出',
  'screen-direction': '越轴',
  'no-shots': '没有分镜',
}

const contViolations = computed<any[]>(() => report.value?.sections?.continuity?.violations || [])
const contWarnings = computed<any[]>(() => report.value?.sections?.continuity?.warnings || [])

const BLOCK_CODES = new Set([
  'table-field', 'unknown-location', 'unknown-character', 'action-actor',
  'action-actor-absent', 'prop-teleport', 'clue-early', 'clue-order', 'screen-direction',
])

const contGroups = computed(() => {
  const map = new Map<string, { key: string; label: string; severity: string; items: string[] }>()
  const feed = (arr: any[], fallbackSeverity: string) => {
    for (const v of arr) {
      const code = String(v?.code || 'other')
      const severity = BLOCK_CODES.has(code) ? 'v' : fallbackSeverity
      if (!map.has(code)) {
        map.set(code, { key: code, label: CODE_LABEL[code] || code, severity, items: [] })
      }
      const g = map.get(code)!
      const shot = v?.shot ? `[${v.shot}] ` : ''
      g.items.push(shot + String(v?.message || ''))
      if (severity === 'v') g.severity = 'v'
    }
  }
  feed(contViolations.value, 'v')
  feed(contWarnings.value, 'w')
  return [...map.values()].sort((a, b) => b.items.length - a.items.length)
})

const contTotal = computed(() => contGroups.value.reduce((n, g) => n + g.items.length, 0))

// 连续性已在上面按 code 聚合；「状态一条都没有」已由 pf-note 说明 ⇒ 这里都不重复
const otherBlockers = computed(() =>
  (report.value?.blockers || []).filter(
    (b: any) => b?.stage && b.stage !== 'continuity' && b.stage !== 'continuity-coverage'))

const STAGE_LABEL: Record<string, string> = {
  assetGate: '资产未验收',
  placeholder: '占位符',
  quality: '质感',
  source: '取数',
  rules: '规则',
}

function stageLabel(s: string) {
  return STAGE_LABEL[s] || s || ''
}

const coverageBlocked = computed(() =>
  (report.value?.blockers || []).some((b: any) => b?.stage === 'continuity-coverage'))

const assetGate = computed(() => report.value?.sections?.assetGate || null)
const assetProblems = computed<any[]>(() => assetGate.value?.problems || [])
const assetWarns = computed<any[]>(() => assetGate.value?.warnings || [])

function text(x: any) {
  if (typeof x === 'string') return x
  return x?.message || JSON.stringify(x)
}

const sourceCounts = computed(() => {
  const c = report.value?.source?.counts || {}
  const keys: [string, string][] = [
    ['storyboards', '分镜'],
    ['storyboard_characters', '镜-角色'],
    ['storyboard_props', '镜-道具'],
    ['continuity_states', '连续性状态'],
    ['scenes', '场景'],
    ['characters', '角色'],
    ['props', '道具'],
  ]
  return keys.filter(([k]) => typeof c[k] === 'number').map(([k, label]) => ({ k: label, v: c[k] }))
})

const nextActions = computed<string[]>(() => report.value?.nextActions || [])
</script>

<style scoped>
.pf-shell { display: inline-flex; }

.pf-trigger { position: relative; gap: 4px; }
.pf-trigger.is-blocked { border-color: var(--error); color: var(--error); }
.pf-badge {
  min-width: 16px; height: 16px; padding: 0 4px; border-radius: 8px;
  background: var(--error); color: #fff; font-size: 10px; line-height: 16px;
  text-align: center; font-variant-numeric: tabular-nums;
}

.pf-overlay {
  position: fixed; inset: 0; z-index: 240;
  background: rgba(24, 33, 50, 0.32);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.pf-panel {
  width: min(760px, 100%);
  max-height: calc(100vh - 48px);
  max-height: calc(100dvh - 48px);
  display: flex; flex-direction: column;
  background: var(--bg-0); border: 1px solid var(--border);
  border-radius: 12px; box-shadow: var(--shadow-panel); overflow: hidden;
}
.pf-head {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  padding: 16px 18px 12px; border-bottom: 1px solid var(--border);
}
.pf-head h3 { margin: 0; font-size: 15px; color: var(--text-0); }
.pf-sub { margin: 4px 0 0; font-size: 11.5px; color: var(--text-3); }
.pf-x {
  border: none; background: transparent; color: var(--text-3);
  font-size: 20px; line-height: 1; cursor: pointer; padding: 0 2px;
}
.pf-x:hover { color: var(--text-1); }

.pf-state { padding: 28px 18px; text-align: center; font-size: 12.5px; color: var(--text-2); }
.pf-err { color: var(--error); }

.pf-body { padding: 14px 18px; overflow-y: auto; flex: 1; }

.pf-summary {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 9px 12px; border-radius: 8px; font-size: 12.5px; margin-bottom: 10px;
}
.pf-summary b { font-size: 13px; }
.pf-summary.ok { background: var(--success-bg); color: var(--success); }
.pf-summary.bad { background: var(--error-bg); color: var(--error); }

.pf-counts { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 11.5px; color: var(--text-3); margin-bottom: 10px; }
.pf-counts b { color: var(--text-1); font-variant-numeric: tabular-nums; }

.pf-note {
  margin: 0 0 12px; padding: 8px 10px; border-left: 3px solid var(--error);
  background: var(--error-bg); color: var(--text-1); font-size: 11.5px; line-height: 1.55;
}

.pf-sec { margin-bottom: 14px; }
.pf-sec h4 {
  display: flex; align-items: baseline; gap: 8px; margin: 0 0 6px;
  font-size: 12px; color: var(--text-1); font-weight: 600;
}
.pf-sec h4 span { font-size: 10.5px; color: var(--text-3); font-weight: 400; }

.pf-group { border: 1px solid var(--border); border-radius: 7px; margin-bottom: 6px; overflow: hidden; }
.pf-group-head {
  display: flex; align-items: center; gap: 8px; width: 100%;
  padding: 7px 10px; background: var(--bg-1); border: none; cursor: pointer;
  font-size: 12px; color: var(--text-1); text-align: left;
}
.pf-group-head:hover { background: var(--bg-hover); }
.pf-caret { color: var(--text-3); transition: transform .15s ease; display: inline-block; }
.pf-caret.open { transform: rotate(90deg); }
.pf-group-name { flex: 1; }
.pf-group-n {
  min-width: 20px; text-align: center; border-radius: 9px; padding: 0 5px;
  font-size: 10.5px; background: var(--bg-3); color: var(--text-2); font-variant-numeric: tabular-nums;
}
.pf-group-n.v { background: var(--error-bg); color: var(--error); }
.pf-group-n.w { background: var(--bg-3); color: var(--text-2); }

.pf-list { margin: 0; padding: 8px 10px 10px 26px; font-size: 11.5px; color: var(--text-2); line-height: 1.7; max-height: 260px; overflow-y: auto; }
.pf-list.is-open { border: 1px solid var(--border); border-radius: 7px; }
.pf-list li { overflow-wrap: anywhere; }
.pf-list.pf-dim { color: var(--text-3); margin-top: 6px; }
.pf-list .pf-bad { color: var(--error); }
.pf-more { list-style: none; margin-left: -12px; color: var(--text-3); font-style: italic; }
.pf-tag {
  display: inline-block; margin-right: 6px; padding: 0 5px; border-radius: 4px;
  font-style: normal; font-size: 10px; background: var(--bg-3); color: var(--text-1);
}

.pf-next { margin: 0; padding-left: 20px; font-size: 11.5px; color: var(--text-2); line-height: 1.75; }
.pf-next li { overflow-wrap: anywhere; }

.pf-foot {
  display: flex; justify-content: flex-end; gap: 8px;
  padding: 12px 18px; border-top: 1px solid var(--border); background: var(--bg-1);
}

@media (max-width: 767px) {
  .pf-overlay { padding: 0; align-items: flex-end; }
  .pf-panel {
    width: 100%; max-height: calc(100vh - 16px); max-height: calc(100dvh - 16px);
    border-radius: 12px 12px 0 0;
  }
  .pf-body { padding: 12px; }
  .pf-foot { flex-wrap: wrap; }
  .pf-foot .btn { flex: 1 1 130px; }
}
</style>

<template>
  <div class="vg-page">
    <div class="settings-scroll">
      <div class="settings-head">
        <h2 class="settings-title">视觉图谱</h2>
        <p class="settings-desc">
          景别 / 构图 / 运镜 / 灯光四类术语，拆镜与提示词共用同一套词表。英文术语点一下就复制，
          直接贴进图片或视频 prompt——中文术语混进英文 prompt 是生图跑偏的常见原因。
        </p>
      </div>

      <!-- 术语图谱 -->
      <section class="card vg-section">
        <div class="vg-toolbar">
          <div class="vg-tabs">
            <button
              v-for="t in tabs"
              :key="t.key"
              class="vg-tab"
              :class="{ active: tab === t.key }"
              @click="tab = t.key"
            >
              {{ t.label }}<span class="vg-tab-count">{{ t.count }}</span>
            </button>
          </div>
          <input v-model="q" class="input vg-search" placeholder="搜中文 / 英文 / 用途…" />
        </div>

        <div v-if="loading" class="vg-empty">加载中…</div>
        <p v-else-if="loadError" class="vg-empty">
          {{ loadError }}
          <button class="btn btn-ghost btn-sm" @click="loadGraph">重试</button>
        </p>
        <div v-else-if="filtered.length" class="vg-grid">
          <article v-for="item in filtered" :key="item.key" class="vg-term">
            <div class="vg-term-top">
              <span class="vg-zh">{{ item.primary }}</span>
              <span class="tag">{{ CATEGORY_LABEL[item.cat] }}</span>
            </div>
            <button class="vg-en" title="点击复制英文术语" @click="copy(item.term.en)">
              <span class="mono">{{ item.term.en }}</span>
              <Copy :size="11" />
            </button>
            <p class="vg-usage">{{ item.term.usage }}</p>
            <p v-if="item.aliases.length" class="vg-alias">别名：{{ item.aliases.join(' / ') }}</p>
            <div v-if="item.term.pairsWith?.length" class="vg-pairs">
              <span class="vg-pairs-label">常搭配</span>
              <code v-for="p in item.term.pairsWith" :key="p" class="vg-pair">{{ p }}</code>
            </div>
          </article>
        </div>
        <p v-else class="vg-empty">{{ q ? `没有匹配「${q}」的术语` : '暂无术语数据' }}</p>
      </section>

      <!-- 分镜自查 -->
      <section class="card vg-section">
        <div class="vg-section-head">
          <div>
            <div class="vg-section-title">分镜自查</div>
            <div class="vg-section-desc">
              粘贴一段分镜描述，看四类术语各覆盖到没有。纯本地匹配，不发请求。
            </div>
          </div>
          <span v-if="probeText.trim()" class="tag tag-accent">{{ probeCovered }} / 4 类已点明</span>
        </div>
        <textarea
          v-model="probeText"
          class="input vg-probe-input"
          rows="4"
          placeholder="例：中景，镜头跟随主角横移，逆光剪影，三分法构图…"
        />
        <div class="vg-probe">
          <div v-for="cat in CATEGORIES" :key="cat" class="vg-probe-row">
            <span class="vg-probe-cat">{{ CATEGORY_LABEL[cat] }}</span>
            <div class="vg-probe-hits">
              <template v-if="probeHits[cat].length">
                <span v-for="h in probeHits[cat]" :key="h.en" class="tag tag-accent">{{ h.matched }} → {{ h.en }}</span>
              </template>
              <span v-else class="dim">{{ probeText.trim() ? '这一类没提到' : '—' }}</span>
            </div>
          </div>
        </div>
        <p v-if="probeText.trim() && !probeCovered" class="vg-hint">
          一处都没认出：分镜里的景别/运镜/灯光常用中文口语写，这里只认词表内的说法。
          换成表里的词能让拆镜与提示词更准。
        </p>
      </section>

      <!-- 本剧引导 -->
      <section class="card vg-section">
        <div class="vg-section-head">
          <div>
            <div class="vg-section-title">拆镜引导</div>
            <div class="vg-section-desc">
              后端按剧的题材/风格生成的引导文本，正是注入拆镜 Agent 的那段。
              {{ guidanceDramaId ? `当前：剧集 #${guidanceDramaId}` : '未指定剧集时是全局通用规则。' }}
            </div>
          </div>
          <div class="vg-section-actions">
            <button class="btn btn-ghost btn-sm" :disabled="guidanceLoading" @click="loadGuidance">
              <RefreshCw :size="12" :class="{ 'vg-spin': guidanceLoading }" /> 读取
            </button>
            <button class="btn btn-ghost btn-sm" :disabled="!guidance" @click="copy(guidance)">
              <Copy :size="12" /> 复制
            </button>
          </div>
        </div>
        <p v-if="guidanceError" class="vg-empty">{{ guidanceError }}</p>
        <pre v-else-if="guidance" class="vg-guidance">{{ guidance }}</pre>
        <p v-else class="vg-empty">{{ guidanceLoading ? '读取中…' : '还没读取。点右上「读取」拉一次。' }}</p>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Copy, RefreshCw } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import { visualGraphAPI } from '~/composables/useApi'
import type { VisualCategory, VisualTerm } from '~/composables/useApi'

const CATEGORIES: VisualCategory[] = ['shot_size', 'composition', 'movement', 'lighting']
const CATEGORY_LABEL: Record<VisualCategory, string> = {
  shot_size: '景别',
  composition: '构图',
  movement: '运镜',
  lighting: '灯光',
}

type TermRow = { key: string; cat: VisualCategory; term: VisualTerm; primary: string; aliases: string[] }

const graph = ref<Partial<Record<VisualCategory, VisualTerm[]>>>({})
const loading = ref(false)
const loadError = ref('')
const q = ref('')
const tab = ref<'all' | VisualCategory>('all')
const probeText = ref('')

/** 四类摊平成一维：标签页、搜索、自查都用这一份，避免三处各写一遍遍历 */
const rows = computed<TermRow[]>(() => {
  const out: TermRow[] = []
  for (const cat of CATEGORIES) {
    for (const term of graph.value[cat] || []) {
      const primary = term.zh[0] || term.en
      out.push({
        key: `${cat}:${term.en}`,
        cat,
        term,
        primary,
        // 主名要单列显示，别名列表里不再重复它
        aliases: term.zh.slice(1),
      })
    }
  }
  return out
})

const tabs = computed(() => [
  { key: 'all' as const, label: '全部', count: rows.value.length },
  ...CATEGORIES.map(cat => ({
    key: cat,
    label: CATEGORY_LABEL[cat],
    count: (graph.value[cat] || []).length,
  })),
])

const filtered = computed(() => {
  const keyword = q.value.trim().toLowerCase()
  return rows.value.filter(item => {
    if (tab.value !== 'all' && item.cat !== tab.value) return false
    if (!keyword) return true
    // 别名也要能搜到：「特写」和「大特写」是两个节点，搜「极特写」也该命中大特写
    return (
      item.term.zh.some(zh => zh.includes(keyword))
      || item.term.en.toLowerCase().includes(keyword)
      || item.term.usage.includes(keyword)
    )
  })
})

/** 自查结果：每类命中的术语（`matched` 是原文里真正出现的那个别名） */
type ProbeHit = { matched: string; en: string }
const probeHits = computed<Record<VisualCategory, ProbeHit[]>>(() => {
  const result: Record<VisualCategory, ProbeHit[]> = {
    shot_size: [], composition: [], movement: [], lighting: [],
  }
  const raw = probeText.value.trim()
  if (!raw) return result
  // 掩码：命中过的字抹掉，否则「中近景」吃掉半截之后「中景」还会在同一段上再命中一次
  const mask = raw.split('')
  for (const cat of CATEGORIES) {
    const candidates = (graph.value[cat] || [])
      .flatMap(term => term.zh.map(zh => ({ term, zh })))
      // 长别名优先，保证「大远景」不会被「远景」抢先
      .sort((a, b) => b.zh.length - a.zh.length)
    const hitEns = new Set<string>()
    for (const { term, zh } of candidates) {
      if (hitEns.has(term.en)) continue
      const idx = mask.join('').indexOf(zh)
      if (idx < 0) continue
      for (let i = idx; i < idx + zh.length; i++) mask[i] = '\u0000'
      hitEns.add(term.en)
      result[cat].push({ matched: zh, en: term.en })
    }
  }
  return result
})
const probeCovered = computed(() => CATEGORIES.filter(cat => probeHits.value[cat].length).length)

const guidance = ref('')
const guidanceError = ref('')
const guidanceLoading = ref(false)
const route = useRoute()
/** 从剧集页跳过来时带 `?drama_id=`，否则用全局默认引导 */
const guidanceDramaId = computed(() => {
  const raw = Number(route.query.drama_id)
  return Number.isFinite(raw) && raw > 0 ? raw : 0
})

async function loadGraph() {
  loading.value = true
  loadError.value = ''
  try {
    const data = await visualGraphAPI.graph()
    graph.value = data?.graph || {}
  } catch (e: any) {
    loadError.value = e?.message || '图谱加载失败'
  } finally {
    loading.value = false
  }
}

async function loadGuidance() {
  guidanceLoading.value = true
  guidanceError.value = ''
  try {
    guidance.value = await visualGraphAPI.guidance(guidanceDramaId.value || undefined)
  } catch (e: any) {
    guidance.value = ''
    guidanceError.value = e?.message || '引导文本获取失败'
  } finally {
    guidanceLoading.value = false
  }
}

async function copy(text: string) {
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    toast.success('已复制')
  } catch {
    // 非 https / 无权限时 clipboard 会直接拒绝，别装作复制成功
    toast.error('复制失败：浏览器不允许访问剪贴板，请手动选中')
  }
}

onMounted(() => {
  loadGraph()
  loadGuidance()
})
</script>

<style scoped>
.vg-page { height: 100%; overflow: hidden; }
.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 980px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; line-height: 1.7; }

.vg-section { padding: 18px; margin-bottom: 18px; }

/* 工具条 */
.vg-toolbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.vg-tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.vg-tab {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; font-size: 12.5px; font-weight: 500;
  color: var(--text-2); background: transparent;
  border: 1px solid transparent; border-radius: 99px;
  cursor: pointer; transition: all 0.16s var(--ease-out);
  font-family: var(--font-body);
}
.vg-tab:hover { background: var(--bg-hover); color: var(--text-0); }
.vg-tab.active { background: var(--accent-bg); border-color: var(--accent); color: var(--accent-text); font-weight: 600; }
.vg-tab-count { font-size: 10.5px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.vg-tab.active .vg-tab-count { color: var(--accent-text); opacity: 0.75; }
.vg-search { width: 240px; margin-left: auto; }

/* 术语卡 */
.vg-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(258px, 1fr)); gap: 10px; }
.vg-term {
  display: flex; flex-direction: column; gap: 6px;
  padding: 12px 13px;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-input);
  transition: border-color 0.18s var(--ease-out), box-shadow 0.18s var(--ease-out), transform 0.18s var(--ease-out);
}
.vg-term:hover { border-color: var(--border-strong); box-shadow: var(--shadow-sm); transform: translateY(-1px); }
.vg-term-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.vg-zh { font-size: 14px; font-weight: 600; color: var(--text-0); font-family: var(--font-display); }
.vg-en {
  display: inline-flex; align-items: center; gap: 5px;
  align-self: flex-start;
  padding: 2px 8px; font-size: 11.5px;
  color: var(--accent-text); background: var(--accent-bg);
  border: 1px dashed transparent; border-radius: var(--radius-sm);
  cursor: pointer; transition: border-color 0.16s, background 0.16s;
}
.vg-en:hover { border-color: var(--accent); background: var(--bg-active); }
.vg-usage { font-size: 12px; color: var(--text-2); line-height: 1.65; margin: 0; }
.vg-alias { font-size: 11px; color: var(--text-3); margin: 0; }
.vg-pairs { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
.vg-pairs-label { font-size: 10px; color: var(--text-3); letter-spacing: 0.04em; }
.vg-pair {
  font-family: var(--font-mono); font-size: 10.5px;
  padding: 1px 6px; border-radius: var(--radius-sm);
  background: var(--bg-2); color: var(--text-2);
}

/* 自查 */
.vg-section-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.vg-section-title { font-size: 14px; font-weight: 600; color: var(--text-0); font-family: var(--font-display); }
.vg-section-desc { font-size: 12px; color: var(--text-3); margin-top: 3px; line-height: 1.6; }
.vg-section-actions { display: flex; gap: 6px; flex-shrink: 0; }
.vg-probe-input { resize: vertical; line-height: 1.7; margin-bottom: 10px; }
.vg-probe { display: flex; flex-direction: column; gap: 7px; }
.vg-probe-row { display: flex; align-items: flex-start; gap: 10px; }
.vg-probe-cat {
  flex-shrink: 0; width: 34px;
  font-size: 11.5px; color: var(--text-3); line-height: 20px;
}
.vg-probe-hits { display: flex; gap: 5px; flex-wrap: wrap; }
.vg-hint { font-size: 11.5px; color: var(--warning); margin: 10px 0 0; line-height: 1.6; }

/* 引导文本 */
.vg-guidance {
  margin: 0; padding: 12px 14px;
  font-family: var(--font-mono); font-size: 11.5px; line-height: 1.8;
  color: var(--text-1); white-space: pre-wrap; word-break: break-word;
  background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius);
  max-height: 420px; overflow-y: auto;
}

.vg-empty { font-size: 12.5px; color: var(--text-3); padding: 6px 0; display: flex; align-items: center; gap: 8px; }
.vg-spin { animation: vg-spin 0.9s linear infinite; }
@keyframes vg-spin { to { transform: rotate(360deg); } }
</style>

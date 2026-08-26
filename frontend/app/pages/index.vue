<template>
  <div class="page">
    <!-- Page Header -->
    <div class="page-head">
      <div class="head-left">
        <h1 class="page-title">短剧项目</h1>
        <p class="page-desc">{{ dramas.length }} 个项目</p>
      </div>
      <div class="head-actions">
        <button class="btn btn-ai" @click="showAutoGen = true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2l2.4 4.9 5.6 1.4-4 3.9.9 5.5L12 15l-4.9 2.7.9-5.5-4-3.9 5.6-1.4z"/>
          </svg>
          AI 一键生成
        </button>
        <button class="btn btn-primary" @click="showCreate = true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          新建项目
        </button>
      </div>
    </div>

    <!-- Stats Bar -->
    <div v-if="stats && !loading" class="stats-bar">
      <div class="stat-total">
        <span class="stat-num">{{ stats.total }}</span>
        <span class="stat-cap">全部项目</span>
      </div>
      <div v-for="s in stats.by_status" :key="s.status" class="stat-chip">
        <span class="stat-dot" :style="{ background: statusColor(s.status) }"></span>
        <span class="stat-cap">{{ statusLabel(s.status) }}</span>
        <span class="stat-num-sm">{{ s.count }}</span>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="loading-state">
      <div class="loading-grid">
        <div v-for="i in 3" :key="i" class="skeleton-card card"></div>
      </div>
    </div>

    <!-- Grid -->
    <div v-else class="grid">
      <div
        v-for="(d, i) in dramas"
        :key="d.id"
        class="card project-card"
        :style="{ animationDelay: `${i * 0.06}s` }"
        @click="navigateTo(`/drama/${d.id}`)"
      >
        <!-- Card film strip decoration -->
        <div class="card-film-strip">
          <span v-for="j in 5" :key="j" class="film-hole"></span>
        </div>

        <div class="card-body">
          <div class="card-header">
            <div class="episode-badge">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/></svg>
              {{ d.episodes?.length || 0 }} 集
            </div>
            <button class="btn btn-ghost btn-icon card-delete" @click.stop="delDrama(d)" title="删除">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
              </svg>
            </button>
          </div>

          <h3 class="project-title">{{ d.title }}</h3>

          <div class="project-meta">
            <span v-if="d.style" class="style-tag">{{ d.style }}</span>
            <span class="meta-item">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              {{ d.characters?.length || 0 }}
            </span>
            <span class="meta-item">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/></svg>
              {{ d.scenes?.length || 0 }}
            </span>
          </div>
        </div>

        <div class="card-footer">
          <div class="progress-mini">
            <div class="progress-mini-track">
              <div class="progress-mini-fill" :style="{ width: getProgress(d) + '%' }"></div>
            </div>
          </div>
          <span class="card-date">{{ fmtDate(d.updated_at || d.updatedAt) }}</span>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="!dramas.length" class="card empty-card" @click="showCreate = true">
        <div class="empty-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round">
            <rect x="3" y="3" width="18" height="18" rx="3"/>
            <line x1="12" y1="8" x2="12" y2="16"/>
            <line x1="8" y1="12" x2="16" y2="12"/>
          </svg>
        </div>
        <p class="empty-title">新建第一个短剧项目</p>
        <p class="empty-desc">从剧本到成片，AI 助力的短剧制作工作台</p>
      </div>
    </div>

    <!-- Create Dialog -->
    <div v-if="showCreate" class="overlay" @click.self="showCreate = false">
      <div class="modal card">
        <div class="modal-header">
          <div class="modal-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
              <rect x="3" y="3" width="18" height="18" rx="3"/>
              <line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
            </svg>
          </div>
          <h2 class="modal-title">新建短剧项目</h2>
          <p class="modal-desc">输入项目基本信息，即可开始制作</p>
        </div>
        <form @submit.prevent="create" class="modal-form">
          <label class="field">
            <span class="field-label">项目名称 <span class="required">*</span></span>
            <input v-model="form.title" class="input" placeholder="例如：都市情感短剧《时光邮局》" required autofocus />
          </label>
          <div class="field-row">
            <label class="field">
              <span class="field-label">计划集数</span>
              <input v-model.number="form.total_episodes" class="input" type="number" min="1" max="100" />
            </label>
            <label class="field">
              <span class="field-label">视觉风格</span>
              <BaseSelect v-model="form.style" :options="styleSelectOptions" placeholder="选择风格" searchable />
            </label>
          </div>
          <div class="modal-actions">
            <button type="button" class="btn" @click="showCreate = false">取消</button>
            <button type="submit" class="btn btn-primary">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              创建项目
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- Auto Gen Dialog -->
    <div v-if="showAutoGen" class="overlay" @click.self="closeAutoGen">
      <div class="modal card auto-modal">
        <!-- 输入态 -->
        <template v-if="!autoRunning">
          <div class="modal-header">
            <div class="modal-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2l2.4 4.9 5.6 1.4-4 3.9.9 5.5L12 15l-4.9 2.7.9-5.5-4-3.9 5.6-1.4z"/>
              </svg>
            </div>
            <h2 class="modal-title">AI 一键生成</h2>
            <p class="modal-desc">输入一句故事梗概，AI 自动完成剧本、角色场景、分镜、图片、视频、配音、合成与拼接</p>
          </div>
          <form @submit.prevent="runAutoGen" class="modal-form">
            <label class="field">
              <span class="field-label">故事梗概 <span class="required">*</span></span>
              <textarea v-model="autoForm.premise" class="input textarea" rows="4" placeholder="例如：外卖小哥意外救下集团千金，从此命运改写…" required></textarea>
            </label>
            <div class="field-row">
              <label class="field">
                <span class="field-label">标题（可选）</span>
                <input v-model="autoForm.title" class="input" placeholder="默认取梗概前 24 字" />
              </label>
              <label class="field">
                <span class="field-label">计划集数</span>
                <input v-model.number="autoForm.episodeCount" class="input" type="number" min="1" max="20" />
              </label>
            </div>
            <div class="field-row">
              <label class="field">
                <span class="field-label">类型（可选）</span>
                <input v-model="autoForm.genre" class="input" placeholder="如：都市、古装、悬疑" />
              </label>
              <label class="field">
                <span class="field-label">视觉风格</span>
                <BaseSelect v-model="autoForm.style" :options="styleSelectOptions" placeholder="默认 realistic" searchable />
              </label>
            </div>
            <div class="field">
              <span class="field-label">生成范围</span>
              <div class="toggle-grid">
                <label v-for="opt in autoMediaOpts" :key="opt.key" class="toggle-item">
                  <input v-model="autoForm[opt.key]" type="checkbox" class="toggle-input" />
                  <span class="toggle-label">{{ opt.label }}</span>
                </label>
              </div>
            </div>
            <div class="modal-actions">
              <button type="button" class="btn" @click="closeAutoGen">取消</button>
              <button type="submit" class="btn btn-ai">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2l2.4 4.9 5.6 1.4-4 3.9.9 5.5L12 15l-4.9 2.7.9-5.5-4-3.9 5.6-1.4z"/>
                </svg>
                开始生成
              </button>
            </div>
          </form>
        </template>

        <!-- 进度态 -->
        <template v-else>
          <div class="modal-header">
            <div class="modal-icon auto-spin">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M21 12a9 9 0 1 1-6.2-8.6"/>
              </svg>
            </div>
            <h2 class="modal-title">AI 生成中…</h2>
            <p class="modal-desc">{{ autoStatus ? `已完成 ${autoStatus.doneCount}/${autoStatus.totalEpisodes} 集${autoStatus.failedCount ? `，${autoStatus.failedCount} 集失败` : ''}` : '正在创建项目与集数…' }}</p>
          </div>
          <div class="auto-progress">
            <div v-if="!autoStatus" class="auto-empty">正在初始化任务…</div>
            <div v-else class="auto-ep-list">
              <div v-for="ep in autoStatus.episodes" :key="ep.id" class="auto-ep">
                <div class="auto-ep-head">
                  <span class="auto-ep-no">第 {{ ep.episodeNumber }} 集</span>
                  <span class="auto-ep-status" :class="autoEpClass(ep.status)">{{ autoStageLabel(ep.status) }}</span>
                </div>
                <div class="auto-ep-meta">
                  <span>分镜 {{ ep.storyboardCount }}</span>
                  <span>图片 {{ ep.imageReadyCount }}</span>
                  <span>视频 {{ ep.videoReadyCount }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="modal-actions">
            <button type="button" class="btn btn-primary" @click="goAutoDrama">查看项目</button>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { toast } from 'vue-sonner'
import { dramaAPI, autoPipelineAPI } from '~/composables/useApi'
import BaseSelect from '~/components/BaseSelect.vue'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()

const dramas = ref([])
const stats = ref(null)
const loading = ref(false)
const showCreate = ref(false)
const form = ref({ title: '', total_episodes: 1, style: '' })

// ===== AI 一键生成 =====
const showAutoGen = ref(false)
const autoForm = ref({
  premise: '',
  title: '',
  genre: '',
  style: '',
  episodeCount: 1,
  withImages: true,
  withVideos: true,
  withCompose: true,
  withMerge: true,
})
const autoRunning = ref(false)
const autoStatus = ref(null)
const autoDramaId = ref(null)
let autoEs = null

const AUTO_STAGE_LABELS = {
  'auto:queued': '排队中',
  'auto:scripting': '生成剧本',
  'auto:extracting': '提取角色/场景',
  'auto:storyboarding': '生成分镜',
  'auto:voicing': '配音',
  'auto:imaging': '生成图片',
  'auto:videoing': '生成视频',
  'auto:composing': '合成',
  'auto:merging': '拼接',
  'auto:done': '已完成',
  'auto:failed': '失败',
}
function autoStageLabel(s) { return AUTO_STAGE_LABELS[s] || (s ? s.replace('auto:', '') : '待开始') }

const autoMediaOpts = [
  { key: 'withImages', label: '图片' },
  { key: 'withVideos', label: '视频' },
  { key: 'withCompose', label: '配音合成' },
  { key: 'withMerge', label: '整集拼接' },
]

function autoEpClass(s) {
  if (s === 'auto:done') return 'is-done'
  if (s === 'auto:failed') return 'is-failed'
  return 'is-running'
}
const styles = ['realistic', 'anime', 'ghibli', 'cinematic', 'comic', 'watercolor']
const styleSelectOptions = computed(() => styles.map(s => ({ label: s, value: s })))

const STATUS_LABELS = {
  draft: '草稿',
  producing: '制作中',
  produced: '已制作',
  completed: '已完成',
  published: '已发布',
  reviewing: '审核中',
  archived: '已归档',
  auto_generating: 'AI 生成中',
}
const STATUS_COLORS = {
  draft: 'var(--text-3)',
  producing: 'var(--info, #3b82f6)',
  produced: 'var(--accent)',
  completed: 'var(--success)',
  published: 'var(--accent)',
  reviewing: 'var(--warning, #f59e0b)',
  archived: 'var(--text-3)',
  auto_generating: 'var(--info, #3b82f6)',
}
function statusLabel(s) { return STATUS_LABELS[s] || s }
function statusColor(s) { return STATUS_COLORS[s] || 'var(--text-3)' }

async function load() {
  loading.value = true
  try {
    const res = await dramaAPI.list()
    dramas.value = res.items || []
    stats.value = (await dramaAPI.stats()) || null
  } catch (e) {
    toast.error(e.message)
  } finally {
    loading.value = false
  }
}

async function create() {
  if (!form.value.title?.trim()) return
  try {
    const d = await dramaAPI.create(form.value)
    showCreate.value = false
    navigateTo(`/drama/${d.id}`)
  } catch (e) {
    toast.error(e.message)
  }
}

async function delDrama(d) {
  if (!(await confirm({ message: `确定删除「${d.title}」？此操作不可恢复。`, danger: true }))) return
  try {
    await dramaAPI.del(d.id)
    toast.success('已删除')
    load()
  } catch (e) {
    toast.error(e.message)
  }
}

async function runAutoGen() {
  if (!autoForm.value.premise?.trim()) return
  autoRunning.value = true
  autoStatus.value = null
  try {
    const res = await autoPipelineAPI.run({ ...autoForm.value })
    autoDramaId.value = res.dramaId
    toast.success('已启动 AI 生成，正在后台制作')
    startAutoStream()
  } catch (e) {
    autoRunning.value = false
    toast.error(e.message)
  }
}

function startAutoStream() {
  stopAutoStream()
  if (!autoDramaId.value) return

  const es = new EventSource(autoPipelineAPI.streamUrl(autoDramaId.value))
  autoEs = es

  // 初始快照（后端先 subscribe 再发 snapshot，全量历史状态）
  es.addEventListener('snapshot', (e) => {
    try {
      const s = JSON.parse(e.data)
      autoStatus.value = s
      if (!s.running) finishAutoStream()
    } catch { /* 忽略解析失败 */ }
  })

  // 单集状态推进（增量：只含 episodeId + status）
  es.addEventListener('status', (e) => {
    try {
      const evt = JSON.parse(e.data)
      applyEpisodeStatus(evt.episodeId, evt.status)
    } catch { /* 忽略解析失败 */ }
  })

  // 媒体就绪进度（增量：ready/total）
  es.addEventListener('media-progress', (e) => {
    try {
      const evt = JSON.parse(e.data)
      applyMediaProgress(evt.episodeId, evt.ready, evt.total)
    } catch { /* 忽略解析失败 */ }
  })

  // ping 心跳无需处理；EventSource 断线后浏览器自动重连，重连后重新收到 snapshot 全量
}

/** 本地合并单集状态推进，并在整剧终止态时收尾 */
function applyEpisodeStatus(episodeId, status) {
  if (!autoStatus.value?.episodes) return
  const ep = autoStatus.value.episodes.find((x) => x.id === episodeId)
  if (ep) ep.status = status
  const eps = autoStatus.value.episodes
  autoStatus.value.doneCount = eps.filter((x) => x.status === 'auto:done').length
  autoStatus.value.failedCount = eps.filter((x) => x.status === 'auto:failed').length
  autoStatus.value.running = autoStatus.value.doneCount + autoStatus.value.failedCount < eps.length
  if (!autoStatus.value.running && eps.length > 0) finishAutoStream()
}

/** 本地合并媒体就绪进度：imaging 阶段更新图片数、videoing 阶段更新视频数 */
function applyMediaProgress(episodeId, ready, total) {
  if (!autoStatus.value?.episodes) return
  const ep = autoStatus.value.episodes.find((x) => x.id === episodeId)
  if (!ep) return
  if (ep.status === 'auto:imaging') ep.imageReadyCount = ready
  else if (ep.status === 'auto:videoing') ep.videoReadyCount = ready
  if (total) ep.storyboardCount = total
}

function finishAutoStream() {
  if (!autoEs) return // 幂等：已收尾则忽略
  stopAutoStream()
  autoRunning.value = false
  toast.success('AI 生成完成')
  load()
}

function stopAutoStream() {
  if (autoEs) { autoEs.close(); autoEs = null }
}

function closeAutoGen() {
  if (autoRunning.value) return
  showAutoGen.value = false
  stopAutoStream()
  autoStatus.value = null
  autoDramaId.value = null
}

function goAutoDrama() {
  if (!autoDramaId.value) return
  navigateTo(`/drama/${autoDramaId.value}`)
}

function fmtDate(s) {
  if (!s) return ''
  const d = new Date(s)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

function getProgress(d) {
  // 多阶段真实进度：剧本 20% + 分镜 20% + 图片 20% + 视频 20% + 配音 20%
  const p = d.progress
  if (!p || !p.total_episodes) return 0
  const scriptRate = p.scripted_episodes / p.total_episodes
  const storyboardRate = p.storyboarded_episodes / p.total_episodes
  const sbTotal = p.storyboards || 0
  const imageRate = sbTotal ? p.images / sbTotal : 0
  const videoRate = sbTotal ? p.videos / sbTotal : 0
  const ttsRate = sbTotal ? p.tts / sbTotal : 0
  return Math.round(((scriptRate + storyboardRate + imageRate + videoRate + ttsRate) / 5) * 100)
}

onMounted(() => {
  load()
  const q = useRoute().query
  if (q.new === '1') showCreate.value = true
  else if (q.auto === '1') showAutoGen.value = true
})
onUnmounted(stopAutoStream)
</script>

<style scoped>
.page {
  padding: 28px 48px 40px;
  overflow-y: auto;
  height: 100%;
  animation: fadeUp 0.35s var(--ease-out) both;
}

/* Page Head */
.page-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 28px;
}
.head-left { display: flex; flex-direction: column; gap: 4px; }
.page-title {
  font-family: var(--font-display);
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-0);
}
.page-desc { font-size: 13px; color: var(--text-3); font-weight: 400; }

/* Stats Bar */
.stats-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 24px;
  padding: 12px 16px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stat-total {
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding-right: 14px;
  margin-right: 4px;
  border-right: 1px solid var(--border);
}
.stat-num {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  color: var(--text-0);
}
.stat-num-sm { font-size: 13px; font-weight: 600; color: var(--text-1); }
.stat-cap { font-size: 12px; color: var(--text-3); }
.stat-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 99px;
}
.stat-dot { width: 7px; height: 7px; border-radius: 50%; }

/* Preset button */
.preset-btn {
  border: 1px dashed var(--border);
  color: var(--accent-text);
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s;
}
.preset-btn:hover {
  border-color: var(--accent);
  background: var(--accent-bg);
  color: var(--accent);
}

/* Grid */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

/* Project Card */
.project-card {
  padding: 0;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: fadeUp 0.4s var(--ease-out) both;
  transition: transform 0.22s var(--ease-out), box-shadow 0.22s var(--ease-out), border-color 0.2s;
}
.project-card:hover {
  border-color: var(--accent);
  box-shadow: var(--shadow-lg);
  transform: translateY(-3px);
}

/* Film strip decoration */
.card-film-strip {
  display: flex;
  justify-content: space-around;
  align-items: center;
  padding: 6px 16px;
  background: var(--bg-2);
  border-bottom: 1px solid var(--border);
}
.film-hole {
  width: 10px; height: 8px;
  background: var(--bg-3);
  border-radius: 2px;
  transition: background 0.2s;
}
.project-card:hover .film-hole:nth-child(2) { background: var(--accent); }
.project-card:hover .film-hole:nth-child(4) { background: var(--accent); opacity: 0.5; }

.card-body { padding: 18px 18px 14px; flex: 1; display: flex; flex-direction: column; gap: 10px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.episode-badge {
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 600;
  color: var(--text-3);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.episode-badge svg { color: var(--accent); }

.card-delete { opacity: 0; transition: opacity 0.15s; }
.project-card:hover .card-delete { opacity: 1; }

.project-title {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  line-height: 1.35;
  color: var(--text-0);
}

.project-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.style-tag {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  background: var(--accent-bg);
  color: var(--accent-text);
  border-radius: 99px;
  border: 1px solid rgba(13,148,136,0.18);
}
.meta-item {
  display: flex; align-items: center; gap: 4px;
  font-size: 12px; color: var(--text-3);
}

.card-footer {
  padding: 10px 18px 14px;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}
.progress-mini { flex: 1; }
.progress-mini-track {
  height: 3px; background: var(--bg-3);
  border-radius: 99px; overflow: hidden;
}
.progress-mini-fill {
  height: 100%;
  background: var(--accent-gradient);
  border-radius: 99px;
  transition: width 0.6s var(--ease-out);
}
.card-date { font-size: 11px; color: var(--text-3); white-space: nowrap; }

/* Loading Skeleton */
.loading-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}
.skeleton-card {
  height: 180px;
  background: linear-gradient(90deg, var(--bg-2) 25%, var(--bg-hover) 50%, var(--bg-2) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border: none;
}
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

/* Empty Card */
.empty-card {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; padding: 56px 32px;
  cursor: pointer;
  border-style: dashed; border-width: 1.5px;
  text-align: center;
  transition: all 0.2s var(--ease-out);
}
.empty-card:hover {
  border-color: var(--accent);
  background: var(--accent-bg);
  transform: translateY(-2px);
}
.empty-icon {
  width: 56px; height: 56px; border-radius: var(--radius-lg);
  background: var(--bg-2);
  display: flex; align-items: center; justify-content: center;
  color: var(--text-3);
  margin-bottom: 4px;
  transition: all 0.2s;
}
.empty-card:hover .empty-icon { background: var(--accent-bg); color: var(--accent); }
.empty-title { font-size: 14px; font-weight: 600; color: var(--text-1); }
.empty-desc { font-size: 12px; color: var(--text-3); max-width: 220px; line-height: 1.6; }

/* Modal */
.modal { padding: 32px; width: 460px; box-shadow: var(--shadow-elevated); animation: scaleIn 0.2s var(--ease-out); }
.modal-header { margin-bottom: 24px; display: flex; flex-direction: column; gap: 6px; }
.modal-icon {
  width: 44px; height: 44px; border-radius: var(--radius);
  background: var(--accent-bg); color: var(--accent);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 4px;
}
.modal-title { font-family: var(--font-display); font-size: 19px; font-weight: 700; }
.modal-desc { font-size: 13px; color: var(--text-3); }
.modal-form { display: flex; flex-direction: column; gap: 16px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field-label { font-size: 12px; font-weight: 600; color: var(--text-1); }
.required { color: var(--error); }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; padding-top: 6px; }

/* Auto Gen */
.auto-modal { width: 520px; }
.textarea {
  resize: vertical;
  min-height: 88px;
  line-height: 1.6;
  font-family: inherit;
  padding: 10px 12px;
}
.toggle-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.toggle-item {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 12px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 99px;
  cursor: pointer;
  transition: all 0.15s;
}
.toggle-item:hover { border-color: var(--accent); }
.toggle-item:has(.toggle-input:checked) {
  background: var(--accent-bg);
  border-color: var(--accent);
}
.toggle-input { accent-color: var(--accent); width: 14px; height: 14px; margin: 0; cursor: pointer; }
.toggle-label { font-size: 12px; font-weight: 500; color: var(--text-1); }

.auto-spin svg { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.auto-progress {
  max-height: 320px;
  overflow-y: auto;
  margin-bottom: 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-2);
}
.auto-empty { padding: 32px 16px; text-align: center; font-size: 13px; color: var(--text-3); }
.auto-ep-list { display: flex; flex-direction: column; }
.auto-ep { padding: 12px 16px; border-bottom: 1px solid var(--border); }
.auto-ep:last-child { border-bottom: none; }
.auto-ep-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.auto-ep-no { font-size: 13px; font-weight: 600; color: var(--text-0); }
.auto-ep-status {
  font-size: 11px; font-weight: 600;
  padding: 3px 10px;
  border-radius: 99px;
  white-space: nowrap;
}
.auto-ep-status.is-running { background: var(--accent-bg); color: var(--accent-text); }
.auto-ep-status.is-done { background: rgba(34,197,94,0.12); color: var(--success); }
.auto-ep-status.is-failed { background: rgba(239,68,68,0.12); color: var(--error); }
.auto-ep-meta {
  display: flex; gap: 14px;
  margin-top: 8px;
  font-size: 11px; color: var(--text-3);
}
</style>

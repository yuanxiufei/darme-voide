<template>
  <div class="settings-layout">
    <!-- 左侧导航 -->
    <aside class="settings-nav">
      <nav class="nav-group">
        <div class="nav-group-label">设置</div>
        <button
          v-for="t in baseTabs"
          :key="t.id"
          :class="['nav-item', { active: tab === t.id }]"
          @click="tab = t.id"
        >
          <component :is="t.icon" :size="15" />
          {{ t.label }}
        </button>
      </nav>
    </aside>

    <!-- 右侧内容区 -->
    <div class="settings-content">
      <!-- ===== 生成历史 ===== -->
      <div v-if="tab === 'history'" class="settings-scroll">
        <div class="settings-head">
          <h2 class="settings-title">生成历史</h2>
          <p class="settings-desc">汇总全部图片与视频生成记录，含模型、状态、耗时与提示词。失败项可展开查看具体原因。</p>
        </div>

        <div class="history-toolbar">
          <div class="history-filters">
            <button v-for="f in historyFilters" :key="f.value" :class="['chip', { active: historyFilter === f.value }]" @click="historyFilter = f.value">
              {{ f.label }}
            </button>
          </div>
          <button class="btn btn-ghost btn-icon" :disabled="historyLoading" title="刷新" @click="loadGenerations">
            <Loader2 v-if="historyLoading" :size="13" class="animate-spin" />
            <RefreshCw v-else :size="13" />
          </button>
        </div>

        <div class="history-stats">
          <div class="history-stat">
            <div class="history-stat-num">{{ historyStats.total }}</div>
            <div class="history-stat-label">总记录</div>
          </div>
          <div class="history-stat ok">
            <div class="history-stat-num">{{ historyStats.success }}</div>
            <div class="history-stat-label">成功</div>
          </div>
          <div class="history-stat warn">
            <div class="history-stat-num">{{ historyStats.processing }}</div>
            <div class="history-stat-label">处理中</div>
          </div>
          <div class="history-stat err">
            <div class="history-stat-num">{{ historyStats.failed }}</div>
            <div class="history-stat-label">失败</div>
          </div>
        </div>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Generation Logs</div>
              <div class="setup-title">生成记录</div>
            </div>
          </div>
          <div v-if="historyLoading" class="config-empty">正在加载生成记录…</div>
          <div v-else-if="!filteredGenerations.length" class="config-empty">暂无生成记录，运行图片 / 视频生成后自动记录。</div>
          <ul v-else class="history-list">
            <li v-for="g in filteredGenerations" :key="`${g.type}-${g.id}`" class="history-item">
              <button class="history-item-head" @click="toggleHistory(g)">
                <span :class="['history-type', g.type]">
                  <ImageIcon v-if="g.type === 'image'" :size="14" />
                  <Film v-else :size="14" />
                </span>
                <span class="history-item-main">
                  <span class="history-item-title">{{ g.type === 'image' ? '图片生成' : '视频生成' }}</span>
                  <span class="history-item-meta">
                    <span class="mono">{{ g.model || '—' }}</span>
                    <span class="dot">·</span>
                    <span>{{ g.provider || '—' }}</span>
                  </span>
                </span>
                <span :class="['status-badge', statusClass(g.status)]">{{ statusLabel(g.status) }}</span>
                <span class="history-item-time">
                  <Clock :size="12" />
                  {{ fmtTime(g.createdAt) }}
                </span>
                <ChevronDown :size="14" :class="['history-chevron', { open: historyExpandedId === `${g.type}-${g.id}` }]" />
              </button>
              <div v-if="historyExpandedId === `${g.type}-${g.id}`" class="history-item-body">
                <div class="history-detail-row"><span class="history-detail-label">模型</span><span class="mono">{{ g.model || '—' }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">服务商</span><span>{{ g.provider || '—' }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">状态</span><span :class="['status-badge', statusClass(g.status)]">{{ statusLabel(g.status) }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">耗时</span><span>{{ fmtElapsed(g.elapsedMs) }}</span></div>
                <div v-if="g.duration != null" class="history-detail-row"><span class="history-detail-label">时长</span><span>{{ g.duration }}s</span></div>
                <div v-if="g.taskId" class="history-detail-row"><span class="history-detail-label">任务 ID</span><span class="mono">{{ g.taskId }}</span></div>
                <div v-if="g.prompt" class="history-detail-block">
                  <span class="history-detail-label">提示词</span>
                  <p class="history-prompt">{{ g.prompt }}</p>
                </div>
                <div v-if="g.errorMsg" class="history-detail-block error">
                  <span class="history-detail-label">失败原因</span>
                  <p class="history-prompt">{{ g.errorMsg }}</p>
                </div>
                <div v-if="g.url" class="history-detail-block">
                  <span class="history-detail-label">产物</span>
                  <a :href="g.url" target="_blank" rel="noopener" class="history-link">打开产物</a>
                </div>
              </div>
            </li>
          </ul>
        </section>
      </div>

      <!-- ===== 数据存储 ===== -->
      <div v-else-if="tab === 'storage'" class="settings-scroll">
        <div class="settings-head">
          <h2 class="settings-title">数据存储</h2>
          <p class="settings-desc">统一管理数据库与图片、视频、音频等生成文件。切换目录时自动迁移旧数据（旧目录保留作安全备份）。</p>
        </div>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Current Storage</div>
              <div class="setup-title">当前存储位置</div>
            </div>
            <button class="btn btn-ghost btn-icon" :disabled="storageLoading" title="刷新" @click="loadStorageInfo">
              <Loader2 v-if="storageLoading" :size="13" class="animate-spin" />
              <RefreshCw v-else :size="13" />
            </button>
          </div>
          <div v-if="storageInfo" class="storage-grid">
            <div class="storage-row">
              <span class="storage-icon"><Database :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">数据根目录</div>
                <div class="mono storage-path">{{ storageInfo.dataRoot }}</div>
              </div>
            </div>
            <div class="storage-row">
              <span class="storage-icon"><FileText :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">数据库文件</div>
                <div class="mono storage-path">{{ storageInfo.dbPath }}</div>
                <div class="dim storage-sub">{{ storageInfo.dbExists ? `已创建 · ${fmtBytes(storageInfo.dbSizeBytes)}` : '尚未创建（首次写入时自动创建）' }}</div>
              </div>
            </div>
            <div class="storage-row">
              <span class="storage-icon"><FolderOpen :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">生成文件目录（图片 / 视频 / 音频）</div>
                <div class="mono storage-path">{{ storageInfo.storagePath }}</div>
                <div class="dim storage-sub">{{ storageInfo.storageExists ? `已占用 ${fmtBytes(storageInfo.storageSizeBytes)}` : '尚未创建（首次生成时自动创建）' }}</div>
              </div>
            </div>
          </div>
          <p v-else class="config-empty">正在加载存储信息…</p>
        </section>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Change Directory</div>
              <div class="setup-title">切换到其他目录</div>
              <div class="setup-desc">填写电脑上的目标目录绝对路径，例如 <span class="mono">D:\drama-data</span>。</div>
            </div>
          </div>
          <div class="storage-form">
            <label class="field">
              <span class="field-label">目标目录路径</span>
              <input v-model="newDataRoot" class="input" placeholder="如 D:\drama-data 或 /home/user/drama-data" />
            </label>
            <label class="storage-check">
              <input type="checkbox" v-model="migrateData" />
              自动迁移旧数据（复制数据库与生成文件，旧目录保留）
            </label>
            <div class="storage-actions">
              <button class="btn btn-primary" :disabled="storageChanging || !newDataRoot.trim()" @click="changeDataRoot">
                <Loader2 v-if="storageChanging" :size="14" class="animate-spin" />
                <HardDrive v-else :size="14" />
                切换并迁移
              </button>
            </div>
            <p class="dim storage-note">切换后当前运行中的实例会立即指向新目录，重启后依然生效。迁移采用复制方式，旧目录不会删除，可随时切回。</p>
          </div>
        </section>
      </div>

      <!-- ===== 画风设置 ===== -->
      <div v-else-if="tab === 'style'" class="settings-scroll">
        <div class="settings-head">
          <h2 class="settings-title">全局默认画风</h2>
          <p class="settings-desc">作为兜底画风。实际优先级：角色画风 &gt; 剧集画风 &gt; 全局默认 &gt; 写实。</p>
        </div>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Global Art Style</div>
              <div class="setup-title">默认画风</div>
              <div class="setup-desc">新建剧集 / 角色未指定画风时使用该画风。不设时回退「写实」。</div>
            </div>
            <button class="btn btn-primary" :disabled="artStyleSaving" @click="saveArtStyle">
              <Loader2 v-if="artStyleSaving" :size="14" class="animate-spin" />
              <Check v-else :size="14" />
              保存
            </button>
          </div>
          <div v-if="artStyleLoading" class="config-empty">正在加载画风设置…</div>
          <div v-else class="art-style-grid">
            <button
              v-for="opt in artStyleOptions"
              :key="opt.value"
              class="art-style-card"
              :class="{ active: artStyle === opt.value }"
              @click="artStyle = opt.value"
            >
              <span class="art-style-check">{{ artStyle === opt.value ? '✓' : '' }}</span>
              <span class="art-style-label">{{ opt.label }}</span>
              <span class="art-style-desc">{{ opt.desc }}</span>
            </button>
            <button class="art-style-card" :class="{ active: artStyle === '' }" @click="artStyle = ''">
              <span class="art-style-check">{{ artStyle === '' ? '✓' : '' }}</span>
              <span class="art-style-label">跟随默认（写实）</span>
              <span class="art-style-desc">不设全局画风，使用系统内置写实兜底</span>
            </button>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ChevronDown, Loader2, HardDrive, Database, FolderOpen, History, Image as ImageIcon, Film, Clock, RefreshCw, FileText, Palette, Check } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import { storageAPI, generationsAPI, appSettingsAPI, type StorageInfo, type GenerationRecord } from '~/composables/useApi'

const tab = ref('history')
const baseTabs = [
  { id: 'history', label: '生成历史', icon: History },
  { id: 'storage', label: '数据存储', icon: HardDrive },
  { id: 'style', label: '画风设置', icon: Palette },
]

// ===== 生成历史 =====
const generations = ref<GenerationRecord[]>([])
const historyLoading = ref(false)
const historyFilter = ref<'all' | 'image' | 'video'>('all')
const historyExpandedId = ref<string | null>(null)

const historyFilters = [
  { value: 'all', label: '全部' },
  { value: 'image', label: '图片' },
  { value: 'video', label: '视频' },
] as const

const filteredGenerations = computed(() => {
  if (historyFilter.value === 'all') return generations.value
  return generations.value.filter((g) => g.type === historyFilter.value)
})

const historyStats = computed(() => {
  const list = generations.value
  const total = list.length
  const success = list.filter((g) => isSuccessStatus(g.status)).length
  const failed = list.filter((g) => isFailedStatus(g.status)).length
  return { total, success, failed, processing: total - success - failed }
})

function isSuccessStatus(s: string): boolean {
  return ['completed', 'succeeded', 'success', 'done'].includes((s || '').toLowerCase())
}
function isFailedStatus(s: string): boolean {
  return ['failed', 'error', 'cancelled', 'canceled'].includes((s || '').toLowerCase())
}
function statusLabel(s: string): string {
  if (isSuccessStatus(s)) return '成功'
  if (isFailedStatus(s)) return '失败'
  const low = (s || '').toLowerCase()
  if (['pending', 'queued', 'processing', 'running', 'generating'].includes(low)) return '处理中'
  return s || '未知'
}
function statusClass(s: string): string {
  if (isSuccessStatus(s)) return 'ok'
  if (isFailedStatus(s)) return 'err'
  return 'warn'
}
function fmtTime(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
function fmtElapsed(ms?: number | null): string {
  if (ms == null) return '—'
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  const totalSec = Math.round(ms / 1000)
  return `${Math.floor(totalSec / 60)}m ${totalSec % 60}s`
}
function toggleHistory(g: GenerationRecord) {
  const key = `${g.type}-${g.id}`
  historyExpandedId.value = historyExpandedId.value === key ? null : key
}
async function loadGenerations() {
  historyLoading.value = true
  try {
    generations.value = await generationsAPI.list({ limit: 200 })
  } catch (e: any) {
    toast.error(e.message || '加载生成历史失败')
  } finally {
    historyLoading.value = false
  }
}

// ===== 数据存储 =====
const storageInfo = ref<StorageInfo | null>(null)
const storageLoading = ref(false)
const storageChanging = ref(false)
const newDataRoot = ref('')
const migrateData = ref(true)

function fmtBytes(n: number | null | undefined): string {
  if (n == null || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`
}

async function loadStorageInfo() {
  storageLoading.value = true
  try { storageInfo.value = await storageAPI.info() }
  catch (e: any) { toast.error(e.message || '加载存储信息失败') }
  finally { storageLoading.value = false }
}

async function changeDataRoot() {
  const target = newDataRoot.value.trim()
  if (!target) { toast.error('请填写目标目录路径'); return }
  storageChanging.value = true
  try {
    storageInfo.value = await storageAPI.change(target, migrateData.value)
    newDataRoot.value = ''
    toast.success(migrateData.value ? '目录已切换，旧数据已自动迁移（旧目录保留）' : '目录已切换')
  } catch (e: any) {
    toast.error(e.message || '切换目录失败')
  } finally { storageChanging.value = false }
}

// ===== 全局画风 =====
const artStyle = ref('')
const artStyleLoading = ref(false)
const artStyleSaving = ref(false)

const artStyleOptions = [
  { value: 'realistic', label: '写实电影', desc: '真人质感、电影级光影与景深' },
  { value: 'anime', label: '日式动漫', desc: '赛璐璐上色、鲜明线条、典型动漫风' },
  { value: 'ghibli', label: '吉卜力', desc: '手绘质感、温暖配色、治愈系' },
  { value: 'cinematic', label: '电影感', desc: '商业电影帧、强氛围与色彩分级' },
  { value: 'comic', label: '美漫漫画', desc: '美式漫画、粗线条、明快对比' },
  { value: 'watercolor', label: '水彩', desc: '水彩晕染、柔和过渡、纸面纹理' },
] as const

async function loadArtStyle() {
  artStyleLoading.value = true
  try {
    const data = await appSettingsAPI.get()
    artStyle.value = data.art_style || ''
  } catch (e: any) {
    toast.error(e.message || '加载画风设置失败')
  } finally { artStyleLoading.value = false }
}

async function saveArtStyle() {
  artStyleSaving.value = true
  try {
    await appSettingsAPI.setArtStyle(artStyle.value)
    toast.success('全局默认画风已保存')
  } catch (e: any) {
    toast.error(e.message || '保存画风设置失败')
  } finally { artStyleSaving.value = false }
}

onMounted(() => { loadStorageInfo(); loadGenerations(); loadArtStyle() })
</script>

<style scoped>
.settings-layout { display: flex; height: 100%; background: var(--bg-base); }

.settings-nav {
  width: 220px; flex-shrink: 0; padding: 16px 10px; border-right: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 14px; background: var(--bg-1);
}
.nav-group { display: flex; flex-direction: column; gap: 4px; }
.nav-group-label {
  font-size: 10px; font-weight: 700; color: var(--text-3);
  letter-spacing: 0.12em; text-transform: uppercase; padding: 0 10px 4px;
}
.nav-item {
  display: flex; align-items: center; gap: 8px; padding: 9px 12px; font-size: 13px;
  border: none; background: none; color: var(--text-2); cursor: pointer;
  border-radius: var(--radius); transition: all 0.12s; text-align: left; width: 100%;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-0); }
.nav-item.active { background: var(--accent-bg); color: var(--accent-text); font-weight: 600; box-shadow: var(--shadow-card); }

.settings-content { flex: 1; overflow: hidden; }
.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 840px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; }

/* Panel */
.setup-panel {
  padding: 18px 18px 16px;
  margin-bottom: 18px;
}
.setup-panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.setup-panel-head.compact { margin-bottom: 12px; }
.setup-kicker {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 4px;
}
.setup-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-0);
}
.setup-desc {
  font-size: 12px;
  color: var(--text-2);
  margin-top: 4px;
}
.config-empty { font-size: 12px; color: var(--text-3); padding: 12px 0; }

/* Storage */
.storage-grid { display: flex; flex-direction: column; gap: 10px; }
.storage-row {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 14px; border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-1);
}
.storage-icon {
  width: 34px; height: 34px; border-radius: 10px; background: var(--accent-bg);
  color: var(--accent); display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.storage-meta { min-width: 0; flex: 1; }
.storage-label { font-size: 11px; color: var(--text-3); margin-bottom: 2px; }
.storage-path {
  font-size: 12px; color: var(--text-1); word-break: break-all;
}
.storage-sub { font-size: 11px; margin-top: 3px; }
.storage-form { display: flex; flex-direction: column; gap: 12px; }
.storage-check { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-2); cursor: pointer; }
.storage-check input { accent-color: var(--accent); }
.storage-actions { display: flex; gap: 8px; }
.storage-note { font-size: 11px; line-height: 1.6; }

/* History */
.history-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
.history-filters { display: flex; gap: 8px; }
.chip {
  padding: 6px 14px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-0); color: var(--text-2); font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.chip:hover { color: var(--text-0); border-color: var(--text-3); }
.chip.active {
  background: var(--accent-bg); border-color: var(--accent); color: var(--accent-text); font-weight: 600;
}
.history-stats {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;
}
.history-stat {
  background: var(--bg-surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px;
}
.history-stat-num { font-size: 20px; font-weight: 700; font-family: var(--font-display); color: var(--text-0); }
.history-stat-label { font-size: 12px; color: var(--text-3); margin-top: 4px; }
.history-stat.ok .history-stat-num { color: #2f9e63; }
.history-stat.warn .history-stat-num { color: #c9973f; }
.history-stat.err .history-stat-num { color: #c95844; }

.history-list { display: flex; flex-direction: column; gap: 8px; margin: 0; padding: 0; list-style: none; }
.history-item {
  border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-0);
  overflow: hidden;
}
.history-item-head {
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 11px 14px;
  background: none; border: none; cursor: pointer; text-align: left;
}
.history-item-head:hover { background: var(--bg-hover); }
.history-type {
  width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center;
  justify-content: center; flex-shrink: 0;
}
.history-type.image { background: rgba(13,148,136,0.14); color: #0d9488; }
.history-type.video { background: rgba(150, 92, 235, 0.16); color: #965ceb; }
.history-item-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.history-item-title { font-size: 13px; font-weight: 600; color: var(--text-0); }
.history-item-meta { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-3); }
.history-item-meta .dot { color: var(--text-3); }
.history-item-time {
  display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--text-3); flex-shrink: 0;
}
.history-chevron { color: var(--text-3); transition: transform 0.18s; flex-shrink: 0; }
.history-chevron.open { transform: rotate(180deg); }

.status-badge {
  font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 999px; white-space: nowrap;
}
.status-badge.ok { background: rgba(47, 158, 99, 0.12); color: #2f9e63; }
.status-badge.warn { background: rgba(201, 151, 63, 0.14); color: #c9973f; }
.status-badge.err { background: rgba(201, 88, 68, 0.12); color: #c95844; }

.history-item-body {
  padding: 12px 14px; border-top: 1px solid var(--border);
  background: var(--bg-surface); display: flex; flex-direction: column; gap: 8px;
}
.history-detail-row { display: flex; gap: 10px; font-size: 12px; color: var(--text-1); }
.history-detail-label {
  width: 64px; flex-shrink: 0; font-size: 11px; color: var(--text-3); padding-top: 2px;
}
.history-detail-block { display: flex; gap: 10px; font-size: 12px; color: var(--text-1); }
.history-detail-block .history-detail-label { width: auto; }
.history-prompt {
  margin: 0; font-size: 12px; color: var(--text-1); line-height: 1.7; flex: 1;
  background: var(--bg-0); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 10px 12px; white-space: pre-wrap; word-break: break-all;
}
.history-detail-block.error .history-detail-label { color: #c95844; }
.history-detail-block.error .history-prompt { color: #c95844; border-color: rgba(201, 88, 68, 0.25); }
.history-link { color: var(--accent); font-size: 12px; }
.history-link:hover { text-decoration: underline; }

/* Field */
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--text-1); }
/* 画风设置 */
.art-style-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}
.art-style-card {
  display: flex; flex-direction: column; align-items: flex-start; gap: 4px;
  padding: 12px 14px; text-align: left; cursor: pointer;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-1); color: var(--text-0);
  transition: all 0.12s; position: relative;
}
.art-style-card:hover { border-color: var(--accent); background: var(--accent-bg); }
.art-style-card.active { border-color: var(--accent); background: var(--accent-bg); box-shadow: var(--shadow-card); }
.art-style-check {
  position: absolute; top: 8px; right: 10px;
  width: 18px; height: 18px; border-radius: 50%;
  border: 1px solid var(--border); background: var(--bg-2);
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: transparent;
}
.art-style-card.active .art-style-check {
  border-color: var(--accent); background: var(--accent); color: #fff;
}
.art-style-label { font-size: 13px; font-weight: 600; }
.art-style-desc { font-size: 11px; color: var(--text-3); line-height: 1.5; }
</style>

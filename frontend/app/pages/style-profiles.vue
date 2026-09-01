<template>
  <div class="sp-page">
    <div class="settings-scroll">
      <div class="settings-head">
        <h2 class="settings-title">风格 Profile</h2>
        <p class="settings-desc">提炼参考素材的 house style（叙事节奏、镜头模式、配音、QC 规则），激活后注入分镜拆解与图片生成，保持多集风格一致。</p>
      </div>

      <section class="setup-panel card">
        <div class="setup-panel-head">
          <div>
            <div class="setup-kicker">Style Profiles</div>
            <div class="setup-title">风格库</div>
            <div class="setup-desc">每个 Profile 可绑定一部剧集（留空为全局）。同剧集内同时只有一个 Profile 处于激活状态。</div>
          </div>
          <div class="sp-head-actions">
            <button class="btn btn-ghost" :disabled="loading" @click="loadProfiles">
              <RefreshCw :size="14" :class="{ 'animate-spin': loading }" /> 刷新
            </button>
            <button class="btn btn-primary" @click="openCreate">
              <Plus :size="14" /> 新建 Profile
            </button>
          </div>
        </div>

        <div v-if="profiles.length" class="sp-grid">
          <article v-for="p in profiles" :key="p.id" class="sp-card" :class="{ active: p.is_active }">
            <div class="sp-card-top">
              <div class="sp-card-name">{{ p.name }}</div>
              <span v-if="p.is_active" class="tag tag-accent">激活中</span>
            </div>
            <div class="sp-card-sub">
              <span>{{ p.drama_id ? `剧集 #${p.drama_id}` : '全局' }}</span>
              <span>{{ fmtDate(p.updated_at) }}</span>
            </div>
            <p class="sp-card-desc">{{ p.description || '暂无描述' }}</p>
            <div class="sp-card-actions">
              <button class="btn btn-ghost btn-sm" @click="openDistill(p)">提炼</button>
              <button class="btn btn-ghost btn-sm" @click="openEdit(p)">编辑</button>
              <button
                v-if="!p.is_active"
                class="btn btn-ghost btn-sm"
                :disabled="activatingId === p.id"
                @click="activateProfile(p)"
              >激活</button>
              <button class="btn btn-ghost btn-sm danger" @click="delProfile(p)">删除</button>
            </div>
          </article>
        </div>
        <p v-else class="config-empty">{{ loading ? '加载中…' : '暂无风格 Profile，点击右上角新建。' }}</p>
      </section>

      <!-- 新建 / 编辑弹窗 -->
      <div v-if="dialog" class="modal-mask">
        <div class="modal-card sp-modal">
          <div class="modal-head">
            <h3>{{ editingId ? '编辑 Profile' : '新建 Profile' }}</h3>
            <button class="modal-close" @click="dialog = false"><X :size="16" /></button>
          </div>
          <div class="modal-body sp-form">
            <label class="field">
              <span class="field-label">名称 *</span>
              <input v-model="form.name" class="input" placeholder="如：古风武侠 · 冷峻写实" />
            </label>
            <label class="field">
              <span class="field-label">剧集绑定</span>
              <select v-model="form.drama_id" class="input">
                <option :value="null">全局（不绑定）</option>
                <option v-for="d in dramaOptions" :key="d.id" :value="d.id">{{ d.title || `剧集 ${d.id}` }}</option>
              </select>
            </label>
            <label class="field">
              <span class="field-label">描述</span>
              <textarea v-model="form.description" class="input sp-textarea" placeholder="一句话概括风格定位" rows="2"></textarea>
            </label>
            <label class="field">
              <span class="field-label">叙事与节奏（storytelling）</span>
              <textarea v-model="form.storytelling" class="input sp-textarea" placeholder="钩子密度、节奏把控、情绪曲线等" rows="3"></textarea>
            </label>
            <label class="field">
              <span class="field-label">镜头模式（shot_patterns）</span>
              <textarea v-model="form.shot_patterns" class="input sp-textarea" placeholder="景别偏好、运镜习惯、转场方式等" rows="3"></textarea>
            </label>
            <label class="field">
              <span class="field-label">配音与字幕（audio_captions）</span>
              <textarea v-model="form.audio_captions" class="input sp-textarea" placeholder="配音风格、字幕密度、BGM 偏好等" rows="3"></textarea>
            </label>
            <label class="field">
              <span class="field-label">QC 规则（qc_rules）</span>
              <textarea v-model="form.qc_rules" class="input sp-textarea" placeholder="硬性质检标准，如禁止出镜、必须闭合等" rows="3"></textarea>
            </label>
            <label class="field">
              <span class="field-label">偏好（preferences）</span>
              <textarea v-model="form.preferences" class="input sp-textarea" placeholder="其他偏好，如色彩倾向、禁忌元素" rows="2"></textarea>
            </label>
          </div>
          <div class="modal-foot">
            <button class="btn btn-ghost" @click="dialog = false">取消</button>
            <button class="btn btn-primary" :disabled="saving" @click="saveProfile">保存</button>
          </div>
        </div>
      </div>

      <!-- 提炼弹窗 -->
      <div v-if="distillDialog" class="modal-mask">
        <div class="modal-card sp-modal">
          <div class="modal-head">
            <h3>提炼 house style</h3>
            <button class="modal-close" @click="closeDistill"><X :size="16" /></button>
          </div>
          <div class="modal-body">
            <p class="sp-distill-hint">用 LLM 分析 Profile 中的文本素材，提炼成结构化风格配置，结果先展示供确认再写入。</p>
            <div v-if="distilling" class="sp-distill-loading">
              <Loader2 :size="18" class="animate-spin" /> 正在提炼…
            </div>
            <template v-else-if="distillResult">
              <div class="sp-distill-result">
                <section v-if="distillResult.storytelling?.summary">
                  <div class="sp-distill-sec-title">叙事</div>
                  <p>{{ distillResult.storytelling.summary }}</p>
                </section>
                <section v-if="distillResult.shot_patterns?.summary">
                  <div class="sp-distill-sec-title">镜头</div>
                  <p>{{ distillResult.shot_patterns.summary }}</p>
                </section>
                <section v-if="distillResult.audio_captions?.summary">
                  <div class="sp-distill-sec-title">配音</div>
                  <p>{{ distillResult.audio_captions.summary }}</p>
                </section>
                <section v-if="distillResult.qc_rules?.summary">
                  <div class="sp-distill-sec-title">QC</div>
                  <p>{{ distillResult.qc_rules.summary }}</p>
                </section>
                <div class="sp-distill-tags" v-if="distillResult.preferences?.length">
                  <span v-for="(t, i) in distillResult.preferences" :key="i" class="tag">{{ t }}</span>
                </div>
              </div>
            </template>
            <p v-else class="config-empty">点击下方「开始提炼」分析素材。</p>
          </div>
          <div class="modal-foot">
            <button class="btn btn-ghost" @click="closeDistill">关闭</button>
            <template v-if="distillResult && !distilling">
              <button class="btn btn-ghost" @click="distillResult = null">重新提炼</button>
              <button class="btn btn-primary" @click="applyDistill">确认写入</button>
            </template>
            <button v-else class="btn btn-primary" :disabled="distilling" @click="startDistill">开始提炼</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Plus, X, RefreshCw, Loader2 } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import { styleProfileAPI, dramaAPI } from '~/composables/useApi'

const loading = ref(false)
const profiles = ref<any[]>([])
const dramaOptions = ref<any[]>([])
const activatingId = ref<number | null>(null)

const dialog = ref(false)
const editingId = ref<number | null>(null)
const saving = ref(false)
const form = ref({
  name: '',
  drama_id: null as number | null,
  description: '',
  storytelling: '',
  shot_patterns: '',
  audio_captions: '',
  qc_rules: '',
  preferences: '',
})

const distillDialog = ref(false)
const distilling = ref(false)
const distillProfileId = ref<number | null>(null)
const distillResult = ref<any>(null)

async function loadProfiles() {
  loading.value = true
  try {
    const res = await styleProfileAPI.list()
    profiles.value = res.profiles || []
  } catch (e: any) {
    toast.error(e?.message || '加载风格 Profile 失败')
  } finally {
    loading.value = false
  }
}

async function loadDramas() {
  try {
    const res = await dramaAPI.list()
    dramaOptions.value = res.items || res || []
  } catch { /* 忽略 */ }
}

function resetForm() {
  form.value = { name: '', drama_id: null, description: '', storytelling: '', shot_patterns: '', audio_captions: '', qc_rules: '', preferences: '' }
}

function openCreate() {
  editingId.value = null
  resetForm()
  dialog.value = true
}

function openEdit(p: any) {
  editingId.value = p.id
  form.value = {
    name: p.name || '',
    drama_id: p.drama_id ?? null,
    description: p.description || '',
    storytelling: p.storytelling || '',
    shot_patterns: p.shot_patterns || '',
    audio_captions: p.audio_captions || '',
    qc_rules: p.qc_rules || '',
    preferences: p.preferences || '',
  }
  dialog.value = true
}

async function saveProfile() {
  if (!form.value.name.trim()) { toast.warning('请填写名称'); return }
  saving.value = true
  const payload: any = {
    name: form.value.name.trim(),
    drama_id: form.value.drama_id,
    description: form.value.description.trim(),
    storytelling: form.value.storytelling.trim(),
    shot_patterns: form.value.shot_patterns.trim(),
    audio_captions: form.value.audio_captions.trim(),
    qc_rules: form.value.qc_rules.trim(),
    preferences: form.value.preferences.trim(),
  }
  try {
    if (editingId.value) await styleProfileAPI.update(editingId.value, payload)
    else await styleProfileAPI.create(payload)
    toast.success('已保存')
    dialog.value = false
    await loadProfiles()
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
  } finally {
    saving.value = false
  }
}

async function activateProfile(p: any) {
  activatingId.value = p.id
  try {
    await styleProfileAPI.activate(p.id)
    toast.success(`已激活「${p.name}」`)
    await loadProfiles()
  } catch (e: any) {
    toast.error(e?.message || '激活失败')
  } finally {
    activatingId.value = null
  }
}

async function delProfile(p: any) {
  if (!confirm(`确定删除风格 Profile「${p.name}」？`)) return
  try {
    await styleProfileAPI.del(p.id)
    toast.success('已删除')
    await loadProfiles()
  } catch (e: any) {
    toast.error(e?.message || '删除失败')
  }
}

function openDistill(p: any) {
  distillProfileId.value = p.id
  distillResult.value = null
  distillDialog.value = true
}

function closeDistill() {
  distillDialog.value = false
  distillResult.value = null
  distillProfileId.value = null
}

async function startDistill() {
  if (!distillProfileId.value) return
  distilling.value = true
  distillResult.value = null
  try {
    const res = await styleProfileAPI.distill(distillProfileId.value)
    distillResult.value = res.result || {}
  } catch (e: any) {
    toast.error(e?.message || '提炼失败')
  } finally {
    distilling.value = false
  }
}

async function applyDistill() {
  if (!distillProfileId.value || !distillResult.value) return
  try {
    await styleProfileAPI.apply(distillProfileId.value, distillResult.value)
    toast.success('已写入 Profile')
    closeDistill()
    await loadProfiles()
  } catch (e: any) {
    toast.error(e?.message || '写入失败')
  }
}

function fmtDate(ts: string | null | undefined) {
  if (!ts) return '—'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ts
  return d.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

onMounted(() => { loadProfiles(); loadDramas() })
</script>

<style scoped>
.sp-page { display: flex; min-height: 100vh; }
.settings-scroll { flex: 1; height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 980px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; }
.setup-panel { padding: 18px 18px 16px; margin-bottom: 18px; }
.setup-panel-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.setup-kicker { font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-3); margin-bottom: 4px; }
.setup-title { font-size: 16px; font-weight: 700; color: var(--text-0); }
.setup-desc { font-size: 12px; color: var(--text-2); margin-top: 4px; }
.config-empty { font-size: 12px; color: var(--text-3); padding: 12px 0; }
.sp-head-actions { display: flex; align-items: center; gap: 8px; }
.sp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }
.sp-card {
  padding: 14px; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-surface);
  display: flex; flex-direction: column; gap: 8px;
}
.sp-card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.sp-card-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.sp-card-name { font-size: 14px; font-weight: 600; color: var(--text-0); }
.sp-card-sub { display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: var(--text-3); }
.sp-card-desc { font-size: 12px; color: var(--text-2); line-height: 1.6; margin: 0; }
.sp-card-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; padding-top: 4px; }
.btn-sm { padding: 3px 8px; font-size: 11px; }
.btn-sm.danger { color: var(--error); }
.modal-mask {
  position: fixed; inset: 0; background: rgba(9, 14, 24, 0.55); backdrop-filter: blur(4px);
  display: flex; align-items: flex-start; justify-content: center; padding: 60px 20px 40px; z-index: 100;
}
.modal-card {
  width: 560px; max-width: 100%; max-height: 82vh; overflow-y: auto;
  background: var(--bg-surface); border: 1px solid var(--border); border-radius: 16px;
  display: flex; flex-direction: column;
}
.modal-head { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
.modal-head h3 { font-size: 15px; font-weight: 600; color: var(--text-0); margin: 0; }
.modal-close { background: none; border: none; color: var(--text-3); cursor: pointer; padding: 4px; }
.modal-close:hover { color: var(--text-1); }
.modal-body { padding: 18px 20px; }
.modal-foot { display: flex; justify-content: flex-end; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--border); }
.sp-form { display: flex; flex-direction: column; gap: 12px; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; color: var(--text-2); font-weight: 500; }
.sp-textarea { resize: vertical; min-height: 52px; font-family: inherit; }
.sp-distill-hint { font-size: 12px; color: var(--text-3); line-height: 1.7; margin: 0 0 12px; }
.sp-distill-loading { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2); padding: 12px 0; }
.sp-distill-result { display: flex; flex-direction: column; gap: 12px; }
.sp-distill-result section p { font-size: 12px; color: var(--text-2); line-height: 1.7; margin: 4px 0 0; }
.sp-distill-sec-title { font-size: 12px; font-weight: 600; color: var(--text-1); }
.sp-distill-tags { display: flex; flex-wrap: wrap; gap: 6px; }
</style>

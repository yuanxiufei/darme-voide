<template>
  <div class="scene-detail">
    <header class="detail-header">
      <button class="back-btn" @click="goBack">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
        返回
      </button>
      <h1>{{ scene?.location || '加载中...' }}</h1>
      <div class="header-actions">
        <button class="btn btn-primary" :disabled="saving" @click="save">{{ saving ? '保存中...' : '保存修改' }}</button>
      </div>
    </header>

    <div v-if="loading" class="loading">加载中...</div>
    <template v-else-if="scene">
      <div class="detail-body">
        <!-- 左侧：图片预览 -->
        <aside class="preview-col">
          <div class="image-card">
            <img v-if="scene.image_url || scene.imageUrl" :src="'/' + (scene.image_url || scene.imageUrl)" :alt="scene.location" />
            <div v-else class="no-image">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              <span>暂无场景图</span>
            </div>
          </div>
          <div class="quick-actions">
            <ModelSelector v-model="imageModel" service-type="image" label="生成模型" />
            <button class="btn btn-primary btn-block" :disabled="imgGen" @click="generateImage">
              {{ imgGen ? '生成中...' : '重新生成场景图' }}
            </button>
          </div>
          <div v-if="errorMsg" class="error-banner">{{ errorMsg }}</div>
        </aside>

        <!-- 右侧：编辑表单 -->
        <main class="form-col">
          <!-- 场景信息 -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg> 场景信息</h3>
            <div class="field-grid-2">
              <label class="field">
                <span>场景名称 / 地点</span>
                <input v-model="form.location" class="input" placeholder="如：凌云阁大殿、竹林小径..." />
              </label>
              <label class="field">
                <span>时间设定</span>
                <select v-model="form.time" class="input">
                  <option value="">未指定</option>
                  <option value="黎明">黎明</option>
                  <option value="清晨">清晨</option>
                  <option value="正午">正午</option>
                  <option value="午后">午后</option>
                  <option value="黄昏">黄昏</option>
                  <option value="夜晚">夜晚</option>
                  <option value="深夜">深夜</option>
                </select>
              </label>
            </div>
            <label class="field">
              <span>场景描述</span>
              <textarea v-model="form.description" class="input" rows="3" placeholder="场景的详细环境描述，包括建筑风格、空间布局、氛围基调..." />
            </label>
          </section>

          <!-- 环境设定 -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg> 环境设定</h3>
            <div class="field-grid-2">
              <label class="field">
                <span>氛围</span>
                <input v-model="form.atmosphere" class="input" placeholder="庄严 / 宁静 / 压抑 / 欢快..." />
              </label>
              <label class="field">
                <span>光线</span>
                <input v-model="form.lighting" class="input" placeholder="自然光 / 烛光 / 月光 / 昏暗..." />
              </label>
            </div>
            <div class="field-grid-2">
              <label class="field">
                <span>天气</span>
                <input v-model="form.weather" class="input" placeholder="晴朗 / 雨天 / 雾气 / 风雪..." />
              </label>
              <label class="field">
                <span>季节</span>
                <input v-model="form.season" class="input" placeholder="春 / 夏 / 秋 / 冬..." />
              </label>
            </div>
            <label class="field">
              <span>视觉风格</span>
              <input v-model="form.style" class="input" placeholder="水墨风 / 赛博朋克 / 古典写实..." />
            </label>
          </section>

          <!-- 图片生成 Prompt -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 图片生成提示词</h3>
            <label class="field">
              <span>自定义 Prompt（留空则自动根据上述信息构建）</span>
              <textarea v-model="form.customPrompt" class="input mono" rows="4" placeholder="highly detailed cinematic environment, [location], [atmosphere]..." />
            </label>
            <p class="hint">留空时系统会根据「地点+时间+氛围+光线+天气+风格」自动组合 prompt。</p>
          </section>
        </main>
      </div>
    </template>
    <div v-else class="empty-state">
      场景不存在或已被删除
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from '#app'

const route = useRoute()
const router = useRouter()
const dramaId = Number(route.params.id)
const sceneId = Number(route.params.sceneId)

// 返回上一级：有浏览历史时后退，直接访问/刷新（无历史）时兜底回项目页
function goBack() {
  const state = window.history.state
  if (state && state.back) {
    router.back()
  } else {
    router.push(`/drama/${dramaId}`)
  }
}

const scene = ref<any>(null)
const loading = ref(true)
const saving = ref(false)
const imgGen = ref(false)
const errorMsg = ref('')
const imageModel = ref('')
const imgPollRef = ref<ReturnType<typeof setInterval> | null>(null)
const imgPollTimeoutRef = ref<ReturnType<typeof setTimeout> | null>(null)

const form = reactive<any>({})
onUnmounted(() => {
  if (imgPollRef.value) clearInterval(imgPollRef.value)
  if (imgPollTimeoutRef.value) clearTimeout(imgPollTimeoutRef.value)
})

onMounted(async () => {
  try {
    const { sceneAPI } = await import('~/composables/useApi')
    const res: any = await sceneAPI.get(sceneId)
    scene.value = res.data || res || null
    if (!scene.value) return
    const s = scene.value
    form.location = s.location || ''
    form.time = s.time || ''
    form.description = s.description || ''
    form.atmosphere = s.atmosphere || ''
    form.lighting = s.lighting || ''
    form.weather = s.weather || ''
    form.season = s.season || ''
    form.style = s.style || ''
    form.customPrompt = s.customPrompt || ''
  } catch (e: any) {
    errorMsg.value = e.message
  } finally {
    loading.value = false
  }
})

async function save() {
  saving.value = true
  try {
    const { sceneAPI } = await import('~/composables/useApi')
    await sceneAPI.update(sceneId, {
      location: form.location,
      time: form.time || null,
      description: form.description,
      atmosphere: form.atmosphere || null,
      lighting: form.lighting || null,
      weather: form.weather || null,
      season: form.season || null,
      style: form.style || null,
      customPrompt: form.customPrompt || null,
    })
    if (scene.value) Object.assign(scene.value, form)
    alert('保存成功')
  } catch (e: any) {
    errorMsg.value = '保存失败: ' + e.message
  } finally {
    saving.value = false
  }
}

async function generateImage() {
  imgGen.value = true
  errorMsg.value = ''
  try {
    const { sceneAPI, dramaAPI } = await import('~/composables/useApi')
    await sceneAPI.generateImage(sceneId, undefined, {
      prompt: form.customPrompt || undefined,
      model: imageModel.value || undefined,
    })
    imgPollRef.value = setInterval(async () => {
      try {
        const data: any = await dramaAPI.get(dramaId)
        const list = data?.scenes || []
        const updated = list.find((s: any) => s.id === sceneId)
        if (updated?.image_url || updated?.imageUrl) {
          clearInterval(imgPollRef.value!)
          imgPollRef.value = null
          scene.value = updated
          imgGen.value = false
        }
      } catch (e) {
        console.warn('[Scene] poll error:', e)
      }
    }, 3000)
    imgPollTimeoutRef.value = setTimeout(() => {
      if (imgPollRef.value) { clearInterval(imgPollRef.value); imgPollRef.value = null }
      imgPollTimeoutRef.value = null
      imgGen.value = false
    }, 120000)
  } catch (e: any) {
    errorMsg.value = e.message
    imgGen.value = false
  }
}

onUnmounted(() => {
  if (imgPollRef.value) clearInterval(imgPollRef.value)
  if (imgPollTimeoutRef.value) clearTimeout(imgPollTimeoutRef.value)
})
</script>

<style scoped>
.scene-detail {
  height: 100%;
  min-height: 0;
  overflow-y: auto;
  background: var(--bg-base);
  font-family: var(--font-body);
}
.detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 28px;
  background: #fff;
  border-bottom: 1px solid rgba(100,120,180,0.12);
  position: sticky;
  top: 0;
  z-index: 10;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.detail-header h1 {
  font-size: 18px;
  font-weight: 700;
  flex: 1;
  margin: 0;
  color: var(--text-0);
}
.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid rgba(100,120,180,0.18);
  background: #fff;
  cursor: pointer;
  font-size: 13px;
  color: #2c3850;
  transition: all .15s;
}
.back-btn:hover { background: rgba(13,148,136,0.06); border-color: rgba(13,148,136,0.25); }

.detail-body {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 24px;
  padding: 24px 28px;
  max-width: 1280px;
  margin: 0 auto;
}
@media (max-width: 800px) {
  .detail-body { grid-template-columns: 1fr; }
}

.preview-col { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 72px; align-self: start; }
.image-card {
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
  border: 1px solid rgba(100,120,180,0.12);
  aspect-ratio: 16/9;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.image-card img { width: 100%; height: 100%; object-fit: cover; display: block; }
.no-image {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
  gap: 8px;
  background: linear-gradient(135deg, #f8f9fc 0%, #eef1f7 100%);
}
.quick-actions { display: flex; flex-direction: column; gap: 10px; }
.btn-block { width: 100%; justify-content: center; }

.form-col { display: flex; flex-direction: column; gap: 20px; }
.form-section {
  background: #fff;
  border-radius: 16px;
  border: 1px solid rgba(100,120,180,0.1);
  padding: 22px 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.form-section h3 {
  font-size: 14px;
  font-weight: 700;
  color: var(--text-0);
  margin: 0 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.form-section h3 svg { opacity: 0.45; }

.field-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 600px) { .field-grid-2 { grid-template-columns: 1fr; } }
.field { display: flex; flex-direction: column; gap: 5px; }
.field span { font-size: 12px; font-weight: 600; color: rgba(40,50,80,0.65); text-transform: uppercase; letter-spacing: 0.3px; }
.input {
  padding: 9px 12px;
  border-radius: 10px;
  border: 1px solid rgba(100,120,180,0.18);
  font-size: 13px;
  outline: none;
  transition: border-color .15s, box-shadow .15s;
  background: #fafbfc;
  color: var(--text-0);
}
.input:focus { border-color: rgba(13,148,136,0.45); box-shadow: 0 0 0 3px rgba(13,148,136,0.08); background: #fff; }
.input::placeholder { color: #b0b8c8; }
textarea.input { resize: vertical; min-height: 60px; line-height: 1.5; }
select.input { cursor: pointer; appearance: auto; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.hint { font-size: 12px; color: #8a94a6; margin: 4px 0 0; line-height: 1.4; }

.error-banner {
  padding: 10px 14px;
  border-radius: 10px;
  background: #fef2f2;
  color: #dc2626;
  font-size: 12px;
  border: 1px solid #fecaca;
}
.loading, .empty-state {
  text-align: center;
  padding: 80px 20px;
  color: #999;
  font-size: 15px;
}

/* 按钮增强 */
.header-actions :deep(.btn) {
  padding: 8px 20px;
  font-size: 13px;
  font-weight: 600;
  border-radius: 10px;
}
.quick-actions :deep(.btn) {
  padding: 10px 16px;
  font-size: 13px;
  font-weight: 600;
  border-radius: 10px;
  transition: all .15s;
}
</style>

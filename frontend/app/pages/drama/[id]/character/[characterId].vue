<template>
  <div class="char-detail">
    <!-- 顶部导航 -->
    <header class="detail-header">
      <button class="back-btn" @click="$router.back()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
        返回
      </button>
      <h1>{{ char?.name || '加载中...' }}</h1>
      <div class="header-actions">
        <button class="btn btn-primary" :disabled="saving" @click="save">{{ saving ? '保存中...' : '保存修改' }}</button>
      </div>
    </header>

    <div v-if="loading" class="loading">加载中...</div>
    <template v-else-if="char">
      <div class="detail-body">
        <!-- 左侧：图片预览 -->
        <aside class="preview-col">
          <div class="image-card">
            <img v-if="char.image_url || char.imageUrl" :src="'/' + (char.image_url || char.imageUrl)" :alt="char.name" />
            <div v-else class="no-image">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>
              <span>暂无形象图</span>
            </div>
          </div>
          <!-- 快捷操作 -->
          <div class="quick-actions">
            <ModelSelector v-model="imageModel" service-type="image" label="生成模型" />
            <button class="btn btn-primary btn-block" :disabled="imgGen" @click="generateImage">
              {{ imgGen ? '生成中...' : '重新生成形象图' }}
            </button>
            <button class="btn btn-block" @click="testVoice">试听声音</button>
          </div>
          <div v-if="errorMsg" class="error-banner">{{ errorMsg }}</div>
        </aside>

        <!-- 右侧：编辑表单 -->
        <main class="form-col">
          <!-- 基本信息 -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg> 基本信息</h3>
            <div class="field-grid-2">
              <label class="field">
                <span>角色名称</span>
                <input v-model="form.name" class="input" />
              </label>
              <label class="field">
                <span>角色定位</span>
                <input v-model="form.role" class="input" placeholder="主角 / 反派 / 配角 / 旁白" />
              </label>
            </div>
            <label class="field">
              <span>角色描述</span>
              <textarea v-model="form.description" class="input" rows="3" placeholder="角色的背景故事、性格特点、在剧情中的作用..." />
            </label>
          </section>

          <!-- 形象设定 -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg> 形象设定</h3>
            <label class="field">
              <span>外貌特征</span>
              <textarea v-model="form.appearance" class="input" rows="3" placeholder="身高、体型、发色、瞳色、面部特征、标志性装扮..." />
            </label>
            <label class="field">
              <span>性格特点</span>
              <textarea v-model="form.personality" class="input" rows="2" placeholder="性格关键词，如：沉稳果断、幽默风趣、阴险狡诈..." />
            </label>
            <div class="field-grid-2">
              <label class="field">
                <span>服装风格</span>
                <input v-model="form.clothing" class="input" placeholder="古装长袍 / 现代西装 / 盔甲战衣..." />
              </label>
              <label class="field">
                <span>武器装备</span>
                <input v-model="form.weapons" class="input" placeholder="长剑 / 匕首 / 法杖 / 无..." />
              </label>
            </div>
          </section>

          <!-- 图片生成 Prompt -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 图片生成提示词</h3>
            <label class="field">
              <span>自定义 Prompt（留空则自动根据上述信息构建）</span>
              <textarea v-model="form.customPrompt" class="input mono" rows="4" placeholder="cinematic portrait of [name], [appearance], [clothing]..." />
            </label>
            <p class="hint">留空时系统会根据「名称+外貌+服装+性格」自动组合 prompt。</p>
          </section>

          <!-- 声音配置 -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg> 声音配置</h3>
            <div class="field-grid-2">
              <label class="field">
                <span>音色描述</span>
                <input v-model="form.voiceStyle" class="input" placeholder="温柔女声 / 沉稳男声 / 童声..." />
              </label>
              <label class="field">
                <span>TTS 模型</span>
                <select v-model="form.voiceModel" class="input">
                  <option value="speech-2.8-hd">MiniMax 2.8 HD</option>
                  <option value="speech-2.6-hd">MiniMax 2.6 HD</option>
                  <option value="speech-2.6">MiniMax 2.6</option>
                </select>
              </label>
            </div>
            <div class="field-grid-3">
              <label class="field">
                <span>语速</span>
                <select v-model="form.voiceSpeed" class="input">
                  <option value="">正常 (1.0x)</option>
                  <option value="0.7">慢速 (0.7x)</option>
                  <option value="0.85">稍慢 (0.85x)</option>
                  <option value="1.15">稍快 (1.15x)</option>
                  <option value="1.3">快速 (1.3x)</option>
                </select>
              </label>
              <label class="field">
                <span>情感表达</span>
                <select v-model="form.voiceEmotion" class="input">
                  <option value="">平静/默认</option>
                  <option value="happy">开心</option>
                  <option value="sad">悲伤</option>
                  <option value="angry">愤怒</option>
                  <option value="excited">兴奋</option>
                  <option value="calm">舒缓</option>
                  <option value="serious">严肃</option>
                  <option value="neutral">中性</option>
                </select>
              </label>
              <label class="field">
                <span>音调</span>
                <select v-model="form.voicePitch" class="input">
                  <option value="">标准 (0)</option>
                  <option value="3">高 (+3)</option>
                  <option value="2">稍高 (+2)</option>
                  <option value="-2">稍低 (-2)</option>
                  <option value="-3">低 (-3)</option>
                  <option value="-5">很低 (-5)</option>
                </select>
              </label>
            </div>
          </section>
        </main>
      </div>
    </template>
    <div v-else class="empty-state">
      角色不存在或已被删除
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRoute } from '#app'

const route = useRoute()

const dramaId = Number(route.params.id)
const characterId = Number(route.params.characterId)

const char = ref<any>(null)
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
    const { characterAPI } = await import('~/composables/useApi')
    // 直接获取单个角色
    const res: any = await characterAPI.get(characterId)
    char.value = res.data || res || null
    if (!char.value) return
    // 回填表单
    const c = char.value
    form.name = c.name
    form.role = c.role || ''
    form.description = c.description || ''
    form.appearance = c.appearance || ''
    form.personality = c.personality || ''
    form.clothing = c.clothing || ''
    form.weapons = c.weapons || ''
    form.customPrompt = c.customPrompt || ''
    form.voiceStyle = c.voiceStyle || ''
    form.voiceModel = c.voiceModel || 'speech-2.8-hd'
    form.voiceSpeed = c.voiceSpeed != null ? String(c.voiceSpeed) : ''
    form.voiceEmotion = c.voiceEmotion || ''
    form.voicePitch = c.voicePitch != null ? String(c.voicePitch) : ''
  } catch (e: any) {
    errorMsg.value = e.message
  } finally {
    loading.value = false
  }
})

async function save() {
  saving.value = true
  try {
    const { characterAPI } = await import('~/composables/useApi')
    await characterAPI.update(characterId, {
      name: form.name,
      role: form.role,
      description: form.description,
      appearance: form.appearance,
      personality: form.personality,
      clothing: form.clothing || null,
      weapons: form.weapons || null,
      customPrompt: form.customPrompt || null,
      voiceStyle: form.voiceStyle,
      voiceModel: form.voiceModel,
      voiceSpeed: form.voiceSpeed ? Number(form.voiceSpeed) : null,
      voiceEmotion: form.voiceEmotion || null,
      voicePitch: form.voicePitch ? Number(form.voicePitch) : null,
    })
    // 刷新本地数据
    if (char.value) Object.assign(char.value, form)
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
    const { characterAPI, dramaAPI } = await import('~/composables/useApi')
    await characterAPI.generateImage(characterId, dramaId, {
      prompt: form.customPrompt || undefined,
      model: imageModel.value || undefined,
    })
    // 轮询等待结果
    imgPollRef.value = setInterval(async () => {
      try {
        const data: any = await dramaAPI.get(dramaId)
        const list = data?.characters || []
        const updated = list.find((c: any) => c.id === characterId)
        if (updated?.image_url || updated?.imageUrl) {
          clearInterval(imgPollRef.value!)
          imgPollRef.value = null
          char.value = updated
          imgGen.value = false
        }
      } catch (e) {
        console.warn('[Character] poll error:', e)
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

async function testVoice() {
  try {
    const { characterAPI } = await import('~/composables/useApi')
    await characterAPI.voiceSample(characterId, dramaId)
    alert('语音试听已提交，请稍后刷新查看')
  } catch (e: any) {
    errorMsg.value = e.message
  }
}

onUnmounted(() => {
  if (imgPollRef.value) clearInterval(imgPollRef.value)
  if (imgPollTimeoutRef.value) clearTimeout(imgPollTimeoutRef.value)
})
</script>

<style scoped>
.char-detail {
  min-height: 100vh;
  background: #f5f6fa;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
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
  color: #1a1a2e;
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
.back-btn:hover { background: rgba(60,90,180,0.06); border-color: rgba(60,90,180,0.25); }

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

/* 预览列 */
.preview-col { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 72px; align-self: start; }
.image-card {
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
  border: 1px solid rgba(100,120,180,0.12);
  aspect-ratio: 3/4;
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

/* 表单列 */
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
  color: #1a1a2e;
  margin: 0 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.form-section h3 svg { opacity: 0.45; flex-shrink: 0; }

.field-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.field-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 600px) {
  .field-grid-2, .field-grid-3 { grid-template-columns: 1fr; }
}
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
  color: #1a1a2e;
}
.input:focus { border-color: rgba(60,90,180,0.45); box-shadow: 0 0 0 3px rgba(60,90,180,0.08); background: #fff; }
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

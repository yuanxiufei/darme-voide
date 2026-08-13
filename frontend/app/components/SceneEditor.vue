<template>
  <Teleport to="body">
    <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
      <div class="modal-panel">
        <div class="modal-header">
          <h3>编辑场景：{{ scene?.location }}</h3>
          <button class="btn-close" @click="$emit('close')">&times;</button>
        </div>

        <!-- 图片预览 -->
        <div v-if="scene?.image_url || scene?.imageUrl" class="editor-preview">
          <img :src="'/' + (scene.image_url || scene.imageUrl)" alt="场景图预览" />
          <span class="preview-label">当前场景图</span>
        </div>

        <div class="modal-body">
          <section class="section">
            <h4>场景信息</h4>
            <div class="grid-2">
              <label class="field">
                <span>地点</span>
                <input v-model="form.location" class="input" />
              </label>
              <label class="field">
                <span>时间</span>
                <input v-model="form.time" class="input" placeholder="白天/夜晚/黄昏" />
              </label>
            </div>
            <label class="field">
              <span>描述</span>
              <textarea v-model="form.description" class="input" rows="2" placeholder="场景的详细描述..." />
            </label>
          </section>

          <section class="section">
            <h4>环境设定</h4>
            <div class="grid-3">
              <label class="field">
                <span>氛围</span>
                <BaseSelect v-model="form.atmosphere" :options="atmosphereOptions" placeholder="选择氛围" />
              </label>
              <label class="field">
                <span>光线</span>
                <BaseSelect v-model="form.lighting" :options="lightingOptions" placeholder="选择光线" />
              </label>
              <label class="field">
                <span>季节</span>
                <BaseSelect v-model="form.season" :options="seasonOptions" placeholder="选择季节" />
              </label>
            </div>
            <div class="grid-2">
              <label class="field">
                <span>天气</span>
                <BaseSelect v-model="form.weather" :options="weatherOptions" placeholder="选择天气" />
              </label>
              <label class="field">
                <span>风格</span>
                <BaseSelect v-model="form.style" :options="styleOptions" placeholder="选择风格" />
              </label>
            </div>
          </section>

          <section class="section">
            <h4>图片生成</h4>
            <label class="field">
              <span>生成 Prompt（留空使用自动生成）</span>
              <textarea v-model="form.prompt" class="input" rows="2" />
            </label>
            <label class="field">
              <span>自定义 Prompt 覆盖</span>
              <textarea v-model="form.customPrompt" class="input" rows="3" placeholder="覆盖自动生成，直接使用此 prompt..." />
            </label>
            <div class="regenerate-row">
              <ModelSelector v-model="imageModel" service-type="image" label="模型" />
              <button class="btn-primary" :disabled="imageGenerating" @click="regenerateImage">
                {{ imageGenerating ? '生成中...' : '重新生成图片' }}
              </button>
            </div>
            <div v-if="imageError" class="error-msg">{{ imageError }}</div>
          </section>
        </div>

        <div class="modal-footer">
          <div v-if="saveError" class="error-msg" style="margin-bottom: 8px;">{{ saveError }}</div>
          <button class="btn-cancel" @click="$emit('close')">取消</button>
          <button class="btn-save" :disabled="saving" @click="save">
            {{ saving ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, reactive, watch } from 'vue'

const props = defineProps<{
  visible: boolean
  scene: any
  episodeId: number
}>()

const emit = defineEmits<{
  close: []
  saved: []
}>()

const form = reactive<any>({})
const saving = ref(false)
const imageModel = ref('')
const imageGenerating = ref(false)
const imageError = ref('')
const saveError = ref('')  // ✅ 新增：保存错误独立显示

const atmosphereOptions = [
  { value: '', label: '--' },
  { value: '浪漫', label: '浪漫' },
  { value: '紧张', label: '紧张' },
  { value: '悲伤', label: '悲伤' },
  { value: '温馨', label: '温馨' },
  { value: '神秘', label: '神秘' },
  { value: '黑暗', label: '黑暗' },
  { value: '欢快', label: '欢快' },
  { value: '庄严', label: '庄严' },
  { value: '史诗', label: '史诗' },
]
const lightingOptions = [
  { value: '', label: '--' },
  { value: '自然光', label: '自然光' },
  { value: '逆光', label: '逆光' },
  { value: '柔光', label: '柔光' },
  { value: '昏暗', label: '昏暗' },
  { value: '金色', label: '金色' },
  { value: '月光', label: '月光' },
  { value: '烛光', label: '烛光' },
  { value: '霓虹', label: '霓虹' },
]
const seasonOptions = [
  { value: '', label: '--' },
  { value: '春', label: '春' },
  { value: '夏', label: '夏' },
  { value: '秋', label: '秋' },
  { value: '冬', label: '冬' },
]
const weatherOptions = [
  { value: '', label: '--' },
  { value: '晴', label: '晴' },
  { value: '多云', label: '多云' },
  { value: '雨', label: '雨' },
  { value: '雪', label: '雪' },
  { value: '雾', label: '雾' },
  { value: '风暴', label: '风暴' },
]
const styleOptions = [
  { value: '', label: '--' },
  { value: '古风', label: '古风' },
  { value: '现代', label: '现代' },
  { value: '科幻', label: '科幻' },
  { value: '奇幻', label: '奇幻' },
  { value: '写实', label: '写实' },
  { value: '二次元', label: '二次元' },
]

watch(() => props.visible && props.scene, (s) => {
  if (!s) return
  form.location = s.location
  form.time = s.time
  form.prompt = s.prompt || ''
  form.description = s.description || ''
  form.atmosphere = s.atmosphere || ''
  form.lighting = s.lighting || ''
  form.weather = s.weather || ''
  form.season = s.season || ''
  form.style = s.style || ''
  form.customPrompt = s.customPrompt || ''
}, { immediate: true })

async function save() {
  saving.value = true
  saveError.value = ''
  try {
    const { sceneAPI } = await import('../composables/useApi')
    await sceneAPI.update(props.scene.id, {
      location: form.location,
      time: form.time,
      prompt: form.prompt,
      description: form.description || null,
      atmosphere: form.atmosphere || null,
      lighting: form.lighting || null,
      weather: form.weather || null,
      season: form.season || null,
      style: form.style || null,
      customPrompt: form.customPrompt || null,
    })
    emit('saved')
    emit('close')
  } catch (err: any) {
    saveError.value = err.message || '保存失败'  // ✅ 使用独立的 saveError
  } finally {
    saving.value = false
  }
}

async function regenerateImage() {
  imageGenerating.value = true
  imageError.value = ''
  try {
    const { sceneAPI } = await import('../composables/useApi')
    await sceneAPI.generateImage(props.scene.id, props.episodeId, {
      prompt: form.customPrompt || form.prompt || undefined,
      model: imageModel.value || undefined,
    })
    emit('saved')
  } catch (err: any) {
    imageError.value = err.message
  } finally {
    imageGenerating.value = false
  }
}
</script>

<style scoped>
.editor-preview {
  position: relative;
  margin: -8px -16px 0;
  border-bottom: 1px solid rgba(100,120,180,0.12);
  background: rgba(0,0,0,0.03);
}
.editor-preview img {
  width: 100%;
  max-height: 200px;
  object-fit: contain;
  display: block;
}
.preview-label {
  position: absolute;
  bottom: 6px;
  right: 10px;
  font-size: 10px;
  color: #fff;
  background: rgba(0,0,0,0.5);
  padding: 2px 8px;
  border-radius: 4px;
}
</style>

<template>
  <Teleport to="body">
    <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
      <div class="modal-panel">
        <div class="modal-header">
          <h3>编辑角色：{{ character?.name }}</h3>
          <button class="btn-close" @click="$emit('close')">&times;</button>
        </div>

        <!-- 图片预览 -->
        <div v-if="character?.image_url || character?.imageUrl" class="editor-preview">
          <img :src="'/' + (character.image_url || character.imageUrl)" alt="角色形象预览" />
          <span class="preview-label">当前形象</span>
        </div>

        <div class="modal-body">
          <!-- 基本信息 -->
          <section class="section">
            <h4>基本信息</h4>
            <div class="grid-2">
              <label class="field">
                <span>名称</span>
                <input v-model="form.name" class="input" />
              </label>
              <label class="field">
                <span>角色定位</span>
                <input v-model="form.role" class="input" placeholder="主角/反派/配角" />
              </label>
            </div>
            <label class="field">
              <span>描述</span>
              <textarea v-model="form.description" class="input" rows="2" />
            </label>
          </section>

          <!-- 形象设定 -->
          <section class="section">
            <h4>形象设定</h4>
            <label class="field">
              <span>外貌</span>
              <textarea v-model="form.appearance" class="input" rows="2" placeholder="身高、体型、发色、眼睛..." />
            </label>
            <label class="field">
              <span>性格</span>
              <textarea v-model="form.personality" class="input" rows="2" placeholder="谨慎、果断、幽默..." />
            </label>
            <label class="field">
              <span>服装</span>
              <input v-model="form.clothing" class="input" placeholder="古装/现代/盔甲/长袍..." />
            </label>
            <label class="field">
              <span>武器</span>
              <input v-model="form.weapons" class="input" placeholder="长剑/匕首/法杖..." />
            </label>
          </section>

          <!-- 图片生成 -->
          <section class="section">
            <h4>图片生成提示词</h4>
            <label class="field">
              <span>自定义 Prompt（留空使用自动生成）</span>
              <textarea v-model="form.customPrompt" class="input" rows="3" placeholder="cinematic portrait of..." />
            </label>
            <div class="regenerate-row">
              <ModelSelector v-model="imageModel" service-type="image" label="模型" />
              <button class="btn-primary" :disabled="imageGenerating" @click="regenerateImage">
                {{ imageGenerating ? '生成中...' : '重新生成图片' }}
              </button>
            </div>
            <div v-if="imageError" class="error-msg">{{ imageError }}</div>
          </section>

          <!-- 声音配置 -->
          <section class="section">
            <h4>声音配置</h4>
            <div class="grid-2">
              <label class="field">
                <span>音色</span>
                <input v-model="form.voiceStyle" class="input" placeholder="温柔女声/沉稳男声..." />
              </label>
              <label class="field">
                <span>TTS 模型</span>
                <BaseSelect v-model="form.voiceModel" :options="voiceModelOptions" placeholder="选择模型" />
              </label>
            </div>
            <div class="grid-3">
              <label class="field">
                <span>语速</span>
                <BaseSelect v-model="form.voiceSpeed" :options="speedOptions" placeholder="正常" />
              </label>
              <label class="field">
                <span>情感</span>
                <BaseSelect v-model="form.voiceEmotion" :options="emotionOptions" placeholder="平静" />
              </label>
              <label class="field">
                <span>音调</span>
                <BaseSelect v-model="form.voicePitch" :options="pitchOptions" placeholder="标准" />
              </label>
            </div>
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
import { ref, reactive, watch, computed } from 'vue'

const props = defineProps<{
  visible: boolean
  character: any
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

const voiceModelOptions = [
  { value: 'speech-2.8-hd', label: 'MiniMax 2.8 HD' },
  { value: 'speech-2.6-hd', label: 'MiniMax 2.6 HD' },
  { value: 'speech-2.6', label: 'MiniMax 2.6' },
]
const speedOptions = [
  { value: '', label: '正常 (1.0x)' },
  { value: '0.7', label: '慢速 (0.7x)' },
  { value: '0.85', label: '稍慢 (0.85x)' },
  { value: '1.15', label: '稍快 (1.15x)' },
  { value: '1.3', label: '快速 (1.3x)' },
]
const emotionOptions = [
  { value: '', label: '平静/默认' },
  { value: 'happy', label: '开心' },
  { value: 'sad', label: '悲伤' },
  { value: 'angry', label: '愤怒' },
  { value: 'excited', label: '兴奋' },
  { value: 'calm', label: '舒缓' },
  { value: 'serious', label: '严肃' },
  { value: 'neutral', label: '中性' },
]
const pitchOptions = [
  { value: '', label: '标准 (0)' },
  { value: '3', label: '高 (+3)' },
  { value: '2', label: '稍高 (+2)' },
  { value: '-2', label: '稍低 (-2)' },
  { value: '-3', label: '低 (-3)' },
  { value: '-5', label: '很低 (-5)' },
]

// 回填表单
watch(() => props.visible && props.character, (c) => {
  if (!c) return
  form.name = c.name
  form.role = c.role
  form.description = c.description
  form.appearance = c.appearance
  form.personality = c.personality
  form.clothing = c.clothing || ''
  form.weapons = c.weapons || ''
  form.customPrompt = c.customPrompt || ''
  form.voiceStyle = c.voiceStyle
  form.voiceModel = c.voiceModel || 'speech-2.8-hd'
  form.voiceSpeed = c.voiceSpeed != null ? String(c.voiceSpeed) : ''
  form.voiceEmotion = c.voiceEmotion || ''
  form.voicePitch = c.voicePitch != null ? String(c.voicePitch) : ''
}, { immediate: true })

async function save() {
  saving.value = true
  saveError.value = ''
  try {
    const { characterAPI } = await import('../composables/useApi')
    await characterAPI.update(props.character.id, {
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
    const { characterAPI } = await import('../composables/useApi')
    await characterAPI.generateImage(props.character.id, props.episodeId, {
      prompt: form.customPrompt || undefined,
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

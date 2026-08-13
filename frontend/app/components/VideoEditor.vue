<template>
  <Teleport to="body">
    <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
      <div class="modal-panel">
        <div class="modal-header">
          <h3>视频生成参数 — 镜头 #{{ paddedIdx }}</h3>
          <button class="btn-close" @click="$emit('close')">&times;</button>
        </div>

        <!-- 图片预览 -->
        <div v-if="storyboard?.image_url || storyboard?.imageUrl" class="editor-preview">
          <img :src="'/' + (storyboard.image_url || storyboard.imageUrl)" alt="镜头帧图预览" />
          <span class="preview-label">镜头帧图</span>
        </div>

        <div class="modal-body">
          <section class="section">
            <h4>参考图</h4>
            <div class="grid-2">
              <label class="field">
                <span>参考模式</span>
                <BaseSelect v-model="form.referenceMode" :options="refModeOptions" placeholder="使用图片" />
              </label>
              <label class="field">
                <span>时长 (秒)</span>
                <input v-model.number="form.duration" type="number" min="3" max="60" class="input" />
              </label>
            </div>
            <div class="grid-2">
              <label class="field">
                <span>首帧图 URL</span>
                <input v-model="form.firstFrameUrl" class="input" placeholder="留空使用镜头图片" />
              </label>
              <label class="field">
                <span>尾帧图 URL</span>
                <input v-model="form.lastFrameUrl" class="input" />
              </label>
            </div>
          </section>

          <section class="section">
            <h4>提示词</h4>
            <label class="field">
              <span>视频 Prompt</span>
              <textarea
                v-model="form.prompt"
                class="input"
                rows="4"
                placeholder="描述这个镜头中应该发生的动作、运镜..."
              />
            </label>
            <label class="field">
              <span>关联角色</span>
              <div v-if="props.characters?.length" class="char-select-grid">
                <label
                  v-for="ch in props.characters"
                  :key="ch.id"
                  :class="['char-chip', selectedCharIds.includes(ch.id) && 'active']"
                  @click.prevent="toggleCharId(ch.id)"
                >
                  <span class="char-dot" :style="{ background: ch.color || '#6a8cff' }" />
                  {{ ch.name }}
                </label>
              </div>
              <div v-else class="dim" style="font-size:12px">当前集无可用角色（先在「角色」Tab 添加）</div>
            </label>
          </section>

          <section class="section">
            <h4>模型选择</h4>
            <ModelSelector v-model="form.model" service-type="video" label="视频生成模型" />
          </section>

          <div v-if="errorMsg" class="error-msg">{{ errorMsg }}</div>
        </div>

        <div class="modal-footer">
          <button class="btn-cancel" @click="$emit('close')">取消</button>
          <button class="btn-save" :disabled="regenerating" @click="regenerate">
            {{ regenerating ? '生成中...' : '重新生成' }}
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
  storyboard: any
  videoRecord?: any
  characters?: any[]   // 可用角色列表（用于关联角色多选）
  configId?: number    // AI 配置 ID（用于视频生成）
}>()

const emit = defineEmits<{
  close: []
  regenerated: []
}>()

const form = reactive<any>({})
const regenerating = ref(false)
const errorMsg = ref('')
const selectedCharIds = ref<number[]>([])  // 多选角色 ID

const paddedIdx = computed(() => {
  const idx = props.storyboard?.index ?? props.storyboard?.id ?? 1
  return String(idx).padStart(2, '0')
})

const refModeOptions = [
  { value: '', label: '使用图片' },
  { value: 'single', label: '单图参考' },
  { value: 'first', label: '首帧驱动' },
  { value: 'first_last', label: '首帧+尾帧' },
]

watch(() => props.visible && (props.storyboard || props.videoRecord), () => {
  const sb = props.storyboard
  const vr = props.videoRecord
  if (!sb) return
  form.model = vr?.model || ''
  form.referenceMode = vr?.referenceMode || vr?.reference_mode || 'first'
  form.duration = vr?.duration || sb.duration || 10
  form.prompt = vr?.prompt || sb.videoPrompt || sb.video_prompt || ''
  form.firstFrameUrl = vr?.firstFrameUrl || vr?.first_frame_url || ''
  form.lastFrameUrl = vr?.lastFrameUrl || vr?.last_frame_url || ''
  form.characterIds = vr?.characterIds || vr?.character_ids || ''
  // 初始化多选角色
  const existingIds = (vr?.characterIds || vr?.character_ids || '')
  selectedCharIds.value = typeof existingIds === 'string'
    ? existingIds.split(',').filter(Boolean).map(Number)
    : (Array.isArray(existingIds) ? existingIds : [])
  form.referenceImageUrls = vr?.referenceImageUrls || vr?.reference_image_urls || null
}, { immediate: true })

function toggleCharId(id: number) {
  const idx = selectedCharIds.value.indexOf(id)
  if (idx >= 0) selectedCharIds.value.splice(idx, 1)
  else selectedCharIds.value.push(id)
}

async function regenerate() {
  regenerating.value = true
  errorMsg.value = ''
  try {
    const { videoAPI } = await import('../composables/useApi')
    const charIdsPayload = selectedCharIds.value.length ? selectedCharIds.value : undefined
    // 如果存在已有 video record，走 regenerate；否则走 generate
    if (props.videoRecord?.id) {
      // 先更新参数
      await videoAPI.update(props.videoRecord.id, {
        prompt: form.prompt,
        model: form.model || undefined,
        reference_mode: form.referenceMode || undefined,
        duration: form.duration,
        character_ids: charIdsPayload,
        first_frame_url: form.firstFrameUrl || undefined,
        last_frame_url: form.lastFrameUrl || undefined,
      })
      await videoAPI.regenerate(props.videoRecord.id, {
        model: form.model || undefined,
        prompt: form.prompt,
        config_id: props.configId,  // ✅ 传递配置 ID
      })
    } else {
      await videoAPI.generate({
        storyboard_id: props.storyboard?.id,
        drama_id: props.storyboard?.dramaId,
        prompt: form.prompt,
        model: form.model || undefined,
        reference_mode: form.referenceMode || 'first',
        duration: form.duration,
        character_ids: charIdsPayload,
        first_frame_url: form.firstFrameUrl || undefined,
        last_frame_url: form.lastFrameUrl || undefined,
        config_id: props.configId,  // ✅ 传递配置 ID
      })
    }
    emit('regenerated')
    emit('close')
  } catch (err: any) {
    errorMsg.value = err.message
  } finally {
    regenerating.value = false
  }
}
</script>

<style scoped>
.char-select-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.char-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid rgba(100,120,180,0.18);
  background: rgba(255,255,255,0.6);
  cursor: pointer;
  font-size: 12px;
  transition: all .15s;
  user-select: none;
}
.char-chip:hover { border-color: rgba(80,110,200,0.35); }
.char-chip.active {
  background: rgba(60,90,180,0.1);
  border-color: rgba(60,90,180,0.4);
  font-weight: 600;
}
.char-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
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

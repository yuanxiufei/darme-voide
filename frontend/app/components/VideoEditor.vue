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
            <div class="grid-3">
              <label class="field">
                <span>参考模式</span>
                <BaseSelect v-model="form.referenceMode" :options="refModeOptions" placeholder="使用图片" />
              </label>
              <label class="field">
                <span>时长 (秒)</span>
                <input v-model.number="form.duration" type="number" :min="durationMin" :max="durationMax" class="input" />
                <span v-if="videoConstraint" class="field-hint">{{ videoConstraint.label }}</span>
              </label>
              <label class="field">
                <span>画幅比例</span>
                <div class="aspect-selector">
                  <button
                    v-for="opt in aspectRatioOptions"
                    :key="opt.value"
                    type="button"
                    :class="['aspect-btn', form.aspectRatio === opt.value && 'active']"
                    :title="`${opt.label} (${opt.desc})`"
                    @click="form.aspectRatio = opt.value"
                  >
                    <component :is="opt.icon" class="aspect-icon" />
                    <span class="aspect-label">{{ opt.label }}</span>
                    <span class="aspect-desc">{{ opt.desc }}</span>
                  </button>
                </div>
              </label>
            </div>
            <div class="grid-2">
              <label class="field">
                <span>首帧图 URL</span>
                <div class="input-row">
                  <input v-model="form.firstFrameUrl" class="input" placeholder="留空使用镜头图片" />
                  <button type="button" class="btn-upload" :disabled="uploading" @click="triggerUpload('first')">
                    {{ uploading && uploadTarget === 'first' ? '上传中…' : '上传' }}
                  </button>
                  <button
                    type="button"
                    class="btn-upload"
                    :disabled="regeneratingFrame === 'first_frame'"
                    title="AI 重新生成首帧图（使用下方「首帧画面内容」）"
                    @click="regenerateFrame('first_frame')"
                  >{{ regeneratingFrame === 'first_frame' ? '生成中…' : 'AI 重生成' }}</button>
                  <button
                    type="button"
                    class="btn-upload"
                    :disabled="!prevLastFrameUrl"
                    :title="prevLastFrameUrl ? '复用上一镜头尾帧作为本镜头首帧，保证镜头连贯' : '上一镜头暂无尾帧图'"
                    @click="reusePrevLastFrame"
                  >复用上镜尾帧</button>
                </div>
              </label>
              <label class="field">
                <span>尾帧图 URL</span>
                <div class="input-row">
                  <input v-model="form.lastFrameUrl" class="input" placeholder="留空使用首帧" />
                  <button type="button" class="btn-upload" :disabled="uploading" @click="triggerUpload('last')">
                    {{ uploading && uploadTarget === 'last' ? '上传中…' : '上传' }}
                  </button>
                  <button
                    type="button"
                    class="btn-upload"
                    :disabled="regeneratingFrame === 'last_frame'"
                    title="AI 重新生成尾帧图（使用下方「尾帧画面内容」）"
                    @click="regenerateFrame('last_frame')"
                  >{{ regeneratingFrame === 'last_frame' ? '生成中…' : 'AI 重生成' }}</button>
                  <button
                    type="button"
                    class="btn-upload"
                    :disabled="!nextFirstFrameUrl"
                    :title="nextFirstFrameUrl ? '复用下一镜头首帧作为本镜头尾帧，保证镜头连贯' : '下一镜头暂无首帧图'"
                    @click="reuseNextFirstFrame"
                  >复用下镜首帧</button>
                </div>
              </label>
            </div>
            <div class="grid-2">
              <label class="field">
                <span>首帧画面内容（AI 重生成用）</span>
                <textarea v-model="form.firstFramePrompt" class="input" rows="2" placeholder="描述首帧画面，留空自动生成" />
              </label>
              <label class="field">
                <span>尾帧画面内容（AI 重生成用）</span>
                <textarea v-model="form.lastFramePrompt" class="input" rows="2" placeholder="描述尾帧画面，留空自动生成" />
              </label>
            </div>
            <div class="grid-2">
              <label class="field">
                <span>过渡效果</span>
                <BaseSelect v-model="form.transitionType" :options="transitionOptions" placeholder="无过渡" />
              </label>
              <label class="field">
                <span>过渡时长 (秒)</span>
                <input v-model.number="form.transitionDuration" type="number" min="0" max="5" step="0.1" class="input" />
              </label>
            </div>
            <input ref="fileInput" type="file" accept="image/png,image/jpeg,image/webp,image/gif,video/mp4,video/quicktime,video/webm" class="hidden-input" @change="onPickFile" />
          </section>

          <section class="section">
            <h4>提示词</h4>
            <div v-if="failReason" class="fail-reason">
              <div class="fail-reason-title">上次生成未通过：</div>
              <div class="fail-reason-body">{{ failReason }}</div>
              <div class="fail-reason-hint">请人工修改上方 Prompt（弱化敏感/暴力/血腥表述），确认无误后再点「重新生成」。</div>
            </div>
            <label class="field">
              <span class="field-label">
                视频 Prompt
                <button class="field-ai-btn" :disabled="optimizing" @click.prevent="optimizePrompt">
                  <Sparkles :size="11" />
                  {{ optimizing ? '优化中…' : 'AI 优化' }}
                </button>
              </span>
              <textarea
                v-model="form.prompt"
                class="input"
                rows="4"
                placeholder="描述这个镜头中应该发生的动作、运镜..."
              />
            </label>
            <label class="field">
              <span>负面提示词（Negative Prompt）</span>
              <textarea
                v-model="form.negativePrompt"
                class="input"
                rows="2"
                placeholder="排除不需要的内容/风格/元素，如：text, watermark, low quality, motion blur..."
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
                  <span class="char-dot" :style="{ background: ch.color || '#14b8a6' }" />
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
import { toast } from 'vue-sonner'
import { Monitor, Smartphone, Square, Sparkles } from 'lucide-vue-next'

const props = defineProps<{
  visible: boolean
  storyboard: any
  videoRecord?: any
  characters?: any[]   // 可用角色列表（用于关联角色多选）
  configId?: number    // AI 配置 ID（用于视频生成）
  failReason?: string  // 上次生成失败/审核拦截原因，带入编辑器引导人工修改提示词
  neighbors?: { prev?: any; next?: any }  // 相邻镜头（用于一键复用首尾帧，保证镜头间视觉连贯）
}>()

const emit = defineEmits<{
  close: []
  regenerated: []
}>()

const form = reactive<any>({})
const regenerating = ref(false)
const regeneratingFrame = ref<'first_frame' | 'last_frame' | null>(null)
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

const aspectRatioOptions = [
  { value: '16:9', label: '横屏', desc: '16:9', icon: Monitor },
  { value: '9:16', label: '竖屏', desc: '9:16', icon: Smartphone },
  { value: '1:1', label: '方形', desc: '1:1', icon: Square },
]

const transitionOptions = [
  { value: 'cut', label: '无过渡（硬切）' },
  { value: 'fade', label: '淡入淡出' },
  { value: 'dissolve', label: '叠化' },
]

// —— 视频模型参数约束（对齐后端各 adapter 的真实 clamp 逻辑）——
// 后端 adapter 差异：volcengine-video 的 normalizeDuration 强制 clamp 到 [4,12]；
// 其余厂商（ali/vidu/minimax）时长自由，比例均支持 16:9/9:16/1:1。
const DURATION_CONSTRAINTS: Array<{ match: RegExp; range: [number, number]; label: string }> = [
  { match: /seedance|doubao|volc|ark/i, range: [4, 12], label: '豆包 Seedance 支持 4–12 秒' },
]

const videoConstraint = computed(() => {
  const m = (form.model || '').toLowerCase()
  for (const c of DURATION_CONSTRAINTS) {
    if (c.match.test(m)) return c
  }
  return null
})
const durationMin = computed(() => videoConstraint.value?.range[0] ?? 3)
const durationMax = computed(() => videoConstraint.value?.range[1] ?? 60)

// 切换模型时自动校验并修正时长，避免选到模型不支持的参数
watch(() => form.model, (newModel, oldModel) => {
  if (!oldModel) return  // 首次初始化不算切换
  const constraint = DURATION_CONSTRAINTS.find(c => c.match.test((newModel || '').toLowerCase()))
  if (!constraint) return
  const d = Number(form.duration)
  const [min, max] = constraint.range
  if (!Number.isFinite(d) || d < min || d > max) {
    const clamped = Math.min(max, Math.max(min, d || 5))
    form.duration = clamped
    toast.warning(`${constraint.label}，时长已自动调整为 ${clamped} 秒`)
  }
})

watch(() => props.visible && (props.storyboard || props.videoRecord), () => {
  const sb = props.storyboard
  const vr = props.videoRecord
  if (!sb) return
  form.model = vr?.model || ''
  form.referenceMode = vr?.referenceMode || vr?.reference_mode || 'first'
  form.duration = vr?.duration || sb.duration || 10
  form.aspectRatio = vr?.aspectRatio || vr?.aspect_ratio || '16:9'
  form.prompt = vr?.prompt || sb.videoPrompt || sb.video_prompt || ''
  form.negativePrompt = vr?.negativePrompt || vr?.negative_prompt || ''
  form.firstFrameUrl = vr?.firstFrameUrl || vr?.first_frame_url || ''
  form.lastFrameUrl = vr?.lastFrameUrl || vr?.last_frame_url || ''
  form.firstFramePrompt = sb.firstFramePrompt || sb.first_frame_prompt || ''
  form.lastFramePrompt = sb.lastFramePrompt || sb.last_frame_prompt || ''
  form.transitionType = sb.transitionType || sb.transition_type || 'cut'
  form.transitionDuration = sb.transitionDuration ?? sb.transition_duration ?? 0.5
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

// ===== 相邻镜头首尾帧一键复用（对齐 gcc KeyframeEditor 的 copyPrevious/copyNext，保证镜头间视觉连贯）=====
// 兼容后端 snake_case（first_frame_image）与前端 camelCase（firstFrameImage）两种字段形态
const getSbFirst = (s: any) => s?.first_frame_image || s?.firstFrameImage || s?.first_frame_url || s?.firstFrameUrl || null
const getSbLast = (s: any) => s?.last_frame_image || s?.lastFrameImage || s?.last_frame_url || s?.lastFrameUrl || null

const prevLastFrameUrl = computed(() => getSbLast(props.neighbors?.prev))
const nextFirstFrameUrl = computed(() => getSbFirst(props.neighbors?.next))

function reusePrevLastFrame() {
  const url = prevLastFrameUrl.value
  if (!url) return
  form.firstFrameUrl = url
  toast.success('已复用上一镜头尾帧作为首帧')
}

function reuseNextFirstFrame() {
  const url = nextFirstFrameUrl.value
  if (!url) return
  form.lastFrameUrl = url
  toast.success('已复用下一镜头首帧作为尾帧')
}

// ===== AI 优化提示词（对齐 gcc KeyframeEditor 的 AI 优化，用户主动触发，与「审核失败自动改写」无关）=====
const optimizing = ref(false)

async function optimizePrompt() {
  const sbId = props.storyboard?.id
  if (!sbId || optimizing.value) return
  optimizing.value = true
  try {
    const res = await fetch(`/api/v1/storyboards/${sbId}/optimize-prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentPrompt: form.prompt || '' }),
    })
    const data = await res.json()
    if (data.code !== 200) throw new Error(data.message || '优化失败')
    const optimized = data.data?.optimizedPrompt
    if (!optimized) throw new Error('未返回优化结果')
    form.prompt = optimized
    toast.success('提示词已优化')
  } catch (err: any) {
    toast.error(err?.message || '提示词优化失败')
  } finally {
    optimizing.value = false
  }
}

// ===== 本地上传首帧/尾帧图 =====
const fileInput = ref<HTMLInputElement | null>(null)
const uploadTarget = ref<'first' | 'last'>('first')
const uploading = ref(false)

function triggerUpload(target: 'first' | 'last') {
  uploadTarget.value = target
  fileInput.value?.click()
}

async function onPickFile(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploading.value = true
  errorMsg.value = ''
  try {
    const { uploadAPI, storyboardAPI } = await import('../composables/useApi')
    const isVideo = file.type.startsWith('video/')
    const frameType = uploadTarget.value === 'first' ? 'first_frame' : 'last_frame'
    if (isVideo) {
      // 视频素材：上传后由后端抽取首帧/尾帧并自动适配到对应镜头
      const sbId = props.storyboard?.id
      if (!sbId) throw new Error('缺少镜头 ID，无法适配视频素材')
      const { url } = await uploadAPI.video(file)
      const res: any = await storyboardAPI.setFrame(sbId, { frame_type: frameType, source_url: url })
      const frameUrl = res?.frame_url || url
      if (uploadTarget.value === 'first') form.firstFrameUrl = frameUrl
      else form.lastFrameUrl = frameUrl
      toast.success('视频素材已抽取' + (uploadTarget.value === 'first' ? '首帧' : '尾帧') + '并适配')
    } else {
      const { url } = await uploadAPI.image(file)
      if (uploadTarget.value === 'first') form.firstFrameUrl = url
      else form.lastFrameUrl = url
    }
  } catch (err: any) {
    errorMsg.value = err.message
  } finally {
    uploading.value = false
    input.value = ''  // 允许重复选择同一文件
  }
}

// ===== 首尾帧独立重新生成 =====
async function regenerateFrame(frameType: 'first_frame' | 'last_frame') {
  const sbId = props.storyboard?.id
  if (!sbId || regeneratingFrame.value) return
  regeneratingFrame.value = frameType
  errorMsg.value = ''
  try {
    const { storyboardAPI } = await import('../composables/useApi')
    const res: any = await storyboardAPI.regenerateFrame(sbId, {
      frame_type: frameType,
      prompt: frameType === 'first_frame' ? form.firstFramePrompt : form.lastFramePrompt,
      negative_prompt: form.negativePrompt || undefined,
    })
    const genId = res?.image_generation_id
    toast.success(frameType === 'first_frame' ? '首帧图已提交生成' : '尾帧图已提交生成')
    if (genId != null) pollFrameResult(genId, frameType)
  } catch (err: any) {
    errorMsg.value = err.message
    toast.error(err?.message || '生成失败')
  } finally {
    regeneratingFrame.value = null
  }
}

// 轮询图片生成结果，完成后自动回填首/尾帧图
function pollFrameResult(genId: number, frameType: 'first_frame' | 'last_frame') {
  const sbId = props.storyboard?.id
  if (!sbId) return
  const deadline = Date.now() + 90_000
  const timer = setInterval(async () => {
    try {
      const { imageAPI } = await import('../composables/useApi')
      const rows: any[] = (await imageAPI.list({ storyboard_id: sbId })) as any
      const rec = rows?.find((r) => r.id === genId)
      if (rec?.status === 'completed' && rec.localPath) {
        clearInterval(timer)
        if (frameType === 'first_frame') form.firstFrameUrl = rec.localPath
        else form.lastFrameUrl = rec.localPath
        toast.success(frameType === 'first_frame' ? '首帧图已更新' : '尾帧图已更新')
      } else if (rec?.status === 'failed') {
        clearInterval(timer)
        toast.error(rec.errorMsg || '生成失败')
      }
    } catch {}
    if (Date.now() > deadline) {
      clearInterval(timer)
      toast.warning('生成仍在进行，请稍后刷新查看')
    }
  }, 3000)
}

async function regenerate() {
  regenerating.value = true
  errorMsg.value = ''
  try {
    const { videoAPI, storyboardAPI } = await import('../composables/useApi')
    // 保存首尾帧画面内容与过渡参数到镜头记录
    if (props.storyboard?.id) {
      await storyboardAPI.update(props.storyboard.id, {
        first_frame_prompt: form.firstFramePrompt || undefined,
        last_frame_prompt: form.lastFramePrompt || undefined,
        transition_type: form.transitionType || undefined,
        transition_duration: form.transitionDuration ?? undefined,
      })
    }
    const charIdsPayload = selectedCharIds.value.length ? selectedCharIds.value : undefined
    // 如果存在已有 video record，走 regenerate；否则走 generate
    if (props.videoRecord?.id) {
      // 先更新参数
      await videoAPI.update(props.videoRecord.id, {
        prompt: form.prompt,
        negative_prompt: form.negativePrompt || undefined,
        model: form.model || undefined,
        reference_mode: form.referenceMode || undefined,
        duration: form.duration,
        aspect_ratio: form.aspectRatio || undefined,
        character_ids: charIdsPayload,
        first_frame_url: form.firstFrameUrl || undefined,
        last_frame_url: form.lastFrameUrl || undefined,
      })
      await videoAPI.regenerate(props.videoRecord.id, {
        model: form.model || undefined,
        prompt: form.prompt,
        negative_prompt: form.negativePrompt || undefined,
        config_id: props.configId,  // ✅ 传递配置 ID
      })
    } else {
      await videoAPI.generate({
        storyboard_id: props.storyboard?.id,
        drama_id: props.storyboard?.dramaId,
        prompt: form.prompt,
        negative_prompt: form.negativePrompt || undefined,
        model: form.model || undefined,
        reference_mode: form.referenceMode || 'first',
        duration: form.duration,
        aspect_ratio: form.aspectRatio || undefined,
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
/* ====== 弹窗基础布局（本组件 Teleport 到 body，需自带 overlay/panel 样式） ====== */
.modal-overlay {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center;
  backdrop-filter: blur(4px);
}
.modal-panel {
  background: var(--bg-0);
  border: 1px solid var(--border);
  border-radius: 12px;
  width: min(680px, 94vw);
  max-height: 85vh;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-elevated);
}
.modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}
.modal-header h3 { margin: 0; font-size: 15px; font-weight: 600; color: var(--text-0); }
.btn-close {
  background: none; border: none; color: var(--text-3);
  font-size: 22px; cursor: pointer; line-height: 1; padding: 0 4px;
}
.btn-close:hover { color: var(--text-0); }
.modal-body {
  padding: 16px 20px; overflow-y: auto;
  flex: 1;
}
.modal-body .section {
  margin-bottom: 20px;
}
.modal-body .section h4 {
  margin: 0 0 10px; font-size: 13px;
  color: var(--accent-text); text-transform: uppercase;
  letter-spacing: 0.5px;
}
.modal-footer {
  display: flex; gap: 8px; justify-content: flex-end;
  padding: 12px 20px;
  border-top: 1px solid var(--border);
}
.btn-save {
  padding: 8px 20px; border-radius: 6px;
  background: var(--accent); color: #fff;
  border: none; font-weight: 600; cursor: pointer;
}
.btn-save:hover { opacity: 0.9; }
.btn-save:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-cancel {
  padding: 8px 16px; border-radius: 6px;
  background: transparent; color: var(--text-2);
  border: 1px solid var(--border); cursor: pointer;
}
.btn-cancel:hover { color: var(--text-0); }
.field {
  display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px;
}
.field > span:not([class]) { font-size: 11px; color: var(--text-3); }
.input, .textarea {
  padding: 8px 10px; border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg-input);
  color: var(--text-0);
  font-size: 13px; font-family: inherit;
  resize: vertical;
}
.input:focus, .textarea:focus {
  outline: none; border-color: var(--accent);
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.error-msg {
  margin-top: 6px; font-size: 12px; color: var(--error);
}
.dim { color: var(--text-3); }
.field-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-1);
  display: flex;
  align-items: center;
  gap: 8px;
}
.field-ai-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 500;
  background: var(--accent-bg);
  color: var(--accent-text);
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s var(--ease-out);
}
.field-ai-btn:hover:not(:disabled) { border-color: var(--accent); }
.field-ai-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.field-hint {
  display: block;
  margin-top: 4px;
  font-size: 11px;
  line-height: 1.4;
  color: #b45309;
}
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
.char-chip:hover { border-color: rgba(13,148,136,0.35); }
.char-chip.active {
  background: rgba(13,148,136,0.1);
  border-color: rgba(13,148,136,0.4);
  font-weight: 600;
}
.char-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.aspect-selector { display: flex; gap: 6px; }
.aspect-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  padding: 8px 6px;
  border-radius: 10px;
  border: 1px solid rgba(100,120,180,0.18);
  background: rgba(255,255,255,0.6);
  cursor: pointer;
  color: #5a6b8f;
  transition: all .15s;
}
.aspect-btn:hover { border-color: rgba(13,148,136,0.35); }
.aspect-btn.active {
  background: rgba(13,148,136,0.1);
  border-color: rgba(13,148,136,0.45);
  color: #0f766e;
  font-weight: 600;
}
.aspect-icon { width: 16px; height: 16px; }
.aspect-label { font-size: 12px; line-height: 1.2; }
.aspect-desc { font-size: 10px; color: #98a6c0; line-height: 1; }
.aspect-btn.active .aspect-desc { color: #14b8a6; }
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
.input-row { display: flex; gap: 6px; align-items: center; }
.input-row .input { flex: 1; min-width: 0; }
.btn-upload {
  flex-shrink: 0;
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid rgba(100,120,180,0.25);
  background: rgba(255,255,255,0.6);
  color: #0f766e;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  transition: all .15s;
}
.btn-upload:hover:not(:disabled) { border-color: rgba(13,148,136,0.45); background: rgba(13,148,136,0.06); }
.btn-upload:disabled { opacity: 0.5; cursor: not-allowed; }
.hidden-input { display: none; }
.fail-reason {
  margin-bottom: 12px;
  padding: 10px 12px;
  border: 1px solid rgba(220, 90, 80, 0.3);
  border-radius: 10px;
  background: rgba(220, 90, 80, 0.08);
  color: #b63b33;
  font-size: 12px;
  line-height: 1.5;
}
.fail-reason-title { font-weight: 600; }
.fail-reason-body { margin-top: 2px; white-space: pre-wrap; word-break: break-word; }
.fail-reason-hint { margin-top: 6px; color: #8a6a22; font-size: 11px; }
</style>

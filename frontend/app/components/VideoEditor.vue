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

            <!-- 参考视频 + 参考图档位（H3 引擎的 ref_videos / ref_image_size） -->
            <div class="grid-2">
              <label class="field">
                <span>参考图档位</span>
                <BaseSelect v-model="form.referenceImageSize" :options="refImageSizeOptions" placeholder="自动" />
              </label>
              <label class="field">
                <span>参考视频 {{ (form.referenceVideoUrls?.length || 0) }}/{{ REF_VIDEO_LIMIT }}</span>
                <div class="input-row">
                  <button type="button" class="btn-upload" @click="addRefVideo">+ 添加</button>
                  <button type="button" class="btn-upload" :disabled="uploadingRefVideo" @click="triggerRefVideoUpload">
                    {{ uploadingRefVideo ? '上传中…' : '上传视频' }}
                  </button>
                  <button
                    type="button"
                    class="btn-upload"
                    :disabled="!prevShotVideoUrl"
                    :title="prevShotVideoUrl ? '把上一镜成片作为本镜参考视频，动作/造型连续' : '上一镜还没有成片'"
                    @click="usePrevShotVideo"
                  >用上镜成片</button>
                </div>
              </label>
            </div>
            <div v-if="form.referenceVideoUrls?.length" class="ref-video-list">
              <div v-for="(u, i) in form.referenceVideoUrls" :key="i" class="input-row">
                <input v-model="form.referenceVideoUrls[i]" class="input" placeholder="参考视频 URL（留空则忽略这条）" />
                <button type="button" class="btn-upload" @click="removeRefVideo(Number(i))">移除</button>
              </div>
            </div>
            <span class="ref-hint">
              参考视频走 H3 的 Ref2VA（声线/动作延续），最多 {{ REF_VIDEO_LIMIT }} 段；只填视频不填参考图时后端路由判为 R2V。
              档位仅 H3 引擎生效（match 快 / max 身份最准），其他厂商忽略。
            </span>
            <input ref="refVideoInput" type="file" accept="video/mp4,video/quicktime,video/webm" class="hidden-input" @change="onPickRefVideo" />

            <!-- 参考音频（H3 Ref2VA 的 ref_audios：出场角色的声线样本） -->
            <label class="field">
              <span>参考音频 {{ (form.referenceAudioUrls?.length || 0) }}/{{ REF_AUDIO_LIMIT }}</span>
              <div class="input-row">
                <button type="button" class="btn-upload" @click="addRefAudio">+ 添加</button>
                <button type="button" class="btn-upload" :disabled="uploadingRefAudio" @click="triggerRefAudioUpload">
                  {{ uploadingRefAudio ? '上传中…' : '上传音频' }}
                </button>
              </div>
            </label>
            <div v-if="form.referenceAudioUrls?.length" class="ref-video-list">
              <div v-for="(u, i) in form.referenceAudioUrls" :key="i" class="input-row">
                <input
                  v-model="form.referenceAudioUrls[i]"
                  class="input"
                  placeholder="参考音频 URL（声线样本，留空则忽略这条）"
                  @input="refAudioTouched = true"
                />
                <button type="button" class="btn-upload" @click="removeRefAudio(Number(i))">移除</button>
              </div>
            </div>
            <span class="ref-hint">
              参考音频走 H3 的 Ref2VA 声线克隆，最多 {{ REF_AUDIO_LIMIT }} 条（引擎侧槽位上限，miniMax/H3 线生效）。
              对话类镜头不填时后端会自动带出场角色的声线样本；手动清空则表示「本镜不要参考音频」，重跑也不会注回。
            </span>
            <input ref="refAudioInput" type="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" class="hidden-input" @change="onPickRefAudio" />
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
// 参考音频「用户动过没」标记 ✓✗：**不能**像参考视频那样一律发数组 —— 后端对**未给**该字段的
// 对话镜头会按 Ref2VA 自动带出场角色声线样本 ✓，一律发 `[]` 等于把这条自动注入**永久关掉** ✓✗
// （新镜头还没有 video 记录 ⇒ 列表本来就是空 ⇒ 用户没动也发空 ⇒ 声线样本再也不来）。
// 所以：未动 ⇒ 不发送；动过 ⇒ 连 `[]` 一起发（空数组 = 明确不要，后端 `_pick` 认这个语义 ✓）。
const refAudioTouched = ref(false)

const paddedIdx = computed(() => {
  const idx = props.storyboard?.index ?? props.storyboard?.id ?? 1
  return String(idx).padStart(2, '0')
})

// reference_mode 必须落在后端 adapter 认识的枚举内（none/single/first_last/multiple）。
// 历史上曾用 'first' 表示「首帧驱动」，但 adapter 三个分支都不识别它 → 请求实际不带任何
// 参考帧，退化成纯文生视频，故统一收敛为 'single'（其语义就是单帧起图 = 首帧驱动）。
const refModeOptions = [
  { value: '', label: '使用图片（自动）' },
  { value: 'single', label: '首帧驱动' },
  { value: 'first_last', label: '首帧+尾帧' },
  { value: 'multiple', label: '多参考图' },
]

// 归一历史记录里的非法 / 别名 reference_mode，避免回填后再次发出 adapter 不识别的模式
function normalizeRefMode(mode?: string | null): string {
  const v = String(mode || '').trim()
  if (v === 'first') return 'single'
  if (v === 'last') return 'first_last'
  return v
}

// ===== 参考素材：参考图档位 + 参考视频（H3 的 ref_image_size / ref_videos）=====
// ⚠️⚠️ 下面两条是**后端真源的镜像** ✗✗，后端**没有** HTTP 口把它们暴露给前端
//    （只有 H3 自己的 `/capabilities`），所以只能手抄一份 ✓：
//    · 档位枚举：核心节点 `comfy_extras/nodes_minimax_h3.py` 的
//      `io.Combo.Input("ref_image_size", options=["match", "max"], default="match")`；
//      H3 wrapper 见 `app/local_services/h3/server.py::_reference_image_size`（默认 match ✓，
//      非法值告警回落 ✓ —— 所以选错**不会报错**，只会静默变成 match ✗）。
//    · 参考视频上限：`app/local_services/h3/server.py::REF_VIDEO_LIMIT`（env
//      `H3_REF_VIDEO_LIMIT`，默认 3 ✓；adapter 侧同样截到 3 ✓，见 `video_adapters.py`
//      的 `body["reference_video"] = [...][:3]`）。
//    后端改枚举/改上限时**必须同步改这里**，否则界面能选、后端静默回落。
const REF_VIDEO_LIMIT = 3
// 参考音频上限同源：H3 主节点 `io.Autogrow.Input("ref_audios", …, min=0, max=3)`；
// `workflow.py::REF_AUDIO_PORT_PREFIX` 按**模板真有的槽位**读；`video_adapters.py` 的 minimax
// 同样截到 3（`body["reference_audio"][:3]`）。模板只有 1 格时多送的那几路会在引擎侧
// 告警丢弃（不会报错）—— 所以界面上限按协议上限写 3，不猜模板。
const REF_AUDIO_LIMIT = 3
const refImageSizeOptions = [
  { value: '', label: '自动（引擎默认 match）' },
  { value: 'match', label: 'match — 匹配首帧尺寸（快）' },
  { value: 'max', label: 'max — 保最大清晰度（慢，身份最准）' },
]

/**
 * 后端 JSON 列（TEXT）→ 数组 ✓✗。
 *
 * ⚠️⚠️ 这一步是**必须**的：`video_generations.reference_*_urls` 是 TEXT 列、存的是
 * JSON **字符串**，而 `row_to_camel` **不解析**它 ✗ ⇒ 拿到的 `vr.referenceImageUrls`
 * 是 `'["a","b"]'` 而不是数组 ✓✗。修前这里直接赋给 form、下游用 `Array.isArray()`
 * 判断 ⇒ **恒 false** ⇒「多参考图模式下回传已有参考图」从来**没发出去过** ✓✗
 * （界面上看着有、提交时静默变 undefined、后端改用自动注入 —— 三方都不报错）。
 * 四种形态都能吃：已是数组 / JSON 串 / 单条裸 URL（老数据）/ 空。
 */
function parseUrlList(raw: any): string[] {
  if (!raw) return []
  if (Array.isArray(raw)) return raw.filter(Boolean).map(String)
  if (typeof raw !== 'string') return []
  const s = raw.trim()
  if (!s) return []
  if (s.startsWith('[')) {
    try {
      const parsed = JSON.parse(s)
      if (Array.isArray(parsed)) return parsed.filter(Boolean).map(String)
    } catch { /* 坏 JSON 当单条裸 URL 处理（比整条丢掉更接近原意） */ }
  }
  return [s]
}

/**
 * 角色 ID 列表的回填解析 ✓✗（与 :func:`parseUrlList` **同病根**，⑯ 只治了参考图那一条）：
 * `video_generations.character_ids` 同样是 TEXT 列存 JSON **字符串**（`'[1,2]'`），原来这里
 * `existingIds.split(',')` 拿到的就是 `['[1', '2]']` ⇒ `map(Number)` ⇒ **`[NaN, NaN]`** ✓✗
 * ⇒ 角色多选**回填全空**（勾选一个都没有），且提交时 `[NaN,NaN]` 经 JSON 变 `[null,null]` ✓✗
 * （库里落两个 null，谁都不报错 ✓）。
 * 五种形态都能吃：数组 / JSON 串 / 逗号串（老数据）/ 单个数字 / 空。
 */
function parseIdList(raw: any): number[] {
  const toIds = (arr: any[]): number[] =>
    arr.map((v) => Number(v)).filter((n) => Number.isFinite(n) && n > 0)
  if (raw === null || raw === undefined || raw === '') return []
  if (Array.isArray(raw)) return toIds(raw)
  if (typeof raw === 'number') return toIds([raw])
  if (typeof raw !== 'string') return []
  const s = raw.trim()
  if (!s) return []
  if (s.startsWith('[')) {
    try {
      const parsed = JSON.parse(s)
      if (Array.isArray(parsed)) return toIds(parsed)
    } catch { /* 坏 JSON 当逗号串处理 */ }
  }
  return toIds(s.split(',').map((x) => x.trim().replace(/^\[|\]$/g, '')))
}

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
  // 空值表示「自动」：交给后端按分镜资产（首帧/尾帧/参考图）决策路线
  form.referenceMode = normalizeRefMode(vr?.referenceMode || vr?.reference_mode)
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
  // 初始化多选角色 —— ⚠️ **必须**过 parseIdList（后端给的是 JSON 字符串，见函数注释 ✗）
  selectedCharIds.value = parseIdList(vr?.characterIds || vr?.character_ids)
  // ⚠️ 必须过 parseUrlList（后端给的是 JSON **字符串**，见函数注释）—— 否则下游
  //    `Array.isArray()` 恒 false ⇒ 多参考图回传**静默失效** ✗
  form.referenceImageUrls = parseUrlList(vr?.referenceImageUrls || vr?.reference_image_urls)
  // 参考**视频**（H3 ref_videos，≤3 路）与参考图档位（match/max）—— 同为 JSON 串/标量
  form.referenceVideoUrls = parseUrlList(vr?.referenceVideoUrls || vr?.reference_video_urls)
  form.referenceImageSize = vr?.referenceImageSize || vr?.reference_image_size || ''
  // 参考**音频**（H3 ref_audios，声线样本）—— 后端**会**给对话镜头自动注入，所以「空」有
  // 两种含义：用户没动过（沿用后端自动收集）／用户主动删光（本镜不要）。用 touched 区分 ✗
  form.referenceAudioUrls = parseUrlList(vr?.referenceAudioUrls || vr?.reference_audio_urls)
  refAudioTouched.value = false
}, { immediate: true })

function toggleCharId(id: number) {
  const idx = selectedCharIds.value.indexOf(id)
  if (idx >= 0) selectedCharIds.value.splice(idx, 1)
  else selectedCharIds.value.push(id)
}

// ===== 相邻镜头首尾帧一键复用（对齐 gcc KeyframeEditor 的 copyPrevious/copyNext，保证镜头间视觉连贯）=====
// 兼容后端 snake_case（first_frame_image）与前端 camelCase（firstFrameImage）两种字段形态
const getSbFirst = (s: any) => s?.first_frame_image || s?.firstFrameImage || s?.first_frame_url || s?.firstFrameUrl || null
// 取上一镜尾帧做顺接参考时，优先真实尾帧（tail_frame_image = 视频实际末帧，像素级连续），设计尾帧兜底
const getSbLast = (s: any) => s?.tail_frame_image || s?.tailFrameImage || s?.last_frame_image || s?.lastFrameImage || s?.last_frame_url || s?.lastFrameUrl || null

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

// ===== 参考视频（H3 Ref2VA 的 ref_videos）=====
const prevShotVideoUrl = computed(
  () => props.neighbors?.prev?.video_url || props.neighbors?.prev?.videoUrl || null,
)

function ensureRefVideoList() {
  if (!Array.isArray(form.referenceVideoUrls)) form.referenceVideoUrls = []
  return form.referenceVideoUrls as string[]
}

function refVideoRoom(): boolean {
  if (ensureRefVideoList().length < REF_VIDEO_LIMIT) return true
  toast.warning(`参考视频最多 ${REF_VIDEO_LIMIT} 段（引擎侧上限）`)
  return false
}

function addRefVideo() {
  if (!refVideoRoom()) return
  ensureRefVideoList().push('')
}

function removeRefVideo(index: number) {
  if (Array.isArray(form.referenceVideoUrls)) form.referenceVideoUrls.splice(index, 1)
}

function usePrevShotVideo() {
  const url = prevShotVideoUrl.value
  if (!url) return
  const list = ensureRefVideoList()
  if (list.includes(url)) {
    toast.info('上一镜成片已经在参考视频里了')
    return
  }
  if (list.length >= REF_VIDEO_LIMIT) {
    toast.warning(`参考视频最多 ${REF_VIDEO_LIMIT} 段（引擎侧上限）`)
    return
  }
  list.push(url)
  toast.success('已把上一镜成片加入参考视频（动作/造型续接）')
}

// 上传参考视频：**不走**抽帧逻辑（那是首/尾帧的那条路，会 setFrame 改分镜），只取 URL 入列
const refVideoInput = ref<HTMLInputElement | null>(null)
const uploadingRefVideo = ref(false)

function triggerRefVideoUpload() {
  if (!refVideoRoom()) return
  refVideoInput.value?.click()
}

async function onPickRefVideo(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploadingRefVideo.value = true
  errorMsg.value = ''
  try {
    const { uploadAPI } = await import('../composables/useApi')
    const { url } = await uploadAPI.video(file)
    const list = ensureRefVideoList()
    // 先清掉「+ 添加」留下的空占位，再插真 URL（否则空串会占掉一个名额）
    const blank = list.indexOf('')
    if (blank >= 0) list.splice(blank, 1)
    if (list.length >= REF_VIDEO_LIMIT) throw new Error(`参考视频最多 ${REF_VIDEO_LIMIT} 段`)
    list.push(url)
    toast.success('参考视频已上传')
  } catch (err: any) {
    errorMsg.value = err.message
  } finally {
    uploadingRefVideo.value = false
    input.value = ''
  }
}

// ===== 参考音频（H3 Ref2VA 的 ref_audios）=====
function ensureRefAudioList() {
  if (!Array.isArray(form.referenceAudioUrls)) form.referenceAudioUrls = []
  return form.referenceAudioUrls as string[]
}

function refAudioRoom(): boolean {
  if (ensureRefAudioList().length < REF_AUDIO_LIMIT) return true
  toast.warning(`参考音频最多 ${REF_AUDIO_LIMIT} 条（引擎侧上限）`)
  return false
}

function addRefAudio() {
  if (!refAudioRoom()) return
  refAudioTouched.value = true
  ensureRefAudioList().push('')
}

function removeRefAudio(index: number) {
  if (!Array.isArray(form.referenceAudioUrls)) return
  form.referenceAudioUrls.splice(index, 1)
  // 删到空也照样标记「动过」⇒ 提交发 `[]` ⇒ 后端清空（否则会被当成没给、旧样本又注回来）
  refAudioTouched.value = true
}

const refAudioInput = ref<HTMLInputElement | null>(null)
const uploadingRefAudio = ref(false)

function triggerRefAudioUpload() {
  if (!refAudioRoom()) return
  refAudioInput.value?.click()
}

async function onPickRefAudio(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploadingRefAudio.value = true
  errorMsg.value = ''
  try {
    const { uploadAPI } = await import('../composables/useApi')
    const { url } = await uploadAPI.audio(file)
    const list = ensureRefAudioList()
    const blank = list.indexOf('')
    if (blank >= 0) list.splice(blank, 1)
    if (list.length >= REF_AUDIO_LIMIT) throw new Error(`参考音频最多 ${REF_AUDIO_LIMIT} 条`)
    list.push(url)
    refAudioTouched.value = true
    toast.success('参考音频已上传')
  } catch (err: any) {
    errorMsg.value = err.message
  } finally {
    uploadingRefAudio.value = false
    input.value = ''
  }
}

// ===== 提交载荷：所见即所存 =====
// 空数组**照发**（后端 `_json_text([])` ⇒ NULL ✓）—— 用户把参考视频删光就是「清空」的意思，
// 发 undefined 会让服务端保留旧值 ⇒ 用户以为删了、重跑还是拿旧素材。
const refVideoPayload = computed(() =>
  parseUrlList(form.referenceVideoUrls).filter((u) => u.trim()),
)
// 多参考图模式下才回传（其余模式由后端按分镜自动注入角色/场景/道具图）
const refImagePayload = computed(() =>
  form.referenceMode === 'multiple' && parseUrlList(form.referenceImageUrls).length
    ? parseUrlList(form.referenceImageUrls)
    : undefined,
)
const refImageSizePayload = computed(() => form.referenceImageSize || undefined)
// 参考音频：**未动过就不发**（见 refAudioTouched 的注释 —— 后端对话镜头会自动注入声线样本）；
// 动过才发，空数组也发（= 明确清空）。
const refAudioPayload = computed(() =>
  refAudioTouched.value ? parseUrlList(form.referenceAudioUrls).filter((u) => u.trim()) : undefined,
)

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
    // ⚠️ 过滤非法值再发（回填 bug 曾让这里带出 `[NaN, NaN]` ⇒ JSON 里变 `[null, null]`）
    const validCharIds = selectedCharIds.value.filter((n) => Number.isFinite(n) && n > 0)
    const charIdsPayload = validCharIds.length ? validCharIds : undefined
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
        // ⚠️ 参考素材**必须**也放进 update ✗✗：后端 PUT 白名单原来没有这四件 ✓✗ ⇒
        //    「更新已有记录」这条路上改参考图/参考视频/档位 = 静默丢弃（本轮已补白名单 ✓）。
        reference_image_urls: refImagePayload.value,
        reference_video_urls: refVideoPayload.value,
        // 参考音频：动过才发（未动 = 不碰库里的旧值；动过且为空 = 清空 ⇒ NULL）
        reference_audio_urls: refAudioPayload.value,
        // ⚠️ update 这条**发空串**而不是 undefined ✗：选「自动」= 用户要**清掉**档位 ✓，
        //    发 undefined 时后端 `key not in body` ⇒ 保留旧值 ✓✗（选了自动，重跑还是 max）。
        //    后端 `_json_text("")` ⇒ NULL ✓，引擎侧回落默认 match ✓。
        reference_image_size: String(form.referenceImageSize ?? ''),
      })
      await videoAPI.regenerate(props.videoRecord.id, {
        model: form.model || undefined,
        prompt: form.prompt,
        negative_prompt: form.negativePrompt || undefined,
        // regenerate 只给「覆盖项」，未给的沿用原记录 ✓；但参考素材**显式发**更稳 ✗：
        // 上面那次 update 若因任何原因没生效，这里就是最后一道（后端同样认这两个 key ✓）。
        reference_video_urls: refVideoPayload.value,
        reference_audio_urls: refAudioPayload.value,
        // 角色关联 / 参考图同属「未给 ⇒ 后端回落原记录」：给了就以本次为准 ✓
        character_ids: charIdsPayload,
        reference_image_urls: refImagePayload.value,
        reference_image_size: refImageSizePayload.value,
        config_id: props.configId,  // ✅ 传递配置 ID
      })
    } else {
      await videoAPI.generate({
        storyboard_id: props.storyboard?.id,
        drama_id: props.storyboard?.dramaId,
        prompt: form.prompt,
        negative_prompt: form.negativePrompt || undefined,
        model: form.model || undefined,
        // 空值 = 自动：交给后端按分镜资产（首帧/尾帧/参考图）决策路线。
        // 此处曾兜底成 'first'，而 adapter 不识别该值 → 请求不带任何参考帧。
        reference_mode: form.referenceMode || undefined,
        duration: form.duration,
        aspect_ratio: form.aspectRatio || undefined,
        character_ids: charIdsPayload,
        first_frame_url: form.firstFrameUrl || undefined,
        last_frame_url: form.lastFrameUrl || undefined,
        // 多参考图模式回传已有参考图；其余模式留空，由后端按分镜自动注入角色/场景/道具图
        reference_image_urls: refImagePayload.value,
        // 参考视频（H3 Ref2VA）：≤3 段；只给视频不给图时后端路由判 R2V/single ✓
        reference_video_urls: refVideoPayload.value,
        // 参考音频（H3 Ref2VA）：未动过不发（留给后端按对话镜头自动注入声线样本）
        reference_audio_urls: refAudioPayload.value,
        // 参考图档位：match（快）/ max（身份最准）；空=引擎默认
        reference_image_size: refImageSizePayload.value,
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
.ref-video-list { display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px; }
.ref-hint {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  line-height: 1.5;
  color: var(--text-3);
}
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

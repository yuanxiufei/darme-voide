<script setup lang="ts">
/**
 * 扩图对话框（移植自上游 `canvas-node-outpaint-dialog.tsx` 的语义，MIT；自研重写 ✓）。
 *
 * ## 为什么值得搬
 *
 * 分镜首帧是 16:9，人物定妆板/竖屏短剧是 9:16，同一张图想换画幅时：
 * 旧办法只有「重新生成」✗ —— 而扩图是**保住主体、只补四周**✓，
 * 后端 `image-qwen21` 的 `outpaint` 变体（`ImagePadForOutpaint` + `ImageScale`）早就绪 ✓，
 * 缺的只是「补多少」这一个确认框 ✓。
 *
 * ## 三条口径
 *
 * 1. **撑大尺寸 = 缩回尺寸** ✓：两个口子不一致会导致整体拉伸变形，见 `utils/canvasOutpaint.ts` 头注。
 * 2. **改数不静默** ✓：超限被压、被吸附 32 都会在下面列出来 —— 用户改 3000 却得到 2080 时必须看见原因。
 * 3. **提示词必填** ✓：只扩画布不给 prompt，模型会把新区域画成纯色/噪声带 ✗。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  OUTPAINT_FEATHERING,
  OUTPAINT_MEGAPIXELS,
  clampPaddings,
  outpaintPlan,
  ratioPaddings,
  type Paddings,
} from '../../utils/canvasOutpaint'

const props = defineProps<{ src: string }>()
const emit = defineEmits<{
  (
    e: 'confirm',
    payload: {
      paddings: Paddings
      width: number
      height: number
      feathering: number
      megapixels: number
      prompt: string
    },
  ): void
  (e: 'close'): void
}>()

const naturalSize = ref<{ width: number; height: number } | null>(null)
const paddings = ref<Paddings>({ left: 0, top: 0, right: 0, bottom: 0 })
const feathering = ref(OUTPAINT_FEATHERING)
const megapixels = ref(OUTPAINT_MEGAPIXELS)
const prompt = ref('')
const error = ref('')
const activeRatio = ref('自由')

const RATIOS = [
  { label: '自由', value: 0 },
  { label: '1:1', value: 1 },
  { label: '4:3', value: 4 / 3 },
  { label: '3:4', value: 3 / 4 },
  { label: '16:9', value: 16 / 9 },
  { label: '9:16', value: 9 / 16 },
]

const plan = computed(() => {
  const natural = naturalSize.value
  if (!natural) return null
  return outpaintPlan(natural, clampPaddings(paddings.value), {
    feathering: feathering.value,
    megapixels: megapixels.value,
  })
})

/** 预览框里原图占的比例（可视化"补在哪一侧" ✓ 比读四个数字快得多） */
const preview = computed(() => {
  const natural = naturalSize.value
  const current = plan.value
  if (!natural || !current) return null
  return {
    '--frame-ratio': `${current.width} / ${current.height}`,
    left: `${(current.paddings.left / current.width) * 100}%`,
    top: `${(current.paddings.top / current.height) * 100}%`,
    width: `${(natural.width / current.width) * 100}%`,
    height: `${(natural.height / current.height) * 100}%`,
  }
})

const changed = computed(() => {
  const current = plan.value
  const natural = naturalSize.value
  if (!current || !natural) return false
  return current.width !== natural.width || current.height !== natural.height
})

function onProbeLoad(ev: Event) {
  const image = ev.target as HTMLImageElement
  naturalSize.value = { width: image.naturalWidth, height: image.naturalHeight }
}

function applyRatio(label: string, value: number) {
  activeRatio.value = label
  const natural = naturalSize.value
  if (!natural) return
  paddings.value = value > 0 ? ratioPaddings(natural, value) : { left: 0, top: 0, right: 0, bottom: 0 }
}

function setPadding(key: keyof Paddings, value: string) {
  const num = Math.max(0, Math.round(Number(value) || 0))
  paddings.value = { ...paddings.value, [key]: num }
  // 手动改边 = 不再是某个固定比例（否则高亮会骗人 ✓）
  activeRatio.value = '自由'
}

function confirm() {
  const current = plan.value
  if (!current) return
  if (!prompt.value.trim()) {
    error.value = '请写新扩出来的区域要画什么（否则模型会补一圈噪声 ✗）'
    return
  }
  if (!changed.value) {
    error.value = '四边都没有留白，扩出来还是原尺寸'
    return
  }
  emit('confirm', {
    paddings: current.paddings,
    width: current.width,
    height: current.height,
    feathering: current.feathering,
    megapixels: current.megapixels,
    prompt: prompt.value.trim(),
  })
}

function onKey(ev: KeyboardEvent) {
  if (ev.key === 'Escape') emit('close')
  if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) confirm()
}

onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="out-overlay" @click.self="emit('close')">
    <div class="out-panel">
      <div class="out-tools">
        <button
          v-for="item in RATIOS"
          :key="item.label"
          class="out-ratio"
          :class="{ active: activeRatio === item.label }"
          @click="applyRatio(item.label, item.value)"
        >
          {{ item.label }}
        </button>
        <span class="out-hint">留白就是模型要补画的部分</span>
      </div>

      <div class="out-body">
        <div class="out-stage">
          <!-- 量尺寸的那张图必须**恒在 DOM 里** ✓：预览框本身要等 naturalSize 才渲染，
               把 @load 挂在框内会死锁（框不出现 ⇒ 图不加载 ⇒ 尺寸永远是 null ✗） -->
          <img :src="props.src" class="out-probe" alt="" @load="onProbeLoad" />
          <div v-if="preview" class="out-frame" :style="preview">
            <img :src="props.src" alt="" class="out-image" draggable="false" />
            <span class="out-tag">原图</span>
          </div>
          <div v-else class="out-loading">加载中…</div>
        </div>

        <div class="out-form">
          <div class="out-grid">
            <label>左 <input :value="paddings.left" type="number" min="0" step="32" @input="setPadding('left', ($event.target as HTMLInputElement).value)" /></label>
            <label>上 <input :value="paddings.top" type="number" min="0" step="32" @input="setPadding('top', ($event.target as HTMLInputElement).value)" /></label>
            <label>右 <input :value="paddings.right" type="number" min="0" step="32" @input="setPadding('right', ($event.target as HTMLInputElement).value)" /></label>
            <label>下 <input :value="paddings.bottom" type="number" min="0" step="32" @input="setPadding('bottom', ($event.target as HTMLInputElement).value)" /></label>
          </div>

          <div class="out-advanced">
            <label>羽化 <input v-model.number="feathering" type="number" min="0" max="128" step="8" /></label>
            <label>像素预算 <input v-model.number="megapixels" type="number" min="0.1" max="8" step="0.1" /> MP</label>
          </div>

          <div v-if="plan" class="out-size">
            输出 {{ plan.width }} × {{ plan.height }}
            <span v-if="naturalSize" class="out-from">（原图 {{ naturalSize.width }} × {{ naturalSize.height }}）</span>
          </div>

          <ul v-if="plan?.warnings.length" class="out-warn">
            <li v-for="line in plan.warnings" :key="line">{{ line }}</li>
          </ul>

          <textarea
            v-model="prompt"
            rows="3"
            class="out-prompt"
            placeholder="新扩出来的区域画什么（例如：延续街道与路灯、房间墙面与踢脚线）"
          ></textarea>
        </div>
      </div>

      <div v-if="error" class="out-error">{{ error }}</div>

      <div class="out-actions">
        <button class="out-btn ghost" @click="emit('close')">取消（Esc）</button>
        <button class="out-btn" :disabled="!plan || !changed" @click="confirm">建扩图卡（原图已连线）</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.out-overlay {
  position: fixed;
  inset: 0;
  z-index: 62;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.78);
}
.out-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: min(1020px, 95vw);
  max-height: 92vh;
  padding: 14px;
  border: 1px solid #2a2f3a;
  border-radius: 10px;
  background: #171a21;
  box-shadow: 0 18px 46px rgba(0, 0, 0, 0.5);
  overflow: auto;
}
.out-tools {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.out-ratio {
  padding: 3px 9px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #22262f;
  color: #d7dbe3;
  font-size: 12px;
  cursor: pointer;
}
.out-ratio.active {
  border-color: #2dd4bf;
  color: #2dd4bf;
}
.out-hint {
  margin-left: auto;
  color: #8b93a2;
  font-size: 12px;
}
.out-body {
  display: flex;
  gap: 14px;
  align-items: flex-start;
}
.out-stage {
  flex: 0 1 auto;
  display: flex;
  justify-content: center;
  align-items: center;
  min-width: 240px;
}
.out-frame {
  position: relative;
  aspect-ratio: var(--frame-ratio);
  height: min(52vh, 460px);
  border: 1px dashed #3b4351;
  border-radius: 6px;
  background: repeating-linear-gradient(45deg, #14171d, #14171d 8px, #191d25 8px, #191d25 16px);
}
.out-probe {
  display: none;
}
.out-image {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border-radius: 3px;
  object-fit: cover;
}
.out-tag {
  position: absolute;
  left: 4px;
  top: 4px;
  padding: 1px 5px;
  border-radius: 3px;
  background: rgba(11, 13, 18, 0.7);
  color: #9aa3b2;
  font-size: 11px;
}
.out-loading {
  color: #8b93a2;
  font-size: 12px;
}
.out-form {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 260px;
}
.out-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
}
.out-grid label,
.out-advanced label {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #8b93a2;
  font-size: 12px;
}
.out-grid input,
.out-advanced input {
  width: 100%;
  padding: 4px 6px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #0f1218;
  color: #e6e9ef;
  font-size: 12px;
}
.out-advanced {
  display: flex;
  gap: 12px;
}
.out-size {
  color: #d7dbe3;
  font-size: 12px;
}
.out-from {
  color: #8b93a2;
}
.out-warn {
  margin: 0;
  padding-left: 16px;
  color: #fbbf24;
  font-size: 11px;
}
.out-prompt {
  width: 100%;
  padding: 7px 9px;
  border: 1px solid #2f3541;
  border-radius: 6px;
  background: #0f1218;
  color: #e6e9ef;
  font-size: 12px;
  resize: vertical;
}
.out-error {
  color: #f87171;
  font-size: 12px;
}
.out-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.out-btn {
  padding: 6px 12px;
  border: 1px solid #2f3541;
  border-radius: 5px;
  background: #2dd4bf;
  color: #0b0d12;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.out-btn:disabled {
  background: #22262f;
  color: #6b7280;
  cursor: default;
}
.out-btn.ghost {
  background: #22262f;
  color: #d7dbe3;
  font-weight: 400;
}
</style>

<template>
  <div class="cg-panel">
    <!-- 标题行：折叠切换 + 激活徽标 + 重置 -->
    <div class="cg-head" @click="expanded = !expanded">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" :style="{ transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform .2s' }"><path d="M9 18l6-6-6-6"/></svg>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20" opacity="0"/><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3.5"/><circle cx="20" cy="5" r="2"/><circle cx="5" cy="19" r="2"/></svg>
      <span class="cg-title">校色调整</span>
      <span v-if="active" class="cg-badge">已启用</span>
      <span class="cg-spacer" />
      <button class="cg-reset" :disabled="!active" @click.stop="reset">重置</button>
    </div>

    <!-- 展开：校色参数 + 预设 -->
    <div v-if="expanded" class="cg-body" @click.stop>
      <div v-for="g in groups" :key="g.title" class="cg-group">
        <div class="cg-group-title">{{ g.title }}</div>
        <label v-for="s in g.sliders" :key="s.path" class="cg-slider">
          <span class="cg-slider-label">{{ s.label }}</span>
          <input
            type="range"
            class="cg-range"
            :min="s.min"
            :max="s.max"
            :step="s.step"
            :value="getPath(local, s.path)"
            @input="onSlider(s, $event)"
          />
          <span class="cg-slider-val">{{ getPath(local, s.path) }}</span>
        </label>
      </div>

      <!-- 预设：加载 / 保存 / 删除 -->
      <div class="cg-presets">
        <div class="cg-group-title">校色预设</div>
        <div class="cg-preset-row">
          <select v-model="selectedPresetId" class="cg-select">
            <option :value="null" disabled>选择预设...</option>
            <option v-for="p in presets" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
          <button class="cg-btn" :disabled="selectedPresetId == null" @click="applyPreset">应用</button>
          <button class="cg-btn cg-btn-danger" :disabled="selectedPresetId == null" @click="deletePreset">删除</button>
        </div>
        <div class="cg-preset-row">
          <input v-model="presetName" class="cg-input" placeholder="预设名称（保存当前参数）" @keyup.enter="savePreset" />
          <button class="cg-btn cg-btn-primary" :disabled="!presetName.trim() || !active" @click="savePreset">保存</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'

const props = defineProps<{ modelValue?: any }>()
const emit = defineEmits<{ (e: 'update:modelValue', v: any): void }>()

interface SliderDef { path: string; label: string; min: number; max: number; step: number }
interface GroupDef { title: string; sliders: SliderDef[] }

// 8 项校色参数定义（与后端 color-grade.ts 的 ColorGradeParams 对齐）
const groups: GroupDef[] = [
  { title: '色彩校准', sliders: [
    { path: 'colorCalibration.red', label: '红通道', min: -100, max: 100, step: 1 },
    { path: 'colorCalibration.green', label: '绿通道', min: -100, max: 100, step: 1 },
    { path: 'colorCalibration.blue', label: '蓝通道', min: -100, max: 100, step: 1 },
  ] },
  { title: '色调映射', sliders: [
    { path: 'toneMapping.gamma', label: '中间调', min: -100, max: 100, step: 1 },
  ] },
  { title: '白平衡修正', sliders: [
    { path: 'whiteBalance.temperature', label: '色温', min: -100, max: 100, step: 1 },
    { path: 'whiteBalance.tint', label: '色调', min: -100, max: 100, step: 1 },
  ] },
  { title: '曝光补偿', sliders: [
    { path: 'exposure', label: '曝光', min: -100, max: 100, step: 1 },
  ] },
  { title: '饱和度微调', sliders: [
    { path: 'saturation', label: '饱和度', min: -100, max: 100, step: 1 },
  ] },
  { title: '对比度优化', sliders: [
    { path: 'contrast', label: '对比度', min: -100, max: 100, step: 1 },
  ] },
  { title: '肤色还原', sliders: [
    { path: 'skinTone', label: '肤色', min: 0, max: 100, step: 1 },
  ] },
  { title: '阴影与高光', sliders: [
    { path: 'shadowsHighlights.shadows', label: '阴影', min: -100, max: 100, step: 1 },
    { path: 'shadowsHighlights.highlights', label: '高光', min: -100, max: 100, step: 1 },
  ] },
]

function defaultGrade(): any {
  return {
    colorCalibration: { red: 0, green: 0, blue: 0 },
    toneMapping: { gamma: 0 },
    whiteBalance: { temperature: 0, tint: 0 },
    exposure: 0,
    saturation: 0,
    contrast: 0,
    skinTone: 0,
    shadowsHighlights: { shadows: 0, highlights: 0 },
  }
}

function clone(v: any): any { return JSON.parse(JSON.stringify(v)) }
function getPath(obj: any, path: string): number {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj) ?? 0
}
function setPath(obj: any, path: string, val: number) {
  const keys = path.split('.')
  let o = obj
  for (let i = 0; i < keys.length - 1; i++) o = o[keys[i]]
  o[keys[keys.length - 1]] = val
}

const expanded = ref(false)
const local = reactive<any>(defaultGrade())
const presets = ref<any[]>([])
const selectedPresetId = ref<number | null>(null)
const presetName = ref('')

const active = computed(() => {
  for (const g of groups) for (const s of g.sliders) if (getPath(local, s.path) !== 0) return true
  return false
})

function syncFromModel(v: any) {
  const merged = { ...defaultGrade(), ...clone(v || {}) }
  Object.assign(local, defaultGrade())
  if (merged.colorCalibration) Object.assign(local.colorCalibration, merged.colorCalibration)
  if (merged.toneMapping) Object.assign(local.toneMapping, merged.toneMapping)
  if (merged.whiteBalance) Object.assign(local.whiteBalance, merged.whiteBalance)
  if (merged.shadowsHighlights) Object.assign(local.shadowsHighlights, merged.shadowsHighlights)
  local.exposure = merged.exposure ?? 0
  local.saturation = merged.saturation ?? 0
  local.contrast = merged.contrast ?? 0
  local.skinTone = merged.skinTone ?? 0
}
watch(() => props.modelValue, syncFromModel, { immediate: true })

function emitUpdate() { emit('update:modelValue', clone(local)) }
function onSlider(s: SliderDef, e: Event) {
  setPath(local, s.path, Number((e.target as HTMLInputElement).value))
  emitUpdate()
}
function reset() {
  Object.assign(local, defaultGrade())
  emitUpdate()
}

onMounted(loadPresets)
async function loadPresets() {
  try {
    const { presetsAPI } = await import('~/composables/useApi')
    const res: any = await presetsAPI.list('colorGrade')
    presets.value = res.data || res || []
  } catch (e) {
    console.warn('[ColorGrade] load presets failed', e)
  }
}
async function applyPreset() {
  const p = presets.value.find(x => x.id === selectedPresetId.value)
  if (!p?.config) return
  syncFromModel(p.config)
  emitUpdate()
}
async function savePreset() {
  const name = presetName.value.trim()
  if (!name) return
  try {
    const { presetsAPI } = await import('~/composables/useApi')
    await presetsAPI.create({ type: 'colorGrade', name, config: clone(local) })
    presetName.value = ''
    await loadPresets()
    alert('预设已保存')
  } catch (e: any) {
    alert('保存预设失败: ' + e.message)
  }
}
async function deletePreset() {
  if (selectedPresetId.value == null) return
  if (!confirm('确认删除该预设？')) return
  try {
    const { presetsAPI } = await import('~/composables/useApi')
    await presetsAPI.del(selectedPresetId.value)
    selectedPresetId.value = null
    await loadPresets()
  } catch (e: any) {
    alert('删除预设失败: ' + e.message)
  }
}
</script>

<style scoped>
.cg-panel {
  border: 1px solid rgba(100, 120, 180, 0.14);
  border-radius: 12px;
  background: #fbfcfe;
  overflow: hidden;
}
.cg-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  color: var(--text-0);
}
.cg-head:hover { background: rgba(13, 148, 136, 0.04); }
.cg-title { font-size: 13px; font-weight: 700; }
.cg-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-bg);
  color: var(--accent-text);
}
.cg-spacer { flex: 1; }
.cg-reset {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-text);
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 6px;
}
.cg-reset:hover:not(:disabled) { background: var(--accent-bg); }
.cg-reset:disabled { color: #c3cbd9; cursor: default; }

.cg-body { padding: 0 14px 14px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 22px; align-items: start; }
.cg-group { display: flex; flex-direction: column; gap: 8px; }
.cg-group-title { font-size: 11px; font-weight: 700; color: rgba(40, 50, 80, 0.55); text-transform: uppercase; letter-spacing: 0.3px; }
.cg-slider { display: grid; grid-template-columns: 52px 1fr 36px; align-items: center; gap: 10px; }
.cg-slider-label { font-size: 12px; color: #4a5670; }
.cg-slider-val {
  font-size: 12px;
  color: var(--accent-text);
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.cg-range { width: 100%; accent-color: var(--accent); cursor: pointer; height: 18px; }

.cg-presets { grid-column: 1 / -1; display: flex; flex-direction: column; gap: 8px; border-top: 1px dashed rgba(100, 120, 180, 0.2); padding-top: 12px; }
@media (max-width: 900px) {
  .cg-body { grid-template-columns: 1fr; }
}
.cg-preset-row { display: flex; gap: 8px; }
.cg-select, .cg-input {
  flex: 1;
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid rgba(100, 120, 180, 0.18);
  font-size: 12px;
  outline: none;
  background: #fff;
  color: var(--text-0);
}
.cg-select:focus, .cg-input:focus { border-color: var(--border-focus); box-shadow: 0 0 0 3px var(--accent-bg); }
.cg-btn {
  padding: 7px 12px;
  border-radius: 8px;
  border: 1px solid rgba(100, 120, 180, 0.2);
  background: #fff;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  color: #2c3850;
  transition: all .15s;
}
.cg-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent-text); }
.cg-btn:disabled { opacity: 0.45; cursor: default; }
.cg-btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.cg-btn-primary:hover:not(:disabled) { background: var(--accent-dark); color: #fff; }
.cg-btn-danger:hover:not(:disabled) { border-color: #dc2626; color: #dc2626; }
</style>

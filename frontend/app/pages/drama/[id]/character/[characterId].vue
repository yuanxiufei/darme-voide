<template>
  <div class="char-detail">
    <!-- 顶部导航 -->
    <header class="detail-header">
      <button class="back-btn" @click="goBack">
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
            <img v-if="currentPreviewUrl" :src="currentPreviewUrl" :alt="char.name" />
            <div v-else class="no-image">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>
              <span>暂无形象图</span>
            </div>
          </div>
          <!-- 服装变体网格：每套造型独立立绘 -->
          <div v-if="costumeOptions.length" class="variation-grid">
            <div
              v-for="c in costumeOptions"
              :key="c"
              class="variation-item"
              :class="{ active: selectedCostume === c }"
              @click="selectVariation(c)"
            >
              <div class="variation-thumb">
                <img v-if="getVariationImage(c)" :src="getVariationImage(c)" :alt="c" />
                <span v-else class="variation-empty">未生成</span>
              </div>
              <span class="variation-name">{{ c }}</span>
            </div>
          </div>
          <!-- 三视图（正面/侧面/背面） -->
          <div class="three-views">
            <div class="tv-head">
              <span class="tv-title">角色三视图</span>
              <button class="btn btn-sm" :disabled="threeGen" @click="generateThreeViews">
                <span v-if="threeGen" class="spinner" />
                {{ threeGen ? '生成中...' : '生成三视图' }}
              </button>
            </div>
            <div class="tv-grid">
              <div v-for="v in ['front','side','back']" :key="v" class="tv-cell">
                <div class="tv-thumb">
                  <img v-if="threeViews[v]?.imageUrl" :src="resolveImageUrl(threeViews[v].imageUrl)" :alt="viewLabel(v)" />
                  <span v-else class="tv-empty">{{ viewLabel(v) }}</span>
                </div>
                <span class="tv-label">{{ viewLabel(v) }}</span>
              </div>
            </div>
          </div>
          <!-- 设定图鉴（服装/武器/首饰独立设定图） -->
          <div class="three-views equip-gallery">
            <div class="tv-head">
              <span class="tv-title">设定图鉴</span>
            </div>
            <div class="tv-grid equip-grid">
              <div v-for="t in equipTypes" :key="t.key" class="tv-cell equip-cell">
                <div class="tv-thumb">
                  <img v-if="equipImages[t.key]?.imageUrl" :src="resolveImageUrl(equipImages[t.key].imageUrl)" :alt="t.label" />
                  <span v-else class="tv-empty">{{ t.label }}</span>
                </div>
                <span class="tv-label">{{ t.label }}</span>
                <button class="btn btn-sm btn-block" :disabled="equipGen === t.key" @click="generateEquip(t.key)">
                  <span v-if="equipGen === t.key" class="spinner" />
                  {{ equipGen === t.key ? '生成中...' : (equipImages[t.key]?.imageUrl ? '重新生成' : '生成') }}
                </button>
              </div>
            </div>
          </div>
          <!-- 快捷操作 -->
          <div class="quick-actions">
            <ModelSelector v-model="imageModel" service-type="image" label="生成模型" />
            <button class="btn btn-primary btn-block" :disabled="imgGen" @click="generateImage">
              <span v-if="imgGen" class="spinner spinner-light" />
              {{ imgGen ? '生成中...' : (selectedCostume ? `生成「${selectedCostume}」造型` : '生成主形象图') }}
            </button>
            <button class="btn btn-block" @click="testVoice">试听声音</button>
            <div class="save-to-lib">
              <button class="btn btn-sm btn-block" @click="saveCharacterToLibrary">保存角色</button>
              <button class="btn btn-sm btn-block" @click="saveWeaponToLibrary">保存武器</button>
              <button class="btn btn-sm btn-block" @click="saveCostumeToLibrary">保存服装</button>
            </div>
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
                <textarea v-model="form.clothing" class="input" rows="2" placeholder="请输入服装风格，例如：休闲、正式、街头" />
              </label>
              <label class="field">
                <span>首饰配饰</span>
                <textarea v-model="form.accessories" class="input" rows="2" placeholder="请输入首饰或配饰描述，例如：银色项链、皮质手链" />
              </label>
            </div>
            <div class="field-grid-2" style="margin-top: 14px;">
              <label class="field">
                <span>武器装备</span>
                <textarea v-model="form.weapons" class="input" rows="2" placeholder="长剑 / 匕首 / 法杖 / 无..." />
              </label>
              <label class="field">
                <span>多套服装变体（可选）</span>
                <textarea v-model="form.costumes" class="input" rows="2" placeholder="输入多个变体名称，用逗号分隔（可选）" />
              </label>
            </div>
            <div class="split-row">
              <button class="btn btn-sm btn-split" type="button" :disabled="splitting" @click="autoSplitVisuals">
                <span v-if="splitting" class="spinner" />
                {{ splitting ? '智能拆分中...' : '智能拆分服装/武器/首饰' }}
              </button>
              <span class="hint">从「外貌特征」自动识别并回填上方服装风格、武器装备、首饰配饰</span>
            </div>
            <p class="hint">服装风格与首饰配饰用于生成对应设定图；多个服装变体用逗号分隔，可在左侧预览区切换不同造型。</p>
          </section>

          <!-- 图片生成 Prompt -->
          <section class="form-section">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 图片生成提示词</h3>
            <label class="field">
              <span>正向提示词</span>
              <textarea v-model="form.customPrompt" class="input mono" rows="4" placeholder="请输入正向提示词，描述期望的服装特征、材质、色彩等" />
            </label>
            <label class="field" style="margin-top: 14px;">
              <span>反向提示词</span>
              <textarea v-model="form.negativePrompt" class="input mono" rows="3" placeholder="请输入反向提示词，描述需排除的元素，如模糊、低质量、变形等" />
            </label>
            <p class="hint">正向提示词描述期望生成的内容（留空时自动根据「名称+外貌+服装+性格」组合）；反向提示词描述需排除的内容。</p>
            <div style="margin-top: 16px;">
              <ColorGradePanel v-model="colorGrade" />
            </div>
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
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from '#app'

const route = useRoute()
const router = useRouter()

const dramaId = Number(route.params.id)
const characterId = Number(route.params.characterId)

// 返回上一级：有浏览历史时后退，直接访问/刷新（无历史）时兜底回项目页
function goBack() {
  const state = window.history.state
  if (state && state.back) {
    router.back()
  } else {
    router.push(`/drama/${dramaId}`)
  }
}

const char = ref<any>(null)
const loading = ref(true)
const saving = ref(false)
const imgGen = ref(false)
const errorMsg = ref('')
const imageModel = ref('')
const imgPollRef = ref<ReturnType<typeof setInterval> | null>(null)
const imgPollTimeoutRef = ref<ReturnType<typeof setTimeout> | null>(null)
const threeGen = ref(false)
const threePollRef = ref<ReturnType<typeof setInterval> | null>(null)
const threePollTimeoutRef = ref<ReturnType<typeof setTimeout> | null>(null)
const equipGen = ref<string | null>(null)
const equipPollRef = ref<ReturnType<typeof setInterval> | null>(null)
const equipPollTimeoutRef = ref<ReturnType<typeof setTimeout> | null>(null)
const colorGrade = ref<any>(null)
const splitting = ref(false)

const form = reactive<any>({})

function parseCostumesArray(text?: string): string[] {
  if (!text) return []
  return text.split(/[,，、]/).map(x => x.trim()).filter(Boolean)
}
function parseCostumesToText(costumes?: string | null): string {
  if (!costumes) return ''
  try {
    const arr = JSON.parse(costumes)
    return Array.isArray(arr) ? arr.join(', ') : ''
  } catch {
    return costumes || ''
  }
}
function parseCostumesFromText(text?: string): string | null {
  if (!text) return null
  const arr = parseCostumesArray(text)
  return arr.length ? JSON.stringify(arr) : null
}
const costumeOptions = computed(() => parseCostumesArray(form.costumes))
const selectedCostume = ref('')

function parseVariations(variations?: string | null): Array<{ name: string; imageUrl: string | null }> {
  if (!variations) return []
  try {
    const arr = JSON.parse(variations)
    if (!Array.isArray(arr)) return []
    return arr.filter((x: any) => x && typeof x === 'object' && typeof x.name === 'string')
  } catch {
    return []
  }
}
const variations = computed(() => parseVariations(char.value?.variations))
function getVariationImage(name: string): string | null {
  const v = variations.value.find(x => x.name === name)
  return v?.imageUrl ? ('/' + v.imageUrl) : null
}
// 当前预览：选中变体有图则显示变体图，否则主图
const currentPreviewUrl = computed(() => {
  const variation = selectedCostume.value ? getVariationImage(selectedCostume.value) : null
  return variation || (char.value?.image_url || char.value?.imageUrl ? '/' + (char.value.image_url || char.value.imageUrl) : null)
})
function selectVariation(costume: string) {
  selectedCostume.value = costume
}

const THREE_VIEW_LABELS: Record<string, string> = { front: '正面', side: '侧面', back: '背面' }
function viewLabel(v: string) { return THREE_VIEW_LABELS[v] || v }
function resolveImageUrl(p?: string | null): string {
  if (!p) return ''
  if (p.startsWith('http://') || p.startsWith('https://') || p.startsWith('data:')) return p
  return p.startsWith('/') ? p : '/' + p
}
function parseThreeViews(v?: string | null): Record<string, any> {
  if (!v) return {}
  try {
    const o = JSON.parse(v)
    return o && typeof o === 'object' ? o : {}
  } catch { return {} }
}
const threeViews = computed(() => parseThreeViews(char.value?.threeViews || char.value?.three_views))

const equipTypes = [
  { key: 'clothing', label: '服装' },
  { key: 'weapon', label: '武器' },
  { key: 'accessory', label: '首饰' },
]
function parseEquipImages(v?: string | null): Record<string, any> {
  if (!v) return {}
  try {
    const o = JSON.parse(v)
    return o && typeof o === 'object' ? o : {}
  } catch { return {} }
}
const equipImages = computed(() => parseEquipImages(char.value?.equipImages || char.value?.equip_images))

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
    form.costumes = parseCostumesToText(c.costumes)
    selectedCostume.value = parseCostumesArray(form.costumes)[0] || ''
    form.customPrompt = c.customPrompt || ''
    form.accessories = c.accessories || ''
    form.negativePrompt = c.negativePrompt || c.negative_prompt || ''
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

async function persistForm() {
  const { characterAPI } = await import('~/composables/useApi')
  await characterAPI.update(characterId, {
    name: form.name,
    role: form.role,
    description: form.description,
    appearance: form.appearance,
    personality: form.personality,
    clothing: form.clothing || null,
    weapons: form.weapons || null,
    costumes: parseCostumesFromText(form.costumes),
    customPrompt: form.customPrompt || null,
    accessories: form.accessories || null,
    negativePrompt: form.negativePrompt || null,
    voiceStyle: form.voiceStyle,
    voiceModel: form.voiceModel,
    voiceSpeed: form.voiceSpeed ? Number(form.voiceSpeed) : null,
    voiceEmotion: form.voiceEmotion || null,
    voicePitch: form.voicePitch ? Number(form.voicePitch) : null,
  })
  // 刷新本地数据
  if (char.value) Object.assign(char.value, form)
}

async function save() {
  saving.value = true
  try {
    await persistForm()
    alert('保存成功')
  } catch (e: any) {
    errorMsg.value = '保存失败: ' + e.message
  } finally {
    saving.value = false
  }
}

async function autoSplitVisuals() {
  if (!form.appearance?.trim()) {
    errorMsg.value = '请先填写「外貌特征」'
    return
  }
  splitting.value = true
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    const result: any = await characterAPI.autoSplitVisuals(characterId, form.appearance)
    // 非空结果回填（覆盖），AI 未识别到的字段保持原值不变
    let filled = 0
    if (result?.clothing) { form.clothing = result.clothing; filled++ }
    if (result?.weapons) { form.weapons = result.weapons; filled++ }
    if (result?.accessories) { form.accessories = result.accessories; filled++ }
    alert(filled > 0 ? '已智能拆分并回填，请确认后保存' : '未识别到服装/武器/首饰信息')
  } catch (e: any) {
    errorMsg.value = '智能拆分失败: ' + e.message
  } finally {
    splitting.value = false
  }
}

async function generateImage() {
  imgGen.value = true
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    const targetCostume = selectedCostume.value
    await characterAPI.generateImage(characterId, undefined, {
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      costume: targetCostume || undefined,
      color_grade: colorGrade.value || undefined,
    })
    // 轮询等待结果：变体生成检查 variations 对应项，主图生成检查 imageUrl
    imgPollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        let done = false
        if (targetCostume) {
          const variation = parseVariations(updated.variations).find((v: any) => v.name === targetCostume)
          done = !!variation?.imageUrl
        } else {
          done = !!(updated.imageUrl || updated.image_url)
        }
        if (done) {
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
    await characterAPI.voiceSample(characterId)
    alert('语音试听已提交，请稍后刷新查看')
  } catch (e: any) {
    errorMsg.value = e.message
  }
}

async function generateThreeViews() {
  threeGen.value = true
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    await characterAPI.generateThreeViews(characterId, undefined, {
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      clothing: form.clothing || undefined,
      weapons: form.weapons || undefined,
      accessories: form.accessories || undefined,
      color_grade: colorGrade.value || undefined,
    })
    threePollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const views = parseThreeViews(updated.threeViews || updated.three_views)
        const done = ['front', 'side', 'back'].every(v => views[v]?.imageUrl)
        if (done) {
          clearInterval(threePollRef.value!)
          threePollRef.value = null
          char.value = updated
          threeGen.value = false
        }
      } catch (e) {
        console.warn('[Character] three-view poll error:', e)
      }
    }, 3000)
    threePollTimeoutRef.value = setTimeout(() => {
      if (threePollRef.value) { clearInterval(threePollRef.value); threePollRef.value = null }
      threePollTimeoutRef.value = null
      threeGen.value = false
    }, 180000)
  } catch (e: any) {
    errorMsg.value = e.message
    threeGen.value = false
  }
}

async function generateEquip(type: 'clothing' | 'weapon' | 'accessory') {
  equipGen.value = type
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    await characterAPI.generateEquipImage(characterId, undefined, {
      type,
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      color_grade: colorGrade.value || undefined,
    })
    equipPollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const data = updated.data || updated
        const imgs = parseEquipImages(data.equipImages || data.equip_images)
        if (imgs[type]?.imageUrl) {
          clearInterval(equipPollRef.value!)
          equipPollRef.value = null
          char.value = data
          equipGen.value = null
        }
      } catch (e) {
        console.warn('[Character] equip poll error:', e)
      }
    }, 3000)
    equipPollTimeoutRef.value = setTimeout(() => {
      if (equipPollRef.value) { clearInterval(equipPollRef.value); equipPollRef.value = null }
      equipPollTimeoutRef.value = null
      equipGen.value = null
    }, 120000)
  } catch (e: any) {
    errorMsg.value = e.message
    equipGen.value = null
  }
}

async function saveWeaponToLibrary() {
  if (!form.weapons) { errorMsg.value = '请先填写武器装备'; return }
  try {
    const { weaponLibraryAPI } = await import('~/composables/useApi')
    await weaponLibraryAPI.fromCharacter(characterId, { name: form.weapons })
    alert('武器已保存到武器库')
  } catch (e: any) {
    errorMsg.value = '保存武器失败: ' + e.message
  }
}

async function saveCostumeToLibrary() {
  if (!form.clothing) { errorMsg.value = '请先填写服装风格'; return }
  try {
    const { costumeLibraryAPI } = await import('~/composables/useApi')
    await costumeLibraryAPI.fromCharacter(characterId, { name: form.clothing })
    alert('服装已保存到服装库')
  } catch (e: any) {
    errorMsg.value = '保存服装失败: ' + e.message
  }
}

async function saveCharacterToLibrary() {
  if (!form.name) { errorMsg.value = '请先填写角色名称'; return }
  try {
    await persistForm()
    const { characterLibraryAPI } = await import('~/composables/useApi')
    await characterLibraryAPI.fromCharacter(characterId)
    alert('角色已保存到角色库')
  } catch (e: any) {
    errorMsg.value = '保存角色到角色库失败: ' + e.message
  }
}

onUnmounted(() => {
  if (imgPollRef.value) clearInterval(imgPollRef.value)
  if (imgPollTimeoutRef.value) clearTimeout(imgPollTimeoutRef.value)
  if (threePollRef.value) clearInterval(threePollRef.value)
  if (threePollTimeoutRef.value) clearTimeout(threePollTimeoutRef.value)
  if (equipPollRef.value) clearInterval(equipPollRef.value)
  if (equipPollTimeoutRef.value) clearTimeout(equipPollTimeoutRef.value)
})
</script>

<style scoped>
.char-detail {
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
  grid-template-columns: 360px 1fr;
  gap: 28px;
  padding: 24px 32px;
  max-width: 1680px;
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
.variation-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.variation-item {
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid transparent;
  background: #fff;
  cursor: pointer;
  transition: border-color .15s, box-shadow .15s;
  display: flex;
  flex-direction: column;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.variation-item.active {
  border-color: #0d9488;
  box-shadow: 0 4px 14px rgba(13,148,136,0.25);
}
.variation-thumb {
  aspect-ratio: 3/4;
  background: #f3f4f6;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.variation-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.variation-empty { font-size: 11px; color: #9ca3af; letter-spacing: 1px; }
.variation-name {
  padding: 6px 4px;
  font-size: 12px;
  text-align: center;
  color: #374151;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.quick-actions { display: flex; flex-direction: column; gap: 10px; }
.btn-block { width: 100%; justify-content: center; }

/* 三视图 */
.three-views {
  background: #fff;
  border-radius: 14px;
  border: 1px solid rgba(100,120,180,0.12);
  padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.tv-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.tv-title { font-size: 13px; font-weight: 700; color: var(--text-0); }
.tv-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.tv-cell { display: flex; flex-direction: column; gap: 5px; }
.tv-thumb {
  aspect-ratio: 3/4;
  border-radius: 10px;
  background: #f3f4f6;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.tv-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.tv-empty { font-size: 11px; color: #9ca3af; letter-spacing: 1px; }
.tv-label { font-size: 11px; text-align: center; color: #6b7280; font-weight: 600; }

/* 设定图鉴（服装/武器/首饰独立设定图） */
.equip-cell .btn { font-size: 11px; padding: 6px 4px; }

/* 保存到资源库 */
.save-to-lib {
  display: flex;
  gap: 8px;
  padding-top: 8px;
  border-top: 1px dashed rgba(100,120,180,0.2);
  margin-top: 4px;
}
.save-to-lib .btn { flex: 1; padding: 8px 6px; font-size: 12px; }
.btn-sm { padding: 7px 12px; font-size: 12px; font-weight: 600; border-radius: 8px; }
.split-row { display: flex; align-items: center; gap: 10px; margin: 12px 0 4px; padding-top: 12px; border-top: 1px dashed rgba(100,120,180,0.18); }
.split-row .hint { margin: 0; }
.btn-split { color: var(--accent-text, #0f766e); border-color: rgba(13,148,136,0.3); background: rgba(13,148,136,0.06); display: inline-flex; align-items: center; gap: 6px; }
.btn-split:hover { background: rgba(13,148,136,0.12); border-color: rgba(13,148,136,0.45); }
.btn-split:disabled { opacity: 0.6; cursor: not-allowed; }
.spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  flex: none;
  border: 2px solid rgba(13,148,136,0.25);
  border-top-color: #0d9488;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
.spinner-light {
  border-color: rgba(255,255,255,0.4);
  border-top-color: #fff;
}
@keyframes spin { to { transform: rotate(360deg); } }

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
  color: var(--text-0);
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

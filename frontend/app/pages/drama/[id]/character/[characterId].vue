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
        <!-- 基本信息 -->
        <section class="form-section">
          <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg> 基本信息</h3>
          <div class="field-grid-2">
            <label class="field">
              <span>角色名称</span>
              <input v-model="form.name" class="input" />
            </label>
            <label class="field">
              <span>角色类型</span>
              <select v-model="form.roleType" class="input">
                <option value="">未指定</option>
                <option value="主角">主角</option>
                <option value="反派">反派</option>
                <option value="配角">配角</option>
                <option value="旁白">旁白</option>
                <option value="其他">其他</option>
              </select>
            </label>
          </div>
          <label class="field">
            <span>角色定位</span>
            <input v-model="form.role" class="input" placeholder="如：霸道总裁男主 / 亦正亦邪的反派" />
          </label>
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
          <label class="field" style="margin-top: 14px;">
            <span>武器装备</span>
            <textarea v-model="form.weapons" class="input" rows="2" placeholder="长剑 / 匕首 / 法杖 / 无..." />
          </label>
          <label class="field" style="margin-top: 14px;">
            <span>多套服装变体（可选）</span>
            <textarea v-model="form.costumes" class="input" rows="2" placeholder="输入多个变体名称，用逗号分隔（可选）" />
          </label>
          <div class="split-row">
            <button class="btn btn-sm btn-split" type="button" :disabled="splitting" @click="autoSplitVisuals">
              <span v-if="splitting" class="spinner" />
              {{ splitting ? '智能拆分中...' : '智能拆分服装/武器/首饰' }}
            </button>
            <span class="hint">从「外貌特征」自动识别并回填上方服装风格、武器装备、首饰配饰</span>
          </div>
          <p class="hint">服装/首饰/武器用于生成各自的三视图设定图；多个服装变体用逗号分隔，可在「角色立绘」模块切换不同造型。</p>
        </section>

        <!-- 图片生成（角色设定总览）：每个生成模块各占一行（全宽），顺序 = 角色立绘 → 全身三视图 → 表情头像 → 服装 → 武器 → 首饰 -->
        <section class="form-section gen-section">
          <div class="sec-head">
            <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 图片生成</h3>
            <div class="gen-toolbar">
              <button class="btn btn-batch" :disabled="batchBusy || anyImageBusy" title="一键重做全套素材：先重新生成全身三视图（作为锚定基准），随后并行重新生成服装 / 武器 / 首饰三视图与表情头像。已有素材也会一并覆盖重做" @click="generateAllAssets">
                <span v-if="batchBusy" class="spinner spinner-light" />
                <span v-else class="batch-ok-icon" />
                {{ batchBusy ? '一键生成中…' : '一键生成全部素材' }}
              </button>
              <ModelSelector v-model="imageModel" service-type="image" label="生成模型" />
            </div>
          </div>

          <!-- 一键生成进度提示：先生成三视图基准，再生成装备三视图与表情（已有素材一并覆盖重做） -->
          <div v-if="batchBusy" class="gen-batch-status">
            <span class="spinner" />
            <span>{{ batchStatus }}</span>
            <span class="gen-batch-meta">已完成 {{ batchProgress }}/{{ batchTotal }}</span>
          </div>

          <!-- 视觉锚定 & 时代背景：参考图自动下发 / 时代注入说明（不影响保存，仅展示本次生成依赖的设定） -->
          <div class="gen-anchorbar">
            <div class="anchor-thumb">
              <img v-if="anchorBaseUrl" :src="anchorBaseUrl" alt="视觉锚定基准图" />
              <div v-else class="anchor-thumb-empty">暂无<br />基准图</div>
            </div>
            <div class="anchor-info">
              <div class="anchor-row">
                <span class="anchor-label">生成锚定</span>
                <select v-model="anchorMode" class="input anchor-select">
                  <option value="auto">自动（三视图优先）</option>
                  <option value="image">仅主立绘</option>
                  <option value="three_views">仅三视图</option>
                  <option value="none">不参考</option>
                </select>
                <span class="anchor-hint">{{ anchorBaseUrl ? '表情 / 装备 / 三视图生成时自动携带左侧基准图参考，保持同角色造型不串脸；主立绘重生成仅支持显式传图' : '先生成主立绘或三视图，后续表情 / 三视图 / 装备会自动以它为基准对齐' }}</span>
              </div>
              <div class="anchor-row anchor-era">
                <template v-if="dramaEra?.era">
                  <span class="anchor-era-tag">{{ dramaEra.era }}</span>
                  <span class="anchor-era-text">{{ dramaEra.summary || '本剧已提炼时代背景，角色资产生成时自动注入时代画面指令' }}</span>
                </template>
                <template v-else>
                  <span class="anchor-era-tag anchor-era-tag-dim">无时代背景</span>
                  <span class="anchor-era-text">本剧尚未提炼时代背景。可在「编辑剧集 → 时代背景」AI 自动提炼，之后角色/道具资产生成会自动注入时代与环境指令。</span>
                </template>
              </div>
            </div>
          </div>

          <div class="gen-layout">
            <!-- ============ 角色立绘 + 全身三视图（各占一行） ============ -->
            <div class="col-main">
              <!-- 角色立绘：提示词 + 生成按钮 + 主形象图 -->
              <div class="gen-module">
                <div class="gen-head">
                  <span class="gen-title">角色立绘</span>
                  <button class="btn btn-sm btn-primary" :disabled="imgGen || batchBusy" @click="generateImage">
                    <span v-if="imgGen" class="spinner spinner-light" />
                    {{ imgGen ? '生成中...' : (selectedCostume ? `生成「${selectedCostume}」造型` : '生成主形象图') }}
                  </button>
                </div>
                <div class="gen-body">
                  <div class="gen-preview">
                    <div class="image-card">
                      <img v-if="currentPreviewUrl" :src="currentPreviewUrl" :alt="char.name" />
                      <div v-else class="no-image">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>
                        <span>暂无形象图</span>
                      </div>
                    </div>
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
                  </div>
                  <div class="gen-prompts">
                    <label class="field pos-field">
                      <span class="plabel plabel-pos">角色立绘 · 正向提示词</span>
                      <textarea v-model="form.customPrompt" class="input mono" rows="4" placeholder="描述角色全身形象（外貌、服装、武器、首饰、画风等）。可点下方「自动生成提示词」或手动填写" />
                    </label>
                    <label class="field neg-field" style="margin-top: 12px;">
                      <span class="plabel plabel-neg">反向提示词</span>
                      <textarea v-model="form.negativePrompt" class="input mono" rows="3" placeholder="描述需排除的元素（如模糊、低质量、变形、多视角拼接）。留空时生成会自动按画风附加默认排除词" />
                    </label>
                    <div class="gen-actions">
                      <button class="btn btn-sm btn-split" type="button" :disabled="promptGen === 'character'" @click="generatePromptFor('character')">
                        <span v-if="promptGen === 'character'" class="spinner" />
                        {{ promptGen === 'character' ? '生成中...' : '根据形象设定生成提示词' }}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 三视图：全身三视图横向长图（左列大图） -->
              <div class="gen-module gen-module-wide">
                <div class="gen-head">
                  <span class="gen-title">全身三视图</span>
                  <button class="btn btn-sm btn-primary" :disabled="threeGen || batchBusy" @click="generateThreeViews">
                    <span v-if="threeGen" class="spinner spinner-light" />
                    {{ threeGen ? '生成中...' : '生成三视图' }}
                  </button>
                </div>
                <div class="gen-body">
                  <div class="gen-preview">
                    <div v-if="threeCombinedUrl" class="gen-media gen-combined">
                      <img :src="threeCombinedUrl" alt="全身三视图（正面/侧面/背面）" />
                    </div>
                    <div v-else class="gen-media gen-empty">暂无全身三视图<br />点击右上角「生成三视图」</div>
                  </div>
                </div>
                <div class="gen-prompts">
                  <p class="hint" style="margin-bottom: 8px;">三视图为<b>一张横向长图</b>：正面在左、侧面在中、背面在右，与角色立绘共享下方提示词。</p>
                  <label class="field pos-field">
                    <span class="plabel plabel-pos">三视图 · 正向提示词</span>
                    <textarea v-model="form.customPrompt" class="input mono" rows="4" placeholder="复用角色立绘提示词，将自动附加「三视图排版」描述" />
                  </label>
                  <label class="field neg-field" style="margin-top: 12px;">
                    <span class="plabel plabel-neg">反向提示词</span>
                    <textarea v-model="form.negativePrompt" class="input mono" rows="3" placeholder="留空时自动附加分屏/网格/文字等排除词，保证三个视角合成在同一张图内" />
                  </label>
                </div>
              </div>
            </div>

            <!-- ============ 表情头像 + 服装/武器/首饰三视图（各占一行） ============ -->
            <div class="col-side">
              <div class="gen-module gen-module-expressions">
                <div class="gen-head">
                  <span class="gen-title">表情头像（{{ expressionReadyCount }}/{{ expressionPresets.length }}）</span>
                  <button class="btn btn-sm btn-primary" :disabled="expressionsBusyActive || batchBusy" @click="generateExpressions()">
                    <span v-if="expressionsBusyActive" class="spinner spinner-light" />
                    {{ expressionsBusyActive ? '生成中...' : (expressionReadyCount > 0 ? '重新生成全部' : '一键生成全部表情') }}
                  </button>
                </div>
                <div class="expressions-grid">
                  <div v-for="e in expressionPresets" :key="e.key" class="expression-item">
                    <div class="expression-img">
                      <img v-if="expressionMap[e.key]?.imageUrl" :src="resolveImageUrl(expressionMap[e.key].imageUrl)" :alt="e.label" />
                      <div v-else class="expression-empty">未生成</div>
                      <div v-if="expressionBusy[e.key]" class="expression-spin"><span class="spinner spinner-light" /></div>
                      <button v-if="expressionMap[e.key]?.imageUrl && !expressionBusy[e.key]" type="button" class="expression-regen" @click="generateExpressions([e.key])">重生成</button>
                    </div>
                    <span class="expression-label">{{ e.label }}</span>
                  </div>
                </div>
                <p class="hint" style="margin-top: 10px;">头肩特写表情组（默认 9 个），用于分镜/表情演出时保持一致。点击单张可单独重生成。</p>
              </div>

              <!-- 服装/武器/首饰三视图：依次竖排，与表情共同构成角色设定总览参考 -->
              <div v-for="t in equipTypes" :key="t.key" class="gen-module equip-module">
                <div class="gen-head">
                  <span class="gen-title">{{ t.label }}三视图</span>
                  <button class="btn btn-sm btn-primary" type="button" :disabled="equipGen === t.key || batchBusy" @click="generateEquip(t.key)">
                    <span v-if="equipGen === t.key" class="spinner spinner-light" />
                    {{ equipGen === t.key ? '生成中...' : (equipImages[t.key]?.imageUrl ? '重新生成' : '生成') }}
                  </button>
                </div>
                <div class="gen-preview">
                  <div v-if="equipImages[t.key]?.imageUrl" class="gen-media gen-combined">
                    <img :src="resolveImageUrl(equipImages[t.key].imageUrl)" :alt="`${t.label}三视图`" />
                  </div>
                  <div v-else class="gen-media gen-empty">暂无{{ t.label }}三视图<br />点击右上角「生成」</div>
                </div>

                <div class="gen-prompts">
                  <label class="field pos-field">
                    <span class="plabel plabel-pos">{{ t.label }} · 正向提示词</span>
                    <textarea :value="form[promptFieldMap[t.key]]" @input="form[promptFieldMap[t.key]] = ($event.target as HTMLTextAreaElement).value" class="input mono" rows="4" :placeholder="`描述${t.label}本体细节（同一物品三个视角并排成一张横图，不出现人物）。可点下方自动生成`" />
                  </label>
                  <label class="field neg-field" style="margin-top: 12px;">
                    <span class="plabel plabel-neg">反向提示词</span>
                    <textarea :value="form[negFieldMap[t.key]]" @input="form[negFieldMap[t.key]] = ($event.target as HTMLTextAreaElement).value" class="input mono" rows="3" placeholder="描述需排除的元素，留空自动按画风附加默认排除词" />
                  </label>
                  <div class="gen-actions">
                    <button class="btn btn-sm btn-split" type="button" :disabled="promptGen === t.key" @click="generatePromptFor(t.key)">
                      <span v-if="promptGen === t.key" class="spinner" />
                      {{ promptGen === t.key ? '生成中...' : '根据形象设定生成提示词' }}
                    </button>
                  </div>
                </div>
                <p class="hint" style="margin-top: 10px;">同一物品正面/侧面/背面并排成一张横向长图（不出现人物），供分镜物品一致性参考。</p>
              </div>
            </div>
          </div>

        </section>

        <!-- 画风（留空跟随剧集） -->
        <section class="form-section">
          <h3><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> 画风</h3>
          <label class="field">
            <span>画风（留空跟随剧集）</span>
            <select v-model="form.style" class="input">
              <option value="">跟随剧集</option>
              <option value="realistic">写实电影</option>
              <option value="anime">日式动漫</option>
              <option value="ghibli">吉卜力</option>
              <option value="cinematic">电影感</option>
              <option value="comic">美漫漫画</option>
              <option value="watercolor">水彩</option>
            </select>
          </label>
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
          <div class="split-row">
            <button class="btn btn-sm btn-split" type="button" @click="testVoice">
              <span v-if="splitting" class="spinner" />
              试听声音
            </button>
            <button class="btn btn-sm btn-split" type="button" @click="saveCharacterToLibrary">保存到角色库</button>
            <button class="btn btn-sm btn-split" type="button" @click="saveCostumeToLibrary">保存服装到服装库</button>
            <button class="btn btn-sm btn-split" type="button" @click="saveWeaponToLibrary">保存武器到武器库</button>
          </div>
          <div v-if="errorMsg" class="error-banner" style="margin-top: 12px;">{{ errorMsg }}</div>
        </section>
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
// 一键生成全套素材：全身三视图为锚定基准先行，装备三视图 + 表情随后并行（已存在也覆盖重生成）
const batchBusy = ref(false)
const batchStatus = ref('')
const batchProgress = ref(0)
const batchTotal = ref(0)
const activePollRef = ref<ReturnType<typeof setInterval> | null>(null)
// 视觉锚定模式（参考图随生成自动下发）：auto / image / three_views / none
const anchorMode = ref<'auto' | 'image' | 'three_views' | 'none'>('auto')
// 剧集时代背景（后端 dramas.era_background AI 提炼 JSON，仅展示用）
const dramaEra = ref<{ era?: string; summary?: string } | null>(null)
const exprPollRef = ref<ReturnType<typeof setInterval> | null>(null)
const exprPollTimeoutRef = ref<ReturnType<typeof setTimeout> | null>(null)
const splitting = ref(false)
const promptGen = ref<string | null>(null)

const form = reactive<any>({})

// 提示词生成对象类型：角色立绘 / 服装 / 武器 / 首饰
const promptTypes = [
  { key: 'character', label: '角色立绘' },
  { key: 'clothing', label: '服装设定' },
  { key: 'weapon', label: '武器设定' },
  { key: 'accessory', label: '首饰设定' },
] as const
type PromptTypeKey = typeof promptTypes[number]['key']

// 各对象类型独立提示词字段映射（角色立绘沿用 customPrompt/negativePrompt，装备各自独立）
const promptFieldMap = {
  character: 'customPrompt',
  clothing: 'clothingPrompt',
  weapon: 'weaponPrompt',
  accessory: 'accessoryPrompt',
} as const
const negFieldMap = {
  character: 'negativePrompt',
  clothing: 'clothingNegativePrompt',
  weapon: 'weaponNegativePrompt',
  accessory: 'accessoryNegativePrompt',
} as const

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

function resolveImageUrl(p?: string | null): string {
  if (!p) return ''
  if (p.startsWith('http://') || p.startsWith('https://') || p.startsWith('data:')) return p
  return p.startsWith('/') ? p : '/' + p
}
// 统一图片 URL 归一化：用于轮询判定"新图已落地"，避免已存在旧图时轮询提前结束
function normUrl(p?: string | null): string {
  return resolveImageUrl(p)
}
function parseThreeViews(v?: string | null): Record<string, any> {
  if (!v) return {}
  try {
    const o = JSON.parse(v)
    return o && typeof o === 'object' ? o : {}
  } catch { return {} }
}
const threeViews = computed(() => parseThreeViews(char.value?.threeViews || char.value?.three_views))
// 视觉锚定基准图缩略图（展示用）：三视图 combined 优先，其次主立绘
const anchorBaseUrl = computed(() => {
  const combined = threeViews.value.combined?.imageUrl
  if (combined) return resolveImageUrl(combined)
  return resolveImageUrl(char.value?.imageUrl || char.value?.image_url || char.value?.localPath || '')
})
const threeCombinedUrl = computed(() => {
  const url = threeViews.value.combined?.imageUrl
  return url ? resolveImageUrl(url) : ''
})

// 服装/武器/首饰图鉴：各自按形象设定字段生成三视图
const equipTypes = [
  { key: 'clothing', label: '服装' },
  { key: 'weapon', label: '武器' },
  { key: 'accessory', label: '首饰' },
] as const
type EquipTypeKey = typeof equipTypes[number]['key']

// ===== 表情头像特写组：key 与后端 EXPRESSION_PRESETS 保持一致 =====
const expressionPresets = [
  { key: 'smile', label: '微笑' },
  { key: 'laugh', label: '大笑' },
  { key: 'mischievous', label: '俏皮' },
  { key: 'angry', label: '愤怒' },
  { key: 'sad', label: '悲伤' },
  { key: 'surprised', label: '惊讶' },
  { key: 'tearful', label: '泪目' },
  { key: 'serious', label: '严肃' },
  { key: 'sobbing', label: '大哭' },
]
const expressionBusy = ref<Record<string, boolean>>({})
const expressionReadyCount = computed(() =>
  Object.values(expressionMap.value).filter((e: any) => e && e.imageUrl).length)
const expressionsBusyActive = computed(() => Object.values(expressionBusy.value).some(Boolean))
// 任一图片生成进行中（单模块按钮启用依赖；一键生成按钮同理禁用）
const anyImageBusy = computed(() =>
  imgGen.value || threeGen.value || !!equipGen.value || expressionsBusyActive.value || batchBusy.value)
function parseEquipImages(v?: string | null): Record<string, any> {
  if (!v) return {}
  try {
    const o = JSON.parse(v)
    return o && typeof o === 'object' ? o : {}
  } catch { return {} }
}
const equipImages = computed(() => parseEquipImages(char.value?.equipImages || char.value?.equip_images))

// 表情头像特写组（存储结构：{ [expressionKey]: { key, imageUrl, prompt, generatedAt } }）
function parseExpressionsRaw(v?: string | null): Record<string, any> {
  if (!v) return {}
  try {
    const o = typeof v === 'string' ? JSON.parse(v) : v
    return o && typeof o === 'object' ? o : {}
  } catch { return {} }
}
const expressionMap = computed(() => parseExpressionsRaw(char.value?.expressions))

onUnmounted(() => {
  if (imgPollRef.value) clearInterval(imgPollRef.value)
  if (imgPollTimeoutRef.value) clearTimeout(imgPollTimeoutRef.value)
  if (threePollRef.value) clearInterval(threePollRef.value)
  if (threePollTimeoutRef.value) clearTimeout(threePollTimeoutRef.value)
  if (equipPollRef.value) clearInterval(equipPollRef.value)
  if (equipPollTimeoutRef.value) clearTimeout(equipPollTimeoutRef.value)
  if (exprPollRef.value) clearInterval(exprPollRef.value)
  if (exprPollTimeoutRef.value) clearTimeout(exprPollTimeoutRef.value)
  if (activePollRef.value) clearInterval(activePollRef.value)
})

onMounted(async () => {
  try {
    const { characterAPI } = await import('~/composables/useApi')
    // 直接获取单个角色
    const res: any = await characterAPI.get(characterId)
    char.value = res.data || res || null
    if (!char.value) return
    // 回填表单（兼容 camelCase / snake_case 双字段名）
    const c = char.value
    const pick = (k: string) => {
      const snake = k.replace(/[A-Z]/g, m => '_' + m.toLowerCase())
      return c[k] ?? c[snake] ?? ''
    }
    form.name = c.name
    form.role = pick('role')
    form.roleType = pick('roleType')
    form.description = pick('description')
    form.appearance = pick('appearance')
    form.personality = pick('personality')
    form.clothing = pick('clothing')
    form.weapons = pick('weapons')
    form.accessories = pick('accessories')
    form.customPrompt = pick('customPrompt')
    form.style = pick('style')
    form.negativePrompt = pick('negativePrompt')
    // 装备提示词：历史遗留的「单图/含人物」文本直接丢弃（否则装备三视图会生成出人）
    const cleanPrompt = (raw: string) => {
      const t = (raw || '').trim()
      if (!t) return ''
      if (t.includes('character appearance') || t.includes('for the character')) return ''
      if (t.includes('single view') && !t.includes('side by side') && !t.includes('three')) return ''
      return t
    }
    const cleanNegative = (raw: string) => {
      const t = (raw || '').trim()
      if (!t) return ''
      return /person|human|hand/i.test(t) ? t : ''
    }
    form.clothingPrompt = cleanPrompt(pick('clothingPrompt'))
    form.clothingNegativePrompt = cleanNegative(pick('clothingNegativePrompt'))
    form.weaponPrompt = cleanPrompt(pick('weaponPrompt'))
    form.weaponNegativePrompt = cleanNegative(pick('weaponNegativePrompt'))
    form.accessoryPrompt = cleanPrompt(pick('accessoryPrompt'))
    form.accessoryNegativePrompt = cleanNegative(pick('accessoryNegativePrompt'))
    form.costumes = parseCostumesToText(c.costumes)
    selectedCostume.value = parseCostumesArray(form.costumes)[0] || ''
    form.voiceStyle = pick('voiceStyle')
    form.voiceModel = c.voiceModel || pick('voiceModel') || 'speech-2.8-hd'
    form.voiceSpeed = c.voiceSpeed != null ? String(c.voiceSpeed) : ''
    form.voiceEmotion = pick('voiceEmotion')
    form.voicePitch = c.voicePitch != null ? String(c.voicePitch) : ''
  } catch (e: any) {
    errorMsg.value = e.message
  } finally {
    loading.value = false
  }
  loadDramaEra()
})

// 拉取剧集时代背景（仅用于页面顶部提示；生成时后端已自动注入，不阻塞角色页）
async function loadDramaEra() {
  try {
    const { dramaAPI } = await import('~/composables/useApi')
    const r: any = await dramaAPI.get(Number(route.params.id))
    const d = r?.data || r
    const raw = d?.era_background ?? d?.eraBackground
    if (raw) {
      const obj = typeof raw === 'string' ? (() => { try { return JSON.parse(raw) } catch { return null } })() : raw
      if (obj) dramaEra.value = obj
    }
  } catch {
    /* 无时代背景或请求失败时静默，不影响角色页 */
  }
}

async function persistForm() {
  const { characterAPI } = await import('~/composables/useApi')
  await characterAPI.update(characterId, {
    name: form.name,
    role: form.role,
    roleType: form.roleType || null,
    description: form.description,
    appearance: form.appearance,
    personality: form.personality,
    clothing: form.clothing || null,
    weapons: form.weapons || null,
    costumes: parseCostumesFromText(form.costumes),
    customPrompt: form.customPrompt || null,
    style: form.style || null,
    accessories: form.accessories || null,
    negativePrompt: form.negativePrompt || null,
    clothingPrompt: form.clothingPrompt || null,
    clothingNegativePrompt: form.clothingNegativePrompt || null,
    weaponPrompt: form.weaponPrompt || null,
    weaponNegativePrompt: form.weaponNegativePrompt || null,
    accessoryPrompt: form.accessoryPrompt || null,
    accessoryNegativePrompt: form.accessoryNegativePrompt || null,
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

// 根据形象设定（名称/外貌/服装/性格等）生成对应对象类型的正向/反向提示词
async function generatePromptFor(type: PromptTypeKey) {
  promptGen.value = type
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    const result: any = await characterAPI.generatePrompt(characterId, {
      type,
      name: form.name,
      appearance: form.appearance,
      personality: form.personality,
      description: form.description,
      clothing: form.clothing,
      weapons: form.weapons,
      accessories: form.accessories,
      costumes: parseCostumesFromText(form.costumes) || undefined,
      style: form.style || undefined,
    })
    const label = promptTypes.find(t => t.key === type)?.label || ''
    if (result?.prompt) form[promptFieldMap[type]] = result.prompt
    if (result?.negativePrompt) form[negFieldMap[type]] = result.negativePrompt
    alert(`${label}提示词已生成（可微调后直接生成图片）`)
  } catch (e: any) {
    errorMsg.value = '生成提示词失败: ' + e.message
  } finally {
    promptGen.value = null
  }
}

async function generateImage() {
  imgGen.value = true
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    const targetCostume = selectedCostume.value
    // 生成前快照当前目标图 URL：重新生成时旧 URL 仍存在，轮询必须等到 URL 变为新图才算完成
    const beforeUrl = targetCostume
      ? normUrl(variations.value.find((v: any) => v.name === targetCostume)?.imageUrl)
      : normUrl(char.value?.image_url || char.value?.imageUrl || null)
    await characterAPI.generateImage(characterId, undefined, {
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      costume: targetCostume || undefined,
    })
    // 轮询等待结果：变体生成检查 variations 对应项，主图生成检查 imageUrl（须与生成前不同 = 新图落地）
    imgPollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        let done = false
        if (targetCostume) {
          const variation = parseVariations(updated.variations).find((v: any) => v.name === targetCostume)
          const cur = normUrl(variation?.imageUrl)
          done = !!cur && cur !== beforeUrl
        } else {
          const cur = normUrl(updated.imageUrl || updated.image_url)
          done = !!cur && cur !== beforeUrl
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
    // 生成前快照当前三视图 URL：重新生成时旧图仍在，须等到 combined 变为新 URL 才算完成
    const beforeUrl = normUrl(threeViews.value.combined?.imageUrl)
    await characterAPI.generateThreeViews(characterId, undefined, {
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      anchor: anchorMode.value,
      clothing: form.clothing || undefined,
      weapons: form.weapons || undefined,
      accessories: form.accessories || undefined,
    })
    threePollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const views = parseThreeViews(updated.threeViews || updated.three_views)
        const cur = normUrl(views.combined?.imageUrl)
        const done = !!cur && cur !== beforeUrl
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

async function generateEquip(type: EquipTypeKey) {
  equipGen.value = type
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    // 生成前快照当前装备图 URL：重新生成时旧图仍在，须等到该项变为新 URL 才算完成
    const beforeUrl = normUrl(equipImages.value[type]?.imageUrl)
    await characterAPI.generateEquipImage(characterId, undefined, {
      type,
      anchor: anchorMode.value,
      prompt: form[promptFieldMap[type]] || undefined,
      negative_prompt: form[negFieldMap[type]] || undefined,
      model: imageModel.value || undefined,
      // 形象设定最新值覆盖：未保存也能按当前服装/武器/首饰生成三视图
      clothing: form.clothing || undefined,
      weapons: form.weapons || undefined,
      accessories: form.accessories || undefined,
      costumes: parseCostumesFromText(form.costumes) || undefined,
    })
    equipPollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const data = updated.data || updated
        const imgs = parseEquipImages(data.equipImages || data.equip_images)
        const cur = normUrl(imgs[type]?.imageUrl)
        if (cur && cur !== beforeUrl) {
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

// 一键批量生成表情头像特写组：默认全部 9 个；传 keys 可单独重生成某几个
async function generateExpressions(keys?: string[]) {
  const pending = keys && keys.length ? keys : expressionPresets.map(e => e.key)
  for (const k of pending) expressionBusy.value[k] = true
  errorMsg.value = ''
  try {
    const { characterAPI } = await import('~/composables/useApi')
    // 生成前快照每个待生成表情的旧 URL：重新生成时须等到该项变为新 URL 才算完成
    const beforeUrls: Record<string, string> = {}
    for (const k of pending) beforeUrls[k] = normUrl(expressionMap.value[k]?.imageUrl)
    await characterAPI.generateExpressions(characterId, {
      keys: pending,
      // 外观/服装最新值覆盖：未保存也能按当前形象生成表情
      appearance: form.appearance || undefined,
      clothing: form.clothing || undefined,
      costumes: parseCostumesFromText(form.costumes) || undefined,
      model: imageModel.value || undefined,
      anchor: anchorMode.value,
    })
    exprPollRef.value = setInterval(async () => {
      try {
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const map = parseExpressionsRaw(updated.expressions)
        if (pending.every(k => {
          const cur = normUrl(map[k]?.imageUrl)
          return !!cur && cur !== beforeUrls[k]
        })) {
          clearInterval(exprPollRef.value!)
          exprPollRef.value = null
          char.value = updated
          for (const k of pending) expressionBusy.value[k] = false
        }
      } catch (e) {
        console.warn('[Character] expression poll error:', e)
      }
    }, 4000)
    exprPollTimeoutRef.value = setTimeout(() => {
      if (exprPollRef.value) { clearInterval(exprPollRef.value); exprPollRef.value = null }
      exprPollTimeoutRef.value = null
      for (const k of pending) expressionBusy.value[k] = false
    }, 300000)
  } catch (e: any) {
    errorMsg.value = e.message
    for (const k of pending) expressionBusy.value[k] = false
  }
}

// 通用轮询：GET 角色并刷新 char.value，直到 isDone(最新数据) 为真；超时返回 timeout
function waitImageDone(isDone: (updated: any) => boolean, timeoutMs: number): Promise<'ok' | 'timeout'> {
  return new Promise(resolve => {
    let settled = false
    if (activePollRef.value) { clearInterval(activePollRef.value); activePollRef.value = null }
    activePollRef.value = setInterval(async () => {
      if (settled) return
      try {
        const { characterAPI } = await import('~/composables/useApi')
        const updated: any = await characterAPI.get(characterId)
        if (!updated) return
        const data = updated.data || updated
        if (data) char.value = data
        if (isDone(data)) {
          settled = true
          clearInterval(activePollRef.value!)
          activePollRef.value = null
          resolve('ok')
        }
      } catch (e) {
        console.warn('[Character] batch poll error:', e)
      }
    }, 4000)
    setTimeout(() => {
      if (settled) return
      settled = true
      if (activePollRef.value) { clearInterval(activePollRef.value); activePollRef.value = null }
      resolve('timeout')
    }, timeoutMs)
  })
}

// 一键生成全部素材：缺啥补啥 + 覆盖已有——点击即把全套（全身三视图 / 服装 / 武器 / 首饰三视图 / 表情）重新生成一遍。
// 顺序依赖：先确保全身三视图落地（作为装备三视图/表情的锚定基准），再并行生成装备三视图与表情。
async function generateAllAssets() {
  if (batchBusy.value) return
  errorMsg.value = ''
  const { characterAPI } = await import('~/composables/useApi')
  // 覆盖已有：已生成的素材也整体重做，保证与最新设定/锚定基准一致
  const needEquips = equipTypes
  const needExprKeys = expressionPresets.map(e => e.key)
  const total = 1 + needEquips.length + 1 // 三视图(1) + 装备三视图 + 表情(1)
  batchBusy.value = true
  batchProgress.value = 0
  batchTotal.value = total
  batchStatus.value = ''
  try {
    const failed: string[] = []
    // ① 全身三视图：先生成并等待落地（后续装备/表情会以其为视觉锚定）
    const threeLabel = '全身三视图'
    batchStatus.value = `正在生成：${threeLabel}（${batchProgress.value + 1}/${total}）`
    const beforeUrl = normUrl(threeViews.value.combined?.imageUrl)
    await characterAPI.generateThreeViews(characterId, undefined, {
      prompt: form.customPrompt || undefined,
      negative_prompt: form.negativePrompt || undefined,
      model: imageModel.value || undefined,
      anchor: anchorMode.value,
      clothing: form.clothing || undefined,
      weapons: form.weapons || undefined,
      accessories: form.accessories || undefined,
    })
    const threeR = await waitImageDone((data: any) => {
      const views = parseThreeViews(data.threeViews || data.three_views)
      const cur = normUrl(views.combined?.imageUrl)
      return !!cur && cur !== beforeUrl
    }, 240000)
    batchProgress.value++
    if (threeR === 'timeout') failed.push(threeLabel)
    // ② 并行生成：装备三视图 + 表情头像（与刚生成的三视图保持同造型）
    const equipBefore: Record<string, string> = {}
    for (const t of needEquips) equipBefore[t.key] = normUrl(equipImages.value[t.key]?.imageUrl)
    const exprBefore: Record<string, string> = {}
    for (const k of needExprKeys) exprBefore[k] = normUrl(expressionMap.value[k]?.imageUrl)
    const subLabels: string[] = [
      ...needEquips.map(t => `${t.label}三视图`),
      `${needExprKeys.length} 个表情`,
    ]
    batchStatus.value = `正在生成：${subLabels.join('、')}（${batchProgress.value + 1}/${total}）`
    const submits: Promise<unknown>[] = []
    for (const t of needEquips) {
      submits.push(characterAPI.generateEquipImage(characterId, undefined, {
        type: t.key,
        anchor: anchorMode.value,
        prompt: form[promptFieldMap[t.key]] || undefined,
        negative_prompt: form[negFieldMap[t.key]] || undefined,
        model: imageModel.value || undefined,
        clothing: form.clothing || undefined,
        weapons: form.weapons || undefined,
        accessories: form.accessories || undefined,
        costumes: parseCostumesFromText(form.costumes) || undefined,
      }))
    }
    submits.push(characterAPI.generateExpressions(characterId, {
      keys: needExprKeys,
      appearance: form.appearance || undefined,
      clothing: form.clothing || undefined,
      costumes: parseCostumesFromText(form.costumes) || undefined,
      model: imageModel.value || undefined,
      anchor: anchorMode.value,
    }))
    await Promise.all(submits)
    const r = await waitImageDone((data: any) => {
      const imgs = parseEquipImages(data.equipImages || data.equip_images)
      const equipOk = needEquips.every(t => {
        const cur = normUrl(imgs[t.key]?.imageUrl)
        return !!cur && cur !== equipBefore[t.key]
      })
      const exprMap = parseExpressionsRaw(data.expressions)
      const exprOk = needExprKeys.every(k => {
        const cur = normUrl(exprMap[k]?.imageUrl)
        return !!cur && cur !== exprBefore[k]
      })
      return equipOk && exprOk
    }, 420000)
    batchProgress.value = total
    if (r === 'timeout') failed.push(subLabels.join('、'))
    if (failed.length) {
      alert(`以下素材可能生成超时（可在对应模块重新生成）：${failed.join('、')}`)
    } else {
      alert('一键生成完成，全套素材已重新生成并自动刷新')
    }
  } catch (e: any) {
    errorMsg.value = '一键生成失败: ' + e.message
  } finally {
    batchBusy.value = false
    batchStatus.value = ''
    batchProgress.value = 0
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
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 24px 32px;
  max-width: 1200px;
  margin: 0 auto;
}
.image-card {
  border-radius: 14px;
  overflow: hidden;
  background: #f8f9fc;
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
.btn-block { width: 100%; justify-content: center; }
.btn-sm { padding: 7px 12px; font-size: 12px; font-weight: 600; border-radius: 8px; }
.btn-primary { background: linear-gradient(135deg, #0d9488, #0f766e); color: #fff; border-color: transparent; display: inline-flex; align-items: center; gap: 6px; }
.btn-primary:hover { filter: brightness(1.05); box-shadow: 0 4px 14px rgba(13,148,136,0.25); }
.btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }

/* 图片生成：每个模块 = 提示词 + 生成按钮 + 对应图片，放在一起 */
.gen-section { padding-top: 20px; }
.gen-toolbar { display: flex; align-items: center; gap: 10px; }
.btn-batch {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, #10b981, #059669);
  border: none; color: #fff; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 8px; cursor: pointer;
  box-shadow: 0 2px 8px rgba(16,185,129,0.25);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.btn-batch:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(16,185,129,0.35); }
.btn-batch:disabled { opacity: 0.6; cursor: not-allowed; box-shadow: none; }
.btn-batch .batch-ok-icon {
  width: 14px; height: 14px; border-radius: 50%;
  background: #fff; position: relative; flex: none;
}
.btn-batch .batch-ok-icon::after {
  content: ''; position: absolute; left: 4px; top: 2px;
  width: 4px; height: 8px; border: solid #059669; border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}
.gen-batch-status {
  display: flex; align-items: center; gap: 10px;
  margin: 10px 0 2px; padding: 9px 14px;
  background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
  border-radius: 10px; color: #059669; font-size: 12px;
}
.gen-batch-meta { margin-left: auto; font-weight: 600; flex: none; }
.gen-batch-status .spinner { border-top-color: #059669; }
.gen-module {
  border: 1px solid rgba(100,120,180,0.12);
  border-radius: 14px;
  background: #fff;
  padding: 16px;
  margin-top: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.gen-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.gen-title { font-size: 13px; font-weight: 700; color: var(--text-0); display: flex; align-items: center; gap: 6px; }
.gen-body { display: grid; grid-template-columns: 280px 1fr; gap: 18px; }
.gen-preview { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.gen-prompts { display: flex; flex-direction: column; min-width: 0; }
.gen-actions { margin-top: 12px; }
.gen-media { border-radius: 12px; overflow: hidden; background: #f3f4f6; border: 1px solid rgba(100,120,180,0.1); }
/* 三视图/设定图按图片实际比例完整展示，不再固定 16:9 裁剪左右视角 */
.gen-combined img { width: 100%; height: auto; max-height: 60vh; object-fit: contain; display: block; background: #111827; }
/* 全宽预览：三视图模块图片占满整行，保证三个视角清晰可见 */
.gen-module-wide .gen-body { grid-template-columns: 1fr; }
.gen-module-wide .gen-prompts { margin-top: 14px; }
.gen-empty {
  aspect-ratio: 16/9;
  max-width: 720px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 12px;
  text-align: center;
  line-height: 1.8;
  background: linear-gradient(135deg, #f8f9fc 0%, #eef1f7 100%);
}
@media (max-width: 760px) {
  .gen-body { grid-template-columns: 1fr; }
  .detail-body { padding: 16px; }
}

/* 图片生成单列布局：每个生成模块各占一行（全宽），从上到下 = 角色立绘 → 全身三视图 → 表情头像 → 服装/武器/首饰 */
.gen-layout { display: block; }
.gen-layout .col-main, .gen-layout .col-side { min-width: 0; display: flex; flex-direction: column; }
.gen-layout .col-main > .gen-module:first-child { margin-top: 0; }

/* 表情头像特写组（独占一行，自适应网格，批量生成独立表情特写） */
.gen-module-expressions { padding: 14px; }
.expressions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.expression-item { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.expression-img {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: 10px;
  overflow: hidden;
  background: linear-gradient(135deg, #f8f9fc 0%, #eef1f7 100%);
  border: 1px solid rgba(100,120,180,0.1);
}
.expression-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.expression-empty {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: #9ca3af; font-size: 11px;
}
.expression-spin {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  background: rgba(15,23,42,0.35);
}
.expression-regen {
  position: absolute; right: 5px; bottom: 5px;
  font-size: 10px; line-height: 1; padding: 4px 7px;
  border: none; border-radius: 6px; cursor: pointer;
  color: #fff; background: rgba(15,23,42,0.55);
  transition: background .15s;
}
.expression-regen:hover { background: rgba(13,148,136,0.85); }
.expression-label { text-align: center; font-size: 11px; color: rgba(40,50,80,0.72); white-space: nowrap; }

/* 服装/武器/首饰三视图：各占一行（全宽），依次排：服装 → 武器 → 首饰 */
.equip-module { padding: 14px; }
.equip-module .gen-head { align-items: flex-start; flex-wrap: wrap; row-gap: 10px; }
.equip-module .gen-preview { align-items: center; }
.equip-module .gen-media img { width: auto; max-width: 100%; max-height: 40vh; margin: 0 auto; }
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

/* 分区头部：标题 + 操作按钮 */
.sec-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.sec-head h3 { margin: 0; }

/* 正/反向提示词分色区分 */
.plabel { display: inline-flex; align-items: center; gap: 6px; }
.plabel::before { content: ''; width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.plabel-pos::before { background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.15); }
.plabel-neg::before { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.15); }
.pos-field textarea { border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.03); }
.pos-field textarea:focus { border-color: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.1); }
.neg-field textarea { border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.03); }
.neg-field textarea:focus { border-color: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.1); }

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
/* ===== 视觉锚定 & 时代背景条（角色图生成） ===== */
.gen-anchorbar { display: flex; gap: 12px; align-items: stretch; margin: 14px 0 6px; padding: 10px 12px; border: 1px dashed #3d3d4d; border-radius: 10px; background: rgba(255,255,255,0.02); }
.anchor-thumb { width: 64px; min-height: 64px; flex: 0 0 auto; border-radius: 8px; overflow: hidden; border: 1px solid #3a3a46; background: #1a1a22; display: flex; align-items: center; justify-content: center; }
.anchor-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.anchor-thumb-empty { color: #777; font-size: 11px; text-align: center; line-height: 1.5; }
.anchor-info { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
.anchor-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.anchor-label { font-size: 12px; color: #aab; flex: 0 0 auto; }
.anchor-hint { font-size: 12px; color: #88889a; }
.anchor-select { width: auto; min-width: 158px; padding: 3px 8px; font-size: 12px; height: auto; }
.anchor-era-tag { font-size: 11px; padding: 2px 10px; border-radius: 20px; background: linear-gradient(135deg, rgba(124,92,255,0.28), rgba(0,188,255,0.18)); color: #d6ccff; border: 1px solid rgba(124,92,255,0.4); white-space: nowrap; }
.anchor-era-tag-dim { background: rgba(255,255,255,0.05); color: #999; border-color: rgba(255,255,255,0.12); }
.anchor-era-text { font-size: 12px; color: #9c9cac; line-height: 1.5; }

</style>

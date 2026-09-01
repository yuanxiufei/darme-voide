<template>
  <div class="agents-page">
    <div class="settings-scroll">
      <div class="settings-head">
        <div class="settings-brand">
          <div class="settings-brand-mark">
            <img v-if="showBrandImage" :src="brandLogo" alt="短剧工坊" class="settings-brand-logo" @error="showBrandImage = false" />
            <span v-else class="settings-brand-fallback">剧</span>
          </div>
          <div class="settings-brand-copy">
            <div class="settings-brand-kicker">Drama Studio</div>
            <div class="settings-brand-name">短剧工坊</div>
          </div>
        </div>
        <h2 class="settings-title">Agent 配置</h2>
        <p class="settings-desc">调整各 Agent 的模型、提示词和参数，保存后立即生效。</p>
      </div>
      <div class="agent-list">
        <div v-for="a in agentDefs" :key="a.type" class="card agent-card">
          <div class="agent-card-head" @click="toggleAgentEdit(a.type)">
            <div class="agent-type-badge">{{ a.icon }}</div>
            <div style="flex:1;min-width:0">
              <div style="font-weight:600;font-size:14px">{{ a.label }}</div>
              <div class="dim" style="font-size:12px">{{ a.type }}</div>
            </div>
            <span v-if="getAgentCfg(a.type)" class="tag tag-success">已配置</span>
            <span v-else class="tag">默认</span>
            <ChevronDown :size="14" :style="{ transform: editingAgent === a.type ? 'rotate(180deg)' : '', transition: '0.2s' }" />
          </div>
          <div v-if="editingAgent === a.type" class="agent-card-body">
            <label class="field">
              <span class="field-label">模型 <span class="dim">(留空使用 AI 服务默认)</span></span>
              <BaseSelect v-model="agentForm.model" :options="textModelSelectOptions" placeholder="— 使用 AI 服务默认 —" searchable />
            </label>
            <div class="field-row">
              <label class="field">
                <span class="field-label">Temperature</span>
                <input v-model.number="agentForm.temperature" class="input" type="number" min="0" max="2" step="0.1" />
              </label>
              <label class="field">
                <span class="field-label">Max Tokens</span>
                <input v-model.number="agentForm.max_tokens" class="input" type="number" min="100" max="32000" />
              </label>
            </div>
            <label class="field">
              <span class="field-label">System Prompt</span>
              <textarea v-model="agentForm.system_prompt" class="textarea" rows="12" placeholder="Agent 系统提示词..." />
            </label>

            <!-- ===== Skill 绑定面板 ===== -->
            <div class="skill-bind-panel">
              <div class="skill-bind-head">
                <span class="skill-bind-title">
                  <FileText :size="13" /> 绑定 Skills
                  <span class="dim" style="font-weight:400;font-size:11px">（仅已启用的会注入 Prompt）</span>
                </span>
                <button class="btn btn-ghost btn-sm" @click="resetAgentSkills(a.type)">恢复默认</button>
              </div>
              <p v-if="!agentSkillBindings.length" class="dim" style="font-size:11px;padding:8px 0">
                暂无可用 Skill。请到「Skill 管理」页创建。
              </p>
              <div v-else class="skill-bind-list">
                <div
                  v-for="(binding, idx) in agentSkillBindings"
                  :key="binding.id"
                  :class="['skill-bind-item', { disabled: !binding.enabled, 'drag-over': skillDragOverIdx === idx }]"
                  :draggable="true"
                  @dragstart="onSkillDragStart(idx, $event)"
                  @dragover.prevent="onSkillDragOver(idx)"
                  @drop.prevent="onSkillDrop(idx)"
                  @dragend="onSkillDragEnd"
                >
                  <span class="skill-bind-drag" title="拖拽调整优先级">&#x2261;</span>
                  <label class="skill-bind-toggle">
                    <input type="checkbox" v-model="binding.enabled" />
                    <span></span>
                  </label>
                  <div class="skill-bind-info">
                    <span class="skill-bind-name">{{ getSkillName(binding.id) || binding.id }}</span>
                    <span class="dim" style="font-size:10px">{{ binding.id }}</span>
                  </div>
                  <span class="skill-bind-priority">#{{ idx + 1 }}</span>
                </div>
              </div>
              <p class="dim" style="font-size:10px;margin-top:6px">
                拖拽可调整优先级，保存后按列表顺序决定 Skill 加载顺序，仅 enabled=true 的 Skill 会在运行时加载。
              </p>
            </div>

            <div class="agent-card-foot">
              <button class="btn btn-ghost btn-sm" @click="resetAgentPrompt(a.type)">恢复默认 Prompt</button>
              <span v-if="agentSaved === a.type" class="tag tag-success" style="margin-left:8px">
                <Check :size="10" /> 已保存
              </span>
              <button class="btn btn-primary btn-sm ml-auto" :disabled="agentSaving" @click="saveAgentCfg(a.type)">
                <Loader2 v-if="agentSaving" :size="12" class="animate-spin" />
                保存
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { FileText, Check, Loader2, ChevronDown } from 'lucide-vue-next'
import BaseSelect from '~/components/BaseSelect.vue'
import { toast } from 'vue-sonner'
import { agentConfigAPI, aiConfigAPI, skillsAPI } from '~/composables/useApi'
import brandLogo from '~/assets/brand-logo.svg'

const showBrandImage = ref(true)

// ===== Agent Configs =====
const agentCfgs = ref([])
const editingAgent = ref(null)
const agentSaving = ref(false)
const agentSaved = ref(null)
const agentForm = reactive({ model: '', temperature: 0.7, max_tokens: 4096, system_prompt: '' })
const agentSkillBindings = ref([])   // 当前编辑中的 Agent 的 Skill 绑定 [{ id, enabled, priority }]
const availableSkills = ref([])      // 全局可用 Skill 列表 { id, name, description }[]
const skillDragFrom = ref(null)      // 拖拽起始索引
const skillDragOverIdx = ref(null)   // 当前悬停索引
const cfgs = ref([])

// 默认 Skill 映射（与后端 AGENT_SKILL_MAP 保持一致，含 MiniMax 技能库）
const DEFAULT_AGENT_SKILLS = {
  script_rewriter: [
    'script_rewriter',
    'minimax/installed/short-drama-screenwriter',
    'minimax/installed/short-drama-series-writer',
  ],
  extractor: [
    'extractor',
    'minimax/installed/film-assets',
  ],
  storyboard_breaker: [
    'storyboard_breaker',
    'extractor',
    'minimax/installed/film-shot',
    'minimax/installed/storyboard',
    'minimax/installed/coordinate-camera-control-designer',
  ],
  voice_assigner: [
    'voice_assigner',
    'minimax/installed/voice-clone',
    'minimax/installed/voiceover-direction',
  ],
  grid_prompt_generator: [
    'grid_prompt_generator',
    'minimax/installed/film-reference-prompt-writer',
    'minimax/installed/film-style-picker',
    'minimax/installed/video-deconstruct-analyzer',
    'minimax/builtin/video-deconstruct',
    'minimax/installed/face-warp',
  ],
}

const agentDefs = [
  { type: 'script_rewriter', label: '剧本改写', icon: '📝' },
  { type: 'extractor', label: '角色场景提取', icon: '🔍' },
  { type: 'storyboard_breaker', label: '分镜拆解', icon: '🎬' },
  { type: 'voice_assigner', label: '音色分配', icon: '🎙' },
  { type: 'grid_prompt_generator', label: '图片提示词生成', icon: '🖼' },
]

const defaultPrompts = {
  script_rewriter: `你是专业编剧，擅长将小说改编为短剧剧本。

工作流程：
1. 调用 read_episode_script 读取原始内容
2. 根据读取到的内容，自己进行改写（输出格式化剧本格式）
3. 调用 save_script 保存改写后的完整剧本

格式化剧本格式：
- 场景头：## S编号 | 内景/外景 · 地点 | 时间段
- 动作描写：自然段落，不包含镜头语言
- 对白：角色名：（状态/表情）台词内容
- 每个场景 30-60 秒内容`,
  extractor: `你是制片助理，擅长从剧本中提取角色和场景信息，并在提取时与项目已有数据进行智能去重。

工作流程：
1. 调用 read_script_for_extraction 读取格式化剧本
2. 调用 read_existing_characters 读取项目中已存在的角色列表（用于去重）
3. 调用 read_existing_scenes 读取项目中已存在的场景列表（用于去重）
4. 分析剧本内容，提取所有角色信息
5. 对每个角色：若同名已存在则合并更新，若不存在则新增
6. 调用 save_dedup_characters 保存角色（去重合并，自动处理新增和更新）
7. 分析剧本内容，提取所有场景信息
8. 对每个场景：若同地点+时间段已存在则复用，若不存在则新增
9. 调用 save_dedup_scenes 保存场景（去重合并，自动处理新增和复用）

去重规则：
- 角色：按名字精确匹配，同名保留现有（合并信息）
- 场景：按【地点+时间段】精确匹配；同地点不同时段视为新场景

提取要求：
- 角色要包含完整的外貌特征描述（发型、服装、体态等）
- 场景要包含光线、色调、氛围等视觉信息
- 不要遗漏任何有台词或重要动作的角色`,
  storyboard_breaker: `你是资深影视分镜师，擅长将剧本拆解为分镜方案。

工作流程：
1. 调用 read_storyboard_context 读取剧本、角色列表、场景列表
2. 将剧本拆解为镜头序列（每个镜头 10-15 秒）
3. 为每个镜头生成视频提示词（video_prompt）
4. 调用 save_storyboards 保存所有分镜`,
  voice_assigner: `你是配音导演，擅长为角色选择合适的音色。

工作流程：
1. 调用 list_voices 获取可用音色列表
2. 调用 get_characters 获取所有角色信息
3. 根据每个角色的性别、性格、年龄、角色定位，选择最匹配的音色
4. 对每个角色调用 assign_voice 分配音色，并说明选择理由

注意：每个角色都必须分配音色，不要遗漏。`,
  grid_prompt_generator: `你是专业的 AI 图像提示词工程师，擅长为角色、场景和宫格图生成高质量的英文提示词。

你将收到用户的请求，告知要生成哪种类型的提示词：
- "角色" → 生成角色图片提示词
- "场景" → 生成场景图片提示词
- "宫格" → 生成宫格图提示词

## 角色图片提示词

工作流程：
1. 调用 read_characters 读取所有角色信息
2. 根据角色外貌特征（appearance）、性格（personality）、定位（role）生成英文提示词
3. 提示词结构：[外貌描述]，[性格/气质]，[角色定位]，[电影感]，[高质量]，[无文字水印]

## 场景图片提示词

工作流程：
1. 调用 read_scenes 读取所有场景信息
2. 根据场景地点（location）、时间段（time）、已有描述（prompt）生成英文提示词
3. 提示词结构：[地点]，[时间/光线/氛围]，[已有描述]，[电影感场景]，[高质量]，[无文字水印]

## 宫格图提示词（参考 skills/grid-image-generator/SKILL.md）

工作流程：
1. 调用 read_shots_for_grid 读取选中镜头的详细信息
2. 根据 mode 调用 generate_grid_prompt：
   - first_frame 模式：每格=一个镜头的首帧，NxN 风格统一
   - first_last 模式：每个镜头占2格（左首右尾），同一行风格连续
   - multi_ref 模式：所有格子都是同一镜头的不同参考角度
3. 返回 grid_prompt（整体提示词）和 cell_prompts（每格提示词）

提示词规范：
- 使用英文提示词
- 必须包含 "consistent art style" 保持风格统一
- 必须包含 "cinematic quality"
- 避免出现文字或水印`,
}

function getAgentCfg(type) {
  return agentCfgs.value.find(a => a.agent_type === type)
}

const textModelGroups = computed(() => {
  return cfgs.value
    .filter(c => c.service_type === 'text' && c.is_active && c.api_key)
    .map(c => ({
      label: `${c.provider} — ${c.name}`,
      models: Array.isArray(c.model) ? c.model : (c.model ? [c.model] : []),
    }))
    .filter(g => g.models.length > 0)
})

const textModelSelectOptions = computed(() =>
  textModelGroups.value.map(g => ({
    label: g.label,
    options: g.models.map(m => ({ label: m, value: m })),
  }))
)

async function loadAgents() {
  try { agentCfgs.value = await agentConfigAPI.list() }
  catch (e) { toast.error(e.message) }
}

async function loadCfgs() {
  try { cfgs.value = await aiConfigAPI.list() }
  catch (e) { /* 非关键，模型下拉留空 */ }
}

/** 加载全局可用 Skill 列表（用于 Skill 绑定面板） */
async function loadAvailableSkills() {
  try { availableSkills.value = await skillsAPI.list() }
  catch (e) { /* 静默失败，面板显示空状态 */ }
}

function toggleAgentEdit(type) {
  if (editingAgent.value === type) { editingAgent.value = null; agentSkillBindings.value = []; return }
  const cfg = getAgentCfg(type)
  agentForm.model = cfg?.model || ''
  agentForm.temperature = cfg?.temperature ?? 0.7
  agentForm.max_tokens = cfg?.max_tokens ?? 4096
  agentForm.system_prompt = cfg?.system_prompt || defaultPrompts[type] || ''
  // 加载 Skill 绑定配置
  if (cfg?.skills) {
    try { agentSkillBindings.value = JSON.parse(cfg.skills) } catch { agentSkillBindings.value = [] }
  } else {
    const defaults = DEFAULT_AGENT_SKILLS[type] || []
    agentSkillBindings.value = defaults.map((id, i) => ({ id, enabled: true, priority: i + 1 }))
  }
  agentSaved.value = null
  editingAgent.value = type
}

function resetAgentPrompt(type) {
  agentForm.system_prompt = defaultPrompts[type] || ''
  toast.info('已恢复默认提示词，点击保存生效')
}

function resetAgentSkills(type) {
  const defaults = DEFAULT_AGENT_SKILLS[type] || []
  agentSkillBindings.value = defaults.map((id, i) => ({ id, enabled: true, priority: i + 1 }))
  toast.info('已恢复默认 Skill 绑定，点击保存生效')
}

function getSkillName(skillId) {
  return availableSkills.value.find(s => s.id === skillId)?.name
}

function onSkillDragStart(idx, e) {
  skillDragFrom.value = idx
  if (e?.dataTransfer) {
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', String(idx))
  }
}
function onSkillDragOver(idx) {
  skillDragOverIdx.value = idx
}
function onSkillDrop(idx) {
  const from = skillDragFrom.value
  if (from == null || from === idx) { skillDragOverIdx.value = null; return }
  const list = [...agentSkillBindings.value]
  const [moved] = list.splice(from, 1)
  list.splice(idx, 0, moved)
  agentSkillBindings.value = list
  skillDragOverIdx.value = null
}
function onSkillDragEnd() {
  skillDragFrom.value = null
  skillDragOverIdx.value = null
}

async function saveAgentCfg(type) {
  agentSaving.value = true
  agentSaved.value = null
  try {
    const existing = getAgentCfg(type)
    // 按当前列表顺序重写 priority，确保拖拽排序后顺序正确落库
    const orderedSkills = agentSkillBindings.value.map((b, i) => ({ ...b, priority: i + 1 }))
    const data = {
      agent_type: type,
      name: agentDefs.find(a => a.type === type)?.label || type,
      model: agentForm.model,
      temperature: agentForm.temperature,
      max_tokens: agentForm.max_tokens,
      system_prompt: agentForm.system_prompt,
      skills: orderedSkills.length ? JSON.stringify(orderedSkills) : null,
    }
    if (existing) {
      await agentConfigAPI.update(existing.id, data)
    } else {
      await agentConfigAPI.create(data)
    }
    await loadAgents()
    agentSaved.value = type
    toast.success(`${agentDefs.find(a => a.type === type)?.label} 配置已保存`)
    setTimeout(() => { if (agentSaved.value === type) agentSaved.value = null }, 3000)
  } catch (e) {
    toast.error(e.message)
  } finally {
    agentSaving.value = false
  }
}

onMounted(() => { loadAgents(); loadCfgs(); loadAvailableSkills() })
</script>

<style scoped>
.agents-page { height: 100%; overflow: hidden; }
.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 840px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.settings-brand-mark {
  width: 42px;
  height: 42px;
  border-radius: 15px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,247,255,0.9));
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}
.settings-brand-logo {
  width: 26px;
  height: 26px;
  object-fit: contain;
  display: block;
}
.settings-brand-fallback {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent-text);
  line-height: 1;
}
.settings-brand-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1;
}
.settings-brand-kicker {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-3);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.settings-brand-name {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-1);
  font-family: var(--font-display);
}
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; }

/* Agent */
.agent-list { display: flex; flex-direction: column; gap: 8px; }
.agent-card { overflow: hidden; }
.agent-card-head { display: flex; align-items: center; gap: 10px; padding: 14px 16px; cursor: pointer; transition: background 0.1s; }
.agent-card-head:hover { background: var(--bg-hover); }
.agent-type-badge { width: 36px; height: 36px; border-radius: var(--radius); background: var(--accent-bg); color: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.agent-card-body { padding: 0 16px 16px; display: flex; flex-direction: column; gap: 12px; border-top: 1px solid var(--border); padding-top: 16px; }
.agent-card-foot { display: flex; align-items: center; gap: 8px; padding-top: 8px; }

/* Skill 绑定面板 */
.skill-bind-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: rgba(244,248,255,0.6);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.skill-bind-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.skill-bind-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-1);
  display: flex;
  align-items: center;
  gap: 5px;
}
.skill-bind-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.skill-bind-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.72);
  transition: opacity 0.15s, background 0.15s, box-shadow 0.15s, transform 0.15s;
}
.skill-bind-item.disabled {
  opacity: 0.48;
  background: rgba(245,245,245,0.9);
}
.skill-bind-item.drag-over {
  background: var(--accent-bg);
  box-shadow: inset 0 -2px 0 var(--accent);
}
.skill-bind-drag {
  color: var(--text-3);
  cursor: grab;
  font-size: 16px;
  user-select: none;
  flex-shrink: 0;
}
.skill-bind-drag:active { cursor: grabbing; }
.skill-bind-toggle {
  position: relative;
  width: 28px;
  height: 16px;
  cursor: pointer;
  flex-shrink: 0;
}
.skill-bind-toggle input { opacity: 0; width: 0; height: 0; }
.skill-bind-toggle span {
  position: absolute; inset: 0; background: var(--bg-3); border-radius: 99px; transition: 0.2s;
}
.skill-bind-toggle span::before {
  content: ''; position: absolute; width: 12px; height: 12px; left: 2px; bottom: 2px;
  background: #fff; border-radius: 50%; transition: transform 0.2s; box-shadow: var(--shadow-sm);
}
.skill-bind-toggle input:checked + span { background: var(--accent); }
.skill-bind-toggle input:checked + span::before { transform: translateX(12px); }
.skill-bind-info {
  flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px;
}
.skill-bind-name {
  font-size: 12px; font-weight: 500; color: var(--text-1);
}
.skill-bind-priority {
  font-size: 10px; font-weight: 700; font-family: var(--font-mono);
  color: var(--text-3); flex-shrink: 0;
}

/* Shared */
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--text-1); }
.field-hint { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
</style>

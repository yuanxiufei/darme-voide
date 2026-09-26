<template>
  <div class="agents-page">
    <div class="settings-scroll">
      <div class="settings-head">
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
                <span class="skill-bind-head-right">
                  <span
                    v-if="skillCharBudget"
                    :class="['skill-bind-usage', { over: isBindOverBudget }]"
                    title="已启用绑定的注入体量合计 / 注入预算（仅 enabled 的 Skill 会计入）"
                  >{{ enabledSkillCount }} 项 · {{ fmtChars(boundChars) }} / {{ fmtChars(skillCharBudget) }} 字符</span>
                  <button class="btn btn-ghost btn-sm" @click="showSkillPicker ? closeSkillPicker() : openSkillPicker()">
                    <Plus :size="12" /> 添加
                  </button>
                  <button class="btn btn-ghost btn-sm" @click="resetAgentSkills(a.type)">恢复默认</button>
                </span>
              </div>

              <!-- 添加绑定：外部库 Skill 无 frontmatter 默认绑定、只能手动挂（skills.vue 也如此指引），
                   而此前本面板只有「恢复默认」、无任何添加控件 → 用户被指到死路。此处为唯一 UI 挂载入口 -->
              <div v-if="showSkillPicker" class="skill-picker">
                <div class="skill-picker-head">
                  <Search :size="12" />
                  <input
                    v-model="skillPickerQuery"
                    class="skill-picker-input"
                    placeholder="搜索 Skill（id / 名称 / 描述）"
                  />
                  <button class="btn btn-ghost btn-sm" @click="closeSkillPicker">完成</button>
                </div>
                <p v-if="!skillPickerCandidates.length" class="dim" style="font-size:11px;padding:8px">
                  {{ skillPickerQuery ? '没有匹配的 Skill' : '所有可用 Skill 都已绑定' }}
                </p>
                <div v-else class="skill-picker-list">
                  <button
                    v-for="s in skillPickerCandidates"
                    :key="s.id"
                    type="button"
                    class="skill-picker-item"
                    @click="addSkillBinding(s.id)"
                  >
                    <span class="skill-picker-main">
                      <span class="skill-picker-name">
                        {{ s.name || s.id }}
                        <span v-if="s.category === 'vendor'" class="skill-bind-tag">{{ s.sourceLabel || '外部库' }}</span>
                        <span
                          v-if="s.missingTools?.length"
                          class="skill-bind-tag warn"
                          :title="`依赖本项目未提供的工具：${s.missingTools.join(', ')}`"
                        >缺 {{ s.missingTools.length }} 工具</span>
                      </span>
                      <span class="dim skill-picker-id">{{ s.id }}</span>
                    </span>
                    <span class="skill-bind-chars">{{ fmtChars(s.charCount ?? 0) }}</span>
                    <Plus :size="12" />
                  </button>
                </div>
              </div>

              <p v-if="!agentSkillBindings.length" class="dim" style="font-size:11px;padding:8px 0">
                {{ availableSkills.length
                  ? '尚未绑定任何 Skill，点击上方「添加」从可用列表中选择。'
                  : '暂无可用 Skill，请到「Skill 管理」页创建。' }}
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
                    <span class="skill-bind-name">
                      {{ getSkillName(binding.id) || binding.id }}
                      <span v-if="vendorLabel(binding.id)" class="skill-bind-tag">{{ vendorLabel(binding.id) }}</span>
                      <span v-if="skillMissing(binding.id)" class="skill-bind-tag danger" title="该 Skill 已从磁盘移除，加载时会被跳过，请关闭开关或点右侧 ✕ 移除">已失效</span>
                      <span
                        v-if="missingToolsTip(binding.id)"
                        class="skill-bind-tag warn"
                        :title="missingToolsTip(binding.id)"
                      >缺 {{ skillMissingTools(binding.id).length }} 个工具</span>
                    </span>
                    <span class="dim" style="font-size:10px">{{ binding.id }}</span>
                  </div>
                  <span class="skill-bind-chars" title="该 Skill 的注入体量（字符，含前置契约与协议字段段）">{{ fmtChars(getSkillChars(binding.id)) }}</span>
                  <span class="skill-bind-priority">#{{ idx + 1 }}</span>
                  <button
                    class="skill-bind-remove"
                    title="移除该绑定（保存后生效）"
                    @click.stop="removeSkillBinding(idx)"
                  ><X :size="12" /></button>
                </div>
              </div>
              <p v-if="isBindOverBudget" class="skill-bind-warn">
                已启用绑定合计已超出注入预算，超出的 Skill 会按列表顺序被跳过（即不生效）。请关闭部分开关，或拖拽降低大 Skill 的优先级。
              </p>
              <p class="dim" style="font-size:10px;margin-top:6px">
                拖拽调整顺序，保存后按此顺序注入（超预算时末尾的会被跳过）；仅勾选的 Skill 会注入。
                默认绑定来自各 SKILL.md 的 frontmatter 声明，外部库 Skill 需在此手动添加。
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
import { FileText, Check, Loader2, ChevronDown, Plus, Search, X } from 'lucide-vue-next'
import BaseSelect from '~/components/BaseSelect.vue'
import { toast } from 'vue-sonner'
import { agentConfigAPI, aiConfigAPI, skillsAPI } from '~/composables/useApi'

// ⚠️⚠️ 2026-09-21：这里所有 `ref([])` 都改成**显式泛型** ✗ ——
//      `ref([])` 推出来是 **`never[]`** ✗（元素 `never` ⇒ 下面每个属性访问都报 TS2339 ✓✗），
//      一趟类型检查能冒出来近百条 ✓。形状取自 `~contracts`（**后端是权威** ✓，本文件是镜像 ✓）。
import type { AgentConfigVO, AiServiceConfigVO, SkillBinding, SkillVO } from '~contracts'

// ===== Agent Configs =====
const agentCfgs = ref<AgentConfigVO[]>([])
const editingAgent = ref<string | null>(null)
const agentSaving = ref(false)
const agentSaved = ref<string | null>(null)
const agentForm = reactive({ model: '', temperature: 0.7, max_tokens: 4096, system_prompt: '' })
const agentSkillBindings = ref<SkillBinding[]>([])   // 当前编辑中的 Agent 的 Skill 绑定
const availableSkills = ref<SkillVO[]>([])           // 全局可用 Skill 列表
const skillCharBudget = ref(0)       // 注入体量硬闸（来自 /skills/meta，超出的 Skill 会被静默跳过）
const skillDragFrom = ref<number | null>(null)   // 拖拽起始索引
const skillDragOverIdx = ref<number | null>(null)   // 当前悬停索引
const showSkillPicker = ref(false)   // 「添加 Skill」候选面板是否展开
const skillPickerQuery = ref('')     // 候选面板搜索词（外部库近 30 个，必须可搜）
const cfgs = ref<AiServiceConfigVO[]>([])

// 出厂默认配置由后端下发（GET /agent-configs/defaults），单一事实来源：
//   提示词 = backend-py/app/services/agent_prompts.py 的 DEFAULT_PROMPTS
//   Skill 绑定 = 各 SKILL.md 的 frontmatter `agents:`（backend-py/app/agent/skills.py 解析）
//   （溯源：TS 时代这两处分别在 backend/src/agents/index.ts 与 agents/skills.ts，已随 Node 后端删除）
// 前端不再本地维护副本 —— 历史副本会与后端漂移（曾长期挂在外部 skill 库上）
type AgentDefault = { name?: string; instructions?: string; skills?: string[] }
const agentDefaults = ref<Record<string, AgentDefault>>({})

/** 某 Agent 的默认提示词（DB 未配置时回显） */
function defaultInstructions(type: string): string {
  return agentDefaults.value[type]?.instructions || ''
}

/** 某 Agent 的默认 Skill 绑定 id 列表 */
function defaultSkillIds(type: string): string[] {
  return agentDefaults.value[type]?.skills || []
}

/** id 列表 → 绑定项（priority 按顺序，默认全部启用） */
function toBindings(ids: string[]) {
  return ids.map((id, i) => ({ id, enabled: true, priority: i + 1 }))
}

const agentDefs = [
  { type: 'script_rewriter', label: '剧本改写', icon: '📝' },
  { type: 'extractor', label: '角色场景提取', icon: '🔍' },
  { type: 'storyboard_breaker', label: '分镜拆解', icon: '🎬' },
  { type: 'voice_assigner', label: '音色分配', icon: '🎙' },
  { type: 'grid_prompt_generator', label: '图片提示词生成', icon: '🖼' },
]


function getAgentCfg(type: string): AgentConfigVO | undefined {
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
  catch (e) { toast.error(e instanceof Error ? e.message : String(e)) }
}

async function loadCfgs() {
  try { cfgs.value = await aiConfigAPI.list() }
  catch (e) { /* 非关键，模型下拉留空 */ }
}

/** 拉取出厂默认配置（默认提示词 + 默认 Skill 绑定），供回显与「恢复默认」使用 */
async function loadAgentDefaults() {
  try {
    const list = await agentConfigAPI.defaults()
    agentDefaults.value = Object.fromEntries((list || []).map((d: AgentConfigVO) => [d.agent_type, d]))
  } catch (e) { /* 非关键：取不到时回显为空，「恢复默认」保持原值 */ }
}

/** 加载全局可用 Skill 列表（用于 Skill 绑定面板）+ 注入体量硬闸 */
async function loadAvailableSkills() {
  try {
    const [list, meta] = await Promise.all([skillsAPI.list(), skillsAPI.meta()])
    availableSkills.value = list
    skillCharBudget.value = meta?.charBudget || 0
  } catch (e) { /* 静默失败，面板显示空状态 */ }
}

/** 单个 Skill 的注入体量（与后端预算闸同口径；缺失按 0 计） */
function getSkillChars(skillId: string): number {
  return availableSkills.value.find(s => s.id === skillId)?.charCount || 0
}

/** 外部技能库 Skill 的来源标签（取 library.yaml 的 label）。
 *  29 个 vendor 分属 2 个库，只标「外部库」无法区分来源；而两个库的用途与体量差异很大，
 *  必须在选择/绑定时就能看出它来自哪个库。非 vendor 返回 ''（供 v-if 使用）。 */
function vendorLabel(skillId: string): string {
  const s = availableSkills.value.find(x => x.id === skillId)
  if (!s || s.category !== 'vendor') return ''
  return s.sourceLabel || '外部库'
}

/** 绑定项指向的 Skill 已不存在（DB 残留引用）：加载器会跳过，需提示用户清理 */
function skillMissing(skillId: string): boolean {
  return availableSkills.value.length > 0 && !availableSkills.value.some(s => s.id === skillId)
}

/** 该 Skill 声明依赖、但本项目（宿主）未提供的工具（后端 missingTools 直出）。
 *  外部技能库多按另一套宿主平台编写，这类 Skill 的正文会教 Agent 调用不存在的工具 ——
 *  绑定后**既不报错也不会生效**，用户只会觉得「勾了没用」，所以必须在绑定现场（以及候选列表里）提示。
 *  仅作预警：MCP 外部工具是运行时动态发现的，存在误报可能，故不做硬拦截。 */
function skillMissingTools(skillId: string): string[] {
  return availableSkills.value.find(s => s.id === skillId)?.missingTools || []
}

/** 缺失工具提示文案（与 v-if 同源，避免模板内重复查找） */
function missingToolsTip(skillId: string): string {
  const list = skillMissingTools(skillId)
  return list.length ? `该 Skill（frontmatter 声明或正文引用）依赖本项目未提供的工具：${list.join(', ')}` : ''
}

/** 已启用绑定的数量（只有 enabled 的会真正注入，统计与预警口径必须与之统一） */
const enabledSkillCount = computed(() => agentSkillBindings.value.filter(b => b.enabled).length)

/** 已启用绑定的注入体量合计 —— 与后端按同一「仅 enabled」口径统计，否则预警会误报 */
const boundChars = computed(() =>
  agentSkillBindings.value.filter(b => b.enabled).reduce((sum, b) => sum + getSkillChars(b.id), 0)
)

/** 合计超出注入预算 → 末尾 Skill 会被静默跳过，必须在绑定现场（而非事后）预警 */
const isBindOverBudget = computed(() => skillCharBudget.value > 0 && boundChars.value > skillCharBudget.value)

/** 体量格式化：≥1 万用 k 记（绑定列表里数字密集，避免一长串） */
function fmtChars(n: number): string {
  const v = Number(n) || 0
  return v >= 10000 ? `${(v / 1000).toFixed(1)}k` : String(v)
}

function toggleAgentEdit(type: string) {
  closeSkillPicker()
  if (editingAgent.value === type) { editingAgent.value = null; agentSkillBindings.value = []; return }
  const cfg = getAgentCfg(type)
  agentForm.model = cfg?.model || ''
  agentForm.temperature = cfg?.temperature ?? 0.7
  agentForm.max_tokens = cfg?.max_tokens ?? 4096
  agentForm.system_prompt = cfg?.system_prompt || defaultInstructions(type)
  // 加载 Skill 绑定配置。⚠️ 回显必须与后端 parseSkillsConfig 同口径（enabled 缺省视为「启用」），
  // 否则缺 enabled 字段的老数据会显示成未勾选、且不计入体量合计，而后端照常注入 —— 显示与实际相反。
  if (cfg?.skills) {
    try {
      const parsed = JSON.parse(cfg.skills)
      agentSkillBindings.value = Array.isArray(parsed)
        ? parsed
            .filter(b => b && typeof b.id === 'string')
            .map(b => ({
              id: b.id,
              enabled: b.enabled !== false,   // 后端 parseSkillsConfig 同口径：缺省视为启用
              priority: typeof b.priority === 'number' ? b.priority : 0,
            }))
        : []
    } catch { agentSkillBindings.value = [] }
  } else {
    agentSkillBindings.value = toBindings(defaultSkillIds(type))
  }
  agentSaved.value = null
  editingAgent.value = type
}

function resetAgentPrompt(type: string) {
  agentForm.system_prompt = defaultInstructions(type) || agentForm.system_prompt
  toast.info('已恢复默认提示词，点击保存生效')
}

function resetAgentSkills(type: string) {
  agentSkillBindings.value = toBindings(defaultSkillIds(type))
  toast.info('已恢复默认 Skill 绑定，点击保存生效')
}

function getSkillName(skillId: string) {
  return availableSkills.value.find(s => s.id === skillId)?.name
}

/**
 * 候选 Skill = 尚未绑定的可用 Skill，按关键词过滤；自有 skill 排在外部库之前。
 * 必要性：外部库 Skill 无 frontmatter 默认绑定，**只能手动挂**（skills.vue 的空状态文案
 * 也把用户指引到这里）—— 此前本面板只有「恢复默认」，没有任何添加控件，用户被指到死路。
 */
const skillPickerCandidates = computed(() => {
  const bound = new Set(agentSkillBindings.value.map(b => b.id))
  const q = skillPickerQuery.value.trim().toLowerCase()
  return availableSkills.value
    .filter(s => !bound.has(s.id))
    .filter(s => !q
      || String(s.id).toLowerCase().includes(q)
      || String(s.name || '').toLowerCase().includes(q)
      || String(s.description || '').toLowerCase().includes(q))
    .sort((a, b) => (a.category === b.category ? 0 : a.category === 'core' ? -1 : 1)
      || String(a.id).localeCompare(String(b.id)))
})

function openSkillPicker() { showSkillPicker.value = true; skillPickerQuery.value = '' }
function closeSkillPicker() { showSkillPicker.value = false; skillPickerQuery.value = '' }

/** 添加绑定：默认启用、置于列表末尾（保存时按列表顺序统一重写 priority，故此处不必算准） */
function addSkillBinding(id: string) {
  if (agentSkillBindings.value.some(b => b.id === id)) return
  agentSkillBindings.value = [
    ...agentSkillBindings.value,
    { id, enabled: true, priority: agentSkillBindings.value.length + 1 },
  ]
  toast.info(`已添加「${getSkillName(id) || id}」，保存后生效`)
}

/** 移除绑定项（含指向已删除 Skill 的「已失效」项）—— 否则失效引用只能永远留在列表里 */
function removeSkillBinding(idx: number) {
  const removed = agentSkillBindings.value[idx]
  if (!removed) return
  agentSkillBindings.value = agentSkillBindings.value.filter((_, i) => i !== idx)
  toast.info(`已移除「${getSkillName(removed.id) || removed.id}」，保存后生效`)
}

function onSkillDragStart(idx: number, e: DragEvent) {
  skillDragFrom.value = idx
  if (e?.dataTransfer) {
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', String(idx))
  }
}
function onSkillDragOver(idx: number) {
  skillDragOverIdx.value = idx
}
function onSkillDrop(idx: number) {
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

async function saveAgentCfg(type: string) {
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
    toast.error(e instanceof Error ? e.message : String(e))
  } finally {
    agentSaving.value = false
  }
}

onMounted(() => { loadAgents(); loadCfgs(); loadAvailableSkills(); loadAgentDefaults() })
</script>

<style scoped>
.agents-page { height: 100%; overflow: hidden; }
.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 840px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
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
.skill-bind-head-right { display: flex; align-items: center; gap: 8px; }
/* 已启用体量合计 / 预算：超预算时转为警示色（超出的 Skill 会被静默跳过） */
.skill-bind-usage { font-size: 10px; font-family: var(--font-mono); color: var(--text-3); white-space: nowrap; }
.skill-bind-usage.over { color: #b45309; font-weight: 600; }
/* 单项注入体量 */
.skill-bind-chars { font-size: 10px; font-family: var(--font-mono); color: var(--text-3); flex-shrink: 0; white-space: nowrap; }
/* 来源 / 失效标记 */
.skill-bind-tag { font-size: 9px; padding: 1px 5px; border-radius: 999px; background: rgba(99,102,241,.1); color: #6366f1; font-weight: 500; margin-left: 4px; }
.skill-bind-tag.danger { background: rgba(220,38,38,.1); color: #dc2626; }
/* 依赖了本项目未提供的工具（预警而非错误：MCP 工具为运行时发现，可能误报） */
.skill-bind-tag.warn { background: rgba(245,158,11,.12); color: #b45309; }
.skill-bind-warn { margin-top: 6px; padding: 6px 9px; border-radius: 6px; background: rgba(245,158,11,.12); color: #b45309; font-size: 10px; line-height: 1.5; }
/* 添加 Skill 候选面板：外部库 Skill 无默认绑定，这里是唯一的 UI 挂载入口 */
.skill-picker { margin-top: 8px; border: 1px solid var(--border); border-radius: 8px; background: rgba(255,255,255,.7); overflow: hidden; }
.skill-picker-head { display: flex; align-items: center; gap: 6px; padding: 6px 8px; border-bottom: 1px solid var(--border); color: var(--text-3); }
.skill-picker-input { flex: 1; min-width: 0; border: none; outline: none; background: transparent; font-size: 11px; color: var(--text-1); }
.skill-picker-list { max-height: 190px; overflow-y: auto; }
.skill-picker-item { display: flex; align-items: center; gap: 8px; width: 100%; padding: 6px 8px; border: none; background: transparent; cursor: pointer; text-align: left; color: var(--text-1); }
.skill-picker-item:hover { background: var(--accent-bg); }
.skill-picker-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.skill-picker-name { font-size: 11px; font-weight: 500; display: flex; align-items: center; }
.skill-picker-id { font-size: 9px; font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
/* 移除绑定项 */
.skill-bind-remove { flex-shrink: 0; border: none; background: transparent; color: var(--text-3); cursor: pointer; padding: 2px; border-radius: 4px; display: flex; }
.skill-bind-remove:hover { color: #dc2626; background: rgba(220,38,38,.08); }
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

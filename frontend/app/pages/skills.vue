<template>
  <div class="skills-layout">
    <!-- Agent 左侧列表 -->
    <aside class="skills-agent-list">
      <div class="skills-agent-title">Skill 来源</div>
      <button
        :class="['skills-agent-item', { active: selectedAgent === 'all' }]"
        @click="selectAgent('all')"
      >
        <span class="agent-type-badge">📦</span>
        <span class="skills-agent-label">全部 Skill</span>
        <span v-if="agentSkillCount('all') > 0" class="skill-count-badge">{{ agentSkillCount('all') }}</span>
      </button>
      <div class="skills-agent-title" style="margin-top:14px">Agent 列表</div>
      <button
        v-for="a in agentDefs"
        :key="a.type"
        :class="['skills-agent-item', { active: selectedAgent === a.type }]"
        @click="selectAgent(a.type)"
      >
        <span class="agent-type-badge">{{ agentIcon(a.type) }}</span>
        <span class="skills-agent-label">{{ a.label }}</span>
        <span v-if="agentSkillCount(a.type) > 0" class="skill-count-badge">{{ agentSkillCount(a.type) }}</span>
      </button>
      <!-- 外部技能库：来自各库 skills/<lib>/library.yaml 的声明，换库 / 加库 / 改展示名零改动 -->
      <template v-if="vendorLibs.length">
        <div class="skills-agent-title" style="margin-top:14px">外部技能库</div>
        <button
          v-for="lib in vendorLibs"
          :key="lib.id"
          :title="lib.declared === false ? `${lib.label}（该库未提供 library.yaml 声明，展示名取目录名）` : (lib.description || lib.label)"
          :class="['skills-agent-item', { active: selectedAgent === 'lib:' + lib.id }]"
          @click="selectAgent('lib:' + lib.id)"
        >
          <span class="agent-type-badge">🧩</span>
          <span class="skills-agent-label">{{ lib.label }}</span>
          <span v-if="lib.skillCount > 0" class="skill-count-badge">{{ lib.skillCount }}</span>
        </button>
      </template>
    </aside>

    <!-- Skill 管理右侧主区域 -->
    <div class="settings-scroll skills-main">
      <div class="settings-head">
        <div style="display:flex;align-items:center;gap:10px">
          <span class="agent-type-badge" style="width:32px;height:32px;font-size:16px">{{ selectedAgentIcon }}</span>
          <div>
            <h2 class="settings-title" style="margin:0">{{ selectedAgentLabel }}</h2>
            <div class="dim" :class="{ 'skill-over-budget': isOverBudget }" style="font-size:12px">{{ selectedAgentSubtitle }}</div>
          </div>
        </div>
        <p class="settings-desc" style="margin-top:10px">Skills 仅作为 Agent 的高级提示词层使用，不影响工作台常规功能入口。</p>
        <div v-if="isOverBudget" class="skill-budget-alert">
          默认绑定合计已超出注入预算，超出部分会按优先级被跳过（即该 Skill 不生效）。可在「Agent 配置 → 绑定 Skills」中精简绑定。
        </div>
        <button v-if="canAddSkill" class="btn btn-primary btn-sm" @click="startAddSkill">
          <Plus :size="13" /> 新增 Skill
        </button>
      </div>

      <!-- 无 skill 提示 -->
      <div v-if="!currentSkills.length" class="step-empty" style="padding:48px 24px">
        <div class="empty-visual">
          <FileText :size="28" />
        </div>
        <div class="empty-title">暂无 Skill</div>
        <div class="empty-desc" v-if="isVendorSelected">外部技能库的 Skill 属「按触发词独立启动」的会话式技能，需在「Agent 配置 → 绑定 Skills」中按需手动启用</div>
        <div class="empty-desc" v-else>点击右上角「新增 Skill」创建第一个提示词文件</div>
      </div>

      <!-- Skill 列表 -->
      <div class="skill-list" v-else>
        <div v-for="s in currentSkills" :key="s.id" class="card skill-card">
          <div class="skill-card-head" @click="toggleSkillEdit(s.id)">
            <FileText :size="14" style="color:var(--accent);flex-shrink:0" />
            <div style="flex:1;min-width:0">
              <div style="font-weight:600;font-size:13px;display:flex;align-items:center;gap:6px;flex-wrap:wrap">
                <span>{{ s.name }}</span>
                <span :class="['skill-cat-badge', s.category]">{{ skillCategoryLabel(s) }}</span>
              </div>
              <div class="dim" style="font-size:11px">{{ s.description }}</div>
              <div class="skill-stat-row">
                <span class="skill-stat-tag" title="实际注入体量（含前置契约与协议字段段，与注入预算同口径）">{{ fmtChars(s.charCount) }} 字符</span>
                <span
                  v-if="s.referenceCount"
                  class="skill-stat-tag warn"
                  title="加载器只读 SKILL.md：这些参考文件不会被注入，正文里顺着引用去找必然落空"
                >📎 {{ s.referenceCount }} 个参考文件（不注入）</span>
                <span
                  v-if="s.missingTools?.length"
                  class="skill-stat-tag warn"
                  :title="`该 Skill（frontmatter 声明或正文引用）依赖本项目未提供的工具：${s.missingTools.join(', ')}。绑定后它会教 Agent 调用这些不存在的工具（既不报错也不生效）；外部库按另一套宿主平台编写属常见情况，故仅作预警`"
                >🧩 缺 {{ s.missingTools.length }} 个工具</span>
                <span
                  v-if="s.protected"
                  class="skill-stat-tag lock"
                  title="项目自带 Skill，受删除保护；如需停用请在「Agent 配置 → 绑定 Skills」中取消勾选"
                >🔒 受保护</span>
              </div>
              <div v-if="(s.boundAgents && s.boundAgents.length) || (s.phases && s.phases.length)" class="skill-meta-row">
                <span v-for="a in s.boundAgents" :key="'a' + a" class="tag">🤖 {{ agentLabel(a) }}</span>
                <span v-for="p in s.phases" :key="'p' + p" class="tag tag-accent">⚙️ {{ p }}</span>
              </div>
            </div>
            <button v-if="!s.protected" class="btn btn-ghost btn-icon" style="margin-right:4px" title="删除 Skill" @click.stop="deleteSkill(s)">
              <Trash2 :size="13" />
            </button>
            <ChevronDown :size="14" :style="{ transform: editingSkill === s.id ? 'rotate(180deg)' : '', transition: '0.2s' }" />
          </div>
          <div v-if="editingSkill === s.id" class="skill-card-body">
            <textarea
              v-model="skillContent"
              class="textarea mono"
              rows="20"
              style="font-size:12px;line-height:1.6"
              placeholder="编写 SKILL.md 内容..."
            />
            <div class="skill-card-foot">
              <span class="dim" style="font-size:11px">skills/{{ s.id }}/SKILL.md</span>
              <span v-if="skillSaved === s.id" class="tag tag-success" style="margin-left:8px">
                <Check :size="10" /> 已保存
              </span>
              <button class="btn btn-primary btn-sm ml-auto" :disabled="skillSaving" @click="saveSkill(s.id)">
                <Loader2 v-if="skillSaving" :size="12" class="animate-spin" />
                保存
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Add Skill Dialog -->
    <div v-if="addSkillDialog" class="overlay" @click.self="addSkillDialog = false">
      <form class="modal card" @submit.prevent="confirmAddSkill">
        <h2 class="modal-title">新增 Skill — {{ selectedAgentLabel }}</h2>
        <label class="field">
          <span class="field-label">Skill 目录名 <span class="dim">(英文，唯一)</span></span>
          <input v-model="newSkillForm.id" class="input" placeholder="如 custom-extraction" />
          <span v-if="newSkillForm.id && !newSkillIdValid" class="field-hint" style="color:#dc2626">
            仅支持字母、数字、连字符、下划线（目录名不能含斜杠与空格）
          </span>
        </label>
        <label class="field">
          <span class="field-label">名称</span>
          <input v-model="newSkillForm.name" class="input" placeholder="如 自定义提取规则" />
        </label>
        <label class="field">
          <span class="field-label">描述</span>
          <input v-model="newSkillForm.description" class="input" placeholder="简短描述此 Skill 的用途" />
        </label>
        <div class="modal-actions">
          <button type="button" class="btn" @click="addSkillDialog = false">取消</button>
          <button type="submit" class="btn btn-primary" :disabled="!newSkillIdValid">创建</button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Plus, FileText, Check, Loader2, ChevronDown, Trash2 } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import { skillsAPI } from '~/composables/useApi'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()

// ===== Skills =====
const selectedAgent = ref('script_rewriter')   // 'all' | Agent 类型 | 'lib:<技能库名>'
const allSkills = ref([])   // SkillVO[]: { id, name, description, category, source, agents, priority, boundAgents, phases }
const skillMeta = ref({ agents: [], sources: [], coreCount: 0, charBudget: 0 })
const editingSkill = ref(null)
const skillContent = ref('')
const skillSaving = ref(false)
const skillSaved = ref(null)
const addSkillDialog = ref(false)
const newSkillForm = reactive({ id: '', name: '', description: '' })
/** 目录名预校验：比后端更严（不允许斜杠，避免误建嵌套目录）→ 即时提示而非等 400 英文报错 */
const newSkillIdValid = computed(() => /^[a-zA-Z0-9_-]+$/.test(newSkillForm.id))

/** Agent 列表：完全来自后端 /skills/meta（显示名与阶段由后端单一维护），前端只补装饰性图标 */
const agentDefs = computed(() => skillMeta.value.agents || [])
/** 外部技能库列表：来自 GET /skills/meta 的 sources（后端按各库 library.yaml 声明解析），换库 / 加库无需改前端 */
const vendorLibs = computed(() => skillMeta.value.sources || [])

/** Agent 图标（纯 UI 装饰；未知 Agent 回退默认值，新增 Agent 不会让前端报错） */
const AGENT_ICONS = {
  script_rewriter: '📝',
  extractor: '🔍',
  storyboard_breaker: '🎬',
  voice_assigner: '🎙',
  grid_prompt_generator: '🖼',
}
function agentIcon(type) {
  return AGENT_ICONS[type] || '🤖'
}

const isVendorSelected = computed(() => selectedAgent.value.startsWith('lib:'))
/** 只有选中具体 Agent 时才允许新增（新建文件落在 skills/<agent>/<id>/ 下） */
const canAddSkill = computed(() => !isVendorSelected.value && selectedAgent.value !== 'all')

const selectedAgentLabel = computed(() => {
  if (selectedAgent.value === 'all') return '全部 Skill'
  if (isVendorSelected.value) {
    const lib = vendorLibs.value.find(l => `lib:${l.id}` === selectedAgent.value)
    return lib ? `${lib.label} 技能库` : selectedAgent.value
  }
  return agentDefs.value.find(a => a.type === selectedAgent.value)?.label || selectedAgent.value
})
const selectedAgentIcon = computed(() => {
  if (selectedAgent.value === 'all') return '📦'
  if (isVendorSelected.value) return '🧩'
  return agentIcon(selectedAgent.value)
})
/** 体量格式化：小于 1 万原样显示，超出用 k 记（避免卡片上一长串数字） */
function fmtChars(n) {
  const v = Number(n) || 0
  return v >= 10000 ? `${(v / 1000).toFixed(1)}k` : String(v)
}

/** 当前选中 Agent 的默认注入体量与预算（后端按注入同口径计算，供显示余量） */
const selectedInjection = computed(() => {
  if (selectedAgent.value === 'all' || isVendorSelected.value) return null
  const a = agentDefs.value.find(x => x.type === selectedAgent.value)
  if (!a) return null
  return { chars: a.charCount || 0, budget: skillMeta.value.charBudget || 0 }
})
/** 默认绑定合计已超出注入预算 → 超出的 skill 会被按优先级静默跳过，必须显式预警 */
const isOverBudget = computed(() => {
  const inj = selectedInjection.value
  return !!inj && inj.budget > 0 && inj.chars > inj.budget
})

/** 副标题：Agent 显示技术标识 + 注入体量余量（便于对照代码与 DB）；技能库显示库说明 */
const selectedAgentSubtitle = computed(() => {
  if (selectedAgent.value === 'all') return `共 ${allSkills.value.length} 个 Skill`
  if (isVendorSelected.value) {
    const lib = vendorLibs.value.find(l => `lib:${l.id}` === selectedAgent.value)
    return lib?.description || `库标识：${lib?.id || ''}`
  }
  const inj = selectedInjection.value
  const usage = inj ? ` · 默认注入 ${fmtChars(inj.chars)} / ${fmtChars(inj.budget)} 字符` : ''
  return `${selectedAgent.value} — Agent Skills${usage}`
})

/** 徽标文案：core = 项目自有；vendor = 归属的外部技能库名（不写死任何库的品牌） */
function skillCategoryLabel(s) {
  if (s.category === 'vendor') return s.sourceLabel || s.source || '外部技能库'
  return '核心'
}
function agentLabel(type) {
  return agentDefs.value.find(a => a.type === type)?.label || type
}

/** 侧栏分组归属：Agent 组看默认绑定（boundAgents）+ 目录归属；技能库组看来源库 */
function inGroup(s, type) {
  if (type === 'all') return true
  if (type.startsWith('lib:')) return s.category === 'vendor' && `lib:${s.source}` === type
  return (s.boundAgents || []).includes(type) || s.id === type || s.id.startsWith(type + '/')
}
function agentSkillCount(type) {
  return allSkills.value.filter(s => inGroup(s, type)).length
}

const currentSkills = computed(() => allSkills.value.filter(s => inGroup(s, selectedAgent.value)))

async function loadAllSkills() {
  try {
    const [list, meta] = await Promise.all([skillsAPI.list(), skillsAPI.meta()])
    allSkills.value = list
    skillMeta.value = meta
  } catch (e) { toast.error(e.message) }
}

/** meta 加载后校正选中项：Agent / 技能库被移除时不至于停在空分组 */
function ensureSelectionValid() {
  const valid = ['all', ...agentDefs.value.map(a => a.type), ...vendorLibs.value.map(l => `lib:${l.id}`)]
  if (!valid.includes(selectedAgent.value)) selectedAgent.value = agentDefs.value[0]?.type || 'all'
}

async function selectAgent(type) {
  selectedAgent.value = type
  editingSkill.value = null
}

function startAddSkill() {
  newSkillForm.id = ''
  newSkillForm.name = ''
  newSkillForm.description = ''
  addSkillDialog.value = true
}

async function confirmAddSkill() {
  if (!newSkillForm.id) return
  const skillId = `${selectedAgent.value}/${newSkillForm.id}`
  try {
    await skillsAPI.create({ id: skillId, name: newSkillForm.name, description: newSkillForm.description })
    addSkillDialog.value = false
    await loadAllSkills()
    toast.success('Skill 创建成功')
  } catch (e) {
    toast.error(e.message)
  }
}

async function deleteSkill(s) {
  if (!(await confirm({
    message: s.category === 'vendor'
      ? `确定删除外部技能库 Skill「${s.id}」？\n该操作将直接从磁盘删除技能文件且不可恢复，如属导入资源需重新导入技能库。`
      : `确定删除 Skill「${s.id}」？`,
    danger: true,
  }))) return
  try {
    await skillsAPI.del(s.id)
    if (editingSkill.value === s.id) editingSkill.value = null
    await loadAllSkills()
    toast.success('已删除')
  } catch (e) {
    toast.error(e.message)
  }
}

async function toggleSkillEdit(id) {
  if (editingSkill.value === id) { editingSkill.value = null; return }
  try {
    const res = await skillsAPI.get(id)
    skillContent.value = res.content
    skillSaved.value = null
    editingSkill.value = id
  } catch (e) { toast.error(e.message) }
}

async function saveSkill(id) {
  skillSaving.value = true
  skillSaved.value = null
  try {
    const res: any = await skillsAPI.update(id, skillContent.value)
    await loadAllSkills()
    skillSaved.value = id
    // 后端会校验 frontmatter：头部被改坏会让该 Skill 静默退出默认注入 → 显式告知而非只报成功
    if (res?.warning) toast.warning(res.warning)
    else toast.success(`已保存`)
    setTimeout(() => { if (skillSaved.value === id) skillSaved.value = null }, 3000)
  } catch (e) {
    toast.error(e.message)
  } finally {
    skillSaving.value = false
  }
}

onMounted(async () => {
  await loadAllSkills()
  ensureSelectionValid()
})
</script>

<style scoped>
.skills-layout { display: flex; height: 100%; overflow: hidden; }
.skills-agent-list {
  width: 200px; flex-shrink: 0; border-right: 1px solid var(--border);
  background: var(--bg-1); display: flex; flex-direction: column;
  overflow-y: auto;
}
.skills-agent-title {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-3); padding: 14px 14px 8px;
}
.skills-agent-item {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 14px; font-size: 13px; cursor: pointer;
  border: none; background: none; color: var(--text-2);
  transition: all 0.12s; width: 100%; text-align: left;
  border-radius: 0;
}
.skills-agent-item:hover { background: var(--bg-hover); color: var(--text-0); }
.skills-agent-item.active { background: var(--accent-bg); color: var(--accent-text); font-weight: 600; }
.skills-agent-label { flex: 1; }
.skill-count-badge {
  font-size: 10px; font-weight: 700; font-family: var(--font-mono);
  background: var(--accent-bg); color: var(--accent-text);
  padding: 1px 5px; border-radius: 99px;
}
.skills-agent-item.active .skill-count-badge { background: rgba(255,255,255,0.2); color: inherit; }
.skills-main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }

.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 900px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; }

/* Skill */
.skill-list { display: flex; flex-direction: column; gap: 8px; }
.skill-card { overflow: hidden; }
.skill-card-head { display: flex; align-items: center; gap: 10px; padding: 12px 16px; cursor: pointer; transition: background 0.1s; }
.skill-card-head:hover { background: var(--bg-hover); }
.skill-card-body { padding: 0 16px 16px; display: flex; flex-direction: column; gap: 10px; border-top: 1px solid var(--border); padding-top: 12px; }
.skill-card-foot { display: flex; align-items: center; gap: 8px; }
.skill-meta-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.skill-cat-badge { font-size: 10px; line-height: 1; padding: 3px 6px; border-radius: 999px; font-weight: 500; white-space: nowrap; background: var(--bg-2); color: var(--text-2); border: 1px solid var(--border); }
.skill-cat-badge.core { background: rgba(13,148,136,.1); color: var(--accent-text); border-color: transparent; }
.skill-cat-badge.vendor { background: rgba(99,102,241,.1); color: #6366f1; border-color: transparent; }
/* 体量 / 参考文件 / 保护状态标记 */
.skill-stat-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 5px; }
.skill-stat-tag { font-size: 10px; line-height: 1; padding: 3px 6px; border-radius: 999px; font-weight: 500; white-space: nowrap; background: var(--bg-2); color: var(--text-3); border: 1px solid var(--border); }
.skill-stat-tag.warn { background: rgba(245,158,11,.12); color: #b45309; border-color: transparent; }
.skill-stat-tag.lock { background: rgba(13,148,136,.1); color: var(--accent-text); border-color: transparent; }
.skill-over-budget { color: #b45309 !important; font-weight: 600; }
.skill-budget-alert { margin-top: 10px; padding: 8px 12px; border-radius: var(--radius); background: rgba(245,158,11,.12); color: #b45309; font-size: 12px; line-height: 1.6; }

/* 空状态 */
.step-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  flex: 1; min-height: 300px; gap: 10px; padding: 46px;
  animation: fadeIn 0.3s var(--ease-out);
}
.empty-visual {
  width: 72px; height: 72px; border-radius: 22px;
  background: rgba(255,255,255,0.8); color: var(--accent);
  border: 1px solid rgba(27, 41, 64, 0.08);
  box-shadow: var(--shadow-sm);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 8px;
}
.empty-title { font-size: 22px; font-weight: 700; font-family: var(--font-display); color: var(--text-0); }
.empty-desc { font-size: 13px; color: var(--text-2); max-width: 420px; text-align: center; line-height: 1.8; }

/* Shared */
.agent-type-badge { width: 36px; height: 36px; border-radius: var(--radius); background: var(--accent-bg); color: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--text-1); }
.field-hint { font-size: 11px; color: var(--text-3); margin-top: 2px; }

.overlay { position: fixed; inset: 0; background: rgba(34,45,66,0.32); backdrop-filter: blur(8px); display: flex; align-items: center; justify-content: center; z-index: 100; animation: fadeIn 0.18s var(--ease-out); }
.modal { padding: 28px; width: 420px; display: flex; flex-direction: column; gap: 12px; box-shadow: var(--shadow-elevated); }
.modal-title { font-family: var(--font-display); font-size: 18px; font-weight: 700; }
.modal-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 6px; }
</style>

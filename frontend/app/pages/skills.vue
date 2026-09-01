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
        <span class="agent-type-badge">{{ a.icon }}</span>
        <span class="skills-agent-label">{{ a.label }}</span>
        <span v-if="agentSkillCount(a.type) > 0" class="skill-count-badge">{{ agentSkillCount(a.type) }}</span>
      </button>
      <button
        :class="['skills-agent-item', { active: selectedAgent === 'minimax' }]"
        @click="selectAgent('minimax')"
      >
        <span class="agent-type-badge">🧩</span>
        <span class="skills-agent-label">MiniMax 技能库</span>
        <span v-if="agentSkillCount('minimax') > 0" class="skill-count-badge">{{ agentSkillCount('minimax') }}</span>
      </button>
    </aside>

    <!-- Skill 管理右侧主区域 -->
    <div class="settings-scroll skills-main">
      <div class="settings-head">
        <div style="display:flex;align-items:center;gap:10px">
          <span class="agent-type-badge" style="width:32px;height:32px;font-size:16px">{{ selectedAgentIcon }}</span>
          <div>
            <h2 class="settings-title" style="margin:0">{{ selectedAgentLabel }}</h2>
            <div class="dim" style="font-size:12px">{{ selectedAgentType }} — Skills</div>
          </div>
        </div>
        <p class="settings-desc" style="margin-top:10px">Skills 仅作为 Agent 的高级提示词层使用，不影响工作台常规功能入口。</p>
        <button v-if="selectedAgent !== 'minimax' && selectedAgent !== 'all'" class="btn btn-primary btn-sm" @click="startAddSkill">
          <Plus :size="13" /> 新增 Skill
        </button>
      </div>

      <!-- 无 skill 提示 -->
      <div v-if="!currentSkills.length" class="step-empty" style="padding:48px 24px">
        <div class="empty-visual">
          <FileText :size="28" />
        </div>
        <div class="empty-title">暂无 Skill</div>
        <div class="empty-desc" v-if="selectedAgent === 'minimax'">MiniMax 技能库已导入项目，可在「Agent 配置」的绑定 Skills 面板中启用</div>
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
                <span :class="['skill-cat-badge', s.category]">{{ skillCategoryLabel(s.category) }}</span>
              </div>
              <div class="dim" style="font-size:11px">{{ s.description }}</div>
              <div v-if="(s.boundAgents && s.boundAgents.length) || (s.phases && s.phases.length)" class="skill-meta-row">
                <span v-for="a in s.boundAgents" :key="'a' + a" class="tag">🤖 {{ agentLabel(a) }}</span>
                <span v-for="p in s.phases" :key="'p' + p" class="tag tag-accent">⚙️ {{ p }}</span>
              </div>
            </div>
            <button class="btn btn-ghost btn-icon" style="margin-right:4px" @click.stop="deleteSkill(s.id)">
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
          <button type="submit" class="btn btn-primary" :disabled="!newSkillForm.id">创建</button>
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
const selectedAgent = ref('script_rewriter')
const allSkills = ref([])   // { id, name, description }[]
const editingSkill = ref(null)
const skillContent = ref('')
const skillSaving = ref(false)
const skillSaved = ref(null)
const addSkillDialog = ref(false)
const newSkillForm = reactive({ id: '', name: '', description: '' })

const agentDefs = [
  { type: 'script_rewriter', label: '剧本改写', icon: '📝' },
  { type: 'extractor', label: '角色场景提取', icon: '🔍' },
  { type: 'storyboard_breaker', label: '分镜拆解', icon: '🎬' },
  { type: 'voice_assigner', label: '音色分配', icon: '🎙' },
  { type: 'grid_prompt_generator', label: '图片提示词生成', icon: '🖼' },
]

const selectedAgentType = computed(() => selectedAgent.value)
const selectedAgentLabel = computed(() => {
  if (selectedAgent.value === 'all') return '全部 Skill'
  if (selectedAgent.value === 'minimax') return 'MiniMax 技能库'
  return agentDefs.find(a => a.type === selectedAgent.value)?.label || ''
})
const selectedAgentIcon = computed(() => {
  if (selectedAgent.value === 'all') return '📦'
  if (selectedAgent.value === 'minimax') return '🧩'
  return agentDefs.find(a => a.type === selectedAgent.value)?.icon || ''
})

const CATEGORY_LABELS = {
  core: '核心',
  'minimax-builtin': 'MiniMax 内置',
  'minimax-installed': 'MiniMax 安装',
  custom: '自定义',
}
function skillCategoryLabel(c) {
  return CATEGORY_LABELS[c] || c
}
function agentLabel(type) {
  return agentDefs.find(a => a.type === type)?.label || type
}

function agentSkillCount(type) {
  if (type === 'all') return allSkills.value.length
  return allSkills.value.filter(s => s.id === type || s.id.startsWith(type + '/')).length
}

const currentSkills = computed(() => {
  if (selectedAgent.value === 'all') return allSkills.value
  return allSkills.value.filter(s => s.id === selectedAgent.value || s.id.startsWith(selectedAgent.value + '/'))
})

async function loadAllSkills() {
  try { allSkills.value = await skillsAPI.list() }
  catch (e) { toast.error(e.message) }
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

async function deleteSkill(id) {
  const isMiniMax = id.startsWith('minimax/')
  if (!(await confirm({
    message: isMiniMax
      ? `确定删除技能库 Skill「${id}」？\n该操作将直接从磁盘删除技能文件且不可恢复，如属导入资源需重新导入技能库。`
      : `确定删除 Skill「${id}」？`,
    danger: true,
  }))) return
  try {
    await skillsAPI.del(id)
    if (editingSkill.value === id) editingSkill.value = null
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
    await skillsAPI.update(id, skillContent.value)
    await loadAllSkills()
    skillSaved.value = id
    toast.success(`已保存`)
    setTimeout(() => { if (skillSaved.value === id) skillSaved.value = null }, 3000)
  } catch (e) {
    toast.error(e.message)
  } finally {
    skillSaving.value = false
  }
}

onMounted(() => { loadAllSkills() })
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
.skill-cat-badge.minimax-installed { background: rgba(99,102,241,.1); color: #6366f1; border-color: transparent; }
.skill-cat-badge.minimax-builtin { background: rgba(168,85,247,.1); color: #a855f7; border-color: transparent; }
.skill-cat-badge.custom { background: rgba(245,158,11,.1); color: #d97706; border-color: transparent; }

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

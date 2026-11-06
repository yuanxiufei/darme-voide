<script setup lang="ts">
import { ref, watch } from 'vue'
import { Pencil } from 'lucide-vue-next'

const props = defineProps<{
  name: string
  tag: string
  value: string | null
  placeholder?: string
  onSave: (value: string) => Promise<void>
}>()

const editing = ref(false)
const draft = ref('')
const saving = ref(false)

watch(
  () => props.value,
  (v) => {
    if (!editing.value) draft.value = v || ''
  },
)

function startEdit() {
  draft.value = props.value || ''
  editing.value = true
}

function cancel() {
  editing.value = false
  draft.value = props.value || ''
}

async function save() {
  if (saving.value) return
  saving.value = true
  try {
    await props.onSave(draft.value)
    editing.value = false
  } catch {
    // 错误提示由 onSave 内部负责（toast），这里保持编辑态让用户继续修改
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="prompt-card" :class="{ editing }">
    <div class="pc-head">
      <div class="pc-name" :title="name">{{ name }}</div>
      <span class="pc-tag">{{ tag }}</span>
    </div>

    <div v-if="!editing" class="pc-view">
      <p class="pc-text" :class="{ dim: !value }">{{ value || placeholder || '（空）' }}</p>
      <button class="btn btn-ghost btn-sm" @click="startEdit">
        <Pencil :size="12" />
        编辑
      </button>
    </div>

    <div v-else class="pc-edit">
      <textarea v-model="draft" class="textarea" rows="4" :placeholder="placeholder" />
      <div class="pc-actions">
        <button class="btn btn-primary btn-sm" :disabled="saving" @click="save">
          {{ saving ? '保存中…' : '保存' }}
        </button>
        <button class="btn btn-ghost btn-sm" :disabled="saving" @click="cancel">取消</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.prompt-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: border-color 0.15s;
}
.prompt-card.editing {
  border-color: var(--accent);
}
.pc-head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.pc-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-1);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}
.pc-tag {
  flex: none;
  font-size: 11px;
  color: var(--accent-text);
  background: var(--accent-bg);
  border-radius: 6px;
  padding: 2px 8px;
}
.pc-view {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}
.pc-text {
  flex: 1;
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-1);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 120px;
  overflow-y: auto;
}
.pc-text.dim {
  color: var(--text-3);
  font-style: italic;
}
.pc-view .btn {
  flex: none;
}
.pc-edit {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pc-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}
</style>

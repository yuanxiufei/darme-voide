<template>
  <div class="model-selector">
    <label v-if="label" class="model-label">{{ label }}</label>
    <BaseSelect
      :model-value="localModel"
      :options="modelOptions"
      :placeholder="loading ? '加载中...' : (placeholder || '选择模型')"
      :disabled="disabled || loading"
      @update:model-value="handleSelect"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'

const props = withDefaults(defineProps<{
  modelValue?: string
  serviceType?: 'image' | 'video' | 'audio' | 'text'
  label?: string
  placeholder?: string
  disabled?: boolean
  configId?: number
}>(), {
  serviceType: 'image',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const allConfigs = ref<any[]>([])
const loading = ref(false)
const localModel = ref(props.modelValue || '')

watch(() => props.modelValue, (v) => { if (v !== localModel.value) localModel.value = v || '' })

function handleSelect(value: string) {
  localModel.value = value
  emit('update:modelValue', value)
}

const modelOptions = computed(() => {
  const opts = allConfigs.value.map((c: any) => ({
    value: c.model || c.name || 'default',
    label: `${c.name || c.model || '默认'}${c.provider ? ` (${c.provider})` : ''}`,
  }))
  return [{ value: '', label: '默认模型' }, ...opts]
})

onMounted(async () => {
  if (!props.serviceType) return
  loading.value = true
  try {
    const { aiConfigAPI } = await import('../composables/useApi')
    const data = await aiConfigAPI.list(props.serviceType)
    allConfigs.value = (data as any)?.items || (Array.isArray(data) ? data : [])
  } catch { /* 使用空列表 */ }
  finally { loading.value = false }
})
</script>

<style scoped>
.model-selector {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 180px;
  max-width: 260px;
  flex: 1 1 auto;
}
.model-label {
  font-size: 12px;
  color: var(--text-3);
  white-space: nowrap;
}
</style>

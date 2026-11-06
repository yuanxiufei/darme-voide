<script setup lang="ts">
/**
 * 画布右键菜单（通用容器）。
 *
 * 一条必须做对的：**点菜单外面才关** ✓ —— 用 `mousedown` 全局监听并判断
 * 是否落在面板内；只在 `click` 上关菜单会让「按住拖到菜单」的操作误关 ✗，
 * 而完全不关则会留下一个吃点击的幽灵层 ✗。
 */
import { onBeforeUnmount, onMounted } from 'vue'

interface MenuItem {
  key: string
  label: string
  danger?: boolean
  disabled?: boolean
}

const props = defineProps<{ x: number; y: number; items: MenuItem[] }>()
const emit = defineEmits<{
  (e: 'select', key: string): void
  (e: 'close'): void
}>()

function onDown(ev: MouseEvent) {
  const target = ev.target as HTMLElement | null
  if (!target?.closest('.cm-panel')) emit('close')
}

onMounted(() => window.addEventListener('mousedown', onDown))
onBeforeUnmount(() => window.removeEventListener('mousedown', onDown))
</script>

<template>
  <div class="cm-panel" :style="{ left: `${props.x}px`, top: `${props.y}px` }" @contextmenu.prevent>
    <button
      v-for="item in props.items"
      :key="item.key"
      class="cm-item"
      :class="{ danger: item.danger }"
      :disabled="item.disabled"
      @click="emit('select', item.key)"
    >
      {{ item.label }}
    </button>
  </div>
</template>

<style scoped>
.cm-panel {
  position: fixed;
  z-index: 90;
  min-width: 168px;
  padding: 4px;
  background: #1b1f27;
  border: 1px solid #2f3541;
  border-radius: 8px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.cm-item {
  text-align: left;
  padding: 6px 10px;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: #e6e8ee;
  font-size: 12.5px;
  cursor: pointer;
}
.cm-item:hover:not(:disabled) {
  background: #262c37;
}
.cm-item:disabled {
  color: #5b6472;
  cursor: default;
}
.cm-item.danger {
  color: #ef7a7a;
}
.cm-item + .cm-item {
  border-top: 1px solid #232833;
}
</style>

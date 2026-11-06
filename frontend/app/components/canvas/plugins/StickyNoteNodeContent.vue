<script setup lang="ts">
/** 便利贴节点（`sticky-note:note`）—— 双击编辑、点色块换色、空着就显示上游文本 ✓。 */
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps<{ values: Record<string, string>; text: string }>()
const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

/** 与源插件同一组配色（源写在 `manifest` 之外，本仓放进 `values.pluginColor` 让用户能改 ✓） */
const COLORS = ['#fde68a', '#fbcfe8', '#bfdbfe', '#bbf7d0', '#fed7aa']

const editing = ref(false)
const box = ref<HTMLTextAreaElement | null>(null)
const color = computed(() => props.values.pluginColor || COLORS[0])
const content = computed(() => props.values.content ?? '')
/**
 * 自己没写就显示上游文本 —— 便利贴最常见的用法就是「接着上游那句话写」✓。
 * 但这只是**显示**：不自动写回 `content`，否则用户一拖动上游文本，便利贴就被悄悄改掉 ✗
 */
const shown = computed(() => content.value || props.text)

watch(editing, async (on) => {
  if (!on) return
  await nextTick()
  box.value?.focus()
})

function nextColor() {
  const index = COLORS.indexOf(color.value)
  emit('patch', { pluginColor: COLORS[(index + 1) % COLORS.length] })
}
</script>

<template>
  <div class="pn-note" :style="{ background: color }" @dblclick.stop="editing = true">
    <textarea
      v-if="editing"
      ref="box"
      class="pn-note-input"
      :value="content || props.text"
      placeholder="写点什么…"
      @pointerdown.stop
      @blur="editing = false"
      @input="emit('patch', { content: ($event.target as HTMLTextAreaElement).value })"
    />
    <p v-else class="pn-note-text" :class="{ 'is-hint': !shown }">
      {{ shown || '双击写点什么…' }}
    </p>
    <button class="pn-note-color" title="换颜色" @pointerdown.stop @click.stop="nextColor" />
    <span v-if="!content && props.text" class="pn-note-from">来自上游</span>
  </div>
</template>

<style scoped>
.pn-note {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  padding: 10px 12px;
  overflow: hidden;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgb(0 0 0 / 35%);
}
.pn-note-text {
  flex: 1;
  margin: 0;
  overflow: hidden;
  font-size: 13px;
  line-height: 1.5;
  color: #1f2937;
  white-space: pre-wrap;
  word-break: break-word;
}
.pn-note-text.is-hint {
  color: #6b7280;
}
.pn-note-input {
  flex: 1;
  width: 100%;
  padding: 0;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  color: #1f2937;
  resize: none;
  background: transparent;
  border: none;
  outline: none;
}
.pn-note-color {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 14px;
  height: 14px;
  cursor: pointer;
  background: rgb(0 0 0 / 18%);
  border: none;
  border-radius: 50%;
}
.pn-note-color:hover {
  background: rgb(0 0 0 / 32%);
}
.pn-note-from {
  position: absolute;
  right: 8px;
  bottom: 6px;
  font-size: 10px;
  color: rgb(0 0 0 / 45%);
}
</style>

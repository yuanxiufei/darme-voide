<script setup lang="ts">
/**
 * Markdown 节点（`markdown:doc`）—— 双击进编辑，Esc/失焦回预览 ✓。
 *
 * 渲染走 `utils/canvasMarkdown.ts`（先转义再拼标签，链接只放行 http(s)/mailto/#）✓。
 */
import { computed, nextTick, ref, watch } from 'vue'

import { renderMarkdown } from '../../../utils/canvasMarkdown'

const props = defineProps<{ values: Record<string, string>; text: string }>()
const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

const editing = ref(false)
const box = ref<HTMLTextAreaElement | null>(null)

/** 与便利贴同一口径：自己没写就用上游文本（只显示、不写回 ✗） */
const source = computed(() => props.values.content || props.text || '')
const html = computed(() => renderMarkdown(source.value))

watch(editing, async (on) => {
  if (!on) return
  await nextTick()
  box.value?.focus()
})
</script>

<template>
  <div class="pn-md" @dblclick.stop="editing = true">
    <textarea
      v-if="editing"
      ref="box"
      class="pn-md-input"
      :value="source"
      placeholder="# 标题&#10;- 列表&#10;**粗体** 与 `代码`"
      @pointerdown.stop
      @keydown.esc.stop="editing = false"
      @blur="editing = false"
      @input="emit('patch', { content: ($event.target as HTMLTextAreaElement).value })"
    />
    <!-- eslint-disable-next-line vue/no-v-html -- renderMarkdown 已转义全部用户文本 -->
    <div v-else class="pn-md-view" v-html="html || '<p class=&quot;is-hint&quot;>双击编辑 Markdown…</p>'" />
  </div>
</template>

<style scoped>
.pn-md {
  width: 100%;
  height: 100%;
  padding: 10px 12px;
  overflow: auto;
}
.pn-md-input {
  width: 100%;
  height: 100%;
  padding: 0;
  font: inherit;
  font-size: 12.5px;
  line-height: 1.6;
  color: #e6e9ef;
  resize: none;
  background: transparent;
  border: none;
  outline: none;
}
.pn-md-view {
  font-size: 12.5px;
  line-height: 1.65;
  color: #d5dae4;
  word-break: break-word;
}
.pn-md-view :deep(h1),
.pn-md-view :deep(h2),
.pn-md-view :deep(h3) {
  margin: 0 0 6px;
  font-size: 15px;
  color: #fff;
}
.pn-md-view :deep(p) {
  margin: 0 0 6px;
}
.pn-md-view :deep(ul),
.pn-md-view :deep(ol) {
  margin: 0 0 6px;
  padding-left: 18px;
}
.pn-md-view :deep(code) {
  padding: 1px 4px;
  font-size: 11.5px;
  background: #0f1218;
  border-radius: 3px;
}
.pn-md-view :deep(pre) {
  margin: 0 0 6px;
  padding: 8px;
  overflow: auto;
  font-size: 11.5px;
  background: #0f1218;
  border-radius: 4px;
}
.pn-md-view :deep(blockquote) {
  margin: 0 0 6px;
  padding-left: 8px;
  color: #9aa3b2;
  border-left: 2px solid #2a2f3a;
}
.pn-md-view :deep(a) {
  color: #2dd4bf;
}
.pn-md-view :deep(hr) {
  margin: 8px 0;
  border: none;
  border-top: 1px solid #2a2f3a;
}
.pn-md-view :deep(.is-hint) {
  color: #6b7280;
}
</style>

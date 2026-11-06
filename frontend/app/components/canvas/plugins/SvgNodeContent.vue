<script setup lang="ts">
/**
 * SVG 节点（`svg:vector`）—— 源码编辑 + `<img src="data:image/svg+xml">` 预览 ✓。
 *
 * ## 为什么用 `<img>` 而不是 `v-html`
 *
 * SVG 里能写 `<script>` 与 `onload`。`v-html` 会把它塞进**当前文档**，等于把用户输入
 * 当代码执行 ✗。放进 `<img>` 的 data URL 里，浏览器按「图片」解析 —— 脚本不执行、
 * 外链不加载，是本仓唯一安全的做法 ✓（代价是 `<img>` 里不能交互，够用）。
 */
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps<{ values: Record<string, string>; text: string }>()
const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

const editing = ref(false)
const box = ref<HTMLTextAreaElement | null>(null)

const source = computed(() => props.values.content ?? '')
/** `encodeURIComponent` 而非 base64：中文/换行都不需要额外编码步骤，出错时还能直接看 ✓ */
const src = computed(() => {
  const body = source.value.trim()
  if (!body) return ''
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(body)}`
})

watch(editing, async (on) => {
  if (!on) return
  await nextTick()
  box.value?.focus()
})
</script>

<template>
  <div class="pn-svg" @dblclick.stop="editing = true">
    <textarea
      v-if="editing"
      ref="box"
      class="pn-svg-input"
      :value="source"
      spellcheck="false"
      placeholder="&lt;svg viewBox=&quot;0 0 100 100&quot;&gt;…&lt;/svg&gt;"
      @pointerdown.stop
      @blur="editing = false"
      @input="emit('patch', { content: ($event.target as HTMLTextAreaElement).value })"
    />
    <img v-else-if="src" class="pn-svg-view" :src="src" alt="SVG" draggable="false" />
    <p v-else class="pn-svg-hint">双击写 SVG 源码</p>
  </div>
</template>

<style scoped>
.pn-svg {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  padding: 8px;
  overflow: hidden;
}
.pn-svg-view {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  /* 透明背景声明（`transparentBackground` 在清单里，渲染上就是这个格）✓ */
  background:
    repeating-conic-gradient(#1b2029 0% 25%, #141922 0% 50%) 0 0 / 12px 12px;
  border-radius: 4px;
}
.pn-svg-input {
  width: 100%;
  height: 100%;
  padding: 6px;
  font: inherit;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px;
  line-height: 1.6;
  color: #d7f9f4;
  resize: none;
  background: #0f1218;
  border: 1px solid #2a2f3a;
  border-radius: 4px;
  outline: none;
}
.pn-svg-hint {
  margin: 0;
  font-size: 12px;
  color: #6b7280;
}
</style>

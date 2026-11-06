<script setup lang="ts">
/**
 * HTML 节点（`html:render`）—— `sandbox` iframe 渲染，`{{input}}` 处填上游文本 ✓。
 *
 * ## 三处不能含糊的地方
 *
 * 1. **`sandbox="allow-scripts"` 不带 `allow-same-origin`** —— 少了后半句，iframe 里的脚本
 *    就能碰到父页的 `localStorage`（整个画布存在那里）✗。只有 `allow-scripts` 时它是
 *    独立不透明源，脚本能跑、但拿不到本仓任何东西 ✓
 * 2. **默认 `pointer-events: none`** —— 不然在 iframe 上按下就会被它吃掉，画布拖不动 ✗。
 *    右上角「交互」按钮才把指针交还给页面内容
 * 3. **`{{input}}` 是占位符、不是追加** —— 用户写的模板里放哪儿就填哪儿；没有占位符时
 *    才把上游文本整段附在末尾（否则「引用上游」这件事会静默失效 ✗）
 */
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps<{ values: Record<string, string>; text: string }>()
const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

/** 上游文本的占位符 —— 写在 script 里而不是模板里（模板里 `{{ }}` 是插值 ✗） */
const INPUT_TOKEN = '{{input}}'

const editing = ref(false)
const interactive = ref(false)
const box = ref<HTMLTextAreaElement | null>(null)

const source = computed(() => props.values.content ?? '')
const rendered = computed(() => {
  const body = source.value
  if (!body.trim()) return ''
  const upstream = props.text || ''
  if (!upstream) return body
  return body.includes(INPUT_TOKEN)
    ? body.split(INPUT_TOKEN).join(upstream)
    : `${body}\n<!-- 上游文本 -->\n<div style="opacity:.75;font-size:12px">${upstream}</div>`
})

watch(editing, async (on) => {
  if (!on) return
  await nextTick()
  box.value?.focus()
})
</script>

<template>
  <div class="pn-html">
    <textarea
      v-if="editing"
      ref="box"
      class="pn-html-input"
      :value="source"
      spellcheck="false"
      placeholder="<h1>Hello</h1>"
      @pointerdown.stop
      @blur="editing = false"
      @input="emit('patch', { content: ($event.target as HTMLTextAreaElement).value })"
    />
    <template v-else>
      <iframe
        v-if="rendered"
        class="pn-html-frame"
        :class="{ 'is-interactive': interactive }"
        sandbox="allow-scripts"
        referrerpolicy="no-referrer"
        :srcdoc="rendered"
      />
      <p v-else class="pn-html-hint">双击写 HTML（<code>{{ INPUT_TOKEN }}</code> 处填上游文本）</p>
    </template>
    <div class="pn-html-bar">
      <button class="pn-html-btn" :disabled="!rendered" @pointerdown.stop @click.stop="interactive = !interactive">
        {{ interactive ? '交互开' : '交互关' }}
      </button>
      <button class="pn-html-btn" @pointerdown.stop @click.stop="editing = !editing">
        {{ editing ? '预览' : '编辑' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.pn-html {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.pn-html-frame {
  width: 100%;
  height: 100%;
  background: #fff;
  border: none;
  /* 不交出指针：否则按在 iframe 上画布拖不动 ✗ */
  pointer-events: none;
}
.pn-html-frame.is-interactive {
  pointer-events: auto;
}
.pn-html-input {
  width: 100%;
  height: 100%;
  padding: 10px 12px;
  font: inherit;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  line-height: 1.6;
  color: #e6e9ef;
  resize: none;
  background: #0f1218;
  border: none;
  outline: none;
}
.pn-html-hint {
  margin: 0;
  padding: 12px;
  font-size: 12px;
  color: #6b7280;
}
.pn-html-hint code {
  padding: 1px 4px;
  background: #0f1218;
  border-radius: 3px;
}
.pn-html-bar {
  position: absolute;
  top: 6px;
  right: 6px;
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.15s;
}
.pn-html:hover .pn-html-bar {
  opacity: 1;
}
.pn-html-btn {
  padding: 2px 6px;
  font-size: 10px;
  color: #cbd5e1;
  cursor: pointer;
  background: rgb(15 18 24 / 88%);
  border: 1px solid #2a2f3a;
  border-radius: 4px;
}
.pn-html-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>

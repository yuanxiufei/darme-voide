<template>
  <Teleport to="body">
    <Transition name="agent-out">
      <div v-if="open && output" class="ao-overlay" @click.self="close">
        <div class="ao-panel" role="dialog" aria-modal="true">
          <header class="ao-head">
            <div class="ao-icon" :class="{ warn: !output.acted }">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <div class="ao-title-wrap">
              <h3 class="ao-title">{{ output.label }} · AI 输出</h3>
              <p class="ao-sub">用时 {{ output.seconds }}s<span v-if="output.toolNames.length"> · 调用 {{ output.toolNames.length }} 个工具</span></p>
            </div>
            <span class="ao-badge" :class="output.acted ? 'ok' : 'warn'">
              {{ output.acted ? '已保存' : '未保存' }}
            </span>
            <button class="ao-close" type="button" aria-label="关闭" @click="close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </header>

          <p v-if="!output.acted" class="ao-hint">
            这次 AI <strong>没有调用任何保存工具</strong>，数据库没有被改动 —— 下面的文字是它的原话，可以据此判断它卡在哪，或直接重试。
          </p>

          <div v-if="output.toolNames.length" class="ao-tools">
            <span v-for="(name, i) in output.toolNames" :key="`${name}-${i}`" class="ao-tool">{{ name }}</span>
          </div>

          <div class="ao-body">
            <pre v-if="output.text" class="ao-text">{{ output.text }}</pre>
            <p v-else class="ao-empty">AI 这次没有返回文字说明。</p>
          </div>

          <footer class="ao-actions">
            <button v-if="output.text" class="btn btn-ghost" type="button" @click="copy">
              {{ copied ? '已复制' : '复制' }}
            </button>
            <button class="btn btn-ghost" type="button" :disabled="busy" @click="retry">
              {{ busy ? '执行中…' : '重试' }}
            </button>
            <button class="btn btn-primary" type="button" @click="close">关闭</button>
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import type { AgentOutput } from '~/composables/useAgent'

const props = defineProps<{
  open: boolean
  output: AgentOutput | null
  /** 重试请求进行中（按钮置灰，避免重复触发） */
  busy?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void
  (e: 'retry'): void
}>()

const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | undefined

function close() {
  emit('update:open', false)
}

function retry() {
  emit('retry')
}

async function copy() {
  const text = props.output?.text
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    if (copiedTimer) clearTimeout(copiedTimer)
    copiedTimer = setTimeout(() => { copied.value = false }, 1500)
  } catch {
    // 剪贴板不可用（非 https / 无权限）时不弹错，用户可手动选中
  }
}

onBeforeUnmount(() => {
  if (copiedTimer) clearTimeout(copiedTimer)
})
</script>

<style scoped>
.ao-overlay {
  position: fixed;
  inset: 0;
  z-index: 9998;
  background: rgba(24, 33, 50, 0.42);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.ao-panel {
  width: 680px;
  max-width: 100%;
  max-height: 82vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-0);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  box-shadow: var(--shadow-panel);
  padding: 20px 22px 18px;
}
.ao-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.ao-icon {
  width: 34px;
  height: 34px;
  flex-shrink: 0;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-bg);
  color: var(--accent);
}
.ao-icon.warn {
  background: var(--error-bg);
  color: var(--error);
}
.ao-title-wrap {
  flex: 1;
  min-width: 0;
}
.ao-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-0);
  margin: 0 0 3px;
  letter-spacing: -0.01em;
}
.ao-sub {
  font-size: 12px;
  color: var(--text-2);
  margin: 0;
}
.ao-badge {
  font-size: 11.5px;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 999px;
  white-space: nowrap;
}
.ao-badge.ok {
  background: var(--accent-bg);
  color: var(--accent-text);
}
.ao-badge.warn {
  background: var(--error-bg);
  color: var(--error);
}
.ao-close {
  border: none;
  background: transparent;
  color: var(--text-2);
  cursor: pointer;
  padding: 4px;
  border-radius: 6px;
  line-height: 0;
}
.ao-close:hover {
  background: var(--bg-input);
  color: var(--text-0);
}
.ao-hint {
  margin: 14px 0 0;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--error-bg);
  color: var(--text-1);
  font-size: 12.5px;
  line-height: 1.65;
}
.ao-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}
.ao-tool {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px;
  padding: 3px 8px;
  border-radius: 6px;
  background: var(--bg-input);
  color: var(--text-2);
  white-space: nowrap;
}
.ao-body {
  margin-top: 12px;
  flex: 1;
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  background: var(--bg-input);
  padding: 12px 14px;
}
.ao-text {
  margin: 0;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.75;
  color: var(--text-1);
  white-space: pre-wrap;
  word-break: break-word;
}
.ao-empty {
  margin: 0;
  font-size: 13px;
  color: var(--text-2);
}
.ao-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
}
.ao-enter-active,
.ao-leave-active {
  transition: opacity 0.15s ease;
}
.ao-enter-active .ao-panel,
.ao-leave-active .ao-panel {
  transition: transform 0.15s ease, opacity 0.15s ease;
}
.ao-enter-from,
.ao-leave-to {
  opacity: 0;
}
.ao-enter-from .ao-panel,
.ao-leave-to .ao-panel {
  transform: translateY(8px) scale(0.98);
}
</style>

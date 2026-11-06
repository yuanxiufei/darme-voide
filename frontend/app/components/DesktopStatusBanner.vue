<!--
  桌面端"本地服务没起来"的横幅。

  ⚠️ 只负责**告知 + 重试**，不负责拉起后端（拉起由壳在 `app_bootstrap` 里做，
     这里只是给用户一个手动重试的入口）。Web 形态下 `isDesktop` 为 false ⇒ 永不渲染。
-->
<template>
  <div v-if="visible" class="desk-banner" role="alert">
    <div class="desk-banner__text">
      <strong>{{ title }}</strong>
      <span>{{ detail }}</span>
    </div>
    <div class="desk-banner__actions">
      <button class="desk-banner__btn" :disabled="busy" @click="onRetry">
        {{ busy ? '启动中…' : '重新启动本地服务' }}
      </button>
      <button v-if="logPath" class="desk-banner__btn desk-banner__btn--ghost" @click="onOpenLog">
        查看日志
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  backendStatus,
  desktopBridgeError,
  isDesktop,
  openPath,
  startBackend,
} from '../composables/useDesktop'

const busy = ref(false)
const lastError = ref<string | null>(null)

const state = computed(() => backendStatus.value?.state ?? 'unknown')
/** `running` 之外的桌面状态都值得提示；`external` 表示复用外部后端，属正常 */
const visible = computed(
  () => isDesktop.value && !['running', 'external'].includes(state.value),
)

const title = computed(() => {
  if (desktopBridgeError.value) return '桌面壳初始化失败'
  if (state.value === 'starting') return '本地服务正在启动'
  if (state.value === 'failed') return '本地服务启动失败'
  if (state.value === 'stopped') return '本地服务未运行'
  return '本地服务状态未知'
})

const detail = computed(() => {
  const err = lastError.value || desktopBridgeError.value || backendStatus.value?.message
  return err || '界面数据依赖本地 Python 服务（默认 127.0.0.1:5790），未启动时全部接口不可用。'
})

const logPath = computed(() => backendStatus.value?.logPath ?? null)

async function onRetry() {
  busy.value = true
  lastError.value = null
  try {
    await startBackend()
  } catch (err: any) {
    lastError.value = err?.message || String(err)
  } finally {
    busy.value = false
  }
}

async function onOpenLog() {
  if (logPath.value) await openPath(logPath.value)
}
</script>

<style scoped>
.desk-banner {
  position: fixed;
  left: 50%;
  bottom: 20px;
  transform: translateX(-50%);
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 16px;
  max-width: min(880px, calc(100vw - 40px));
  padding: 12px 16px;
  border: 1px solid rgba(239, 83, 80, 0.5);
  border-radius: 10px;
  background: #2a1618;
  color: #ffdede;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
  font-size: 13px;
  line-height: 1.5;
}
.desk-banner__text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.desk-banner__text span {
  color: #f0b6b6;
  word-break: break-all;
}
.desk-banner__actions {
  display: flex;
  gap: 8px;
  flex: none;
}
.desk-banner__btn {
  padding: 6px 12px;
  border-radius: 7px;
  border: 1px solid rgba(239, 83, 80, 0.6);
  background: rgba(239, 83, 80, 0.18);
  color: #ffdede;
  cursor: pointer;
  font-size: 12px;
  white-space: nowrap;
}
.desk-banner__btn:disabled {
  opacity: 0.6;
  cursor: default;
}
.desk-banner__btn--ghost {
  border-color: rgba(255, 255, 255, 0.2);
  background: transparent;
}
</style>

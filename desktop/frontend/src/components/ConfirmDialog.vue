<template>
  <Teleport to="body">
    <Transition name="confirm">
      <div v-if="state.visible" class="confirm-overlay" @click.self="settle(false)">
        <div class="confirm-panel" role="dialog" aria-modal="true">
          <div class="confirm-icon" :class="{ danger: state.danger }">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h3 class="confirm-title">{{ state.title }}</h3>
          <p class="confirm-message">{{ state.message }}</p>
          <div class="confirm-actions">
            <button class="btn btn-ghost" @click="settle(false)">{{ state.cancelText }}</button>
            <button class="btn" :class="state.danger ? 'btn-danger' : 'btn-primary'" @click="settle(true)">
              {{ state.confirmText }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { useConfirm } from '~/composables/useConfirm'
const { state, settle } = useConfirm()
</script>

<style scoped>
.confirm-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(24, 33, 50, 0.4);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.confirm-panel {
  width: 400px;
  max-width: 100%;
  background: var(--bg-0);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  box-shadow: var(--shadow-panel);
  padding: 24px;
}
.confirm-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-bg);
  color: var(--accent);
  margin-bottom: 14px;
}
.confirm-icon.danger {
  background: var(--error-bg);
  color: var(--error);
}
.confirm-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-0);
  margin: 0 0 8px;
  letter-spacing: -0.01em;
}
.confirm-message {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--text-2);
  margin: 0;
  white-space: pre-line;
  word-break: break-word;
}
.confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 22px;
}
.btn-danger {
  background: var(--error);
  color: #fff;
  border-color: transparent;
  box-shadow: 0 8px 20px rgba(210, 79, 102, 0.24);
}
.btn-danger:hover {
  background: var(--error);
  box-shadow: 0 10px 24px rgba(210, 79, 102, 0.32);
}
.confirm-enter-active,
.confirm-leave-active {
  transition: opacity 0.15s ease;
}
.confirm-enter-active .confirm-panel,
.confirm-leave-active .confirm-panel {
  transition: transform 0.15s ease, opacity 0.15s ease;
}
.confirm-enter-from,
.confirm-leave-to {
  opacity: 0;
}
.confirm-enter-from .confirm-panel,
.confirm-leave-to .confirm-panel {
  transform: translateY(8px) scale(0.98);
}
</style>

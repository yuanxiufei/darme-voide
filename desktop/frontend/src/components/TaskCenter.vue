<template>
  <div class="tc-root">
    <transition name="tc-panel">
      <section v-if="open" class="tc-panel" role="dialog" aria-label="任务中心">
        <header class="tc-head">
          <span class="tc-head-title">任务</span>
          <span v-if="summary.running" class="tc-chip run">进行中 {{ summary.running }}</span>
          <span v-if="summary.failed" class="tc-chip bad">失败 {{ summary.failed }}</span>
          <button class="tc-icon-btn" title="收起" @click="open = false">✕</button>
        </header>

        <div class="tc-list">
          <p v-if="!sorted.length" class="tc-empty">没有任务</p>
          <article
            v-for="t in sorted"
            :key="t.id"
            class="tc-item"
            :class="t.status"
          >
            <div class="tc-item-head">
              <span class="tc-badge" :class="t.kind">{{ KIND_LABEL[t.kind] }}</span>
              <span class="tc-title" :title="t.title">{{ t.title }}</span>
              <button class="tc-icon-btn sm" title="移除" @click="tc.remove(t.id)">✕</button>
            </div>

            <p v-if="t.subtitle" class="tc-sub">{{ t.subtitle }}</p>

            <!-- 进行中：阶段文字 + （有真实百分比时才画条；没有就画不确定条纹，不编造数字） -->
            <template v-if="t.status === 'running' || t.status === 'queued'">
              <div v-if="t.progress !== null" class="tc-bar"><i :style="{ width: `${t.progress}%` }" /></div>
              <div v-else class="tc-bar indet"><i /></div>
              <!-- 取消只给**真的能停**的任务（发起方声明 onCancel）；已提交的渲染撤不回来，就不画 -->
              <div v-if="t.onCancel" class="tc-actions">
                <button class="tc-btn ghost" @click="tc.requestCancel(t.id)">取消</button>
                <p class="tc-meta">{{ t.stage }} · {{ taskElapsed(t, now) }}</p>
              </div>
              <p v-else class="tc-meta">{{ t.stage }} · {{ taskElapsed(t, now) }}</p>
            </template>

            <!-- 失败：归因标签 + 建议 + 原始文案 + 两个动作 -->
            <template v-else-if="t.status === 'failed'">
              <p class="tc-fail">
                <span class="tc-fail-tag" :class="t.failure?.kind">{{ t.failure?.label || '失败' }}</span>
                <span class="tc-fail-hint">{{ t.failure?.hint }}</span>
              </p>
              <p v-if="t.failure?.message" class="tc-raw" :title="t.failure.message">{{ t.failure.message }}</p>
              <div class="tc-actions">
                <!-- 主按钮按归因决定：可重试的（网络/超时/未知）给「重试」；
                     要改东西的（缺参数/模型/额度）给「去处理」——两类错**解法不同**，
                     原来只有「重试」，等于把改参数的人往同一个坑里推 ✓✗ -->
                <button
                  v-if="t.failure?.action.kind === 'retry' && t.retry"
                  class="tc-btn primary"
                  @click="tc.retryTask(t.id)"
                >{{ t.failure?.action.label || '重试' }}</button>
                <button
                  v-if="gotoOf(t)"
                  :class="['tc-btn', t.failure?.action.kind === 'goto' ? 'primary' : '']"
                  @click="go(t)"
                >{{ t.failure?.action.label || '去看看' }}</button>
                <!-- 「去处理」类错也给重试（次按钮）：用户可能就是想先再撞一次，
                     但视觉上不许它抢主按钮位 —— 主按钮必须是那一类错真正的解法 ★ -->
                <button
                  v-if="t.failure?.action.kind === 'goto' && t.retry"
                  class="tc-btn ghost"
                  @click="tc.retryTask(t.id)"
                >仍重试</button>
                <span class="tc-meta">{{ taskElapsed(t, now) }}</span>
              </div>
            </template>

            <p v-else class="tc-meta">
              {{ t.status === 'canceled' ? '已取消' : '完成' }} · {{ taskElapsed(t, now) }}
            </p>
          </article>
        </div>

        <footer v-if="hasFinished" class="tc-foot">
          <button class="tc-btn" @click="tc.clearFinished()">清空已完成</button>
        </footer>
      </section>
    </transition>

    <button
      v-if="!open && summary.total > 0"
      class="tc-fab"
      :class="{ bad: summary.failed > 0, run: summary.running > 0 && summary.failed === 0 }"
      :title="fabTitle"
      @click="open = true"
    >
      <span v-if="summary.running" class="tc-spin" aria-hidden="true" />
      <span v-else class="tc-fab-mark" aria-hidden="true">✓</span>
      <span class="tc-fab-num">{{ summary.failed || summary.running }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
/**
 * 任务中心（右下角）—— `useTaskCenter` 的可见面。
 *
 * 为什么是「全局挂载 + 抽屉」而不是每个页面各做一块：
 * 本仓的任务最怕的就是「切页即丢」（`pendingVideoIds` 是页面内 ref ✓✗），
 * 挂在 `app.vue` 上才能保证「去改个参数再回来，那条还在跑/还失败着」。
 *
 * 视觉纪律：
 *   - 有真实百分比才画百分比条；没有就画不确定条纹（**不写 95% 这种假读数** ✓✗）；
 *   - 失败行必须给出「哪一类 + 下一步干什么 + 一个按钮」，不允许只丢一句红字；
 *   - 成功后自动折叠到最下面，失败永远排最上面。
 */
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useTaskCenter, taskElapsed, failureRoute, type TaskItem, type TaskKind } from '~/composables/useTaskCenter'

const tc = useTaskCenter()
const router = useRouter()
const open = ref(false)
const now = ref(Date.now())

const KIND_LABEL: Record<TaskKind, string> = {
  image: '图',
  video: '视',
  tts: '音',
  compose: '合',
  merge: '拼',
  agent: '脑',
  export: '导',
  pipeline: '线',
}

const RANK: Record<string, number> = { failed: 0, running: 1, queued: 2, canceled: 3, success: 4 }

// `tc.summary` 是 ComputedRef —— 模板里的 `summary.running` **不会**自动解包（它不是扁平 ref），
// 所以在这里摊成一个本地 computed（顶层 ref 才会被 Vue 模板解包 ✓）
const summary = computed(() => tc.summary.value)

const sorted = computed(() =>
  [...tc.items].sort((a, b) => {
    const r = (RANK[a.status] ?? 9) - (RANK[b.status] ?? 9)
    return r !== 0 ? r : b.startedAt - a.startedAt
  }),
)

const hasFinished = computed(() => tc.items.some(t => t.status === 'success' || t.status === 'canceled'))
const fabTitle = computed(() => {
  const { running, failed } = tc.summary.value
  if (failed) return `${failed} 个任务失败${running ? `，${running} 个进行中` : ''}`
  return `${running} 个任务进行中`
})

function gotoOf(t: TaskItem): string {
  // 只有「该去改东西」的失败才画这个按钮：可重试的错硬塞一个跳转只会分散注意力
  if (t.failure && t.failure.action.kind !== 'goto') return ''
  return failureRoute(t)
}

function go(t: TaskItem) {
  const target = gotoOf(t)
  if (!target) return
  router.push(target)
  open.value = false
}

let timer: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  // 1 秒一刷只为算耗时；任务为空时也照跑（一次 setInterval 的代价可忽略）
  timer = setInterval(() => { now.value = Date.now() }, 1000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.tc-root {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 9000;
  font-family: var(--font-body);
}

/* ---------- 浮动入口 ---------- */
.tc-fab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 16px;
  height: 44px;
  border: none;
  border-radius: 22px;
  background: var(--accent-gradient);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--shadow-lg);
  transition: transform 0.15s var(--ease-out), box-shadow 0.15s;
}
.tc-fab:hover { transform: translateY(-1px); box-shadow: var(--shadow-xl); }
.tc-fab.bad { background: linear-gradient(135deg, #e06a7d, var(--error)); }
.tc-fab.run { background: var(--accent-gradient); }
.tc-fab-mark { font-size: 13px; opacity: 0.9; }
.tc-fab-num { font-variant-numeric: tabular-nums; }

.tc-spin {
  width: 12px;
  height: 12px;
  border: 2px solid rgba(255, 255, 255, 0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: tc-rotate 0.8s linear infinite;
}
@keyframes tc-rotate { to { transform: rotate(360deg); } }

/* ---------- 面板 ---------- */
.tc-panel {
  width: 372px;
  max-height: min(68vh, 560px);
  display: flex;
  flex-direction: column;
  background: var(--bg-0);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-xl);
  overflow: hidden;
}

.tc-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-base);
}
.tc-head-title { font-size: 13px; font-weight: 700; color: var(--text-0, #1e293b); }

.tc-chip {
  font-size: 11px;
  padding: 2px 7px;
  border-radius: 10px;
  font-variant-numeric: tabular-nums;
}
.tc-chip.run { color: var(--accent-text); background: var(--accent-bg); }
.tc-chip.bad { color: var(--error); background: var(--error-bg); }

.tc-icon-btn {
  margin-left: auto;
  width: 22px;
  height: 22px;
  border: none;
  background: transparent;
  color: var(--text-3, #94a3b8);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 12px;
  line-height: 1;
}
.tc-icon-btn:hover { background: var(--bg-hover); color: var(--text-1, #334155); }
.tc-icon-btn.sm { width: 18px; height: 18px; opacity: 0; font-size: 11px; }
.tc-item:hover .tc-icon-btn.sm { opacity: 1; }

.tc-list { flex: 1; overflow-y: auto; padding: 6px; }
.tc-empty { margin: 24px 0; text-align: center; font-size: 12px; color: var(--text-3, #94a3b8); }

.tc-item {
  padding: 9px 10px;
  border-radius: var(--radius);
  border: 1px solid transparent;
}
.tc-item + .tc-item { margin-top: 2px; }
.tc-item:hover { background: var(--bg-hover); }
.tc-item.failed { background: var(--error-bg); border-color: rgba(210, 79, 102, 0.18); }
.tc-item.success .tc-title { color: var(--text-2, #64748b); }

.tc-item-head { display: flex; align-items: center; gap: 7px; }

.tc-badge {
  flex: none;
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  font-size: 10px;
  font-weight: 700;
  border-radius: 5px;
  color: var(--accent-text);
  background: var(--accent-bg);
}
.tc-badge.video { color: var(--info); background: var(--info-bg); }
.tc-badge.tts { color: var(--warning); background: var(--warning-bg); }
.tc-badge.export { color: var(--success); background: var(--success-bg); }

.tc-title {
  flex: 1;
  min-width: 0;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text-0, #1e293b);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tc-sub {
  margin: 3px 0 0 25px;
  font-size: 11.5px;
  color: var(--text-2, #64748b);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tc-meta {
  margin: 4px 0 0 25px;
  font-size: 11px;
  color: var(--text-3, #94a3b8);
  font-variant-numeric: tabular-nums;
}

/* 进度条：确定 / 不确定 */
.tc-bar {
  margin: 6px 0 0 25px;
  height: 3px;
  border-radius: 2px;
  background: var(--bg-3);
  overflow: hidden;
}
.tc-bar i { display: block; height: 100%; background: var(--accent-gradient); transition: width 0.4s var(--ease-out); }
.tc-bar.indet i {
  width: 35%;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: tc-sweep 1.3s ease-in-out infinite;
}
@keyframes tc-sweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(320%); }
}

.tc-fail { display: flex; gap: 6px; align-items: flex-start; margin: 6px 0 0 25px; }
.tc-fail-tag {
  flex: none;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 4px;
  color: var(--error);
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(210, 79, 102, 0.3);
}
.tc-fail-hint { font-size: 11px; line-height: 1.45; color: var(--text-1, #334155); }

.tc-raw {
  margin: 4px 0 0 25px;
  font-size: 10.5px;
  line-height: 1.4;
  color: var(--text-3, #94a3b8);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tc-actions { display: flex; align-items: center; gap: 6px; margin: 7px 0 0 25px; }
.tc-actions .tc-meta { margin: 0 0 0 auto; }

.tc-btn {
  padding: 3px 10px;
  font-size: 11.5px;
  border-radius: 6px;
  border: 1px solid var(--border-strong);
  background: var(--bg-0);
  color: var(--text-1, #334155);
  cursor: pointer;
}
.tc-btn:hover { background: var(--bg-hover); }
.tc-btn.primary { border-color: var(--accent); color: var(--accent-text); background: var(--accent-bg); }
/* 「仍重试」：属于次选动作，压成无框文字，不让它跟主按钮抢 */
.tc-btn.ghost { border-color: transparent; background: transparent; color: var(--text-3, #94a3b8); }
.tc-btn.ghost:hover { color: var(--text-1, #334155); text-decoration: underline; }

.tc-foot {
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  background: var(--bg-base);
  text-align: right;
}

/* 进出场 */
.tc-panel-enter-active, .tc-panel-leave-active { transition: opacity 0.16s, transform 0.16s var(--ease-out); }
.tc-panel-enter-from, .tc-panel-leave-to { opacity: 0; transform: translateY(8px) scale(0.98); }

@media (max-width: 560px) {
  .tc-root { right: 12px; bottom: 12px; }
  .tc-panel { width: calc(100vw - 24px); max-height: 62vh; }
}
</style>

<template>
  <Teleport to="body">
    <Transition name="ob">
      <div v-if="visible" class="ob-overlay">
        <div class="ob-panel">
          <!-- 顶部：品牌 + 跳过 -->
          <div class="ob-top">
            <div class="ob-brand">
              <div class="ob-brand-mark">剧</div>
              <div class="ob-brand-text">
                <span class="ob-brand-name">短剧工坊</span>
                <span class="ob-brand-sub">Drama Studio</span>
              </div>
            </div>
            <button v-if="step < 4" class="ob-skip" @click="finish">跳过</button>
          </div>

          <!-- 进度点 -->
          <div class="ob-dots">
            <span
              v-for="i in 5"
              :key="i"
              class="ob-dot"
              :class="{ active: i - 1 === step, done: i - 1 < step }"
            />
          </div>

          <!-- 页面内容 -->
          <div class="ob-body">
            <!-- 1. 欢迎 -->
            <div v-if="step === 0" class="ob-page">
              <div class="ob-hero">
                <div class="ob-hero-glow"></div>
                <div class="ob-logo">剧</div>
                <svg class="ob-spark" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2l2.4 4.9 5.6 1.4-4 3.9.9 5.5L12 15l-4.9 2.7.9-5.5-4-3.9 5.6-1.4z"/>
                </svg>
              </div>
              <h1 class="ob-title">嗨，创作者</h1>
              <p class="ob-sub">把你的故事，变成会动的短剧</p>
              <p class="ob-desc">只需一段剧本，AI 帮你搞定剩下的一切</p>
              <div class="ob-actions">
                <button class="ob-btn ob-btn-primary" @click="next">看看怎么玩</button>
                <button class="ob-link" @click="finish">稍后了解，直接开始</button>
              </div>
            </div>

            <!-- 2. 工作流 -->
            <div v-else-if="step === 1" class="ob-page">
              <h2 class="ob-title">四步做出短剧</h2>
              <p class="ob-desc">从剧本到成片，AI 全程助力</p>
              <div class="ob-steps">
                <div v-for="s in workflowSteps" :key="s.number" class="ob-step">
                  <div class="ob-step-num">{{ s.number }}</div>
                  <div class="ob-step-body">
                    <h3 class="ob-step-title">{{ s.title }}</h3>
                    <p class="ob-step-desc">{{ s.description }}</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- 3. 亮点 -->
            <div v-else-if="step === 2" class="ob-page">
              <h2 class="ob-title">核心亮点</h2>
              <p class="ob-desc">为短剧制作量身打造</p>
              <div class="ob-highlights">
                <div v-for="h in highlights" :key="h.title" class="ob-hl">
                  <div class="ob-hl-icon">{{ h.icon }}</div>
                  <h3 class="ob-hl-title">{{ h.title }}</h3>
                  <p class="ob-hl-desc">{{ h.description }}</p>
                </div>
              </div>
            </div>

            <!-- 4. AI 服务配置 -->
            <div v-else-if="step === 3" class="ob-page">
              <h2 class="ob-title">连接你的 AI 服务</h2>
              <p class="ob-desc">在「设置」中配置图片 / 视频 / 音频 / 文本模型服务，支持多厂商、多模型自动切换</p>
              <div class="ob-config">
                <div class="ob-config-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                  </svg>
                </div>
                <p class="ob-config-text">文本 · 图片 · 视频 · 音频，四种能力分别配置对应模型服务，无需填写 API Key 也可先体验</p>
              </div>
              <div class="ob-actions">
                <button class="ob-btn ob-btn-primary" @click="goSettings">前往设置</button>
              </div>
            </div>

            <!-- 5. 行动 -->
            <div v-else class="ob-page">
              <h2 class="ob-title">现在就开始创作</h2>
              <p class="ob-desc">选一个方式，马上体验</p>
              <div class="ob-quick">
                <button v-for="q in quickStarts" :key="q.id" class="ob-q" @click="quickStart(q.id)">
                  <div class="ob-q-icon">{{ q.icon }}</div>
                  <div class="ob-q-body">
                    <h3 class="ob-q-title">{{ q.title }}</h3>
                    <p class="ob-q-desc">{{ q.description }}</p>
                  </div>
                </button>
              </div>
              <div class="ob-actions">
                <button class="ob-btn ob-btn-primary" @click="quickStart('create')">创建我的第一部短剧</button>
                <p class="ob-hint">以后可在顶部「帮助」中重新查看引导</p>
              </div>
            </div>
          </div>

          <!-- 底部导航 -->
          <div class="ob-footer">
            <button v-if="step > 0" class="ob-btn" @click="prev">上一步</button>
            <span class="ob-footer-spacer"></span>
            <button v-if="step < 4" class="ob-btn ob-btn-primary" @click="next">下一步</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
const STORAGE_KEY = 'ds_onboarding_completed'

// 全局打开信号（供顶部「帮助」按钮重新触发）
const openSignal = useState('onboarding-open', () => false)

const step = ref(0)
const visible = ref(false)
const completed = ref(false)

const workflowSteps = [
  { number: '①', title: '写剧本', description: 'AI 自动提取角色和场景' },
  { number: '②', title: '定形象', description: '一键生成角色定妆照' },
  { number: '③', title: '排分镜', description: '关键帧驱动视频生成' },
  { number: '④', title: '导成片', description: '合并导出完整短剧' },
]

const highlights = [
  { icon: '🎬', title: '首尾帧锁定', description: '上一镜的结尾，就是下一镜的开头' },
  { icon: '👔', title: '角色衣橱', description: '同一角色，多套造型随时切换' },
  { icon: '🎨', title: '风格统一', description: '真人、动漫、3D 任选，全片一致' },
]

const quickStarts = [
  { id: 'autogen', icon: '✨', title: 'AI 一键生成', description: '输入故事梗概，AI 全流程制作' },
  { id: 'example', icon: '📁', title: '看看示例项目', description: '先逛逛别人怎么做的' },
]

function next() {
  if (step.value < 4) step.value++
}

function prev() {
  if (step.value > 0) step.value--
}

function finish() {
  completed.value = true
  visible.value = false
  openSignal.value = false
  try { localStorage.setItem(STORAGE_KEY, '1') } catch { /* 忽略存储失败 */ }
}

function goSettings() {
  finish()
  navigateTo('/settings')
}

function quickStart(id) {
  finish()
  if (id === 'autogen') navigateTo('/?auto=1')
  else if (id === 'example') navigateTo('/library')
  else navigateTo('/?new=1')
}

// 初始：未完成引导则自动弹出
onMounted(() => {
  let done = false
  try { done = localStorage.getItem(STORAGE_KEY) === '1' } catch { /* 忽略 */ }
  completed.value = done
  if (!done) visible.value = true
})

// 监听外部「帮助」按钮重新打开
watch(openSignal, (v) => {
  if (v) {
    step.value = 0
    visible.value = true
  }
})
</script>

<style scoped>
.ob-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(8, 10, 16, 0.72);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

.ob-panel {
  width: 480px;
  max-width: calc(100vw - 32px);
  max-height: calc(100vh - 32px);
  overflow-y: auto;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg, 16px);
  padding: 24px 28px 20px;
  box-shadow: var(--shadow-elevated, 0 24px 64px rgba(0, 0, 0, 0.5));
}

/* 顶部 */
.ob-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ob-brand { display: flex; align-items: center; gap: 10px; }
.ob-brand-mark {
  width: 30px; height: 30px;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-2); border-radius: var(--radius);
  border: 1px solid var(--border);
  font-family: var(--font-display);
  font-size: 15px; font-weight: 700; color: var(--accent-text);
}
.ob-brand-text { display: flex; flex-direction: column; line-height: 1; }
.ob-brand-name { font-family: var(--font-display); font-size: 14px; font-weight: 700; color: var(--text-0); }
.ob-brand-sub { font-size: 10px; color: var(--text-3); margin-top: 1px; letter-spacing: 0.04em; }
.ob-skip {
  background: none; border: none; cursor: pointer;
  font-size: 12px; color: var(--text-3);
  transition: color 0.15s;
}
.ob-skip:hover { color: var(--text-0); }

/* 进度点 */
.ob-dots {
  display: flex; justify-content: center; gap: 8px;
  margin: 18px 0 22px;
}
.ob-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--bg-3);
  transition: all 0.2s var(--ease-out);
}
.ob-dot.active { background: var(--accent); transform: scale(1.3); }
.ob-dot.done { background: var(--accent-text); opacity: 0.6; }

/* 内容 */
.ob-body { min-height: 300px; display: flex; flex-direction: column; }
.ob-page { display: flex; flex-direction: column; align-items: center; text-align: center; }

.ob-title {
  font-family: var(--font-display);
  font-size: 24px; font-weight: 700; color: var(--text-0);
  margin: 0 0 6px;
}
.ob-sub { font-size: 17px; color: var(--text-1); margin: 0 0 4px; }
.ob-desc { font-size: 13px; color: var(--text-3); margin: 0; line-height: 1.6; }

/* 欢迎 hero */
.ob-hero { position: relative; margin-bottom: 22px; }
.ob-hero-glow {
  position: absolute; inset: -28px;
  background: radial-gradient(circle, rgba(13,148,136,0.25), rgba(168,85,247,0.15), transparent 70%);
  border-radius: 50%;
  filter: blur(20px);
}
.ob-logo {
  position: relative;
  width: 84px; height: 84px;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: 22px;
  font-family: var(--font-display);
  font-size: 40px; font-weight: 700; color: var(--accent-text);
}
.ob-spark {
  position: absolute; top: -6px; right: -8px;
  color: var(--accent);
  animation: ob-pulse 1.6s ease-in-out infinite;
}
@keyframes ob-pulse { 0%, 100% { opacity: 0.5; transform: scale(1); } 50% { opacity: 1; transform: scale(1.15); } }

/* 按钮 */
.ob-actions { display: flex; flex-direction: column; align-items: center; gap: 14px; margin-top: 24px; }
.ob-btn {
  padding: 9px 22px;
  border-radius: var(--radius);
  font-size: 13px; font-weight: 600;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--bg-2); color: var(--text-1);
  transition: all 0.18s var(--ease-out);
}
.ob-btn:hover { border-color: var(--accent); color: var(--text-0); }
.ob-btn-primary {
  background: var(--accent); color: #fff; border-color: transparent;
  font-size: 14px; padding: 10px 28px;
}
.ob-btn-primary:hover { filter: brightness(1.06); color: #fff; }
.ob-link {
  background: none; border: none; cursor: pointer;
  font-size: 12px; color: var(--text-3);
  transition: color 0.15s;
}
.ob-link:hover { color: var(--text-0); }
.ob-hint { font-size: 10px; color: var(--text-3); margin: 0; }

/* 工作流步骤 */
.ob-steps { display: flex; flex-direction: column; gap: 12px; width: 100%; margin-top: 22px; }
.ob-step {
  display: flex; align-items: center; gap: 14px;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 13px 16px;
  text-align: left;
}
.ob-step-num {
  width: 34px; height: 34px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-bg); border-radius: var(--radius);
  font-size: 15px; font-weight: 700; color: var(--accent-text);
}
.ob-step-title { font-size: 14px; font-weight: 600; color: var(--text-0); margin: 0; }
.ob-step-desc { font-size: 12px; color: var(--text-3); margin: 2px 0 0; }

/* 亮点 */
.ob-highlights { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; width: 100%; margin-top: 22px; }
.ob-hl {
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 12px;
  display: flex; flex-direction: column; align-items: center; gap: 8px;
}
.ob-hl-icon { font-size: 24px; }
.ob-hl-title { font-size: 13px; font-weight: 600; color: var(--text-0); margin: 0; }
.ob-hl-desc { font-size: 11px; color: var(--text-3); margin: 0; line-height: 1.5; }

/* AI 服务配置 */
.ob-config {
  margin-top: 22px;
  display: flex; flex-direction: column; align-items: center; gap: 14px;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 22px 20px;
}
.ob-config-icon {
  width: 52px; height: 52px;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-bg); border-radius: 50%; color: var(--accent);
}
.ob-config-text { font-size: 13px; color: var(--text-2); margin: 0; line-height: 1.7; max-width: 320px; }

/* 快速开始 */
.ob-quick { display: flex; flex-direction: column; gap: 12px; width: 100%; margin-top: 22px; }
.ob-q {
  display: flex; align-items: center; gap: 14px;
  width: 100%; text-align: left;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px;
  cursor: pointer;
  transition: all 0.18s var(--ease-out);
}
.ob-q:hover { border-color: var(--accent); background: var(--bg-hover); }
.ob-q-icon {
  width: 44px; height: 44px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-bg); border-radius: var(--radius);
  font-size: 20px;
}
.ob-q-title { font-size: 14px; font-weight: 600; color: var(--text-0); margin: 0; }
.ob-q-desc { font-size: 12px; color: var(--text-3); margin: 2px 0 0; }

/* 底部导航 */
.ob-footer { display: flex; align-items: center; margin-top: 20px; }
.ob-footer-spacer { flex: 1; }

/* 过渡 */
.ob-enter-active, .ob-leave-active { transition: opacity 0.25s var(--ease-out); }
.ob-enter-from, .ob-leave-to { opacity: 0; }
.ob-enter-active .ob-panel, .ob-leave-active .ob-panel { transition: transform 0.25s var(--ease-out); }
.ob-enter-from .ob-panel, .ob-leave-to .ob-panel { transform: translateY(12px) scale(0.98); }
</style>

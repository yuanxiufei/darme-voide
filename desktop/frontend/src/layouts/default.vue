<template>
  <div class="shell">
    <!-- Header -->
    <header class="header">
      <!-- ≤1023px 才出现：导航收进右侧抽屉，避免 10 个导航项在平板/手机上挤出屏幕 -->
      <button
        class="nav-toggle"
        :class="{ open: navOpen }"
        :aria-expanded="navOpen"
        aria-label="导航菜单"
        @click="navOpen = !navOpen"
      >
        <span></span><span></span><span></span>
      </button>
      <div class="header-left">
        <button class="brand" @click="navigateTo('/')">
          <div class="brand-mark">
            <img v-if="showBrandImage" :src="brandLogo" alt="短剧工坊" class="brand-logo" @error="showBrandImage = false" />
            <span v-else class="brand-fallback">剧</span>
          </div>
          <div class="brand-text">
            <span class="brand-name">短剧工坊</span>
            <span class="brand-sub">Drama Studio</span>
          </div>
        </button>
      </div>

      <nav class="header-nav" :class="{ open: navOpen }" @click="navOpen = false">
        <RouterLink to="/" class="nav-link" :class="{ active: route.path === '/' }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          <span>项目</span>
        </RouterLink>
        <RouterLink to="/library" class="nav-link" :class="{ active: route.path.startsWith('/library') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            <line x1="8" y1="7" x2="16" y2="7"/>
            <line x1="8" y1="11" x2="13" y2="11"/>
          </svg>
          <span>资源库</span>
        </RouterLink>
        <RouterLink to="/skills" class="nav-link" :class="{ active: route.path.startsWith('/skills') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="9" y1="13" x2="15" y2="13"/>
            <line x1="9" y1="17" x2="13" y2="17"/>
          </svg>
          <span>Skill</span>
        </RouterLink>
        <!-- 只认 `/knowledge` 与 `/knowledge/<子路径>`：`/knowledge-loop` 是另一页，不能一起亮 -->
        <RouterLink to="/knowledge" class="nav-link" :class="{ active: route.path === '/knowledge' || route.path.startsWith('/knowledge/') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            <path d="M9 7h7M9 11h7"/>
          </svg>
          <span>知识库</span>
        </RouterLink>
        <RouterLink to="/knowledge-loop" class="nav-link" :class="{ active: route.path.startsWith('/knowledge-loop') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12a9 9 0 1 1-3.5-7.1"/>
            <polyline points="21 3 21 9 15 9"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
          <span>闭环</span>
        </RouterLink>
        <RouterLink to="/models" class="nav-link" :class="{ active: route.path.startsWith('/models') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="4" width="16" height="16" rx="2"/>
            <rect x="9" y="9" width="6" height="6"/>
            <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>
          </svg>
          <span>模型</span>
        </RouterLink>
        <RouterLink to="/agents" class="nav-link" :class="{ active: route.path.startsWith('/agents') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="8" width="16" height="12" rx="2"/>
            <path d="M12 8V4"/>
            <circle cx="12" cy="3" r="1.2"/>
            <line x1="9" y1="13" x2="9" y2="15"/>
            <line x1="15" y1="13" x2="15" y2="15"/>
          </svg>
          <span>Agent</span>
        </RouterLink>
        <RouterLink to="/style-profiles" class="nav-link" :class="{ active: route.path.startsWith('/style-profiles') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2l2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 7.7l5.4-.8z"/>
            <path d="M6 20l2-2M18 20l-2-2"/>
          </svg>
          <span>风格</span>
        </RouterLink>
        <RouterLink to="/visual-graph" class="nav-link" :class="{ active: route.path.startsWith('/visual-graph') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"/>
            <circle cx="12" cy="12" r="3"/>
            <path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>
          </svg>
          <span>视觉</span>
        </RouterLink>
        <RouterLink to="/canvas" class="nav-link" :class="{ active: route.path.startsWith('/canvas') }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="4" width="7" height="6" rx="1"/>
            <rect x="14" y="12" width="7" height="6" rx="1"/>
            <path d="M10 7h4v10h-4"/>
          </svg>
          <span>画布</span>
        </RouterLink>
        <RouterLink to="/settings" class="nav-link" :class="{ active: route.path === '/settings' }">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
          <span>设置</span>
        </RouterLink>
      </nav>

      <div class="header-right">
        <button class="help-btn" title="使用引导" @click="openHelp">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </button>
        <div class="film-strip">
          <span class="film-frame"></span>
          <span class="film-frame"></span>
          <span class="film-frame"></span>
        </div>
      </div>
    </header>

    <!-- 抽屉遮罩：点击空白处关闭 -->
    <div v-if="navOpen" class="nav-scrim" @click="navOpen = false"></div>

    <main class="content">
      <slot />
    </main>
  </div>
</template>

<script setup>
import brandLogo from '~/assets/brand-logo.svg'

const route = useRoute()
const showBrandImage = ref(true)

/* 移动端抽屉：路由变化必关（否则点完链接抽屉还盖在页面上） */
const navOpen = ref(false)
watch(() => route.path, () => { navOpen.value = false })

function onKeydown(e) {
  if (e.key === 'Escape') navOpen.value = false
}
onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))

function openHelp() {
  useState('onboarding-open', () => false).value = true
}
</script>

<style scoped>
.shell {
  display: flex; flex-direction: column;
  height: 100vh; overflow: hidden;
  background: var(--bg-base);
}

/* === Header === */
.header {
  display: flex; align-items: center;
  height: 56px; flex-shrink: 0;
  padding: 0 24px;
  background: var(--bg-1);
  border-bottom: 1px solid var(--border);
  gap: 32px;
}

.header-left { display: flex; align-items: center; }

.brand {
  display: flex; align-items: center; gap: 10px;
  background: none; border: none; cursor: pointer; padding: 0;
  text-decoration: none; border-radius: var(--radius);
  transition: opacity 0.15s;
}
.brand:hover { opacity: 0.75; }
.brand-mark {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-2); border-radius: var(--radius);
  border: 1px solid var(--border);
  overflow: hidden;
}
.brand-logo {
  width: 22px;
  height: 22px;
  object-fit: contain;
  display: block;
}
.brand-fallback {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 700;
  color: var(--accent-text);
  line-height: 1;
}
.brand-text { display: flex; flex-direction: column; align-items: flex-start; line-height: 1; }
.brand-name {
  font-family: var(--font-display);
  font-size: 15px; font-weight: 700;
  color: var(--text-0);
  letter-spacing: -0.01em;
}
.brand-sub {
  font-size: 10px; font-weight: 400;
  color: var(--text-3); margin-top: 1px;
  letter-spacing: 0.04em;
}

/* Nav */
.header-nav { display: flex; gap: 4px; flex: 1; }
.nav-link {
  display: flex; align-items: center; gap: 7px;
  padding: 7px 14px; border-radius: var(--radius);
  font-size: 13px; font-weight: 500;
  color: var(--text-2); text-decoration: none;
  transition: all 0.18s var(--ease-out);
  border: 1px solid transparent;
}
.nav-link:hover {
  background: var(--bg-hover); color: var(--text-0);
  border-color: var(--border);
}
.nav-link.active {
  background: var(--accent-bg);
  color: var(--accent-text);
  border-color: rgba(13,148,136,0.18);
  font-weight: 600;
}

.header-right { display: flex; align-items: center; margin-left: auto; gap: 12px; }

.help-btn {
  display: flex; align-items: center; justify-content: center;
  width: 30px; height: 30px;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); cursor: pointer;
  color: var(--text-2);
  transition: all 0.18s var(--ease-out);
}
.help-btn:hover { color: var(--text-0); border-color: var(--accent); }

/* Film strip decoration */
.film-strip {
  display: flex; align-items: center; gap: 3px;
  padding: 6px 10px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.film-frame {
  width: 8px; height: 10px;
  background: var(--bg-3);
  border-radius: 1.5px;
  transition: background 0.2s;
}
.film-frame:nth-child(2) { background: var(--accent); opacity: 0.6; }
.film-frame:nth-child(3) { opacity: 0.3; }

/* Content */
.content { flex: 1; overflow: hidden; display: flex; flex-direction: column; min-height: 0; }

/* === Mobile / Tablet: hamburger + drawer === */
.nav-toggle {
  display: none;
  flex-direction: column; justify-content: center; gap: 4px;
  width: 34px; height: 34px; padding: 0 8px; flex-shrink: 0;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: var(--radius); cursor: pointer;
  position: relative; z-index: 130;   /* 必须高于 scrim，否则打开后点不到 */
}
.nav-toggle span {
  display: block; height: 2px; border-radius: 2px;
  background: var(--text-1);
  transition: transform 0.22s var(--ease-out), opacity 0.18s;
}
.nav-toggle.open span:nth-child(1) { transform: translateY(6px) rotate(45deg); }
.nav-toggle.open span:nth-child(2) { opacity: 0; }
.nav-toggle.open span:nth-child(3) { transform: translateY(-6px) rotate(-45deg); }
.nav-scrim { display: none; }

@media (max-width: 1023px) {
  .header { height: 54px; padding: 0 16px; gap: 12px; }
  .brand-text { display: none; }
  .film-strip { display: none; }
  .nav-toggle { display: inline-flex; }
  .header-nav {
    position: fixed; z-index: 120;
    top: 0; right: 0; bottom: 0;
    width: min(84vw, 320px);
    flex: none;
    flex-direction: column; align-items: stretch; justify-content: flex-start;
    gap: 4px; padding: 68px 14px 24px;
    background: var(--bg-1);
    border-left: 1px solid var(--border);
    box-shadow: var(--shadow-xl);
    overflow-y: auto;
    transform: translateX(102%);
    transition: transform 0.24s var(--ease-out);
  }
  .header-nav.open { transform: translateX(0); }
  .nav-link { padding: 11px 12px; font-size: 14px; }
  .nav-link svg { width: 17px; height: 17px; flex-shrink: 0; }
  .nav-scrim {
    display: block; position: fixed; inset: 0; z-index: 110;
    background: rgba(35, 48, 74, 0.36);
    backdrop-filter: blur(2px);
    animation: fadeIn 0.18s var(--ease-out);
  }
}

@media (max-width: 767px) {
  .header { padding: 0 12px; }
}
</style>

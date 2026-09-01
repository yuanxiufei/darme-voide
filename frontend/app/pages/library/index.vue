<template>
  <div class="lib-page">
    <header class="lib-header">
      <div class="lib-title">
        <h2>资源库</h2>
        <p class="lib-sub">管理角色、场景、兵器、服装、物品模板，提升内容复用效率</p>
      </div>
    </header>

    <!-- 分类卡片 -->
    <div class="lib-hub">
      <NuxtLink to="/library/characters" class="hub-card">
        <div class="hub-icon" style="background: rgba(99, 102, 241, 0.12); color: #818cf8;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="8" r="4"/>
            <path d="M5.6 21.3c.7-3.7 3.5-6.6 6.4-6.6s5.7 2.9 6.4 6.6"/>
          </svg>
        </div>
        <div class="hub-body">
          <h3>角色库</h3>
          <p>视觉形象设定 + 声音配置，多维度标签分类</p>
          <span class="hub-stat">{{ stats.characters }} 个模板</span>
        </div>
      </NuxtLink>

      <NuxtLink to="/library/scenes" class="hub-card">
        <div class="hub-icon" style="background: rgba(16, 185, 129, 0.12); color: #34d399;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <path d="M21 15l-5-5L5 21"/>
          </svg>
        </div>
        <div class="hub-body">
          <h3>场景库</h3>
          <p>环境描述、氛围设定、光线等视觉元素</p>
          <span class="hub-stat">{{ stats.scenes }} 个模板</span>
        </div>
      </NuxtLink>

      <NuxtLink to="/library/weapons" class="hub-card">
        <div class="hub-icon" style="background: rgba(245, 158, 11, 0.12); color: #fbbf24;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 17.5L3 6V3h3l11.5 11.5"/>
            <path d="M21 3l-5 5"/>
            <path d="M18 6l3 3"/>
            <line x1="3" y1="21" x2="9" y2="15"/>
          </svg>
        </div>
        <div class="hub-body">
          <h3>兵器库</h3>
          <p>兵器类别、属性、外观分类管理</p>
          <span class="hub-stat">{{ stats.weapons }} 个模板</span>
        </div>
      </NuxtLink>

      <NuxtLink to="/library/costumes" class="hub-card">
        <div class="hub-icon" style="background: rgba(236, 72, 153, 0.12); color: #f472b6;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2l2 4.2 4.6.7-3.3 3.2.8 4.6L12 12.9 7.9 14.7l.8-4.6L5.4 6.9l4.6-.7z"/>
          </svg>
        </div>
        <div class="hub-body">
          <h3>服装库</h3>
          <p>按风格、部位、材质维度分类检索</p>
          <span class="hub-stat">{{ stats.costumes }} 个模板</span>
        </div>
      </NuxtLink>

      <NuxtLink to="/library/props" class="hub-card">
        <div class="hub-icon" style="background: rgba(139, 92, 246, 0.12); color: #a78bfa;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2l2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 7.7l5.4-.8z"/>
            <path d="M3 20h18"/>
          </svg>
        </div>
        <div class="hub-body">
          <h3>物品库</h3>
          <p>道具、信物、线索、法器资产，设定图注入分镜参考</p>
          <span class="hub-stat">{{ stats.props }} 个模板</span>
        </div>
      </NuxtLink>
    </div>

    <!-- 最近添加 -->
    <section class="recent-section" v-if="recentItems.length">
      <h3 class="section-title">最近添加</h3>
      <div class="recent-grid">
        <div v-for="item in recentItems" :key="item.type + item.id" class="recent-item">
          <span class="recent-badge" :class="item.type">{{ item.typeLabel }}</span>
          <span class="recent-name">{{ item.name }}</span>
          <span class="recent-time">{{ formatDate(item.created_at || item.createdAt) }}</span>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { characterLibraryAPI, sceneLibraryAPI, weaponLibraryAPI, costumeLibraryAPI, propsLibraryAPI } from '~/composables/useApi'

const stats = reactive({ characters: 0, scenes: 0, weapons: 0, costumes: 0, props: 0 })
const recentItems = ref([] as any[])

onMounted(async () => {
  try {
    const [cr, sr, wr, cor, pr] = await Promise.all([
      characterLibraryAPI.list({ pageSize: 1 }),
      sceneLibraryAPI.list({ pageSize: 1 }),
      weaponLibraryAPI.list({ pageSize: 1 }),
      costumeLibraryAPI.list({ pageSize: 1 }),
      propsLibraryAPI.list(),
    ])
    stats.characters = cr.total ?? 0
    stats.scenes = sr.total ?? 0
    stats.weapons = wr.total ?? 0
    stats.costumes = cor.total ?? 0
    stats.props = (Array.isArray(pr) ? pr : pr?.items || []).length

    // 获取最近条目
    const [rc, rs, rw, rco, rp] = await Promise.all([
      characterLibraryAPI.list({ pageSize: 4, sortBy: 'created_at', sortOrder: 'desc' }),
      sceneLibraryAPI.list({ pageSize: 4, sortBy: 'created_at', sortOrder: 'desc' }),
      weaponLibraryAPI.list({ pageSize: 4, sortBy: 'created_at', sortOrder: 'desc' }),
      costumeLibraryAPI.list({ pageSize: 4, sortBy: 'created_at', sortOrder: 'desc' }),
      propsLibraryAPI.list(),
    ])
    const propArr = Array.isArray(rp) ? rp : rp?.items || []
    recentItems.value = [
      ...(rc.items || []).map((i: any) => ({ ...i, type: 'char', typeLabel: '角色' })),
      ...(rs.items || []).map((i: any) => ({ ...i, type: 'scene', typeLabel: '场景' })),
      ...(rw.items || []).map((i: any) => ({ ...i, type: 'weapon', typeLabel: '兵器' })),
      ...(rco.items || []).map((i: any) => ({ ...i, type: 'costume', typeLabel: '服装' })),
      ...propArr.slice(0, 4).map((i: any) => ({ ...i, type: 'prop', typeLabel: '物品' })),
    ]
      .sort((a: any, b: any) => new Date(b.created_at || b.createdAt).getTime() - new Date(a.created_at || a.createdAt).getTime())
      .slice(0, 8)
  } catch (err: any) {
    console.error('[LibraryPage] Failed to load stats:', err?.message ?? err)
  }
})

function formatDate(d: string) {
  if (!d) return ''
  return new Date(d).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}
</script>

<style scoped>
.lib-page {
  height: 100%; overflow-y: auto;
  padding: 32px 40px;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
}

.lib-header {
  margin-bottom: 32px;
}
.lib-header h2 {
  font-family: var(--font-display);
  font-size: 26px; font-weight: 700;
  color: var(--text-0);
  letter-spacing: -0.02em;
}
.lib-sub {
  font-size: 14px; color: var(--text-2);
  margin-top: 6px;
}

/* Hub Grid */
.lib-hub {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 40px;
}

.hub-card {
  display: flex; align-items: flex-start; gap: 18px;
  padding: 24px;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  text-decoration: none;
  transition: all 0.2s var(--ease-out);
  cursor: pointer;
}
.hub-card:hover {
  border-color: var(--accent);
  box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  transform: translateY(-1px);
}
.hub-icon {
  width: 52px; height: 52px;
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius);
}
.hub-body h3 {
  font-size: 16px; font-weight: 600;
  color: var(--text-0);
  margin-bottom: 4px;
}
.hub-body p {
  font-size: 13px; color: var(--text-2);
  line-height: 1.5; margin-bottom: 8px;
}
.hub-stat {
  font-size: 12px; color: var(--text-3);
  background: var(--bg-2);
  padding: 2px 8px; border-radius: 10px;
  display: inline-block;
}

/* Recent Section */
.section-title {
  font-size: 14px; font-weight: 600;
  color: var(--text-1);
  margin-bottom: 14px;
}
.recent-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.recent-item {
  display: flex; flex-direction: column; gap: 4px;
  padding: 12px 14px;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: border-color 0.15s;
}
.recent-item:hover { border-color: var(--accent); }
.recent-badge {
  font-size: 10px; font-weight: 600;
  padding: 1px 6px;
  border-radius: 8px;
  align-self: flex-start;
  text-transform: uppercase;
}
.recent-badge.char  { background: rgba(99,102,241,0.1); color: #818cf8; }
.recent-badge.scene { background: rgba(16,185,129,0.1); color: #34d399; }
.recent-badge.weapon { background: rgba(245,158,11,0.1); color: #fbbf24; }
.recent-badge.costume { background: rgba(236,72,153,0.1); color: #f472b6; }
.recent-badge.prop { background: rgba(139,92,246,0.1); color: #a78bfa; }
.recent-name {
  font-size: 13px; font-weight: 500;
  color: var(--text-0);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.recent-time {
  font-size: 11px; color: var(--text-3);
}

@media (max-width: 900px) {
  .lib-hub { grid-template-columns: 1fr; }
  .recent-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>

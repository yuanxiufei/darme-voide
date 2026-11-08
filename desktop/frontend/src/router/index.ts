/**
 * 桌面端路由表（显式声明，替代 Nuxt 的文件系统路由）。
 *
 * ## 为什么用 hash 模式
 *
 * Tauri 把前端资源放在自定义协议下（Windows `http://tauri.localhost`，
 * macOS/Linux `tauri://localhost`）。history 模式下，深链接刷新会被壳当作
 * 静态文件路径去查、查不到就 404 ⇒ 只能用 hash。
 *
 * Web 端（Nuxt）继续用 history 模式 —— 两个工程的路由策略互不影响，
 * 这正是"两份独立代码"的收益之一。
 *
 * ## 与 `pages/` 目录的一一对应
 *
 * 20 条路由 = 20 个页面文件。目录名 `[id]` / `[characterId]` 是**历史命名**，
 * 保留是为了复制时不改文件名；路由参数名在这里显式声明。
 */
import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'home', component: () => import('../pages/index.vue') },
  { path: '/models', name: 'models', component: () => import('../pages/models.vue') },
  { path: '/settings', name: 'settings', component: () => import('../pages/settings.vue') },
  { path: '/agents', name: 'agents', component: () => import('../pages/agents.vue') },
  { path: '/skills', name: 'skills', component: () => import('../pages/skills.vue') },
  { path: '/knowledge', name: 'knowledge', component: () => import('../pages/knowledge.vue') },
  {
    path: '/knowledge-loop',
    name: 'knowledge-loop',
    component: () => import('../pages/knowledge-loop.vue'),
  },
  {
    path: '/style-profiles',
    name: 'style-profiles',
    component: () => import('../pages/style-profiles.vue'),
  },
  {
    path: '/visual-graph',
    name: 'visual-graph',
    component: () => import('../pages/visual-graph.vue'),
  },
  { path: '/canvas', name: 'canvas', component: () => import('../pages/canvas.vue') },

  { path: '/library', name: 'library', component: () => import('../pages/library/index.vue') },
  {
    path: '/library/characters',
    name: 'library-characters',
    component: () => import('../pages/library/characters.vue'),
  },
  {
    path: '/library/costumes',
    name: 'library-costumes',
    component: () => import('../pages/library/costumes.vue'),
  },
  {
    path: '/library/props',
    name: 'library-props',
    component: () => import('../pages/library/props.vue'),
  },
  {
    path: '/library/scenes',
    name: 'library-scenes',
    component: () => import('../pages/library/scenes.vue'),
  },
  {
    path: '/library/weapons',
    name: 'library-weapons',
    component: () => import('../pages/library/weapons.vue'),
  },

  { path: '/drama/:id', name: 'drama', component: () => import('../pages/drama/[id]/index.vue') },
  {
    path: '/drama/:id/prompts',
    name: 'drama-prompts',
    component: () => import('../pages/drama/[id]/prompts.vue'),
  },
  {
    path: '/drama/:id/character/:characterId',
    name: 'drama-character',
    component: () => import('../pages/drama/[id]/character/[characterId].vue'),
  },
  {
    path: '/drama/:id/scene/:sceneId',
    name: 'drama-scene',
    component: () => import('../pages/drama/[id]/scene/[sceneId].vue'),
  },
  {
    path: '/drama/:id/episode/:episodeNumber',
    name: 'drama-episode',
    component: () => import('../pages/drama/[id]/episode/[episodeNumber].vue'),
    // 替代 Nuxt 的 definePageMeta({ layout: 'studio' })
    meta: { layout: 'studio' },
  },

  // 兜底：未知路径回首页（桌面端没有 404 页面的意义）
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
  scrollBehavior(_to, _from, saved) {
    return saved ?? { top: 0 }
  },
})

export default router

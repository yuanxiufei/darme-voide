<!--
  桌面端根组件 —— 替代 Web 端 `frontend/app/app.vue` 里的
  `<NuxtLayout><NuxtPage /></NuxtLayout>`。

  差异说明（这是两端**允许且应当**分叉的地方）：
  - Nuxt：布局由 `definePageMeta({ layout })` + `layouts/` 自动解析
  - 桌面：布局由 `route.meta.layout` + 显式 map 解析

  其余四件（Toaster / Onboarding / ConfirmDialog / TaskCenter / DesktopStatusBanner）
  与 Web 端保持同一套组件与同一套挂载位置 —— 尤其 TaskCenter 必须挂在最外层，
  否则切页会丢任务（原因见 composables/useTaskCenter.ts 文件头）。
-->
<template>
  <component :is="layout">
    <RouterView />
  </component>

  <Toaster position="top-right" :duration="3000" />
  <Onboarding />
  <ConfirmDialog />
  <TaskCenter />
  <!-- 桌面端专用：Web 形态下 isDesktop 恒为 false，不会渲染 -->
  <DesktopStatusBanner />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Toaster } from 'vue-sonner'
import DefaultLayout from './layouts/default.vue'
import StudioLayout from './layouts/studio.vue'

const route = useRoute()

const layouts = {
  default: DefaultLayout,
  studio: StudioLayout,
} as const

const layout = computed(() => {
  const key = route.meta.layout as keyof typeof layouts | undefined
  return (key && layouts[key]) || DefaultLayout
})
</script>

<style>
/* 顺序不能反：.prod-card 全局化后与 studio.css 的 .card 同特异性，靠加载顺序取胜 */
@import './assets/studio.css';
@import './assets/episode-panels.css';
</style>

<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
  <Toaster position="top-right" :duration="3000" />
  <Onboarding />
  <ConfirmDialog />
  <!-- 任务中心：挂在最外层，才能做到「切页不丢」（详情见 composables/useTaskCenter.ts 文件头） -->
  <TaskCenter />
</template>

<script setup>
import { Toaster } from 'vue-sonner'
</script>

<style>
@import url('./assets/studio.css');
/* 生产面板骨架（.prod-* 等）：被拆出的面板组件要用，必须留在全局层。
   ⚠️ 顺序不能反 —— `.prod-card` 全局化后与 studio.css 的 `.card` 同特异性，靠加载顺序取胜 */
@import url('./assets/episode-panels.css');
</style>

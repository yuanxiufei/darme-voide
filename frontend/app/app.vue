<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
  <Toaster position="top-right" :duration="3000" />
  <Onboarding />
  <ConfirmDialog />
</template>

<script setup>
import { Toaster } from 'vue-sonner'
</script>

<style>
@import url('./assets/studio.css');
</style>

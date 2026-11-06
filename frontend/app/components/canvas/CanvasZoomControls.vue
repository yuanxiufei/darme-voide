<script setup lang="ts">
/**
 * 缩放控件（右下角常驻）。
 *
 * 两处口径：① 按钮缩放以**画布中心**为锚 ✓（不是左上角，否则越点越偏）；
 * ② 百分比四舍五入到整数 ✓ —— 显示 `103.7%` 只会让人分不清「到底复位没有」✗。
 */
import { Maximize2, Minus, Plus } from 'lucide-vue-next'

const props = defineProps<{ scale: number }>()
const emit = defineEmits<{
  (e: 'zoom', factor: number): void
  (e: 'reset'): void
}>()
</script>

<template>
  <div class="zc" @mousedown.stop>
    <button class="zc-btn" title="缩小" @click="emit('zoom', 1 / 1.25)"><Minus :size="13" /></button>
    <span class="zc-val">{{ Math.round(props.scale * 100) }}%</span>
    <button class="zc-btn" title="放大" @click="emit('zoom', 1.25)"><Plus :size="13" /></button>
    <button class="zc-btn" title="复位（100%）" @click="emit('reset')"><Maximize2 :size="13" /></button>
  </div>
</template>

<style scoped>
.zc {
  display: flex;
  align-items: center;
  gap: 2px;
  background: rgba(23, 26, 33, 0.92);
  border: 1px solid #2a2f3a;
  border-radius: 6px;
  padding: 2px;
}
.zc-btn {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: #cbd2dd;
  cursor: pointer;
}
.zc-btn:hover {
  background: #22262f;
}
/* 触屏：26px 的缩放手柄点不准，抬到 34px；容器随之变高，不影响桌面鼠标档 */
@media (pointer: coarse) {
  .zc-btn { width: 34px; height: 34px; }
  .zc-val { min-width: 48px; }
}
.zc-val {
  min-width: 44px;
  text-align: center;
  font-size: 11px;
  color: #9aa4b2;
  font-variant-numeric: tabular-nums;
}
</style>

<script setup lang="ts">
/**
 * 两张图重叠 + 滑杆对比（选「哪一版」用）。
 *
 * 口径来自上游 `canvas-image-compare-modal.tsx`：**同一位置重叠**、拖动滑杆看边界，
 * 比左右并排更容易看出「脸有没有变、构图差了哪一点」✓。
 * 实现按本仓重写：上层的裁剪用 `clip-path: inset`（左图露左侧 pos%）——
 * 用 `width` 裁剪会让图片被**拉伸**而不是被裁掉 ✗。
 */
import { ref } from 'vue'
import { X } from 'lucide-vue-next'

const props = defineProps<{ left?: string; right?: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const pos = ref(50)
</script>

<template>
  <div class="ic-mask" @mousedown.self="emit('close')">
    <div class="ic-box">
      <img v-if="props.right" :src="props.right" class="ic-img" alt="" />
      <div class="ic-clip" :style="{ clipPath: `inset(0 ${100 - pos}% 0 0)` }">
        <img v-if="props.left" :src="props.left" class="ic-img" alt="" />
      </div>
      <div class="ic-divider" :style="{ left: `${pos}%` }" />
      <input v-model.number="pos" type="range" min="0" max="100" step="1" class="ic-range" />
      <button class="ic-close" title="关闭" @click="emit('close')"><X :size="16" /></button>
      <span class="ic-tag ic-tag-l">左：{{ props.left ? 'A' : '空' }}</span>
      <span class="ic-tag ic-tag-r">右：{{ props.right ? 'B' : '空' }}</span>
    </div>
  </div>
</template>

<style scoped>
.ic-mask {
  position: fixed;
  inset: 0;
  z-index: 95;
  display: grid;
  place-items: center;
  background: rgba(6, 8, 12, 0.82);
}
.ic-box {
  position: relative;
  width: min(78vw, 900px);
  max-height: 82vh;
  background: #12151c;
  border: 1px solid #2f3541;
  border-radius: 10px;
  overflow: hidden;
  padding: 12px;
}
.ic-img {
  display: block;
  width: 100%;
  max-height: 70vh;
  object-fit: contain;
  background: #0b0d12;
  border-radius: 6px;
}
.ic-clip {
  position: absolute;
  inset: 12px;
}
.ic-divider {
  position: absolute;
  top: 12px;
  bottom: 44px;
  width: 1px;
  background: #2dd4bf;
  pointer-events: none;
}
.ic-range {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 10px;
  width: calc(100% - 24px);
  accent-color: #2dd4bf;
}
.ic-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 6px;
  background: rgba(15, 17, 21, 0.8);
  color: #e6e8ee;
  cursor: pointer;
}
.ic-tag {
  position: absolute;
  top: 16px;
  font-size: 11px;
  padding: 2px 7px;
  border-radius: 4px;
  background: rgba(15, 17, 21, 0.8);
  color: #9aa4b2;
}
.ic-tag-l {
  left: 16px;
}
.ic-tag-r {
  right: 56px;
}
</style>

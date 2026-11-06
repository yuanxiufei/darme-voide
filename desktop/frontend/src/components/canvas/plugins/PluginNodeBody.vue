<script setup lang="ts">
/**
 * 插件节点正文分发器 —— 画布给插件留的**唯一**渲染槽 ✓。
 *
 * `CanvasNodeCard.vue` 只认内置五类（text/image/video/config/group），其余类型一律进这里；
 * 这里按 `type` 查 `utils/canvasPlugins.ts` 的清单找组件。
 *
 * ## 三种状态都必须画出来，不许空白
 *
 * | 情况 | 表现 |
 * | --- | --- |
 * | 已登记 + 组件在 | 正常渲染 |
 * | 已登记 + 组件缺失 | 明确报「清单登记了但没接组件」（真 bug，不能装作渲染成功 ✗） |
 * | 未登记 | 报「未安装的插件：<type>」，并保留节点本身（删了等于吞掉用户数据 ✗） |
 *
 * 插件自定义键统一落 `node.values`，改值只发 `patch`，由卡片合并后交给 `useCanvas.update` ✓。
 */
import { computed } from 'vue'

import type { CanvasNode } from '../../../composables/useCanvas'
import { pluginOf, pluginValuesOf } from '../../../utils/canvasPlugins'
import HtmlNodeContent from './HtmlNodeContent.vue'
import MarkdownNodeContent from './MarkdownNodeContent.vue'
import PanoramaNodeContent from './PanoramaNodeContent.vue'
import StickyNoteNodeContent from './StickyNoteNodeContent.vue'
import SvgNodeContent from './SvgNodeContent.vue'

const props = defineProps<{
  node: CanvasNode
  /** 上游文本（text 节点正文 / 其它插件节点输出的文本） */
  text?: string
  /** 上游图片（URL 列表，顺序＝连线顺序） */
  images?: string[]
}>()

const emit = defineEmits<{ (e: 'patch', patch: Record<string, string>): void }>()

/** 组件名 → 组件（名字与后端 `contentComponent` 一字不差，见 `utils/canvasPlugins.ts`） */
const COMPONENTS: Record<string, unknown> = {
  HtmlNodeContent,
  MarkdownNodeContent,
  PanoramaNodeContent,
  StickyNoteNodeContent,
  SvgNodeContent,
}

const spec = computed(() => pluginOf(props.node.type))
const component = computed(() => (spec.value ? COMPONENTS[spec.value.contentComponent] : undefined))
// 缺键的档（老画布 / 手工改过 `values`）在这里补默认值 ✓ —— 让 5 个组件各自兜一遍
// 会把「默认值是什么」写 5 份，改一处漏四处 ✗
const values = computed(() => pluginValuesOf(props.node.type, props.node.values))

function onPatch(patch: Record<string, string>) {
  emit('patch', patch)
}
</script>

<template>
  <component
    :is="component"
    v-if="component"
    :values="values"
    :text="props.text || ''"
    :images="props.images || []"
    @patch="onPatch"
  />
  <div v-else-if="spec" class="pn-missing">
    <p>插件「{{ spec.title }}」已登记，但没找到渲染组件 <code>{{ spec.contentComponent }}</code></p>
  </div>
  <div v-else class="pn-missing">
    <p>未安装的插件：<code>{{ props.node.type }}</code></p>
    <p class="pn-missing-hint">节点内容还在，装上插件即可恢复显示</p>
  </div>
</template>

<style scoped>
.pn-missing {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  padding: 12px;
  font-size: 12px;
  color: #9aa3b2;
  text-align: center;
  border: 1px dashed #3a4152;
  border-radius: 6px;
}
.pn-missing p {
  margin: 0;
}
.pn-missing code {
  padding: 1px 4px;
  font-size: 11px;
  color: #cbd5e1;
  background: #0f1218;
  border-radius: 3px;
}
.pn-missing-hint {
  font-size: 11px;
  color: #6b7280;
}
</style>

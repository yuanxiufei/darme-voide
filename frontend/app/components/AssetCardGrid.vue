<script setup lang="ts">
/**
 * 资产卡片网格：角色形象与场景图片两个面板共用。
 *
 * 为什么不是「两个面板各写一份」：这两处卡片是逐行同构的（封面 + 生成状态徽标 +
 * 名字 + 元信息 + 状态点 + 模型选择 + 编辑/生成按钮），差异只在**取哪个字段当标题**、
 * **封面方图还是横图**、**点进去是哪条路由**。前两项由调用方按 `items` 算好送进来，
 * 第三项走 `open` 事件，组件本身不认识 `name` / `location` 这类业务字段。
 *
 * 归一在页面侧做的理由：后端同一字段有 snake/camel 两套写法（`image_url` / `imageUrl`），
 * 组件里再判一遍就是第三处重复，也会把「哪个字段算图」的知识散进组件。
 */
interface AssetCardItem {
  id: number
  /** 卡片主标题（角色名 / 地点名） */
  title: string
  /** 副标题（角色定位 / 时间） */
  meta: string
  /** 已带前导 `/` 的图片地址；null = 还没出图 */
  imageUrl: string | null
  /** 该条目是否正在跑生成任务 */
  pending: boolean
  /** 点开大图时查看器的标题 */
  previewTitle: string
  /** 原始对象，供 `name-extra` 插槽与事件回传时取业务字段 */
  raw: any
}

const props = withDefaults(defineProps<{
  items: AssetCardItem[]
  /** 场景卡用横版封面（16:9），角色卡用方图 */
  wide?: boolean
  /** 当前选中的模型名（空串 = 用默认模型） */
  modelValueOf: (item: AssetCardItem) => string
  /** 编辑按钮的悬停说明 */
  editTitle?: string
}>(), {
  wide: false,
  editTitle: '编辑详情',
})

const emit = defineEmits<{
  /** 点卡片本体（进详情页） */
  open: [item: AssetCardItem]
  /** 点封面图，交给页面的全屏查看器 */
  preview: [src: string, title: string]
  edit: [item: AssetCardItem]
  generate: [item: AssetCardItem]
  'update:model': [item: AssetCardItem, value: string]
}>()

/** 三态文案：有图 → 已生成；在跑 → 生成中；否则待生成。徽标与状态点共用同一判定 */
function statusOf(item: AssetCardItem) {
  if (item.imageUrl) return { text: '已生成', badge: 'is-ready' }
  if (item.pending) return { text: '生成中', badge: 'is-pending' }
  return { text: '待生成', badge: '' }
}
</script>

<template>
  <div class="asset-grid">
    <div v-for="item in items" :key="item.id" class="card asset-card clickable" @click="emit('open', item)">
      <div :class="['asset-cover', { wide: props.wide }]">
        <img
          v-if="item.imageUrl"
          :src="item.imageUrl"
          class="previewable-image"
          @click.stop="emit('preview', item.imageUrl, item.previewTitle)"
        />
        <div v-else class="asset-cover-empty">
          <slot name="empty-icon" />
        </div>
        <span class="asset-cover-badge" :class="statusOf(item).badge">{{ statusOf(item).text }}</span>
      </div>

      <div class="asset-body">
        <div class="asset-name-row">
          <div class="asset-name">{{ item.title }}</div>
          <slot name="name-extra" :item="item" />
        </div>
        <div class="asset-meta dim">{{ item.meta }}</div>
      </div>

      <div class="asset-foot">
        <div class="foot-left">
          <span :class="['dot', item.imageUrl && 'ok', item.pending && 'pending']" />
          <span class="dim foot-status">{{ statusOf(item).text }}</span>
        </div>
        <div class="foot-right">
          <ModelSelector
            :model-value="modelValueOf(item)"
            service-type="image"
            placeholder="默认模型"
            @update:model-value="(v: string) => emit('update:model', item, v)"
            @click.stop
          />
          <button class="btn btn-sm btn-edit" :title="props.editTitle" @click.stop="emit('edit', item)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            编辑
          </button>
          <button class="btn btn-sm btn-primary" :disabled="item.pending" @click.stop="emit('generate', item)">
            {{ item.pending ? '生成中' : '生成' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 「视频」与「合成」两个生产面板的共用骨架：按镜头遍历的媒体卡片网格。
 *
 * 为什么不拆成两个组件：两者 90% 同构（封面、序号、已合成徽标、状态点、失败行、主按钮），
 * 差异只有三处**业务语义**——封面看哪个视频、状态点标什么、主按钮什么条件下能点。
 * 三个差异都写在下面的 `mode` 分支里，各带注释；页面只传数据与三个事件。
 *
 * 依赖 `~/utils/storyboard` 的纯取值函数（帧/视频/配音判定）—— 那些函数与页面状态无关，
 * 上一轮已经从剧集页抽出去了，所以组件不需要父级透传 `hasVid` 之类的判断。
 */
import {
  getComposedVideoUrl, getStoryboardCover, getVideoUrl, hasComposed, hasImg, hasTTS, hasVid,
} from '~/utils/storyboard'

type MediaMode = 'videos' | 'compose'

const props = withDefaults(defineProps<{
  sbs: any[]
  /** videos＝单镜视频面板；compose＝成片合成面板 */
  mode: MediaMode
  /** 工具条右侧的「x/y 已完成」计数文案（两个面板口径不同，由页面给） */
  countLabel: string
  /** 本集待生成成本估算（后端估算接口的原始对象，没有就不显示） */
  costEstimate?: any
  /** 成本徽标的悬停说明 */
  estimateTitle?: string
  /** 该镜头是否正在跑任务（决定按钮禁用与状态点脉冲） */
  isPending: (id: number) => boolean
  /** 该镜头的失败信息（空串＝没有失败） */
  failMessage: (id: number) => string
}>(), {
  costEstimate: null,
  estimateTitle: '',
})

const emit = defineEmits<{
  /** 点主按钮（生成视频 / 合成） */
  act: [sb: any]
  /** 点失败行里的「编辑提示词」（仅 videos 模式显示） */
  edit: [sb: any]
  /** 点封面图，交给页面的全屏查看器 */
  preview: [src: string, title: string]
}>()

function pad(n: number) {
  return String(n).padStart(2, '0')
}

/**
 * 封面用哪个视频：
 * - videos 模式只认**单镜视频**（`video_url`）。即使已经合成过也照旧显示原视频——
 *   原视频旁边的「已合成」徽标就是在说「这个镜头的成片版另有其物」，这里换掉反而看不出差别。
 * - compose 模式优先**合成成片**，没有才退回单镜视频（合成前的预览）。
 */
function coverVideoOf(sb: any): string | null {
  if (props.mode === 'compose') return getComposedVideoUrl(sb) || getVideoUrl(sb)
  return getVideoUrl(sb)
}

/** 状态点：label 会显示在点右侧，pending 让点带脉冲圈 */
function dotsOf(sb: any): { label: string; on: boolean; pending?: boolean }[] {
  const pending = props.isPending(sb.id)
  if (props.mode === 'compose') {
    return [
      { label: '视频', on: hasVid(sb) },
      { label: '配音', on: hasTTS(sb) },
      { label: pending ? '合成中' : '合成', on: hasComposed(sb), pending },
    ]
  }
  return [
    { label: '图', on: hasImg(sb) },
    { label: pending ? '视频生成中' : '视频', on: hasVid(sb), pending },
  ]
}

function actionLabel(sb: any): string {
  const pending = props.isPending(sb.id)
  if (props.mode === 'compose') {
    if (pending) return '合成中'
    return hasComposed(sb) ? '重新合成' : '开始合成'
  }
  return pending ? '生成中' : '生成视频'
}

/**
 * 主按钮能不能点。
 * compose 多一条前置：**没有单镜视频就没法合成**——否则用户会得到一个转圈几十秒、
 * 最后报「缺少输入」的失败卡片。
 */
function actionDisabled(sb: any): boolean {
  if (props.isPending(sb.id)) return true
  if (props.mode === 'compose') return !hasVid(sb)
  return false
}
</script>

<template>
  <div class="prod-content">
    <div class="prod-section-bar">
      <span class="dim" style="font-size:12px">{{ sbs.length }} 个镜头</span>
      <span class="tag mono">{{ countLabel }}</span>
      <span
        v-if="costEstimate && costEstimate.total_cost != null"
        class="tag mono cost-est"
        :title="estimateTitle"
      >本集待生成 ≈ ¥{{ Number(costEstimate.total_cost).toFixed(2) }}</span>
      <div class="ml-auto flex gap-1">
        <slot name="toolbar" />
      </div>
    </div>

    <div class="prod-grid">
      <div v-for="(sb, i) in sbs" :key="sb.id" class="card prod-card">
        <div class="prod-cover">
          <video
            v-if="coverVideoOf(sb)"
            :src="'/' + coverVideoOf(sb)"
            class="prod-video"
            controls
            preload="metadata"
            playsinline
          />
          <img
            v-else-if="hasImg(sb)"
            :src="'/' + getStoryboardCover(sb)"
            class="previewable-image"
            @click.stop="emit('preview', '/' + getStoryboardCover(sb), `镜头 #${pad(i + 1)} 参考图`)"
          />
          <div v-else class="prod-cover-empty">
            <slot name="empty-icon" />
          </div>
          <span class="prod-idx">#{{ pad(i + 1) }}</span>
          <span v-if="hasComposed(sb)" class="prod-overlay-badge">已合成</span>
        </div>

        <div class="prod-info">
          <div class="prod-desc truncate">{{ sb.description || sb.title || '—' }}</div>
          <div class="prod-meta-line">{{ sb.shot_type || sb.shotType || '未设景别' }} · {{ sb.duration || 10 }}s</div>
          <div class="prod-dots">
            <template v-for="dot in dotsOf(sb)" :key="dot.label">
              <span :class="['dot', dot.on && 'ok', dot.pending && 'pending']" />
              <span style="font-size:10px">{{ dot.label }}</span>
            </template>
          </div>
          <div v-if="failMessage(sb.id)" class="prod-error">
            <span>{{ failMessage(sb.id) }}</span>
            <button v-if="mode === 'videos'" class="prod-error-link" @click="emit('edit', sb)">编辑提示词</button>
          </div>
        </div>

        <div class="prod-actions">
          <button class="btn btn-sm" :disabled="actionDisabled(sb)" @click="emit('act', sb)">
            <slot name="action-icon" />
            {{ actionLabel(sb) }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

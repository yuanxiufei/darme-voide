<script setup lang="ts">
/**
 * 配音列表：一个镜头一张卡，多人对话拆成「一条台词一张卡」。
 *
 * 为什么要把「状态映射」放在这里而不是页面：匹配状态是后端给的字符串
 * （`matched` / `no_voice` / `not_found` / `narrator`），但「该显示什么字、什么颜色」
 * 纯粹是这一处的展示决定。以前它散在页面里 5 个函数（`getMatchStatusLabel` /
 * `getMatchStatusDot` / `getDLMatchStatusColor` / `getDubMatchStatusLabel` /
 * `getDubMatchDotClass`），而且每个函数都要先回头读 `dubMatchCache` —— 模板里一行
 * 台词会触发三次「进缓存查一次」。现在页面算好 `status` 字面量送进来，组件只做映射。
 *
 * 两种卡片的口径**故意不一致**，是历史行为，不要「顺手统一」：
 * - 多人卡：音频与「已生成」逐条台词各查一次 TTS 列表；
 * - 单人卡：标签/音频都取整镜字段（`sb.tts_audio_url`）与验证缓存的第一条。
 */
interface DubLineVM {
  speaker: string
  text: string
  /** 后端匹配状态字面量，null = 没验证过 */
  status: string | null
  /** 该条台词的音频地址（不带前导 `/`） */
  audioUrl: string | null
}

interface DubSingleVM {
  status: string | null
  text: string
  /** 景别 / 时长 / 地点，单人卡才有这一行 */
  meta: string[]
  hasTTS: boolean
  audioUrl: string | null
}

interface DubRowVM {
  key: number | string
  /** 原始分镜对象，只用于回传给页面的生成动作 */
  sb: any
  /** 已补零的镜头号 */
  num: string
  /** 所有说话人（多人时以「 / 」并列） */
  speaker: string
  multi: boolean
  peopleCount: number
  lines: DubLineVM[]
  /** 单人卡专用；多人卡为 null */
  single: DubSingleVM | null
}

const props = defineProps<{
  rows: DubRowVM[]
  /** 角色音色匹配有问题的条数 */
  issueCount: number
  /** 音色配置变更后未重新生成的条数 */
  staleCount: number
}>()

const emit = defineEmits<{
  generate: [row: DubRowVM]
}>()

/** `matched` 是正常状态，不出标签——满屏「已匹配」只是噪音 */
function statusLabel(status: string | null) {
  if (!status) return ''
  const map: Record<string, string> = {
    matched: '',
    narrator: '旁白',
    no_voice: '未配置音色',
    not_found: '角色不存在',
  }
  return map[status] || ''
}

function statusDot(status: string | null) {
  if (!status) return ''
  const map: Record<string, string> = {
    matched: 'match-dot match-dot-ok',
    narrator: 'match-dot match-dot-ok',
    no_voice: 'match-dot match-dot-warn',
    not_found: 'match-dot match-dot-err',
  }
  return map[status] || ''
}

/** 卡片左侧竖条：角色不存在=红、没配音色=黄，其余（含未验证）用主色 */
function statusBorder(status: string | null) {
  if (status === 'not_found') return 'var(--danger, #e74c3c)'
  if (status === 'no_voice') return 'var(--warning, #f59e0b)'
  return 'var(--accent)'
}
</script>

<template>
  <div class="dub-grid">
    <div v-if="props.issueCount > 0" class="dub-validation-banner">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span>{{ props.issueCount }} 个角色存在音色匹配问题，生成配音将使用默认音色</span>
    </div>
    <div v-if="props.staleCount > 0" class="dub-validation-banner dub-stale-banner">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span>{{ props.staleCount }} 个配音音色已过期（角色音色配置已变更），建议重新生成配音</span>
    </div>

    <template v-for="row in rows" :key="row.key">
      <!-- 多人对话：一条台词一张卡 -->
      <template v-if="row.multi">
        <div
          v-for="(line, ti) in row.lines"
          :key="`${row.key}-${ti}`"
          class="card dub-card dub-sub-card"
          :style="{ borderLeftColor: statusBorder(line.status) }"
        >
          <div class="dub-head">
            <div class="dub-copy">
              <div v-if="ti === 0" class="dub-title">
                <span class="frame-num">#{{ row.num }}</span>
                <span class="frame-badge">{{ row.speaker }}</span>
                <span class="tag tag-info" style="font-size:10px;padding:1px 6px">{{ row.peopleCount }}人</span>
              </div>
              <div class="dub-line-item" style="margin-top:4px">
                <span class="dub-line-speaker">{{ line.speaker }}</span>
                <span class="dub-line-text">{{ line.text }}</span>
                <span v-if="statusLabel(line.status)" class="dub-match-tag" :class="statusDot(line.status)">{{ statusLabel(line.status) }}</span>
              </div>
            </div>
            <span class="tag" :class="line.audioUrl ? 'tag-success' : ''">{{ line.audioUrl ? '已生成' : '待生成' }}</span>
          </div>
          <div class="dub-foot">
            <audio v-if="line.audioUrl" :src="'/' + line.audioUrl" controls preload="none" class="dub-audio" />
            <button class="btn btn-sm ml-auto" @click="emit('generate', row)">生成配音</button>
          </div>
        </div>
      </template>

      <!-- 单人对话 -->
      <div v-else class="card dub-card">
        <div class="dub-head">
          <div class="dub-copy">
            <div class="dub-title">
              <span class="frame-num">#{{ row.num }}</span>
              <span class="frame-badge">{{ row.speaker }}</span>
              <span v-if="row.single && statusLabel(row.single.status)" class="dub-match-tag" :class="statusDot(row.single.status)">{{ statusLabel(row.single.status) }}</span>
            </div>
            <div class="dub-desc">{{ (row.single && row.single.text) || '未填写文本' }}</div>
          </div>
          <span class="tag" :class="row.single && row.single.hasTTS ? 'tag-success' : ''">{{ row.single && row.single.hasTTS ? '已生成' : '待生成' }}</span>
        </div>
        <div class="dub-meta">
          <span v-for="(m, mi) in (row.single ? row.single.meta : [])" :key="mi" class="dim">{{ m }}</span>
        </div>
        <div class="dub-foot">
          <audio v-if="row.single && row.single.audioUrl" :src="'/' + row.single.audioUrl" controls preload="none" class="dub-audio" />
          <div v-else class="dim" style="font-size:12px">尚未生成语音文件</div>
          <button class="btn btn-sm ml-auto" @click="emit('generate', row)">生成配音</button>
        </div>
      </div>
    </template>
  </div>
</template>

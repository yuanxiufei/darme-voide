<script setup lang="ts">
/**
 * 画布右侧对话面板 —— 一句话操作画布。
 *
 * 三条可核对的性质：
 *
 * | 性质 | 怎么保证 |
 * | --- | --- |
 * | 可回溯 | 指令与回执都留在 `turns`，可重发、可看失败明细 |
 * | 可撤销 | 落地前存 `exportDoc()`，撤销就是 `importDoc()` 回去（画布之后再动过就禁用） |
 * | 真能停 | `AbortSignal` 直通 `fetch`：中止后 op 一条都没回来（不是「界面不转圈但照写」） |
 *
 * 「实时同步」靠 `useCanvas()` 的模块级单例 —— 面板与画布读**同一批 ref**，
 * 顶部状态条同帧，不轮询；AI 落地后按 `touched` 选中那批卡，用户立刻看见动了哪几张。
 *
 * 有意不做：不持久化对话（`before` 快照体积大、revision 也会对不上，留一堆点不动的撤销更差）；
 * 不自动搬视口（AI 在另一头建卡时硬拽镜头会丢掉用户正在看的东西，要过去点「跳到改动」）。
 */
import { computed, nextTick, ref, watch } from 'vue'
import { ChevronRight, Eraser, Send, Sparkles, Square, Target, Undo2 } from 'lucide-vue-next'
import { NODE_TYPE_LABEL, useCanvas } from '../../composables/useCanvas'
import { runCanvasAgent } from '../../utils/canvasAgent'
import { useEngineRun } from '../../composables/useEngineRun'
import { isRunPending, normalizeRunVerdict, type RunAnswer, type RunOverride } from '../../utils/canvasOps'

const props = defineProps<{
  open: boolean
  /** 画布可视区尺寸（`focusNode` 要用；不给就只能选中不能飞过去） */
  stage?: { width: number; height: number }
  /**
   * 代点生成：模型给 `run_generation` 时调它，答复说 `ok` 才算真提交。
   * 不给（或没提交成）⇒ 该卡落进 `pendingRun`，由用户自己点卡上的「生成」。
   *
   * 可以返回 promise：同一批 op 刚建的卡还没挂载、或档位清单还在路上，
   * 它只能等一次才知道真答案 ✓ —— 面板同样 `await` 后再写回执，不抢跑 ✗。
   * 也可以返回 `{ ok: false, reason }`：理由写进回执明细（别丢成一句「看卡片」✗）。
   */
  generate?: (nodeId: string, override?: RunOverride) => RunAnswer | Promise<RunAnswer>
}>()
const emit = defineEmits<{
  (e: 'update:open', value: boolean): void
}>()

interface Turn {
  id: string
  role: 'user' | 'assistant'
  text: string
  applied?: number
  pendingRun?: string[]
  /** 这一轮真提交了生成的配置卡（卡片自己跑，回执要等它跑完） */
  startedRun?: string[]
  errors?: string[]
  rejected?: string[]
  /** 不影响落地、但要如实说一声的事（例：整批自检没跑成） */
  warnings?: string[]
  touched?: string[]
  /** 落地前的画布文档 —— 撤销用 */
  before?: string
  /** 落地后的版本号 —— 只有画布停在这里时才允许撤销 */
  afterRevision?: number
  undone?: boolean
  failed?: string
  stopped?: boolean
  at: number
}

const canvas = useCanvas()
/**
 * 引擎状态是**模块级单例**：卡片写、这里读 ✓。
 *
 * 为什么读它：回执写「已提交生成 1」之后就再无下文的体验是断的 ——
 * 用户得自己在画布上找那张卡、盯它转圈，才知道到底跑没跑成 ✗。
 * 同一份状态读过来，回执就能从「已提交」走成「生成中 62%」→「已出图 1」/「生成失败」✓。
 */
const engine = useEngineRun()
const { nodes, edges, selection, revision } = canvas

const turns = ref<Turn[]>([])
const draft = ref('')
const busy = ref(false)
const stage = ref('')
const listEl = ref<HTMLElement | null>(null)
const boxEl = ref<HTMLTextAreaElement | null>(null)
const bump = ref(false)

let controller: AbortController | null = null
let seq = 0
let bumpTimer: ReturnType<typeof setTimeout> | null = null

/** 选中项的友好名（「图片卡、文本卡」比一屏 uuid 有用） */
const selectionLabel = computed(() => {
  const picked = nodes.value.filter((node) => selection.value.includes(node.id))
  if (!picked.length) return ''
  const names = picked.slice(0, 3).map((node) => {
    const label = NODE_TYPE_LABEL[node.type as keyof typeof NODE_TYPE_LABEL] || node.type
    return node.title ? `${label}「${node.title}」` : label
  })
  return picked.length > 3 ? `${names.join('、')} 等 ${picked.length} 项` : names.join('、')
})

const statusText = computed(() => {
  if (busy.value) return stage.value || '正在处理…'
  if (selection.value.length) return `已引用选中 ${selection.value.length} 项`
  return '说一句要做什么'
})

/**
 * 这一轮真提交的卡现在跑到哪了 ✓ —— 回执不是一次性标签，而是**活的**：
 * 「已提交生成 1」→「生成中 62%」→「已出图 1」/「生成失败 1」。
 *
 * 依据是 `useEngineRun.stateOf(id)`（卡片在写、这里在读）：
 * 引擎只报 `{phase, step, total}`，百分比是 composable 自己插出来的 ✓，
 * 所以这里直接读 `percent` 就好，别自己再按 step 算一套（两套口径必然对不上 ✗）。
 */
function runBadge(turn: Turn) {
  // `stateOf` 给的是 computed（卡片那边也读 `.value`）—— 取一次，下面按值判断 ✓
  const rows = (turn.startedRun || []).map((id) => ({ id, state: engine.stateOf(id).value }))
  if (!rows.length) return null
  const running = rows.filter((row) => row.state.status === 'queued' || row.state.status === 'running')
  const failed = rows.filter((row) => row.state.status === 'failed')
  const done = rows.filter((row) => row.state.status === 'succeeded')
  if (failed.length) {
    const reason = failed[0].state.error || '引擎没给原因'
    return {
      cls: 'tag tag-error cc-tagbtn',
      text: `生成失败 ${failed.length}`,
      title: `点一下选中这几张卡看现场 · 第一条：${reason}`,
      ids: rows.map((row) => row.id),
    }
  }
  if (running.length) {
    const pct = Math.max(...running.map((row) => row.state.percent || 0))
    const phase = engine.phaseLabel(running[0].state.phase) || '准备中'
    return {
      cls: 'tag cc-tagbtn',
      text: `生成中 ${pct}%`,
      title: `${running.length}/${rows.length} 张在跑 · 当前：${phase}`,
      ids: rows.map((row) => row.id),
    }
  }
  if (done.length) {
    return {
      cls: 'tag tag-success cc-tagbtn',
      text: `已出图 ${done.length}`,
      title: '成品已自动带进画布，连在原卡右侧',
      ids: rows.map((row) => row.id),
    }
  }
  return { cls: 'tag tag-success cc-tagbtn', text: `已提交生成 ${rows.length}`, title: '等引擎接单…', ids: rows.map((row) => row.id) }
}

const EXAMPLES = [
  { label: '建三张分镜卡并连起来', text: '在空白处建三张分镜卡片，按顺序从左到右排开并依次连起来' },
  { label: '改选中配置卡的提示词', text: '把选中的配置卡的提示词改成「雨夜，青石长街，灯笼晃动」' },
  { label: '删掉空白文本卡', text: '删掉画布上所有没有文字的文本卡' },
  { label: '把选中的卡收进分组', text: '把选中的卡片收进一个分组，标题叫「开场三镜」' },
]

watch(
  () => revision.value,
  () => {
    bump.value = true
    if (bumpTimer) clearTimeout(bumpTimer)
    bumpTimer = setTimeout(() => { bump.value = false }, 700)
  },
)

watch([() => turns.value.length, busy], () => { scrollToEnd() })

function scrollToEnd() {
  nextTick(() => {
    const el = listEl.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

function push(turn: Omit<Turn, 'id' | 'at'>) {
  turns.value.push({ ...turn, id: `turn-${++seq}`, at: Date.now() })
}

/*
 * 对话记录落 localStorage —— **只存文本与回执摘要，不存 `before` 快照**。
 *
 * `before` 是一整份画布文档（节点多时几十上百 KB），存进配额既顶上限、revision
 * 也早对不上；留一堆点不动的「撤销」比不给撤销更差 ✗。所以刷新后历史还在，
 * 那些轮次自然不满足 `canUndo`（它要求有 `before` 且版本没变）。
 */
const LOG_KEY = 'darme.canvas.chatlog'
const LOG_MAX = 40

function persistLog() {
  try {
    const slim = turns.value.slice(-LOG_MAX).map(({ before, ...rest }) => {
      void before
      return rest
    })
    localStorage.setItem(LOG_KEY, JSON.stringify(slim))
  } catch {
    // 隐私模式 / 配额满：不记而已，不影响这一轮对话 ✓
  }
}

function restoreLog() {
  try {
    const raw = localStorage.getItem(LOG_KEY)
    if (!raw) return
    const list = JSON.parse(raw) as Turn[]
    if (!Array.isArray(list) || !list.length) return
    turns.value = list.slice(-LOG_MAX)
    // 续上序号：不然新 turn 的 id 会和恢复出来的撞 key ✓
    seq = list.length
  } catch {
    // 坏数据（手改过 / 旧结构）就当没有，不拦用户 ✓
  }
}

watch(turns, persistLog, { deep: true })

function issues(turn: Turn): string[] {
  return [...(turn.errors || []), ...(turn.rejected || [])]
}

function canUndo(turn: Turn): boolean {
  return (
    turn.role === 'assistant'
    && !!turn.before
    && !turn.undone
    && !turn.stopped
    && turn.afterRevision !== undefined
    && revision.value === turn.afterRevision
  )
}

/** 不可撤销时要说清是哪种原因：刷新后的历史没有快照，跟"你后来改过画布"不是一回事 */
function undoHint(turn: Turn): string {
  if (canUndo(turn)) return '把画布还原到这条指令之前'
  if (turn.undone) return '这一步已经撤销过了'
  if (!turn.before) return '这条是上次会话的记录（快照没留），只能「重新执行」'
  return '画布之后又改动过，不能撤销这一步'
}

async function send(raw?: string) {
  const text = (raw ?? draft.value).trim()
  if (!text || busy.value) return

  draft.value = ''
  autosize()
  push({ role: 'user', text })

  busy.value = true
  stage.value = '正在读取画布并规划…'
  controller = new AbortController()
  // 撤销点必须在**落地之前**存：等 op 落完再存，撤销就等于什么都没做
  const before = canvas.exportDoc()
  // 真提交了哪些卡的生成（卡片自己跑；没提交成的会留在 pendingRun 里）
  const startedRun: string[] = []

  try {
    const outcome = await runCanvasAgent(text, {
      signal: controller.signal,
      // 没有 `generate` 注入就别传 `onGenerate`：否则执行器会以为"有人接"，把 run_generation
      // 当已落地跳过，而实际上一条都没跑 ✗（pendingRun 也就不会提示用户 ✗✗）
      onGenerate: props.generate
        ? (op) => {
          // `CanvasOp` 是联合类型，只有 `run_generation` 带 `nodeId`/`mode`/`prompt`
          const run = op as { nodeId?: string; mode?: string; prompt?: string }
          if (!run.nodeId) return { ok: false, reason: '这条生成动作没带 nodeId，没法定位卡片' }
          // 落成局部常量：下面 `await` 进的闭包里，TS 不会保住对 `run.nodeId` 的收窄 ✓
          const nodeId = run.nodeId
          /*
           * `mode`/`prompt` 必须一起带下去 ✓（源同口径 `generateNodeRef(nodeId, mode, prompt)`）：
           * 模型常常先写一张新的文本卡、再让这张配置卡去跑 —— 只传 nodeId
           * 就等于把模型这一轮写的那句话丢掉，跑出来还是卡上旧提示词 ✗。
           */
          const override: RunOverride = { mode: run.mode, prompt: run.prompt }
          /*
           * 卡片闸没过（缺档位/缺件/缺首帧/类别不符）⇒ 答复 `ok:false`，执行器记 skipped、
           * 面板也不把这条算进「已提交生成」✓
           *
           * ⚠️ 卡片**可能先给一个 promise**（刚建的卡还没挂载 / 档位清单还在路上）——
           * 那就等它真答复再记账 ✓✗：同步当 `true` 处理过一版，实测结果是
           * 「模型让 zimage 卡跑 video，卡片如实拒绝，回执却写已提交生成 1」。
           *
           * 答复里的 `reason` **不要在这里消费**：它跟着返回值上到执行器，
           * 由 skipped 的明细渲染（写两处必然漂移，而用户只看得到其中一处 ✗）。
           */
          const settle = (value: unknown) => {
            const verdict = normalizeRunVerdict(value)
            if (verdict.ok) startedRun.push(nodeId)
            return verdict
          }
          const answer = props.generate?.(nodeId, override)
          return isRunPending(answer) ? answer.then(settle) : settle(answer)
        }
        : undefined,
    })
    stage.value = '正在落地…'

    push({
      role: 'assistant',
      text: outcome.text || (outcome.applied ? '已完成。' : '没听出要改画布哪一部分 —— 试试说「建一张图」「删掉选中的卡」。'),
      applied: outcome.applied,
      pendingRun: outcome.pendingRun,
      startedRun,
      errors: outcome.errors,
      rejected: outcome.rejected,
      warnings: outcome.warnings,
      touched: outcome.touched,
      before,
      afterRevision: revision.value,
    })

    // 落地后把这批卡选中：用户立刻看见「AI 动了哪几张」
    if (outcome.touched.length) canvas.setSelection(outcome.touched)
  } catch (error) {
    const aborted = controller?.signal.aborted === true
      || (error instanceof Error && error.name === 'AbortError')
    if (aborted) {
      // 中止发生在 fetch 层，op 根本没回来 —— 如实说「画布没动」
      push({ role: 'assistant', text: '已停止。画布没有改动。', stopped: true })
    } else {
      const message = error instanceof Error ? error.message : String(error)
      push({ role: 'assistant', text: '这句没跑通。', failed: message })
    }
  } finally {
    busy.value = false
    stage.value = ''
    controller = null
  }
}

/** 停：真中止请求（op 一条都不会落），不是只让界面别转圈 */
function stop() {
  controller?.abort()
}

function undo(turn: Turn) {
  if (!canUndo(turn) || !turn.before) return
  canvas.importDoc(turn.before)
  turn.undone = true
}

/** 重发：拿这条回执前面那句用户原话再跑一遍 */
function retry(turn: Turn) {
  const index = turns.value.indexOf(turn)
  for (let i = index - 1; i >= 0; i -= 1) {
    if (turns.value[i].role === 'user') {
      send(turns.value[i].text)
      return
    }
  }
}

/** 跳到改动：选中 + 把视口挪到第一个还在的节点 */
function jumpTo(turn: Turn) {
  const ids = turn.touched || []
  if (!ids.length) return
  canvas.setSelection(ids)
  const alive = nodes.value.find((node) => ids.includes(node.id))
  if (alive && props.stage?.width) {
    canvas.focusNode(alive.id, { width: props.stage.width, height: props.stage.height })
  }
}

function reset() {
  turns.value = []
  try {
    localStorage.removeItem(LOG_KEY)
  } catch {
    // 同 persistLog：清不掉也就是下次还看得见旧记录，不影响画布 ✓
  }
}

// 面板是常驻组件（挂在画布页上）⇒ setup 里恢复一次即可 ✓
restoreLog()

function autosize() {
  const el = boxEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(140, el.scrollHeight)}px`
}

function onKeydown(event: KeyboardEvent) {
  // 中文输入法组字期间回车是「选词」，不是「发送」
  if (event.isComposing || event.keyCode === 229) return
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    send()
  }
}

function timeOf(at: number) {
  const date = new Date(at)
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}
</script>

<template>
  <aside class="cc" :class="{ 'is-open': props.open }" aria-label="画布对话面板">
    <header class="cc-head">
      <span class="cc-orb"><Sparkles :size="13" /></span>
      <div class="cc-head-text">
        <b>画布助手</b>
        <span>{{ statusText }}</span>
      </div>
      <button class="btn btn-ghost btn-icon" title="清空对话记录" @click="reset">
        <Eraser :size="14" />
      </button>
      <button class="btn btn-ghost btn-icon cc-collapse" title="收起（Ctrl / ⌘ + J）" @click="emit('update:open', false)">
        <ChevronRight :size="16" />
      </button>
    </header>

    <!-- 与画布同源的实时状态：改一下就跳，不轮询 -->
    <div class="cc-stat" :class="{ 'is-bump': bump }">
      <span>{{ nodes.length }} 节点</span>
      <i>·</i>
      <span>{{ edges.length }} 连线</span>
      <i>·</i>
      <span :class="{ 'is-live': selection.length }">选中 {{ selection.length }}</span>
      <i>·</i>
      <span class="cc-rev">r{{ revision }}</span>
    </div>

    <div ref="listEl" class="cc-list">
      <div v-if="!turns.length" class="cc-empty">
        <p class="cc-empty-lead">用一句话改画布：建卡、连线、改提示词、删掉多余的。</p>
        <p class="cc-empty-note">选中的卡片会自动作为上下文带上；改动画布后可以逐条撤销。</p>
        <button v-for="item in EXAMPLES" :key="item.label" class="cc-example" @click="send(item.text)">
          {{ item.label }}
        </button>
      </div>

      <div v-for="turn in turns" :key="turn.id" class="cc-turn" :class="`is-${turn.role}`">
        <div class="cc-meta">
          <span>{{ turn.role === 'user' ? '我' : '画布助手' }}</span>
          <span>{{ timeOf(turn.at) }}</span>
        </div>

        <div class="cc-bubble">
          <p class="cc-text">{{ turn.text }}</p>

          <div v-if="turn.role === 'assistant'" class="cc-tags">
            <span v-if="turn.undone" class="tag">已撤销</span>
            <span v-else-if="turn.stopped" class="tag">已停止</span>
            <span v-else-if="turn.applied" class="tag tag-success">落地 {{ turn.applied }} 条</span>
            <span v-else class="tag">画布未改</span>
            <button
              v-if="runBadge(turn)"
              :class="runBadge(turn)!.cls"
              :title="runBadge(turn)!.title"
              @click="canvas.setSelection(runBadge(turn)!.ids)"
            >
              {{ runBadge(turn)!.text }}
            </button>
            <button
              v-if="turn.pendingRun?.length"
              class="tag tag-warning cc-tagbtn"
              title="选中这几张配置卡，点卡上的「运行」开始生成"
              @click="canvas.setSelection(turn.pendingRun || [])"
            >
              待运行 {{ turn.pendingRun.length }}
            </button>
            <button
              v-if="turn.touched?.length && !turn.undone"
              class="tag tag-info cc-tagbtn"
              title="选中这批卡片并把视口挪过去"
              @click="jumpTo(turn)"
            >
              <Target :size="11" /> 跳到改动 {{ turn.touched.length }}
            </button>
          </div>

          <details v-if="issues(turn).length" class="cc-details">
            <summary>{{ issues(turn).length }} 条未生效</summary>
            <ul>
              <li v-for="(line, index) in issues(turn)" :key="index">{{ line }}</li>
            </ul>
          </details>

          <!-- 提醒与「未生效」分开：这些事**不影响落地**，混进上面那栏会被读成失败 ✗ -->
          <details v-if="turn.warnings?.length" class="cc-details">
            <summary>{{ turn.warnings.length }} 条提醒</summary>
            <ul>
              <li v-for="(line, index) in turn.warnings" :key="index">{{ line }}</li>
            </ul>
          </details>

          <p v-if="turn.failed" class="cc-fail">{{ turn.failed }}</p>

          <div v-if="turn.role === 'assistant'" class="cc-acts">
            <button
              class="btn btn-ghost btn-sm"
              :disabled="!canUndo(turn)"
              :title="undoHint(turn)"
              @click="undo(turn)"
            >
              <Undo2 :size="12" /> 撤销这一步
            </button>
            <button class="btn btn-ghost btn-sm" :disabled="busy" @click="retry(turn)">重新执行</button>
          </div>
        </div>
      </div>

      <div v-if="busy" class="cc-turn is-assistant">
        <div class="cc-bubble cc-typing">
          <span class="cc-dot" /><span class="cc-dot" /><span class="cc-dot" />
          <em>{{ stage || '正在处理…' }}</em>
        </div>
      </div>
    </div>

    <div class="cc-compose">
      <div v-if="selectionLabel" class="cc-ref">
        <span class="cc-ref-key">引用选中</span>
        <span class="cc-ref-val">{{ selectionLabel }}</span>
        <button class="cc-ref-clear" title="取消引用（只清选中，不改画布）" @click="canvas.setSelection([])">×</button>
      </div>

      <!-- 输入框与发送同排：原来按钮另起一行，手机抽屉里凭白吃掉一行高度 ✗ -->
      <div class="cc-row">
        <textarea
          ref="boxEl"
          v-model="draft"
          class="textarea cc-input"
          rows="1"
          placeholder="例如：把这三张卡连成一条流程"
          @input="autosize"
          @keydown="onKeydown"
        />
        <button v-if="busy" class="btn btn-sm cc-stop" @click="stop">
          <Square :size="12" /> 停止
        </button>
        <button v-else class="btn btn-primary btn-sm" :disabled="!draft.trim()" @click="send()">
          <Send :size="12" /> 发送
        </button>
      </div>
      <span class="cc-hint">Enter 发送 · Shift+Enter 换行</span>
    </div>
  </aside>
</template>

<style scoped>
/*
 * 配色一律走全局令牌。画布舞台本身是暗色网格（那是「工作台」），
 * 面板属于**应用外壳**，跟站内其它页面同色系才不会像贴上去的一块黑砖。
 */
.cc {
  display: flex;
  flex-direction: column;
  width: 360px;
  flex: 0 0 360px;
  min-height: 0;
  background: var(--panel-bg, #fff);
  border: 1px solid var(--border, #e3e8f0);
  border-radius: var(--radius, 10px);
  box-shadow: var(--shadow, 0 6px 20px rgba(15, 23, 42, 0.06));
  overflow: hidden;
}

.cc-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 8px 8px 12px;
  border-bottom: 1px solid var(--border, #e3e8f0);
}

.cc-orb {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  flex: 0 0 auto;
  border-radius: 8px;
  background: var(--accent-bg, #e6fffb);
  color: var(--accent, #0d9488);
}

.cc-head-text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.cc-head-text b {
  font-size: 13px;
  line-height: 1.3;
  color: var(--text-0, #0f172a);
}

.cc-head-text span {
  font-size: 11px;
  color: var(--text-3, #8a94a6);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cc-stat {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  font-size: 11px;
  color: var(--text-3, #8a94a6);
  background: var(--bg-1, #f8fafc);
  border-bottom: 1px solid var(--border, #e3e8f0);
  transition: background-color 0.3s ease, color 0.3s ease;
}

.cc-stat i {
  font-style: normal;
  opacity: 0.45;
}

.cc-stat .is-live {
  color: var(--accent, #0d9488);
  font-weight: 600;
}

/* 版本跳动 = 画布内容变了：闪一下，让「同步」看得见 */
.cc-stat.is-bump {
  background: var(--accent-bg, #e6fffb);
  color: var(--accent, #0d9488);
}

.cc-rev {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  opacity: 0.75;
}

.cc-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.cc-empty {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 4px 2px;
}

.cc-empty-lead {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--text-1, #334155);
}

.cc-empty-note {
  margin: 0 0 6px;
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--text-3, #8a94a6);
}

.cc-example {
  text-align: left;
  padding: 8px 10px;
  font-size: 12px;
  color: var(--text-1, #334155);
  background: var(--bg-1, #f8fafc);
  border: 1px solid var(--border, #e3e8f0);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease;
}

.cc-example:hover {
  border-color: var(--accent, #0d9488);
  background: var(--accent-bg, #e6fffb);
  color: var(--accent-text, #0f766e);
}

.cc-turn {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.cc-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 10.5px;
  color: var(--text-3, #8a94a6);
}

.cc-turn.is-user .cc-meta {
  flex-direction: row-reverse;
}

.cc-bubble {
  padding: 8px 10px;
  border-radius: 10px;
  font-size: 12.5px;
  line-height: 1.65;
  min-width: 0;
}

.cc-turn.is-user .cc-bubble {
  align-self: flex-end;
  max-width: 88%;
  background: var(--accent-bg, #e6fffb);
  border: 1px solid var(--accent-border, #b8eee6);
  color: var(--accent-text, #0f766e);
}

.cc-turn.is-assistant .cc-bubble {
  background: var(--bg-1, #f8fafc);
  border: 1px solid var(--border, #e3e8f0);
  color: var(--text-1, #334155);
}

.cc-text {
  margin: 0;
  overflow-wrap: anywhere;
}

.cc-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}

.cc-tagbtn {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  cursor: pointer;
  border: 1px solid transparent;
}

.cc-details {
  margin-top: 6px;
  font-size: 11.5px;
  color: var(--text-2, #64748b);
}

.cc-details summary {
  cursor: pointer;
  color: var(--warning, #b45309);
}

.cc-details ul {
  margin: 4px 0 0;
  padding-left: 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.cc-fail {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--error, #b91c1c);
  overflow-wrap: anywhere;
}

.cc-acts {
  display: flex;
  gap: 4px;
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--border, #e3e8f0);
}

.cc-typing {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--text-3, #8a94a6);
}

.cc-typing em {
  font-style: normal;
  font-size: 11.5px;
  margin-left: 4px;
}

.cc-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent, #0d9488);
  animation: cc-blink 1.1s ease-in-out infinite;
}

.cc-dot:nth-child(2) { animation-delay: 0.15s; }
.cc-dot:nth-child(3) { animation-delay: 0.3s; }

@keyframes cc-blink {
  0%, 100% { opacity: 0.25; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(-2px); }
}

.cc-compose {
  padding: 10px;
  border-top: 1px solid var(--border, #e3e8f0);
  background: var(--panel-bg, #fff);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.cc-ref {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 6px;
  font-size: 11px;
  border-radius: 6px;
  background: var(--accent-bg, #e6fffb);
  border: 1px solid var(--accent-border, #b8eee6);
  color: var(--accent-text, #0f766e);
}

.cc-ref-key {
  flex: 0 0 auto;
  font-weight: 600;
}

.cc-ref-val {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cc-ref-clear {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  padding: 0 2px;
}

.cc-input {
  width: 100%;
  resize: none;
  min-height: 38px;
  max-height: 140px;
  font-size: 12.5px;
  line-height: 1.6;
}

/* 输入框吃掉剩余宽度、按钮贴底（多行输入时按钮跟着往下走，不悬在半空） */
.cc-row {
  display: flex;
  align-items: flex-end;
  gap: 6px;
}

.cc-row .cc-input {
  flex: 1;
  min-width: 0;
}

.cc-row .btn {
  flex: 0 0 auto;
}

.cc-hint {
  display: block;
  margin-top: 4px;
  font-size: 10.5px;
  color: var(--text-3, #8a94a6);
}

.cc-stop {
  color: var(--error, #b91c1c);
  border-color: var(--error-border, #f3c6c6);
}

/*
 * 桌面档：收起 = **不占位**。留一条 360px 的空栏比没有动画更难看，
 * 而且画布会白白少掉三分之一宽；抽屉档（≤1023）走下面的 transform 位移。
 */
@media (min-width: 1024px) {
  .cc:not(.is-open) { display: none; }
}

/* 1024–1279：右栏收窄一档，否则画布放不下一张卡 */
@media (min-width: 1024px) and (max-width: 1279px) {
  .cc { width: 320px; flex: 0 0 320px; }
}

/*
 * ≤1023：面板改成**底部抽屉**。右栏在窄屏会挤掉画布本体（最需要看的就是画布），
 * 抽屉只占 62vh，且画布仍可滚 —— 比把画布压成 200px 宽强得多。
 */
@media (max-width: 1023px) {
  .cc {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    width: 100%;
    flex: none;
    height: min(62vh, 560px);
    z-index: 140;
    border-radius: 14px 14px 0 0;
    border-bottom: 0;
    padding-bottom: env(safe-area-inset-bottom, 0px);
    /* --panel-bg 是 88% 半透明（浮层用），抽屉压在暗色画布上会透出底下的网格，
       正文对比度掉得厉害（实测能看见画布方块 ✗）⇒ 抽屉改成不透明底 */
    background: var(--bg-0, #fff);
    box-shadow: 0 -10px 30px rgba(15, 23, 42, 0.18);
    transform: translateY(105%);
    transition: transform 0.24s ease;
    pointer-events: none;
  }

  /* 抽屉拉手：不然用户不知道这层能收（收起入口只有箭头 + 点遮罩） */
  .cc::before {
    content: '';
    position: absolute;
    top: 6px;
    left: 50%;
    width: 36px;
    height: 4px;
    margin-left: -18px;
    background: var(--border-strong, #bcc9d9);
    border-radius: 2px;
  }

  .cc.is-open {
    transform: translateY(0);
    pointer-events: auto;
  }

  /* 箭头语义随形态变：右栏是「往右收」，抽屉是「往下收」 */
  .cc-collapse svg {
    transform: rotate(90deg);
  }
}

@media (max-width: 767px) {
  .cc {
    height: min(70vh, 620px);
  }

  .cc-list {
    padding: 10px;
  }

  .cc-turn.is-user .cc-bubble {
    max-width: 94%;
  }

  /* 触控档：发送/停止按钮抬到 34px，避免小按钮点不中 */
  .cc-row {
    gap: 6px;
  }

  .cc-row .btn {
    min-height: var(--tap-min, 34px);
    padding-inline: 14px;
  }

  .cc-hint {
    display: none;
  }
}
</style>

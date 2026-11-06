<template>
  <span class="pp-shell">
    <button
      class="btn pp-trigger"
      title="预设框架一键开局：随机 Variation Card（固定 5 镜）→ 建项目 + 第 1 集 + 分镜 → 批量生首帧"
      @click="open"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
      框架开局
    </button>

    <div v-if="shown" class="pp-overlay" @click.self="close">
      <div class="pp-panel" role="dialog" aria-modal="true" aria-label="预设框架一键开局">
        <header class="pp-head">
          <div class="pp-head-copy">
            <h3>预设框架 · 一键开局</h3>
            <p class="pp-sub">Variation Card（固定 5 镜）→ 建项目 + 第 1 集 + 5 条分镜 → 批量生首帧</p>
          </div>
          <button class="pp-x" aria-label="关闭" @click="close">×</button>
        </header>

        <div class="pp-body">
          <!-- ① 卡：主题家族 + 5 镜配比 -->
          <section class="pp-cardhead">
            <div class="pp-family">
              <span class="pp-family-k">主题家族</span>
              <b class="pp-family-v">{{ card?.themeFamily || '—' }}</b>
            </div>
            <button class="btn pp-mini" :disabled="cardLoading" @click="rollCard">
              {{ cardLoading ? '换卡中…' : '换一张' }}
            </button>
          </section>

          <div v-if="cardLoading && !card" class="pp-state">正在取卡…</div>
          <p v-else-if="cardError" class="pp-state pp-err">{{ cardError }}</p>

          <ul v-if="card" class="pp-shots">
            <li v-for="s in card.shots" :key="s.shotIndex" class="pp-shot">
              <span class="pp-shot-no">#{{ s.shotIndex }}</span>
              <div class="pp-shot-main">
                <div class="pp-shot-lead">{{ s.mainFocalPoint }} · {{ s.activity }}</div>
                <div class="pp-chips">
                  <span class="pp-chip">{{ s.compositionPattern }}</span>
                  <span class="pp-chip">{{ s.thematicClue }}</span>
                  <span class="pp-chip">{{ s.characterLayout }}</span>
                  <span class="pp-chip">{{ s.spaceType }} / {{ s.foregroundFrame }}</span>
                  <span class="pp-chip">{{ s.cameraPosition }}</span>
                  <span class="pp-chip">{{ s.lightStructure }}</span>
                  <span class="pp-chip">{{ s.windDirection }}</span>
                  <span v-if="s.livingElement" class="pp-chip pp-chip-live">{{ s.livingElement }}</span>
                </div>
              </div>
            </li>
          </ul>

          <p v-if="card" class="pp-note">
            <code>THEME_FAMILY_A</code> / <code>ACTIVITY_1</code> 这类是后端的<b>占位数据池</b>（原实现即如此，
            等具体领域知识替换），不是乱码；5 镜条数由后端固定，前端不裁剪也不补。
          </p>

          <!-- ② 建剧表单 -->
          <div v-if="!run" class="pp-form">
            <label class="pp-field">
              <span>项目名称</span>
              <input
                v-model="title"
                class="input"
                maxlength="120"
                :disabled="starting"
                placeholder="例如：框架开局 · THEME_FAMILY_A"
              />
            </label>
            <label class="pp-field">
              <span>描述（可选）</span>
              <input
                v-model="description"
                class="input"
                maxlength="400"
                :disabled="starting"
                placeholder="留空则不给描述"
              />
            </label>
            <label class="pp-check">
              <input v-model="autoImages" type="checkbox" :disabled="starting" />
              <span>
                建剧后立刻批量生首帧（5 张并发入队；单张数十秒。异步出图失败<b>不会</b>回写镜头状态，
                进度只看下面的「首帧 ✓」列）
              </span>
            </label>
          </div>

          <!-- ③ 进度 / 结果 -->
          <div v-if="run" class="pp-run">
            <div class="pp-run-head">
              <b>项目 #{{ run.dramaId }} · 第 1 集</b>
              <span class="pp-counts">
                首帧 {{ summary.imagesGenerated }}/{{ summary.totalShots }} · 视频 {{ summary.videosGenerated }}/{{ summary.totalShots }}
              </span>
              <button class="btn pp-mini" :disabled="refreshing" @click="refresh()">
                {{ refreshing ? '刷新中…' : '刷新' }}
              </button>
            </div>

            <ul class="pp-shots">
              <li v-for="row in runRows" :key="row.sbId" class="pp-shot">
                <span class="pp-shot-no">#{{ row.no }}</span>
                <div class="pp-shot-main">
                  <div class="pp-shot-lead">{{ row.lead }}</div>
                  <div class="pp-chips">
                    <span class="pp-chip pp-chip-state" :class="row.tone">{{ row.label }}</span>
                    <span class="pp-chip">{{ row.img }}</span>
                    <span class="pp-chip">{{ row.vid }}</span>
                  </div>
                </div>
              </li>
            </ul>

            <p v-if="pendingImages && enqueuedCount" class="pp-note">
              还有 {{ summary.totalShots - summary.imagesGenerated }} 镜没落首帧，其中 {{ enqueuedCount }} 镜
              <b>已在出图队列里</b>（`storyboards.status` 在后端是入队态，落库后「首帧 ✓」才变）。
              视频只能等首帧落库后再生成 —— 没有首帧的镜头会被后端<b>跳过</b>且不计入 videoGenIds。
            </p>
            <p v-else-if="pendingImages" class="pp-note">
              还有 {{ summary.totalShots - summary.imagesGenerated }} 镜没有首帧，也没在队列里 ——
              先「生成首帧」再「生成视频」（没有首帧的镜头会被后端<b>跳过</b>）。
            </p>
            <p v-else-if="summary.totalShots > 0" class="pp-note pp-ok">
              首帧齐了，可以直接生视频（图生视频，逐镜提交，比出图慢）。
            </p>
          </div>

          <p v-if="starting" class="pp-state pp-wait">
            已提交 {{ elapsed }} 秒…（建剧 + 5 张首帧并发，通常 1–4 分钟；关掉这个窗口<b>不会中断后端</b>，跑完会提示）
          </p>
          <p v-if="err" class="pp-state pp-err">{{ err }}</p>
        </div>

        <footer class="pp-foot">
          <template v-if="!run">
            <button class="btn" @click="close">取消</button>
            <button class="btn btn-primary" :disabled="!canStart" @click="start">
              {{ starting ? `开局中… ${elapsed}s` : '一键开局' }}
            </button>
          </template>
          <template v-else>
            <button class="btn" @click="close">关闭</button>
            <button v-if="pendingImages" class="btn" :disabled="busy" @click="doImages">
              {{ imaging ? '生首帧中…' : '生成首帧' }}
            </button>
            <button class="btn" :disabled="busy || summary.imagesGenerated === 0" @click="doVideos">
              {{ videoing ? '生视频中…' : '生成视频' }}
            </button>
            <button class="btn btn-primary" @click="goDrama">查看项目</button>
          </template>
        </footer>
      </div>
    </div>
  </span>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { toast } from 'vue-sonner'
import { presetFrameworkAPI } from '~/composables/useApi'
import type { PresetFrameworkStatus, VariationCard } from '~/composables/useApi'

/**
 * 预设框架一键管线入口（后端 `routers/preset_framework.py` 的 6 条端点）。
 *
 * 语义按后端实况抄，不按"想当然"写：
 * - 一键开局 = `full-pipeline`（建剧 + 批量首帧），只有显式 `autoGenerateImages:false` 才跳过生图；
 * - 分步开局 = `create`（只建结构）→ 结果面板里再点「生成首帧」= `generate-images`；
 * - 「生成视频」= `generate-videos`，**没有首帧的镜头由后端跳过**，故先看 `summary.imagesGenerated`；
 * - `status/{dramaId}` 是唯一进度真源（每镜 status/imageUrl/videoUrl + summary），面板打开时轮询。
 */

interface Run {
  dramaId: number
  episodeId: number
  storyboardIds: number[]
  variationCard: VariationCard
}

/**
 * ⚠️ `storyboards.status` 在本域是**入队态**，不是完成态：`trigger_image_generation()` 在
 * `generate_image()` 返回 id（只是 insert 了一条 processing 记录）时就写 `image_ready`，
 * 异步出图/出视频的成败**不回写**这一列。真正的完成态是两列 URL：
 * `first_frame_image` / `video_url`（由 worker + webhooks 落库）。
 * 所以下面一律先看 URL、再看 status —— 反过来写会得到「首帧就绪但首帧为空、0/5」的自相矛盾。
 */
function stateOf(r: { imageUrl: string | null; videoUrl: string | null; status: string }) {
  if (r.videoUrl) return { label: '视频就绪', tone: 'ok', enqueued: false }
  if (r.status === 'video_failed') return { label: '视频失败', tone: 'bad', enqueued: false }
  if (r.status === 'video_ready' || r.status === 'generating_video') {
    return { label: '视频入队中', tone: 'warn', enqueued: false }
  }
  if (r.imageUrl) return { label: '首帧就绪', tone: 'ok', enqueued: false }
  if (r.status === 'image_failed') return { label: '首帧失败', tone: 'bad', enqueued: false }
  if (r.status === 'image_ready' || r.status === 'generating_image') {
    return { label: '首帧入队中', tone: 'warn', enqueued: true }
  }
  return { label: '待生成', tone: 'idle', enqueued: false }
}

const EMPTY_SUMMARY = { totalShots: 0, imagesGenerated: 0, videosGenerated: 0 }

const shown = ref(false)
const card = ref<VariationCard | null>(null)
const cardLoading = ref(false)
const cardError = ref('')

const title = ref('')
const description = ref('')
const autoImages = ref(true)

const starting = ref(false)
const imaging = ref(false)
const videoing = ref(false)
const refreshing = ref(false)
const elapsed = ref(0)
const err = ref('')

const run = ref<Run | null>(null)
const status = ref<PresetFrameworkStatus | null>(null)

let tick: ReturnType<typeof setInterval> | null = null
let poll: ReturnType<typeof setInterval> | null = null

const busy = computed(() => imaging.value || videoing.value || starting.value)
const canStart = computed(() => !!card.value && title.value.trim().length > 0 && !starting.value)
const summary = computed(() => status.value?.summary || EMPTY_SUMMARY)
const pendingImages = computed(
  () => !!status.value && summary.value.imagesGenerated < summary.value.totalShots,
)

const runRows = computed(() => {
  const rows = status.value?.storyboards || []
  const shots = (status.value?.variationCard || run.value?.variationCard)?.shots || []
  return rows.map((r, i) => {
    const shot = shots.find((s) => s.shotIndex === r.storyboardNumber) || shots[i] || null
    const st = stateOf(r)
    return {
      sbId: r.id,
      no: r.storyboardNumber ?? i + 1,
      lead: shot ? `${shot.mainFocalPoint} · ${shot.activity}` : `分镜 #${r.id}`,
      label: st.label,
      tone: st.tone,
      enqueued: st.enqueued,
      img: r.imageUrl ? '首帧 ✓' : '首帧 —',
      vid: r.videoUrl ? '视频 ✓' : '视频 —',
    }
  })
})

const enqueuedCount = computed(() => runRows.value.filter((r) => r.enqueued).length)

function open() {
  shown.value = true
  window.addEventListener('keydown', onKey)
  if (!card.value) rollCard()
  if (run.value) {
    refresh(true)
    startPoll()
  }
}

function close() {
  shown.value = false
  stopPoll()
  window.removeEventListener('keydown', onKey)
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') close()
}

function stopPoll() {
  if (poll) {
    clearInterval(poll)
    poll = null
  }
}

function startPoll() {
  stopPoll()
  poll = setInterval(() => {
    if (shown.value && run.value) refresh(true)
  }, 3000)
}

async function rollCard() {
  if (cardLoading.value) return
  cardLoading.value = true
  cardError.value = ''
  try {
    const next = await presetFrameworkAPI.variationCard(card.value?.themeFamily)
    card.value = next
    // 换卡时同步改名：只在"空"或"还是自动生成的名字"时覆盖，别冲掉用户手打的
    if (!title.value.trim() || title.value.startsWith('框架开局 · ')) {
      title.value = `框架开局 · ${next.themeFamily}`
    }
  } catch (e: any) {
    cardError.value = e?.message || '取卡失败'
  } finally {
    cardLoading.value = false
  }
}

async function refresh(silent = false) {
  if (!run.value) return
  if (!silent) refreshing.value = true
  try {
    const s = await presetFrameworkAPI.status(run.value.dramaId)
    status.value = s
    // 首帧齐了又没有在跑的活 ⇒ 进度不会再变，停掉轮询（点按钮会重新起）
    if (!busy.value && s.summary.imagesGenerated >= s.summary.totalShots) stopPoll()
  } catch (e: any) {
    if (!silent) err.value = e?.message || '取状态失败'
  } finally {
    if (!silent) refreshing.value = false
  }
}

async function start() {
  if (!canStart.value || !card.value) return
  err.value = ''
  starting.value = true
  elapsed.value = 0
  tick = setInterval(() => {
    elapsed.value += 1
  }, 1000)
  const payload = {
    title: title.value.trim(),
    description: description.value.trim() || undefined,
    variationCard: card.value,
  }
  try {
    const res = autoImages.value
      ? await presetFrameworkAPI.fullPipeline(payload)
      : await presetFrameworkAPI.create(payload)
    run.value = {
      dramaId: res.dramaId,
      episodeId: res.episodeId,
      storyboardIds: res.storyboardIds || [],
      variationCard: card.value,
    }
    await refresh(true)
    startPoll()
    toast.success(
      shown.value
        ? `框架开局完成：项目 #${res.dramaId}`
        : `框架开局完成：项目 #${res.dramaId}（点「框架开局」看进度）`,
    )
  } catch (e: any) {
    err.value = e?.message || '开局失败'
    toast.error(err.value)
  } finally {
    if (tick) {
      clearInterval(tick)
      tick = null
    }
    starting.value = false
    elapsed.value = 0
  }
}

async function doImages() {
  if (!run.value || busy.value) return
  err.value = ''
  imaging.value = true
  startPoll()
  try {
    const r = await presetFrameworkAPI.generateImages({
      dramaId: run.value.dramaId,
      episodeId: run.value.episodeId,
      storyboardIds: run.value.storyboardIds,
      variationCard: run.value.variationCard,
    })
    const n = (r?.imageGenIds || []).length
    if (n) toast.success(`已提交 ${n} 张首帧`)
    else toast.error('一张都没提交：后端没拿到图（查后端日志与引擎状态）')
  } catch (e: any) {
    err.value = e?.message || '生首帧失败'
    toast.error(err.value)
  } finally {
    imaging.value = false
    await refresh(true)
  }
}

async function doVideos() {
  if (!run.value || busy.value) return
  err.value = ''
  videoing.value = true
  startPoll()
  try {
    const r = await presetFrameworkAPI.generateVideos({
      dramaId: run.value.dramaId,
      episodeId: run.value.episodeId,
      storyboardIds: run.value.storyboardIds,
      variationCard: run.value.variationCard,
    })
    const n = (r?.videoGenIds || []).length
    if (n) toast.success(`已提交 ${n} 段视频`)
    else toast.error('一段都没提交：多半是这些镜头还没有首帧')
  } catch (e: any) {
    err.value = e?.message || '生视频失败'
    toast.error(err.value)
  } finally {
    videoing.value = false
    await refresh(true)
  }
}

function goDrama() {
  const id = run.value?.dramaId
  if (!id) return
  close()
  navigateTo(`/drama/${id}`)
}

onBeforeUnmount(() => {
  stopPoll()
  if (tick) clearInterval(tick)
  window.removeEventListener('keydown', onKey)
})
</script>

<style scoped>
.pp-shell { display: inline-flex; }

.pp-overlay {
  position: fixed; inset: 0; z-index: 240;
  background: rgba(24, 33, 50, 0.32);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.pp-panel {
  width: min(720px, 100%);
  max-height: calc(100vh - 48px);
  max-height: calc(100dvh - 48px);
  display: flex; flex-direction: column;
  background: var(--bg-0); border: 1px solid var(--border);
  border-radius: 12px; box-shadow: var(--shadow-panel); overflow: hidden;
}
.pp-head {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  padding: 16px 18px 12px; border-bottom: 1px solid var(--border);
}
.pp-head h3 { margin: 0; font-size: 15px; color: var(--text-0); }
.pp-sub { margin: 4px 0 0; font-size: 11.5px; color: var(--text-3); }
.pp-x {
  border: none; background: transparent; color: var(--text-3);
  font-size: 20px; line-height: 1; cursor: pointer; padding: 0 2px;
}
.pp-x:hover { color: var(--text-1); }

.pp-body { padding: 14px 18px; overflow-y: auto; flex: 1; }

.pp-state { padding: 14px 0; text-align: center; font-size: 12.5px; color: var(--text-2); line-height: 1.7; }
.pp-err { color: var(--error); }
.pp-wait { padding: 10px 0 2px; color: var(--text-2); }

.pp-cardhead {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 9px 12px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-1); margin-bottom: 10px;
}
.pp-family { display: flex; align-items: baseline; gap: 8px; min-width: 0; }
.pp-family-k { font-size: 10.5px; color: var(--text-3); }
.pp-family-v { font-size: 13px; color: var(--text-0); overflow-wrap: anywhere; }
.pp-mini { padding: 4px 9px; font-size: 11.5px; flex: none; }

.pp-shots { list-style: none; margin: 0 0 10px; padding: 0; }
.pp-shot {
  display: flex; gap: 10px; align-items: flex-start;
  padding: 8px 0; border-bottom: 1px dashed var(--border);
}
.pp-shot:last-child { border-bottom: none; }
.pp-shot-no {
  flex: none; min-width: 26px; font-size: 11px; color: var(--text-3);
  font-variant-numeric: tabular-nums; padding-top: 2px;
}
.pp-shot-main { min-width: 0; flex: 1; }
.pp-shot-lead { font-size: 12px; color: var(--text-1); overflow-wrap: anywhere; }
.pp-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.pp-chip {
  padding: 1px 6px; border-radius: 4px; background: var(--bg-2); color: var(--text-3);
  font-size: 10.5px; line-height: 1.6; overflow-wrap: anywhere;
}
.pp-chip-live { background: var(--accent-bg); color: var(--accent-text); }
.pp-chip-state.idle { background: var(--bg-3); color: var(--text-2); }
.pp-chip-state.warn { background: var(--warning-bg); color: var(--warning); }
.pp-chip-state.ok { background: var(--success-bg); color: var(--success); }
.pp-chip-state.bad { background: var(--error-bg); color: var(--error); }

.pp-note {
  margin: 0 0 12px; padding: 8px 10px; border-left: 3px solid var(--border);
  background: var(--bg-1); color: var(--text-2); font-size: 11.5px; line-height: 1.65;
  overflow-wrap: anywhere;
}
.pp-note code { font-size: 10.5px; color: var(--text-1); }
.pp-note.pp-ok { border-left-color: var(--success); color: var(--text-1); }

.pp-form { display: flex; flex-direction: column; gap: 10px; margin-top: 4px; }
.pp-field { display: flex; flex-direction: column; gap: 4px; font-size: 11.5px; color: var(--text-3); }
.pp-check {
  display: flex; align-items: flex-start; gap: 7px;
  font-size: 11.5px; color: var(--text-2); line-height: 1.6;
}
.pp-check input { margin-top: 2px; accent-color: var(--accent); flex: none; }
.pp-check code { font-size: 10.5px; }

.pp-run-head {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 9px 12px; border-radius: 8px; background: var(--bg-1);
  border: 1px solid var(--border); margin-bottom: 10px;
}
.pp-run-head b { font-size: 12.5px; color: var(--text-0); }
.pp-counts { font-size: 11px; color: var(--text-3); font-variant-numeric: tabular-nums; margin-left: auto; }

.pp-foot {
  display: flex; justify-content: flex-end; gap: 8px;
  padding: 12px 18px; border-top: 1px solid var(--border); background: var(--bg-1);
}

@media (max-width: 767px) {
  .pp-overlay { padding: 0; align-items: flex-end; }
  .pp-panel {
    width: 100%; max-height: calc(100vh - 16px); max-height: calc(100dvh - 16px);
    border-radius: 12px 12px 0 0;
  }
  .pp-body { padding: 12px; }
  .pp-counts { margin-left: 0; width: 100%; }
  .pp-foot { flex-wrap: wrap; }
  .pp-foot .btn { flex: 1 1 130px; }
}
</style>

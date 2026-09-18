<script setup lang="ts">
// 开跑前体检面板 —— 调 `POST /api/v1/production/preflight { episodeId }` ✓
//
// 设计要点（都来自后端那几步踩出来的结论 ✓）：
// 1. **只给 episodeId** ✓ —— 后端自己取数组装 ✓，前端不认识 plan/manifest 的形状 ✓；
// 2. ⭐ `ready` 之外**必须显示 `nextActions`** ✓ —— 这份清单是**按成本排序**的
//    （先零成本的文字/结构 ✓ 最后才花钱 ✓），比"红了几条"有用得多 ✓；
// 3. ⭐⭐ **`ready=false` 且没有分镜时**，后端会明说「无从体检 ≠ 通过」✓ ——
//    面板**原样显示**它 ✓，不自己改写成人话（改写就会把"没判"说成"通过"✗）；
// 4. ⭐ "**没判**"与"通过"在界面上**必须能区分** ✗ —— 覆盖率来自 source.notes ✓。
import { ref, watch, onMounted } from 'vue'
import { toast } from 'vue-sonner'
import { productionAPI } from '~/composables/useApi'

const props = defineProps<{ episodeId: number | null }>()

type Finding = { stage?: string; message?: string; shot?: string; code?: string }
type Report = {
  ready?: boolean
  blockers?: Finding[]
  warnings?: Finding[]
  nextActions?: string[]
  source?: { episodeId?: number; counts?: Record<string, number>; notes?: string[] }
}

const loading = ref(false)
const report = ref<Report | null>(null)
const error = ref('')

async function load() {
  if (!props.episodeId) {
    report.value = null
    return
  }
  loading.value = true
  error.value = ''
  try {
    report.value = await productionAPI.preflightForEpisode(props.episodeId)
  } catch (err: any) {
    // ⚠️ 读不到就**如实说读不到** ✓ —— 不保留上一次的结果冒充本次 ✗
    report.value = null
    error.value = err?.message || '体检请求失败'
    toast.error(`体检失败：${error.value}`)
  } finally {
    loading.value = false
  }
}

watch(() => props.episodeId, load)
onMounted(load)

const STAGE_LABEL: Record<string, string> = {
  source: '数据',
  placeholders: '占位符',
  polish: '质感层',
  continuity: '连续性',
  assetGate: '验收门',
  input: '输入',
}
function stageLabel(stage?: string) {
  return STAGE_LABEL[stage || ''] || stage || '—'
}
</script>

<template>
  <section class="pf">
    <header class="pf-head">
      <div class="pf-title">
        <span class="pf-dot" :class="report?.ready ? 'ok' : 'blocked'"></span>
        开跑前体检
        <span v-if="report" class="pf-verdict" :class="report.ready ? 'ok' : 'blocked'">
          {{ report.ready ? '可以开跑' : '还不能开跑' }}
        </span>
        <span v-else-if="loading" class="pf-verdict">体检中…</span>
        <span v-else class="pf-verdict">未选集</span>
      </div>
      <button class="pf-btn" :disabled="loading || !episodeId" @click="load">
        {{ loading ? '…' : '重新体检' }}
      </button>
    </header>

    <p class="pf-hint">
      只做**事前**检查 ✓（还没花钱就能看出问题 ✓）；`ready` 只代表**检查过了** ✓
      不代表生成质量 ✓。
    </p>

    <p v-if="error" class="pf-error">体检没跑成：{{ error }}（这不等于通过 ✗）</p>

    <template v-if="report">
      <!-- ⭐ 行动项：按成本排序 ✓（先零成本、后花钱 ✓） -->
      <div v-if="report.nextActions?.length" class="pf-block">
        <div class="pf-block-title">接下来做什么（按成本从低到高 ✓）</div>
        <ol class="pf-actions">
          <li v-for="(action, index) in report.nextActions" :key="index">{{ action }}</li>
        </ol>
      </div>

      <div v-if="report.blockers?.length" class="pf-block">
        <div class="pf-block-title blocked">阻断（{{ report.blockers.length }}）</div>
        <ul class="pf-list">
          <li v-for="(item, index) in report.blockers" :key="index">
            <span class="pf-chip">{{ stageLabel(item.stage) }}</span>
            <span v-if="item.shot" class="pf-shot">{{ item.shot }}</span>
            {{ item.message }}
          </li>
        </ul>
      </div>

      <div v-if="report.warnings?.length" class="pf-block">
        <div class="pf-block-title">建议（{{ report.warnings.length }}）—— 不阻断 ✓</div>
        <ul class="pf-list">
          <li v-for="(item, index) in report.warnings" :key="index">
            <span class="pf-chip">{{ stageLabel(item.stage) }}</span>
            {{ item.message }}
          </li>
        </ul>
      </div>

      <!-- ⭐ 覆盖率 / 取数痕迹：用来区分「通过」与「没判」✓ -->
      <div v-if="report.source?.notes?.length" class="pf-block">
        <div class="pf-block-title">数据与覆盖率（「没判」和「通过」要分得清 ✓）</div>
        <ul class="pf-list notes">
          <li v-for="(note, index) in report.source.notes" :key="index">{{ note }}</li>
        </ul>
      </div>

      <div v-if="!report.blockers?.length && !report.warnings?.length"
           class="pf-empty">没有发现阻断或建议 ✓</div>
    </template>
  </section>
</template>

<style scoped>
.pf {
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 10px;
  padding: 14px 16px;
  margin: 14px 0 18px;
  background: rgba(15, 23, 42, 0.35);
}
.pf-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.pf-title { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; }
.pf-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.pf-dot.ok { background: #22c55e; }
.pf-dot.blocked { background: #ef4444; }
.pf-verdict { font-size: 12px; font-weight: 500; opacity: 0.75; }
.pf-verdict.ok { color: #22c55e; opacity: 1; }
.pf-verdict.blocked { color: #ef4444; opacity: 1; }
.pf-btn {
  font-size: 12px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
  border: 1px solid rgba(148, 163, 184, 0.35); background: transparent; color: inherit;
}
.pf-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.pf-hint { font-size: 12px; opacity: 0.6; margin: 8px 0 0; }
.pf-error { font-size: 12px; color: #f59e0b; margin: 8px 0 0; }
.pf-block { margin-top: 12px; }
.pf-block-title { font-size: 12px; font-weight: 600; opacity: 0.8; margin-bottom: 6px; }
.pf-block-title.blocked { color: #ef4444; opacity: 1; }
.pf-actions { margin: 0; padding-left: 20px; font-size: 13px; line-height: 1.7; }
.pf-list { margin: 0; padding-left: 0; list-style: none; font-size: 13px; line-height: 1.7; }
.pf-list li { display: flex; gap: 6px; align-items: baseline; }
.pf-list.notes li { opacity: 0.7; font-size: 12px; display: block; }
.pf-chip {
  flex: none; font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: rgba(148, 163, 184, 0.18);
}
.pf-shot { flex: none; font-size: 11px; opacity: 0.6; font-family: ui-monospace, monospace; }
.pf-empty { margin-top: 12px; font-size: 13px; opacity: 0.7; }
</style>

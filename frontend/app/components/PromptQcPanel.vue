<script setup lang="ts">
// 提示词生产质检台（独立组件，最小侵入挂到提示词页）
//
// 两件事，都来自 reference/short-drama-agent 的生产契约：
//   ① 占位符解析：`<location>L1</location>` → **具体场景描述**（模型不认识 L1 这种内部编号 ✗）
//   ② 五段质感层：逼出**物理锚点**（真实镜头/调色/质感/瑕疵），并把「只有空话」逐条报出来
//
// ⚠️ 两个端点都是**纯函数、毫秒级**（不碰 DB、不调模型）⇒ 这里可以随时点、随手改 ✓
// ⚠️ 返回形状按后端真键渲染（不猜）：resolve → text/ok/used/unresolved/durationsMs/
//    residualTags/bareIds/problems；polish → sections/text/ok/issues/vagueTerms/inserted
// ⚠️ 面板**只呈现**，不自己判分（判据只在前端一次都不用写）
import { computed, ref } from 'vue'
import { promptToolsAPI } from '~/composables/useApi'

const props = defineProps<{
  /** 可选的映射表（locations/roles/props/clues）；不传 ⇒ 未映射编号会被如实报出来 */
  maps?: Record<string, Record<string, string>>
}>()

const raw = ref('')
const resolved = ref<Record<string, any> | null>(null)
const busyResolve = ref(false)
const error = ref('')

// 质感层输入：只收连续性镜头卡上**真实存在**的字段（不凭空补默认值 ✗）
const themeTags = ref('')
const characterScene = ref('')
const lens = ref('')
const palette = ref('')
const texture = ref('')
const shotType = ref('')
const handheld = ref(false)
const innerMonologue = ref(false)

const polished = ref<Record<string, any> | null>(null)
const busyPolish = ref(false)

const sections = computed<[string, string][]>(() => {
  const raw = polished.value?.sections
  return raw && typeof raw === 'object' ? Object.entries(raw as Record<string, string>) : []
})

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : []
}

async function runResolve() {
  if (!raw.value.trim()) return
  busyResolve.value = true
  error.value = ''
  try {
    resolved.value = (await promptToolsAPI.resolve({
      text: raw.value,
      maps: props.maps ?? {},
    })) as Record<string, any>
  } catch (err: any) {
    error.value = err?.message || '解析失败'
  } finally {
    busyResolve.value = false
  }
}

async function runPolish() {
  busyPolish.value = true
  error.value = ''
  try {
    polished.value = (await promptToolsAPI.polish({
      themeTags: themeTags.value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
      characterScene: characterScene.value,
      lens: lens.value,
      palette: palette.value,
      texture: texture.value,
      shotType: shotType.value,
      handheld: handheld.value,
      innerMonologue: innerMonologue.value,
      // 分镜切片：没有就传空串，**不**拿别的字段凑（凑出来的切片会变成假镜头 ✗）
      slice: '',
    })) as Record<string, any>
  } catch (err: any) {
    error.value = err?.message || '质感层生成失败'
  } finally {
    busyPolish.value = false
  }
}
</script>

<template>
  <section class="qc-panel">
    <header class="qc-head">
      <h2 class="qc-title">提示词生产质检</h2>
      <p class="qc-sub">
        这两步都是<strong>本地纯计算</strong>（毫秒级、不花钱、不落库）—— 先把提示词弄干净，
        再送去生成。
      </p>
    </header>

    <p v-if="error" class="qc-error">{{ error }}</p>

    <!-- ① 占位符解析 -->
    <div class="qc-block">
      <div class="qc-block-head">
        <span class="qc-step">① 占位符 → 具体内容</span>
        <button class="qc-btn" :disabled="busyResolve || !raw.trim()" @click="runResolve">
          {{ busyResolve ? '解析中…' : '解析' }}
        </button>
      </div>
      <textarea
        v-model="raw"
        class="qc-input"
        rows="4"
        placeholder="把提示词粘进来，例如：<location>L1</location> 里，<role>R5</role> 缓缓抬头，目标时长 <duration-ms>6000</duration-ms>"
      />
      <p class="qc-hint">
        模型<strong>不认识</strong> L1 / R5 这类内部编号 ✗ —— 没解析就送出去，会被当成文字生成。
      </p>

      <div v-if="resolved" class="qc-result">
        <div class="qc-badges">
          <span class="qc-chip" :class="resolved.ok ? 'ok' : 'bad'">
            {{ resolved.ok ? '可以送模型' : '还不能送模型' }}
          </span>
        </div>
        <pre class="qc-text">{{ resolved.text }}</pre>

        <ul v-if="list(resolved.problems).length" class="qc-list bad">
          <li v-for="(item, index) in list(resolved.problems)" :key="index">{{ item }}</li>
        </ul>
        <p v-if="list(resolved.unresolved).length" class="qc-warn">
          未映射编号 {{ list(resolved.unresolved).length }} 处 —— 请补映射表（locations / roles / props / clues）。
        </p>
        <p v-if="list(resolved.residualTags).length" class="qc-warn">
          仍有类标签片段：{{ list(resolved.residualTags).slice(0, 6).join(' ') }}
          —— 常见原因：开闭标签写错 / 嵌套 / 跨行。
        </p>
        <p v-if="list(resolved.bareIds).length" class="qc-warn">
          裸编号（只警告，不阻断）：{{ list(resolved.bareIds).join(' ') }}
        </p>
      </div>
    </div>

    <!-- ② 五段质感层 -->
    <div class="qc-block">
      <div class="qc-block-head">
        <span class="qc-step">② Mx-Shell 五段质感层</span>
        <button class="qc-btn" :disabled="busyPolish" @click="runPolish">
          {{ busyPolish ? '生成中…' : '生成质感层' }}
        </button>
      </div>
      <div class="qc-grid">
        <input v-model="themeTags" class="qc-field" placeholder="主题标签（逗号分隔，至少 3 个）" />
        <input v-model="characterScene" class="qc-field" placeholder="锁定角色与场景" />
        <input v-model="lens" class="qc-field" placeholder="镜头（如 35mm 定焦）" />
        <input v-model="palette" class="qc-field" placeholder="调色（如 冷绿灰）" />
        <input v-model="texture" class="qc-field" placeholder="质感（如 皮肤毛孔可见）" />
        <input v-model="shotType" class="qc-field" placeholder="景别（如 中近景）" />
      </div>
      <div class="qc-toggles">
        <label><input v-model="handheld" type="checkbox" /> 手持镜头</label>
        <label><input v-model="innerMonologue" type="checkbox" /> 内心独白</label>
      </div>

      <div v-if="polished" class="qc-result">
        <div class="qc-badges">
          <span class="qc-chip" :class="polished.ok ? 'ok' : 'bad'">
            {{ polished.ok ? '约束够硬' : '有软约束' }}
          </span>
        </div>
        <div v-for="[key, value] in sections" :key="key" class="qc-section">
          <span class="qc-section-key">{{ key }}</span>
          <span class="qc-section-val">{{ value }}</span>
        </div>
        <pre v-if="polished.text" class="qc-text">{{ polished.text }}</pre>
        <ul v-if="list(polished.issues).length" class="qc-list bad">
          <li v-for="(item, index) in list(polished.issues)" :key="index">{{ item }}</li>
        </ul>
      </div>
    </div>
  </section>
</template>

<style scoped>
.qc-panel {
  margin-bottom: 16px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-2);
}
.qc-head {
  margin-bottom: 10px;
}
.qc-title {
  margin: 0 0 4px;
  font-size: 14px;
  font-weight: 600;
}
.qc-sub {
  margin: 0;
  font-size: 12px;
  color: var(--text-3);
}
.qc-error {
  margin: 8px 0;
  font-size: 12px;
  color: var(--danger, #e5484d);
}
.qc-block {
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.qc-block + .qc-block {
  margin-top: 12px;
}
.qc-block-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.qc-step {
  font-size: 13px;
  font-weight: 600;
}
.qc-btn {
  padding: 5px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-1);
  color: var(--text-1);
  font-size: 12px;
  cursor: pointer;
}
.qc-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.qc-input,
.qc-field {
  width: 100%;
  padding: 7px 9px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-1);
  color: var(--text-1);
  font-size: 12px;
  font-family: inherit;
}
.qc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 6px;
}
.qc-toggles {
  display: flex;
  gap: 16px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-2);
}
.qc-hint,
.qc-warn {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--text-3);
}
.qc-warn {
  color: var(--warning, #d97706);
}
.qc-result {
  margin-top: 10px;
}
.qc-badges {
  margin-bottom: 6px;
}
.qc-chip {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
}
.qc-chip.ok {
  background: var(--success-bg, #10362a);
  color: var(--success, #30a46c);
}
.qc-chip.bad {
  background: var(--danger-bg, #3a1216);
  color: var(--danger, #e5484d);
}
.qc-text {
  margin: 6px 0 0;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--bg-1);
  color: var(--text-2);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.qc-list {
  margin: 6px 0 0;
  padding-left: 18px;
  font-size: 12px;
}
.qc-list.bad {
  color: var(--danger, #e5484d);
}
.qc-section {
  display: flex;
  gap: 8px;
  padding: 3px 0;
  font-size: 12px;
  color: var(--text-2);
}
.qc-section-key {
  flex: 0 0 84px;
  color: var(--text-3);
}
</style>

<template>
  <div class="models-page">
    <div class="settings-scroll">
      <div class="settings-head">
        <div class="settings-brand">
          <div class="settings-brand-mark">
            <img v-if="showBrandImage" :src="brandLogo" alt="短剧工坊" class="settings-brand-logo" @error="showBrandImage = false" />
            <span v-else class="settings-brand-fallback">剧</span>
          </div>
          <div class="settings-brand-copy">
            <div class="settings-brand-kicker">Drama Studio</div>
            <div class="settings-brand-name">短剧工坊</div>
          </div>
        </div>
        <h2 class="settings-title">AI 服务配置</h2>
        <p class="settings-desc">先用推荐模板快速落配置，再按服务类型微调。工作台创建集时会锁定所选图片、视频和音频能力。</p>
      </div>

      <section class="setup-panel card">
        <div class="setup-panel-head">
          <div>
            <div class="setup-kicker">Quick Setup</div>
            <div class="setup-title">推荐配置</div>
            <div class="setup-desc">一键写入文本、图片、视频、音频四类推荐配置，适合作为开箱默认方案。</div>
          </div>
          <button class="btn btn-primary" @click="presetDialog = true">
            <Sparkles :size="14" /> 一键配置
          </button>
        </div>
        <div class="preset-grid">
          <article v-for="preset in presetCards" :key="preset.serviceType" class="preset-card">
            <div class="preset-card-top">
              <span class="preset-service">{{ preset.label }}</span>
              <span class="tag tag-accent">{{ preset.provider }}</span>
            </div>
            <div class="preset-model mono">{{ preset.model }}</div>
            <div class="preset-base mono">{{ preset.baseUrl }}</div>
          </article>
        </div>
      </section>

      <section class="setup-panel card">
        <div class="setup-panel-head">
          <div>
            <div class="setup-kicker">Local Models</div>
            <div class="setup-title">本地模型</div>
            <div class="setup-desc">无需 API Key，一键初始化 Ollama / SD / 本地视频 / 语音四类本地服务。</div>
          </div>
          <button class="btn btn-ghost" :disabled="localLoading" @click="initLocalModels">
            <Loader2 v-if="localLoading" :size="14" class="animate-spin" />
            <Server v-else :size="14" />
            初始化本地模型
          </button>
        </div>
        <div v-if="localConfigs.length" class="local-grid">
          <article v-for="lc in localConfigs" :key="lc.id" class="preset-card">
            <div class="preset-card-top">
              <span class="preset-service">{{ serviceMeta[lc.service_type]?.label || lc.service_type }}</span>
              <span class="tag tag-accent">本地</span>
            </div>
            <div class="preset-model mono">{{ fmtModel(lc.model) }}</div>
            <div class="preset-base mono">{{ lc.base_url || '未设置 Base URL' }}</div>
          </article>
        </div>
        <p v-else class="config-empty">尚未初始化本地模型，点击右上角按钮一键创建。</p>

        <!-- Ollama 本地模型管理 -->
        <div class="ollama-panel">
          <div class="ollama-head">
            <div class="ollama-title">
              <Bot :size="14" />
              <span>Ollama 模型管理</span>
              <span class="ollama-state" :class="{ on: ollamaStatus?.running }">{{ ollamaStatus?.running ? '运行中' : '未运行' }}</span>
            </div>
            <div class="ollama-head-actions">
              <button class="btn btn-ghost btn-icon" :disabled="ollamaBusy" title="刷新检测" @click="refreshOllama">
                <Loader2 v-if="ollamaBusy" :size="13" class="animate-spin" />
                <RefreshCw v-else :size="13" />
              </button>
              <button v-if="!ollamaStatus?.running" class="btn btn-ghost btn-sm" :disabled="ollamaBusy" @click="startOllama">
                <Play :size="13" /> 启动 Ollama
              </button>
            </div>
          </div>
          <p v-if="ollamaStatus?.message" class="ollama-msg">{{ ollamaStatus.message }}</p>
          <div v-if="ollamaModels.length" class="ollama-models">
            <div v-for="m in ollamaModels" :key="m.name" class="ollama-model">
              <span class="ollama-model-name mono">{{ m.name }}</span>
              <span class="ollama-model-size">{{ m.size_label }}</span>
              <button class="btn btn-ghost btn-xs" :disabled="ollamaBusy" title="创建/切换本地文本配置使用该模型" @click="useOllamaModel(m.name)">
                <Plus :size="12" /> 使用
              </button>
            </div>
          </div>
          <div v-else class="ollama-empty">{{ ollamaStatus?.running ? '本机暂无已安装模型，可在下方输入模型名下载' : 'Ollama 未运行，先点击「启动 Ollama」' }}</div>
          <div class="ollama-pull">
            <input v-model="ollamaPullName" class="input" placeholder="输入模型名，如 qwen3:8b / qwen2.5:7b" @keyup.enter="pullOllamaModel" />
            <button class="btn btn-primary btn-sm" :disabled="ollamaPulling || !ollamaPullName.trim()" @click="pullOllamaModel">
              <Loader2 v-if="ollamaPulling" :size="13" class="animate-spin" />
              <Download v-else :size="13" />
              {{ ollamaPulling ? '下载中…' : '下载模型' }}
            </button>
          </div>
          <div v-if="ollamaPulling" class="ollama-progress">
            <div class="gpu-bar"><div class="gpu-bar-fill" :style="{ width: pullProgress + '%' }" /></div>
            <div class="ollama-progress-label">{{ pullStatusText }}</div>
          </div>
        </div>
      </section>

      <section class="setup-panel card">
        <div class="setup-panel-head">
          <div>
            <div class="setup-kicker">GPU Monitor</div>
            <div class="setup-title">GPU 显存监控</div>
            <div class="setup-desc">查看本地模型显存占用、租约状态与队列，支持一键释放全部显存。</div>
          </div>
          <div class="gpu-head-actions">
            <button class="btn btn-ghost btn-icon" :disabled="gpuLoading" title="刷新" @click="loadGpuStatus">
              <Loader2 v-if="gpuLoading" :size="13" class="animate-spin" />
              <RefreshCw v-else :size="13" />
            </button>
            <button class="btn btn-ghost btn-sm" :disabled="gpuLoading || !gpu?.loadedModels?.length" @click="releaseAllGpu">
              <Monitor :size="13" /> 释放显存
            </button>
          </div>
        </div>
        <div v-if="gpu" class="gpu-grid">
          <div class="gpu-card">
            <div class="gpu-label">显存占用</div>
            <div class="gpu-value">{{ gpu.hardware ? gpu.usedVRAM_GB.toFixed(1) + ' / ' + gpu.totalVRAM_GB + ' GB' : '0 GB' }}</div>
            <div class="gpu-bar"><div class="gpu-bar-fill" :style="{ width: vramPercent + '%' }" /></div>
          </div>
          <div class="gpu-card">
            <div class="gpu-label">租约状态</div>
            <div class="gpu-value">{{ gpu.isLocked ? '锁定中' : '空闲' }}</div>
            <div class="gpu-sub">{{ gpu.holder || '无持有任务' }} · 队列 {{ gpu.queueLength }}</div>
          </div>
          <div class="gpu-card">
            <div class="gpu-label">已加载模型</div>
            <div class="gpu-value">{{ gpu.loadedModels?.length || 0 }} 个</div>
            <div class="gpu-sub truncate">{{ gpu.loadedModels?.join(', ') || '—' }}</div>
          </div>
          <div v-if="gpu.hardware" class="gpu-card">
            <div class="gpu-label">{{ gpu.hardware.gpuName }}</div>
            <div class="gpu-value">{{ gpu.hardware.utilizationPercent }}% · {{ gpu.hardware.temperatureC }}°C</div>
            <div class="gpu-sub">{{ fmtMB(gpu.hardware.usedMemoryMB) }} / {{ fmtMB(gpu.hardware.totalMemoryMB) }} 已用</div>
          </div>
        </div>
        <div v-if="gpu && !gpu.hardware" class="gpu-notice">
          未检测到 NVIDIA 独显（nvidia-smi 不可用）。当前仅显示软件估算的显存/租约状态，真实 GPU 名称、利用率、温度需在装有 NVIDIA 显卡的机器上查看。
        </div>
        <p v-else-if="!gpuLoading" class="config-empty">暂无 GPU 数据，点击刷新获取。无 NVIDIA GPU 时仅显示租约状态。</p>
      </section>

      <section class="setup-panel card">
        <div class="setup-panel-head compact">
          <div>
            <div class="setup-title">Token 用量统计</div>
            <div class="setup-desc">累计所有 Agent 调用的 token 消耗（输入 / 输出 / 总计）。</div>
          </div>
        </div>
        <div v-if="tokenStats" class="token-stats">
          <div class="token-stat-card">
            <div class="token-stat-num">{{ fmtTokens(tokenStats.totalTokens) }}</div>
            <div class="token-stat-label">总 Token</div>
          </div>
          <div class="token-stat-card">
            <div class="token-stat-num">{{ fmtTokens(tokenStats.totalInputTokens) }}</div>
            <div class="token-stat-label">输入 Token</div>
          </div>
          <div class="token-stat-card">
            <div class="token-stat-num">{{ fmtTokens(tokenStats.totalOutputTokens) }}</div>
            <div class="token-stat-label">输出 Token</div>
          </div>
          <div class="token-stat-card">
            <div class="token-stat-num">{{ tokenStats.runs }}</div>
            <div class="token-stat-label">Agent 调用次数</div>
          </div>
        </div>
        <table v-if="tokenStats && tokenStats.byScope?.length" class="token-table">
          <thead>
            <tr><th>Agent</th><th>调用</th><th>输入</th><th>输出</th><th>总计</th></tr>
          </thead>
          <tbody>
            <tr v-for="row in tokenStats.byScope" :key="row.scope">
              <td>{{ row.scope }}</td>
              <td>{{ row.runs }}</td>
              <td>{{ fmtTokens(row.inputTokens) }}</td>
              <td>{{ fmtTokens(row.outputTokens) }}</td>
              <td>{{ fmtTokens(row.totalTokens) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="config-empty">暂无 token 用量记录，运行任意 Agent 任务后自动统计。</p>
      </section>

      <section class="setup-panel card">
        <div class="setup-panel-head compact">
          <div>
            <div class="setup-title">快捷模板</div>
            <div class="setup-desc">选择服务类型后，直接用模板填充推荐的 `provider / base URL / model`。</div>
          </div>
        </div>
        <div class="template-row">
          <button
            v-for="st in serviceTypes"
            :key="st.type"
            class="template-type-chip"
            @click="startAddCfg(st.type)"
          >
            {{ st.label }}
          </button>
        </div>
      </section>

      <div class="sections">
        <section v-for="st in serviceTypes" :key="st.type">
          <div class="section-head">
            <div>
              <span class="section-title">{{ st.label }}</span>
              <div class="section-subtitle">{{ serviceMeta[st.type].desc }}</div>
            </div>
            <span v-if="countActive(st.type)" class="tag tag-accent">{{ countActive(st.type) }} 已启用</span>
            <button class="btn btn-ghost btn-sm ml-auto" @click="startAddCfg(st.type)"><Plus :size="13" /> 添加</button>
          </div>
          <div class="config-list">
            <template v-if="onlineByType(st.type).length">
              <div class="config-group-label">在线 API</div>
              <div v-for="c in onlineByType(st.type)" :key="c.id" class="card config-row">
                <div class="config-info">
                  <div class="config-main">
                    <div class="config-line">
                      <span class="config-provider">{{ c.provider }}</span>
                      <span class="config-name">{{ c.name || `${c.provider}-${c.service_type}` }}</span>
                      <span class="tag tag-accent">在线</span>
                    </div>
                    <span class="config-model mono truncate">{{ fmtModel(c.model) }}</span>
                    <span class="config-base mono truncate">{{ c.base_url || '未设置 Base URL' }}</span>
                  </div>
                </div>
                <span :class="['tag', c.api_key ? 'tag-success' : 'tag-error']">{{ c.api_key ? '已配置' : '无密钥' }}</span>
                <button class="btn btn-ghost btn-sm" @click="testExistingCfg(c)">测试</button>
                <label class="toggle"><input type="checkbox" :checked="c.is_active" @change="toggleCfg(c)"><span /></label>
                <button class="btn btn-ghost btn-icon" @click="startEditCfg(c)"><Pencil :size="13" /></button>
                <button class="btn btn-ghost btn-icon" @click="delCfg(c.id)"><Trash2 :size="13" /></button>
              </div>
            </template>
            <template v-if="localByType(st.type).length">
              <div class="config-group-label">本地模型</div>
              <div v-for="c in localByType(st.type)" :key="c.id" class="card config-row">
                <div class="config-info">
                  <div class="config-main">
                    <div class="config-line">
                      <span class="config-provider">{{ c.provider }}</span>
                      <span class="config-name">{{ c.name || `${c.provider}-${c.service_type}` }}</span>
                      <span class="tag tag-local">本地</span>
                    </div>
                    <span class="config-model mono truncate">{{ fmtModel(c.model) }}</span>
                    <span class="config-base mono truncate">{{ c.base_url || '未设置 Base URL' }}</span>
                  </div>
                </div>
                <button class="btn btn-ghost btn-sm" @click="testExistingCfg(c)">测试</button>
                <label class="toggle"><input type="checkbox" :checked="c.is_active" @change="toggleCfg(c)"><span /></label>
                <button class="btn btn-ghost btn-icon" @click="startEditCfg(c)"><Pencil :size="13" /></button>
                <button class="btn btn-ghost btn-icon" @click="delCfg(c.id)"><Trash2 :size="13" /></button>
              </div>
            </template>
            <p v-if="!byType(st.type).length" class="config-empty">暂无配置</p>
          </div>
        </section>
      </div>
    </div>

    <!-- AI Config Dialog -->
    <div v-if="cfgDialog" class="overlay" @click.self="cfgDialog = false">
      <form class="modal card config-modal" @submit.prevent="saveCfg">
        <div class="config-modal-head">
          <div>
            <div class="setup-kicker">{{ cfgEditId ? 'Edit Config' : 'New Config' }}</div>
            <h2 class="modal-title">{{ cfgEditId ? '编辑服务配置' : `添加${serviceMeta[cfgForm.service_type].label}服务` }}</h2>
            <div class="modal-note">推荐先选择模板，系统会自动填入更合理的 `Base URL` 与默认模型。</div>
          </div>
          <span class="tag tag-accent">{{ serviceMeta[cfgForm.service_type].label }}</span>
        </div>
        <div class="preset-picker">
          <button
            v-for="preset in presetsByType(cfgForm.service_type)"
            :key="`${cfgForm.service_type}-${preset.provider}`"
            type="button"
            class="preset-pill"
            @click="applyProviderPreset(cfgForm.service_type, preset.provider)"
          >
            {{ preset.label }}
          </button>
        </div>
        <label class="field">
          <span class="field-label">配置名称</span>
          <input v-model="cfgForm.name" class="input" placeholder="如 默认图像服务" />
        </label>
        <label class="field"><span class="field-label">服务商</span>
          <BaseSelect v-model="cfgForm.provider" :options="providerSelectOptions" placeholder="选择服务商" searchable />
        </label>
        <label class="field">
          <span class="field-label">优先级</span>
          <input v-model.number="cfgForm.priority" class="input" type="number" min="0" max="999" />
          <span class="field-hint">数值越高越优先。工作台默认会优先使用同类型里优先级最高的启用配置。</span>
        </label>
        <label class="field"><span class="field-label">API Key</span><input v-model="cfgForm.api_key" class="input" type="password" placeholder="sk-..." /></label>
        <label class="field"><span class="field-label">Base URL</span><input v-model="cfgForm.base_url" class="input" placeholder="https://..." /></label>
        <div class="endpoint-hint">
          <span class="dim">实际端点前缀：</span>
          <span class="mono">{{ endpointHint }}</span>
        </div>
        <label class="field"><span class="field-label">模型（逗号分隔）</span><input v-model="cfgForm.modelStr" class="input" placeholder="model-name" /></label>
        <label v-if="cfgForm.service_type === 'image'" class="field">
          <span class="field-label">负面提示词（可选）</span>
          <textarea v-model="cfgForm.negative_prompt" class="input" rows="2" placeholder="如 low quality, blurry, distorted face, watermark, text" />
          <span class="field-hint">图片生成时统一排除的内容；留空则用各服务商默认值。</span>
        </label>
        <div v-if="cfgTestResult" class="test-result" :class="{ ok: cfgTestResult.reachable, bad: !cfgTestResult.reachable }">
          <div class="test-result-head">
            <span class="tag" :class="cfgTestResult.reachable ? 'tag-success' : 'tag-error'">{{ cfgTestResult.status || 'ERROR' }}</span>
            <span>{{ cfgTestResult.message }}</span>
          </div>
          <div class="mono test-result-url">{{ cfgTestResult.method }} {{ cfgTestResult.url }}</div>
          <div v-if="cfgTestResult.response_preview" class="mono test-result-preview">{{ cfgTestResult.response_preview }}</div>
        </div>
        <div v-if="cfgModelsResult" class="models-result">
          <div class="models-result-head">
            <span class="tag" :class="cfgModelsResult.listable ? 'tag-accent' : 'tag'">{{ cfgModelsResult.listable ? `${cfgModelsResult.models_count} 个模型` : '不支持列举' }}</span>
            <span>{{ cfgModelsResult.message }}</span>
          </div>
          <div v-if="cfgModelsResult.model_checks && cfgModelsResult.model_checks.length" class="models-exists-list">
            <div v-for="c in cfgModelsResult.model_checks" :key="c.model" class="models-exists" :class="{ ok: c.exists, bad: !c.exists }">
              {{ c.exists ? '✓' : '✗' }} {{ c.model }}
            </div>
          </div>
          <div v-else-if="cfgModelsResult.model" class="models-exists" :class="{ ok: cfgModelsResult.model_exists, bad: !cfgModelsResult.model_exists }">
            {{ cfgModelsResult.model_exists ? '✓' : '✗' }} {{ cfgModelsResult.model }}
          </div>
          <div v-if="cfgModelsResult.models.length" class="models-list">
            <button
              v-for="m in cfgModelsResult.models"
              :key="m"
              type="button"
              class="model-chip mono"
              :class="{ active: isCurrentModel(m) }"
              :title="`点击填入模型名`"
              @click="cfgForm.modelStr = m"
            >{{ m }}</button>
          </div>
        </div>
        <div class="modal-actions">
          <button type="button" class="btn btn-ghost" :disabled="cfgTesting" @click="testDraftCfg">
            <Loader2 v-if="cfgTesting" :size="12" class="animate-spin" />
            <span v-else>测试配置</span>
          </button>
          <button type="button" class="btn btn-ghost" :disabled="cfgModelsLoading" @click="listDraftModels">
            <Loader2 v-if="cfgModelsLoading" :size="12" class="animate-spin" />
            <span v-else>列出模型</span>
          </button>
          <button type="button" class="btn" @click="cfgDialog = false">取消</button>
          <button type="submit" class="btn btn-primary">保存</button>
        </div>
      </form>
    </div>

    <!-- Quick Preset Dialog -->
    <div v-if="presetDialog" class="overlay" @click.self="presetDialog = false">
      <form class="modal card config-modal" @submit.prevent="applyQuickPreset">
        <div class="config-modal-head">
          <div>
            <div class="setup-kicker">Quick Preset</div>
            <h2 class="modal-title">一键配置</h2>
            <div class="modal-note">按推荐链路自动创建或更新 4 条服务配置，并同时初始化 5 个 Agent 的默认模型。</div>
          </div>
          <span class="tag tag-success">推荐</span>
        </div>
        <div class="preset-grid-form">
          <label class="field">
            <span class="field-label">API Key <span class="dim">(统一用于文本 / 图片 / 视频 / 音频)</span></span>
            <input v-model="presetForm.apiKey" class="input" type="password" placeholder="输入统一 API Key" />
          </label>
        </div>
        <div class="preset-grid compact">
          <article v-for="preset in presetCards" :key="`${preset.serviceType}-${preset.provider}`" class="preset-card">
            <div class="preset-card-top">
              <span class="preset-service">{{ preset.label }}</span>
              <span class="tag tag-accent">{{ preset.provider }}</span>
            </div>
            <div class="preset-model mono">{{ preset.model }}</div>
            <div class="preset-base mono">{{ preset.baseUrl }}</div>
          </article>
        </div>
        <div class="modal-actions">
          <button type="button" class="btn" @click="presetDialog = false">取消</button>
          <button type="submit" class="btn btn-primary">创建并启用</button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Plus, Pencil, Trash2, Loader2, Bot, Sparkles, Monitor, RefreshCw, Server, Play, Download } from 'lucide-vue-next'
import BaseSelect from '~/components/BaseSelect.vue'
import { toast } from 'vue-sonner'
import { aiConfigAPI, aiProvidersAPI, traceAPI } from '~/composables/useApi'
import brandLogo from '~/assets/brand-logo.svg'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()

const showBrandImage = ref(true)

// ===== AI Service Configs =====
const cfgs = ref([])
const cfgDialog = ref(false)
const cfgEditId = ref(null)
const presetDialog = ref(false)
const cfgTesting = ref(false)
const cfgTestResult = ref(null)
const cfgModelsLoading = ref(false)
const cfgModelsResult = ref<any>(null)
const cfgForm = reactive({ name: '', provider: '', api_key: '', base_url: '', modelStr: '', service_type: 'text', priority: 0, negative_prompt: '' })
const presetForm = reactive({ apiKey: '' })
const serviceTypes = [{ type: 'text', label: '文本' }, { type: 'image', label: '图片' }, { type: 'video', label: '视频' }, { type: 'audio', label: '音频' }]
const providers = ref<string[]>(['ali', 'chatfire', 'gemini', 'minimax', 'ollama', 'openai', 'openrouter', 'vidu', 'volcengine'])
const providerSelectOptions = computed(() => providers.value.map(p => ({ label: p, value: p })))
const serviceMeta = {
  text: { label: '文本', desc: '剧本改写、角色场景提取、分镜拆解等 Agent 文本能力' },
  image: { label: '图片', desc: '角色图、场景图、镜头图与首尾帧等静态图像生成' },
  video: { label: '视频', desc: '镜头视频生成，支持单图、多图和首尾帧模式' },
  audio: { label: '音频', desc: '角色试听、旁白与对白语音生成' },
}
const providerPresets = ref<Record<string, Record<string, { label: string; baseUrl: string; models: string[] }>>>({
  text: {
    chatfire: { label: '推荐', baseUrl: 'https://api.chatfire.site', models: ['gemini-3-pro-preview'] },
    openrouter: { label: 'OpenRouter 推荐', baseUrl: 'https://openrouter.ai/api', models: ['google/gemini-3-flash-preview'] },
    openai: { label: 'OpenAI 推荐', baseUrl: 'https://api.openai.com', models: ['gpt-4.1-mini'] },
  },
  image: {
    chatfire: { label: '推荐', baseUrl: 'https://api.chatfire.site', models: ['doubao-seedream-4-5-251128'] },
    gemini: { label: 'Gemini 推荐', baseUrl: 'https://api.chatfire.site', models: ['gemini-3-pro-image-preview'] },
    volcengine: { label: '火山推荐', baseUrl: 'https://ark.cn-beijing.volces.com', models: ['doubao-seedream-4-0-250828'] },
  },
  video: {
    volcengine: { label: '火山引擎', baseUrl: 'https://api.chatfire.site/volcengine', models: ['doubao-seedance-1-5-pro-251215'] },
    vidu: { label: 'Vidu 推荐', baseUrl: 'https://api.vidu.com', models: ['viduq3-turbo'] },
    ali: { label: '阿里推荐', baseUrl: 'https://dashscope.aliyuncs.com', models: ['wan2.6-i2v-flash'] },
  },
  audio: {
    minimax: { label: 'MiniMax', baseUrl: 'https://api.chatfire.site/minimax', models: ['speech-2.8-hd'] },
  },
})
const presetCards = ref([
  { serviceType: 'text', label: '文本', provider: 'chatfire', baseUrl: 'https://api.chatfire.site', model: 'gemini-3-pro-preview', priority: 100 },
  { serviceType: 'image', label: '图片', provider: 'gemini', baseUrl: 'https://api.chatfire.site', model: 'gemini-3-pro-image-preview', priority: 99 },
  { serviceType: 'video', label: '视频', provider: 'volcengine', baseUrl: 'https://api.chatfire.site/volcengine', model: 'doubao-seedance-1-5-pro-251215', priority: 98 },
  { serviceType: 'audio', label: '音频', provider: 'minimax', baseUrl: 'https://api.chatfire.site/minimax', model: 'speech-2.8-hd', priority: 97 },
])
const endpointPrefixes = ref<Record<string, string>>({
  chatfire: '/v1',
  openai: '/v1',
  openrouter: '/v1',
  minimax: '/v1',
  gemini: '/v1beta',
  volcengine: '/api/v3',
  ali: '/api/v1',
  vidu: '/ent/v2',
})

const endpointHint = computed(() => {
  const provider = cfgForm.provider
  const base = cfgForm.base_url || 'https://...'
  const prefix = endpointPrefixes.value[provider] || ''
  if (!provider) return '选择服务商后显示推荐端点前缀'
  return `${base}${prefix}`
})

function byType(t) { return cfgs.value.filter(c => c.service_type === t) }
function onlineByType(t) { return cfgs.value.filter(c => c.service_type === t && !c.is_local) }
function localByType(t) { return cfgs.value.filter(c => c.service_type === t && c.is_local) }
function countActive(t) { return byType(t).filter(c => c.is_active).length }
function fmtModel(m) { return Array.isArray(m) ? m.join(', ') : m || '—' }
function presetsByType(type) {
  const group = providerPresets.value[type] || {}
  return Object.entries(group).map(([provider, preset]) => ({ provider, ...preset }))
}
function applyProviderPreset(type, provider) {
  const preset = providerPresets.value[type]?.[provider]
  if (!preset) return
  cfgForm.provider = provider
  cfgForm.base_url = preset.baseUrl
  cfgForm.modelStr = preset.models.join(', ')
  cfgForm.name = `${preset.label}-${serviceMeta[type].label}`
}

// 从后端 /ai-providers 拉取服务商目录，覆盖内置硬编码（后端不可用时静默降级到硬编码）
async function loadProviders() {
  try {
    const rows: any[] = await aiProvidersAPI.list()
    if (!Array.isArray(rows) || !rows.length) return

    const provList: string[] = []
    const seenProvider = new Set<string>()
    const prefixMap: Record<string, string> = {}
    const presets: Record<string, Record<string, { label: string; baseUrl: string; models: string[] }>> = {}
    const cards: any[] = []
    const priorityByType: Record<string, number> = { text: 100, image: 99, video: 98, audio: 97 }

    for (const r of rows) {
      const provider = r.provider
      const st = r.service_type
      if (!provider || !st) continue

      if (!seenProvider.has(provider)) { seenProvider.add(provider); provList.push(provider) }
      if (r.endpoint_prefix) prefixMap[provider] = r.endpoint_prefix

      if (!presets[st]) presets[st] = {}
      presets[st][provider] = {
        label: r.display_name || provider,
        baseUrl: r.default_url || '',
        models: Array.isArray(r.preset_models) ? r.preset_models : [],
      }

      if (r.is_recommended) {
        const models = Array.isArray(r.preset_models) ? r.preset_models : []
        cards.push({
          serviceType: st,
          label: serviceMeta[st]?.label || st,
          provider,
          baseUrl: r.default_url || '',
          model: models[0] || '',
          priority: priorityByType[st] ?? 0,
        })
      }
    }

    if (provList.length) providers.value = provList
    if (Object.keys(prefixMap).length) endpointPrefixes.value = prefixMap
    if (Object.keys(presets).length) providerPresets.value = presets
    if (cards.length) presetCards.value = cards
  } catch (e) {
    // 后端服务商目录不可用时，保留内置硬编码
  }
}

async function loadCfgs() { try { cfgs.value = await aiConfigAPI.list() } catch (e) { toast.error(e.message) } }
async function toggleCfg(c) { try { await aiConfigAPI.update(c.id, { is_active: !c.is_active }); loadCfgs() } catch (e: any) { toast.error(e?.message || '切换失败') } }
async function delCfg(id) { try { await aiConfigAPI.del(id); toast.success('已删除'); loadCfgs() } catch (e: any) { toast.error(e?.message || '删除失败') } }
function startAddCfg(t) {
  cfgEditId.value = null
  cfgTestResult.value = null
  cfgModelsResult.value = null
  Object.assign(cfgForm, { name: '', provider: '', api_key: '', base_url: '', modelStr: '', service_type: t, priority: 0, negative_prompt: '' })
  const firstPreset = presetsByType(t)[0]
  if (firstPreset) applyProviderPreset(t, firstPreset.provider)
  cfgDialog.value = true
}
function startEditCfg(c) {
  cfgEditId.value = c.id
  cfgTestResult.value = null
  cfgModelsResult.value = null
  Object.assign(cfgForm, {
    name: c.name || '',
    provider: c.provider,
    api_key: c.api_key || '',
    base_url: c.base_url || '',
    modelStr: fmtModel(c.model),
    service_type: c.service_type,
    priority: c.priority ?? 0,
    negative_prompt: c.negative_prompt || '',
  })
  cfgDialog.value = true
}
async function testCfgPayload(payload) {
  cfgTesting.value = true
  try {
    cfgTestResult.value = await aiConfigAPI.test(payload)
    if (cfgTestResult.value.reachable) toast.success('端点已响应')
    else toast.warning('端点未通过测试')
  } catch (e) {
    toast.error(e.message)
  } finally {
    cfgTesting.value = false
  }
}
async function testDraftCfg() {
  await testCfgPayload({
    service_type: cfgForm.service_type,
    provider: cfgForm.provider,
    api_key: cfgForm.api_key,
    base_url: cfgForm.base_url,
    model: cfgForm.modelStr.split(',').map(s => s.trim()).filter(Boolean),
  })
}
function isCurrentModel(m: string) {
  const cur = cfgForm.modelStr.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean)
  return cur.includes(String(m).toLowerCase())
}
async function listDraftModels() {
  if (!cfgForm.provider) { toast.warning('请先选择服务商'); return }
  cfgModelsLoading.value = true
  try {
    const models = cfgForm.modelStr.split(',').map((s) => s.trim()).filter(Boolean)
    cfgModelsResult.value = await aiConfigAPI.models({
      provider: cfgForm.provider,
      api_key: cfgForm.api_key,
      base_url: cfgForm.base_url,
      models,
    })
    const r = cfgModelsResult.value
    if (r?.listable && r?.reachable) toast.success(r.message)
    else if (r?.listable) toast.warning(r.message)
    else toast(r.message)
  } catch (e: any) {
    toast.error(e.message)
  } finally {
    cfgModelsLoading.value = false
  }
}
async function testExistingCfg(c) {
  startEditCfg(c)
  await testCfgPayload({
    service_type: c.service_type,
    provider: c.provider,
    api_key: c.api_key || '',
    base_url: c.base_url || '',
    model: Array.isArray(c.model) ? c.model : [],
  })
}
/** 保存前校验模型是否存在于平台列表；存在缺失时弹红字确认。返回 false 表示用户取消保存。 */
async function ensureModelsValidated(models: string[]): Promise<boolean> {
  let checks: Array<{ model: string; exists: boolean }> | null = null

  // 复用最近一次「列出模型」结果（仅当模型集合完全一致）
  const cached = cfgModelsResult.value?.model_checks
  if (Array.isArray(cached)) {
    const set = new Set((cached as any[]).map((c) => String(c.model).toLowerCase()))
    const same = cached.length === models.length && models.every((m) => set.has(m.toLowerCase()))
    if (same) checks = cached
  }

  if (!checks) {
    try {
      const res = await aiConfigAPI.models({
        provider: cfgForm.provider,
        api_key: cfgForm.api_key,
        base_url: cfgForm.base_url,
        models,
      })
      cfgModelsResult.value = res
      checks = res?.model_checks || null
    } catch {
      checks = null // 校验失败不阻断保存
    }
  }

  if (!checks || !checks.length) return true
  const missing = checks.filter((c) => !c.exists).map((c) => c.model)
  if (!missing.length) return true

  return await confirm({
    title: '模型可能不存在',
    message: `以下模型未出现在平台模型列表中，可能无法调用：\n${missing.join('、')}\n\n是否仍要保存？`,
    confirmText: '仍要保存',
    cancelText: '返回修改',
    danger: true,
  })
}

async function saveCfg() {
  if (!cfgForm.provider) { toast.warning('选择服务商'); return }
  const models = cfgForm.modelStr.split(',').map(s => s.trim()).filter(Boolean)

  // 保存前校验：平台支持列举时检测模型是否存在，缺失则红字预警确认
  if (models.length && !(await ensureModelsValidated(models))) return

  try {
    if (cfgEditId.value) await aiConfigAPI.update(cfgEditId.value, { name: cfgForm.name, provider: cfgForm.provider, api_key: cfgForm.api_key, base_url: cfgForm.base_url, model: models, priority: cfgForm.priority, negative_prompt: cfgForm.negative_prompt })
    else await aiConfigAPI.create({ service_type: cfgForm.service_type, provider: cfgForm.provider, name: cfgForm.name || `${cfgForm.provider}-${cfgForm.service_type}`, api_key: cfgForm.api_key, base_url: cfgForm.base_url, model: models, priority: cfgForm.priority, negative_prompt: cfgForm.negative_prompt })
    cfgDialog.value = false; toast.success('已保存'); loadCfgs()
  } catch (e) { toast.error(e.message) }
}
async function applyQuickPreset() {
  if (!presetForm.apiKey) {
    toast.warning('请填写 API Key')
    return
  }
  try {
    await aiConfigAPI.quickPreset(presetForm.apiKey)
    await loadCfgs()
    presetDialog.value = false
    toast.success('推荐配置已写入')
  } catch (e) {
    toast.error(e.message)
  }
}

// ===== 本地模型 & GPU 监控 =====
const localLoading = ref(false)
const localConfigs = ref<any[]>([])
const gpu = ref<any>(null)
const gpuLoading = ref(false)

const vramPercent = computed(() => {
  if (!gpu.value?.hardware) return 0
  const t = gpu.value?.totalVRAM_GB
  if (!t) return 0
  return Math.min(100, Math.round((gpu.value.usedVRAM_GB / t) * 100))
})

function fmtMB(mb: number) {
  if (mb == null) return '—'
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`
}

async function initLocalModels() {
  localLoading.value = true
  try {
    await aiConfigAPI.quickLocal()
    await loadCfgs()
    await loadLocalConfigs()
    toast.success('本地模型已初始化')
  } catch (e: any) { toast.error(e.message) } finally { localLoading.value = false }
}

async function loadLocalConfigs() {
  try { localConfigs.value = await aiConfigAPI.configsLocal() } catch (e: any) { /* 非关键，静默 */ }
}

async function loadGpuStatus() {
  gpuLoading.value = true
  try { gpu.value = await aiConfigAPI.gpuStatus() } catch (e: any) { toast.error(e.message) } finally { gpuLoading.value = false }
}

// ===== Ollama 模型管理 =====
const ollamaStatus = ref<any>(null)
const ollamaModels = computed(() => ollamaStatus.value?.models || [])
const ollamaBusy = ref(false)
const ollamaPulling = ref(false)
const ollamaPullName = ref('')
const pullProgress = ref(0)
const pullStatusText = ref('')

async function refreshOllama() {
  ollamaBusy.value = true
  try { ollamaStatus.value = await aiConfigAPI.ollamaStatus() } catch (e: any) { toast.error(e.message) } finally { ollamaBusy.value = false }
}

async function startOllama() {
  ollamaBusy.value = true
  try {
    const res: any = await aiConfigAPI.ollamaStart()
    toast(res.message || (res.started ? 'Ollama 已启动' : 'Ollama 未启动'))
    await refreshOllama()
  } catch (e: any) { toast.error(e.message) } finally { ollamaBusy.value = false }
}

async function pullOllamaModel() {
  const name = ollamaPullName.value.trim()
  if (!name || ollamaPulling.value) return
  ollamaPulling.value = true
  pullProgress.value = 0
  pullStatusText.value = '连接 Ollama…'
  try {
    const res = await aiConfigAPI.ollamaPull(name, undefined, (ev: any) => {
      if (ev.status) pullStatusText.value = ev.status
      const total = Number(ev.total || 0)
      const completed = Number(ev.completed || 0)
      if (total > 0) pullProgress.value = Math.min(100, Math.round((completed / total) * 100))
    })
    if (res?.ok === false) { toast.error(res.error || '下载失败'); return }
    toast.success(`模型 ${name} 下载完成`)
    await refreshOllama()
  } catch (e: any) { toast.error(e.message) } finally { ollamaPulling.value = false }
}

async function useOllamaModel(model: string) {
  ollamaBusy.value = true
  try {
    const all: any[] = await aiConfigAPI.list('text')
    const local = all?.find((c: any) => c.provider === 'ollama' || String(c.base_url || '').includes('11434'))
    if (local) {
      await aiConfigAPI.update(local.id, { ...local, model: [model] })
      toast.success(`文本配置已切换为 ${model}`)
    } else {
      await aiConfigAPI.create({
        service_type: 'text',
        provider: 'ollama',
        name: '文本(本地)',
        base_url: 'http://localhost:11434',
        model: [model],
        priority: 85,
      })
      toast.success(`已创建本地文本配置（${model}）`)
    }
    await loadCfgs()
    await loadLocalConfigs()
  } catch (e: any) { toast.error(e.message) } finally { ollamaBusy.value = false }
}

async function releaseAllGpu() {
  if (!(await confirm({ message: '确认释放全部本地模型显存？进行中的生成任务可能被中断。', danger: true }))) return
  gpuLoading.value = true
  try {
    await aiConfigAPI.gpuReleaseAll()
    toast.success('已释放全部显存')
    await loadGpuStatus()
  } catch (e: any) { toast.error(e.message) } finally { gpuLoading.value = false }
}

const tokenStats = ref<any>(null)
async function loadTokenStats() {
  try {
    tokenStats.value = await traceAPI.stats()
  } catch {
    tokenStats.value = null
  }
}

function fmtTokens(n: number): string {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k'
  return String(n)
}

onMounted(() => { loadCfgs(); loadLocalConfigs(); loadGpuStatus(); refreshOllama(); loadProviders(); loadTokenStats() })
</script>

<style scoped>
.models-page { height: 100%; overflow: hidden; }
.settings-scroll { height: 100%; overflow-y: auto; padding: 36px 48px; max-width: 840px; margin: 0 auto; animation: fadeUp 0.3s var(--ease-out); }
.settings-head { margin-bottom: 24px; }
.settings-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.settings-brand-mark {
  width: 42px;
  height: 42px;
  border-radius: 15px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,247,255,0.9));
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}
.settings-brand-logo {
  width: 26px;
  height: 26px;
  object-fit: contain;
  display: block;
}
.settings-brand-fallback {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent-text);
  line-height: 1;
}
.settings-brand-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1;
}
.settings-brand-kicker {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-3);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.settings-brand-name {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-1);
  font-family: var(--font-display);
}
.settings-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.settings-desc { font-size: 13px; color: var(--text-2); margin-top: 4px; }

/* AI Config */
.setup-panel {
  padding: 18px 18px 16px;
  margin-bottom: 18px;
}
.setup-panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.setup-panel-head.compact { margin-bottom: 12px; }
.setup-kicker {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 4px;
}
.setup-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-0);
}
.setup-desc {
  font-size: 12px;
  color: var(--text-2);
  margin-top: 4px;
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.preset-grid.compact {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 8px;
}
.preset-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.82);
  padding: 12px 13px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.preset-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.preset-service { font-size: 12px; font-weight: 600; }
.preset-model { font-size: 12px; color: var(--text-1); }
.preset-base { font-size: 11px; color: var(--text-3); }
.template-row { display: flex; flex-wrap: wrap; gap: 8px; }
.template-type-chip {
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.82);
  color: var(--text-1);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  cursor: pointer;
  transition: 0.15s;
}
.template-type-chip:hover {
  border-color: var(--accent);
  color: var(--accent-text);
  background: var(--accent-bg);
}
.sections { display: flex; flex-direction: column; gap: 24px; }
.section-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.section-title { font-size: 13px; font-weight: 600; }
.section-subtitle { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.config-list { display: flex; flex-direction: column; gap: 6px; }
.config-group-label { font-size: 11px; font-weight: 600; color: var(--text-3); margin: 8px 0 2px; letter-spacing: 0.4px; }
.tag-local { background: var(--warning-bg); color: var(--warning); }
.config-row { display: flex; align-items: center; gap: 8px; padding: 10px 14px; }
.config-info { flex: 1; display: flex; align-items: center; gap: 10px; min-width: 0; }
.config-main { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.config-line { display: flex; align-items: center; gap: 8px; min-width: 0; }
.config-provider { font-size: 13px; font-weight: 600; }
.config-name { font-size: 12px; color: var(--text-2); }
.config-model { font-size: 11px; color: var(--text-2); }
.config-base { font-size: 11px; color: var(--text-3); }
.config-empty { font-size: 12px; color: var(--text-3); padding: 12px 0; }

/* Token 用量统计 */
.token-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }
.token-stat-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.token-stat-num { font-size: 20px; font-weight: 700; font-family: var(--font-display); color: var(--text-0); }
.token-stat-label { font-size: 12px; color: var(--text-3); margin-top: 4px; }
.token-table { width: 100%; margin-top: 14px; border-collapse: collapse; font-size: 12px; }
.token-table th, .token-table td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
.token-table th { color: var(--text-3); font-weight: 600; }
.token-table td { color: var(--text-1); }

/* 本地模型 & GPU 监控 */
.local-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; margin-top: 12px; }
.gpu-head-actions { display: flex; align-items: center; gap: 8px; }
.gpu-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }
.gpu-card { padding: 12px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-1); }
.gpu-label { font-size: 11px; color: var(--text-3); margin-bottom: 6px; }
.gpu-value { font-size: 16px; font-weight: 700; color: var(--text-0); }
.gpu-sub { font-size: 11px; color: var(--text-3); margin-top: 4px; }
.gpu-bar { height: 6px; background: var(--bg-hover); border-radius: 999px; margin-top: 8px; overflow: hidden; }
.gpu-bar-fill { height: 100%; background: var(--accent); border-radius: 999px; transition: width 0.3s; }
.gpu-notice { margin-top: 10px; padding: 9px 12px; border: 1px dashed var(--border); border-radius: var(--radius); background: var(--bg-1); font-size: 12px; color: var(--text-2); line-height: 1.6; }

/* Ollama 模型管理 */
.ollama-panel { margin-top: 16px; padding: 14px 14px 16px; border: 1px dashed var(--border); border-radius: var(--radius); background: var(--bg-1); }
.ollama-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.ollama-title { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--text-0); }
.ollama-state { font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 99px; background: var(--bg-3); color: var(--text-3); }
.ollama-state.on { background: var(--accent-bg); color: var(--accent); }
.ollama-head-actions { display: flex; align-items: center; gap: 8px; }
.ollama-msg { margin-top: 10px; font-size: 12px; color: var(--text-2); }
.ollama-models { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto; }
.ollama-model { display: flex; align-items: center; gap: 10px; padding: 7px 10px; background: var(--bg-0); border: 1px solid var(--border); border-radius: var(--radius-sm); }
.ollama-model-name { flex: 1; min-width: 0; font-size: 12px; color: var(--text-1); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ollama-model-size { font-size: 11px; color: var(--text-3); flex-shrink: 0; }
.ollama-empty { margin-top: 10px; font-size: 12px; color: var(--text-3); }
.ollama-pull { display: flex; gap: 8px; margin-top: 12px; }
.ollama-pull .input { flex: 1; min-width: 0; }
.ollama-progress { margin-top: 10px; }
.ollama-progress-label { margin-top: 6px; font-size: 11px; color: var(--text-3); }

.toggle { position: relative; width: 30px; height: 17px; cursor: pointer; flex-shrink: 0; }
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle span { position: absolute; inset: 0; background: var(--bg-3); border-radius: 99px; transition: 0.2s; }
.toggle span::before { content: ''; position: absolute; width: 13px; height: 13px; left: 2px; bottom: 2px; background: var(--bg-0); border-radius: 50%; transition: 0.2s; box-shadow: var(--shadow); }
.toggle input:checked + span { background: var(--accent); }
.toggle input:checked + span::before { transform: translateX(13px); }

/* Shared */
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--text-1); }
.field-hint { font-size: 11px; color: var(--text-3); margin-top: 2px; }

.overlay { position: fixed; inset: 0; background: rgba(34,45,66,0.32); backdrop-filter: blur(8px); display: flex; align-items: center; justify-content: center; z-index: 100; animation: fadeIn 0.18s var(--ease-out); }
.modal { padding: 28px; width: 420px; display: flex; flex-direction: column; gap: 12px; box-shadow: var(--shadow-elevated); }
.modal-title { font-family: var(--font-display); font-size: 18px; font-weight: 700; }
.modal-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 6px; }
.config-modal { width: min(720px, calc(100vw - 40px)); max-height: calc(100vh - 48px); overflow-y: auto; }
.config-modal-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.modal-note {
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-2);
}
.preset-picker {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.preset-pill {
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.72);
  color: var(--text-1);
  border-radius: 999px;
  padding: 8px 11px;
  font-size: 12px;
  cursor: pointer;
}
.preset-pill:hover {
  border-color: var(--accent);
  background: var(--accent-bg);
  color: var(--accent-text);
}
.endpoint-hint {
  margin-top: -4px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px dashed var(--border);
  background: rgba(244,248,255,0.72);
  font-size: 12px;
}
.test-result {
  display: flex;
  flex-direction: column;
  gap: 8px;
  border-radius: 14px;
  padding: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.72);
}
.test-result.ok { border-color: rgba(74, 167, 92, 0.28); }
.test-result.bad { border-color: rgba(201, 88, 68, 0.28); }
.test-result-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-1);
}
.test-result-url,
.test-result-preview {
  font-size: 11px;
  color: var(--text-3);
  word-break: break-all;
}
.models-result {
  display: flex;
  flex-direction: column;
  gap: 8px;
  border-radius: 14px;
  padding: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.72);
  margin-top: 12px;
}
.models-result-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-1);
}
.models-exists {
  font-size: 12px;
  font-weight: 600;
}
.models-exists-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.models-exists.ok { color: #3f9d5c; }
.models-exists.bad { color: #c95844; }
.models-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 180px;
  overflow-y: auto;
}
.model-chip {
  padding: 3px 9px;
  font-size: 11px;
  border-radius: 99px;
  border: 1px solid var(--border);
  background: var(--bg-1);
  color: var(--text-2);
  cursor: pointer;
  transition: all 0.15s ease;
}
.model-chip:hover { border-color: var(--accent); color: var(--accent); }
.model-chip.active { border-color: var(--accent); background: var(--accent-bg); color: var(--accent); }
.preset-grid-form {
  display: grid;
  grid-template-columns: repeat(1, minmax(0, 1fr));
  gap: 10px;
}

@media (max-width: 900px) {
  .preset-grid,
  .preset-grid.compact {
    grid-template-columns: 1fr;
  }
}
</style>

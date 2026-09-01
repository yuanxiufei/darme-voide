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

        <!-- 本地服务状态 -->
        <div class="runtime-panel">
          <div class="ollama-head">
            <div class="ollama-title">
              <Server :size="14" />
              <span>本地服务状态</span>
              <span class="ollama-state" :class="{ on: runtimeHealth?.all_running }">
                {{ runtimeHealth ? `${runtimeHealth.running_count}/${runtimeHealth.total} 运行中` : '检测中…' }}
              </span>
            </div>
            <div class="ollama-head-actions">
              <button class="btn btn-ghost btn-icon" :disabled="runtimeBusy" title="刷新检测" @click="refreshRuntimeHealth">
                <Loader2 v-if="runtimeBusy" :size="13" class="animate-spin" />
                <RefreshCw v-else :size="13" />
              </button>
            </div>
          </div>
          <div class="runtime-grid">
            <div v-for="s in runtimeHealth?.services ?? []" :key="s.key" class="runtime-item">
              <span class="runtime-dot" :class="{ on: s.running }" />
              <div class="runtime-meta">
                <span class="runtime-label">{{ s.label }}</span>
                <span class="runtime-sub mono">
                  {{ s.base_url }}
                  <template v-if="s.running"> · {{ s.latency_ms }}ms</template>
                  <template v-else-if="s.error"> · {{ s.error }}</template>
                </span>
              </div>
              <span v-if="s.registered_count" class="tag tag-accent">{{ s.registered_count }} 配置</span>
            </div>
          </div>
        </div>

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
              <button class="btn btn-ghost btn-icon" :disabled="ollamaBusy" title="删除该模型" @click="deleteOllamaModel(m.name)">
                <Trash2 :size="12" />
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

        <!-- 本地模型扫描与注册 -->
        <div class="scan-panel">
          <div class="ollama-head">
            <div class="ollama-title">
              <ScanSearch :size="15" />
              本地模型扫描与注册
            </div>
            <div class="ollama-head-actions">
              <select v-model="selectedDrive" class="input scan-drive-select" :disabled="scanLoading" title="选择要扫描的盘符">
                <option value="">全部目录</option>
                <option v-for="d in drives" :key="d.root" :value="d.root">{{ d.letter }} 盘 ({{ d.root }})</option>
              </select>
              <label class="scan-auto-toggle" title="扫描完成后自动注册所有可注册模型">
                <input type="checkbox" v-model="autoRegister" />
                <span>自动注册</span>
              </label>
              <button v-if="scanLoading" class="btn btn-ghost btn-sm" @click="cancelScan">
                <X :size="14" /> 取消
              </button>
              <button class="btn btn-ghost btn-sm" :disabled="scanLoading" @click="scanLocalModels">
                <Loader2 v-if="scanLoading" :size="14" class="animate-spin" />
                <ScanSearch v-else :size="14" />
                {{ scanLoading ? '扫描中…' : '扫描本地模型' }}
              </button>
            </div>
          </div>

          <div v-if="scanProgress && !scanProgress.done" class="scan-progress">
            <div class="gpu-bar"><div class="gpu-bar-fill" :style="{ width: scanProgressPercent + '%' }" /></div>
            <div class="scan-progress-label">
              已扫描 <b>{{ scanProgress.scannedFiles }}</b> 个文件，发现 <b>{{ scanProgress.foundModels }}</b> 个模型
              <span v-if="scanProgress.currentDir" class="scan-progress-dir">{{ scanProgress.currentDir }}</span>
            </div>
          </div>

          <p class="scan-desc">扫描电脑里的模型文件并识别类型，勾选即可一键注册。可设置「模型存储目录」作为下载/存放的主目录，散落在电脑任意位置的其他模型则添加为额外扫描目录一并调用。</p>

          <div class="scan-roots">
            <div class="scan-roots-label">
              <HardDrive :size="13" />
              <span>模型存储目录</span>
              <span class="scan-roots-hint">下载/存放模型的主目录，扫描时优先扫描</span>
            </div>
            <div class="scan-roots-input-row">
              <input
                v-model="modelDirInput"
                class="input"
                type="text"
                placeholder="如 D:\models 或 D:\code\ComfyUI\ComfyUI\models"
                @keyup.enter="saveModelDir"
              />
              <button class="btn btn-ghost btn-sm" :disabled="modelDirInput.trim().replace(/[\\/]+$/, '') === modelDir" @click="saveModelDir">
                <Check :size="14" /> 保存
              </button>
            </div>

            <div class="hf-download">
              <div class="scan-roots-label">
                <Download :size="13" />
                <span>下载模型</span>
                <span class="scan-roots-hint">权重下载到上面的「模型存储目录」</span>
              </div>
              <div class="hf-source-row">
                <button
                  v-for="s in HF_SOURCES"
                  :key="s.value"
                  type="button"
                  class="hf-source-btn"
                  :class="{ active: hfSource === s.value }"
                  @click="hfSource = s.value"
                >
                  {{ s.label }}
                </button>
              </div>
              <div class="scan-roots-input-row">
                <input
                  v-model="hfRepo"
                  class="input"
                  type="text"
                  placeholder="如 Qwen/Qwen3-4B"
                  @keyup.enter="listHfFiles"
                />
                <button class="btn btn-ghost btn-sm" :disabled="!hfRepo.trim() || hfListing" @click="listHfFiles">
                  <Loader2 v-if="hfListing" :size="14" class="animate-spin" />
                  <ScanSearch v-else :size="14" /> 浏览文件
                </button>
                <button class="btn btn-primary btn-sm" :disabled="!hfRepo.trim() || hfDownloading || !hfSelectedFiles.length" @click="startHfDownload">
                  <Loader2 v-if="hfDownloading" :size="14" class="animate-spin" />
                  <Download v-else :size="14" /> 下载{{ hfSelectedFiles.length ? ` (${hfSelectedFiles.length})` : '' }}
                </button>
              </div>

              <div v-if="hfFiles.length" class="hf-file-list">
                <label class="hf-file-row">
                  <input type="checkbox" class="scan-check" :checked="hfSelectedFiles.length === hfFiles.length" @change="toggleAllHf" />
                  <span class="hf-file-name">全选</span>
                </label>
                <label v-for="f in hfFiles" :key="f.name" class="hf-file-row">
                  <input type="checkbox" class="scan-check" :checked="hfSelectedFiles.includes(f.name)" @change="toggleHfFile(f.name)" />
                  <span class="hf-file-name mono">{{ f.name }}</span>
                  <span class="hf-file-size">{{ formatSize(f.size) }}</span>
                </label>
              </div>

              <div v-if="hfProgress" class="hf-progress">
                <div class="hf-progress-track"><div class="hf-progress-fill" :style="{ width: hfPercent + '%' }" /></div>
                <div class="hf-progress-text">{{ hfProgressText }}</div>
              </div>
            </div>
          </div>

          <div class="scan-roots">
            <div class="scan-roots-label">
              <FolderOpen :size="13" />
              <span>额外扫描目录</span>
              <span class="scan-roots-hint">电脑里其他位置的模型也一并扫描</span>
            </div>
            <div class="scan-roots-input-row">
              <input
                v-model="customRootInput"
                class="input"
                type="text"
                placeholder="如 D:\models\LLM，回车或点添加"
                @keyup.enter="addCustomRoot"
              />
              <button class="btn btn-ghost btn-sm" :disabled="!customRootInput.trim()" @click="addCustomRoot">
                <Plus :size="14" /> 添加
              </button>
            </div>
            <div v-if="customRoots.length" class="scan-roots-chips">
              <span v-for="r in customRoots" :key="r" class="scan-root-chip">
                <FolderOpen :size="12" class="scan-root-chip-icon" />
                <span class="scan-root-chip-path">{{ r }}</span>
                <button class="scan-root-chip-x" title="移除" @click="removeCustomRoot(r)"><X :size="12" /></button>
              </span>
            </div>
          </div>

          <p v-if="scanError" class="scan-error">{{ scanError }}</p>

          <div v-if="scanResult" class="scan-summary">
            <span class="scan-summary-total">共识别 <b>{{ scanResult.total }}</b> 个模型文件</span>
            <span v-if="scanResult.truncated" class="tag tag-warning">结果已截断</span>
            <span v-for="(count, kind) in scanResult.byKind" :key="kind" class="tag" :class="`kind-tag-${kind}`">{{ KIND_META[kind]?.label || kind }} {{ count }}</span>
          </div>

          <div v-if="scanModels.length" class="scan-toolbar">
            <div class="scan-filters">
              <button
                v-for="k in ['all', 'text', 'image', 'video', 'audio', 'unknown']"
                :key="k"
                class="scan-filter"
                :class="{ active: scanFilter === k }"
                @click="scanFilter = k"
              >{{ k === 'all' ? '全部' : KIND_META[k]?.label || k }}</button>
            </div>
            <div class="scan-toolbar-actions">
              <button class="btn btn-ghost btn-sm" :disabled="!registerableModels.length" @click="toggleSelectAll">
                {{ allSelected ? '取消全选' : '全选可注册' }}
              </button>
              <button class="btn btn-primary btn-sm" :disabled="registerLoading || !selectedCount" @click="registerSelected">
                <Loader2 v-if="registerLoading" :size="14" class="animate-spin" />
                注册所选{{ selectedCount ? ` (${selectedCount})` : '' }}
              </button>
            </div>
          </div>

          <div v-if="scanModels.length" class="scan-list">
            <label
              v-for="m in scanModels"
              :key="m.path"
              class="scan-item"
              :class="{ disabled: !isRegisterable(m) }"
            >
              <input
                type="checkbox"
                class="scan-check"
                :checked="selectedPaths.has(m.path)"
                :disabled="!isRegisterable(m)"
                @change="toggleSelect(m.path)"
              />
              <div class="scan-item-main">
                <div class="scan-item-name mono">{{ m.filename }}</div>
                <div class="scan-item-meta">
                  <span class="tag" :class="`kind-tag-${m.kind}`">{{ KIND_META[m.kind]?.label || m.kind }}</span>
                  <span class="tag" :class="m.role === 'standalone' ? 'tag-accent' : ''">{{ ROLE_LABEL[m.role] || m.role }}</span>
                  <span class="scan-item-path">{{ m.dirname }}</span>
                </div>
                <div v-if="m.suggested?.note" class="scan-item-note"><Info :size="12" /> {{ m.suggested.note }}</div>
              </div>
              <div class="scan-item-right">
                <span class="scan-item-size">{{ m.sizeHuman }}</span>
                <span v-if="m.runtime" class="scan-item-runtime">{{ m.runtime }}</span>
                <button
                  v-if="isInsideModelDir(m)"
                  class="btn btn-ghost btn-icon scan-item-del"
                  :disabled="deletingPath === m.path"
                  title="删除本地文件"
                  @click.stop.prevent="deleteModelFile(m)"
                >
                  <Loader2 v-if="deletingPath === m.path" :size="14" class="animate-spin" />
                  <Trash2 v-else :size="14" />
                </button>
              </div>
            </label>
          </div>
          <p v-else-if="!scanLoading && !scanError" class="scan-empty">点击「扫描本地模型」开始识别本机可用的模型文件。</p>

          <div v-if="registerResult?.skipped?.length" class="scan-skipped">
            <div class="scan-skipped-title">已跳过 {{ registerResult.skipped.length }} 项：</div>
            <div v-for="s in registerResult.skipped" :key="s.filename" class="scan-skipped-item">
              <span class="mono">{{ s.filename }}</span> — {{ s.reason }}
            </div>
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
        <label v-if="cfgForm.service_type === 'video'" class="field">
          <span class="field-label">FL2VA Checkpoint（动作/空镜/转场，本地 H3）</span>
          <input v-model="cfgForm.checkpoint_fl2va" class="input" placeholder="minimax_h3_fl2va_pruned_int8_convrot" />
        </label>
        <label v-if="cfgForm.service_type === 'video'" class="field">
          <span class="field-label">Ref2VA Checkpoint（说话/对话，本地 H3）</span>
          <input v-model="cfgForm.checkpoint_ref2va" class="input" placeholder="minimax_h3_ref2va_pruned_int8_convrot" />
          <span class="field-hint">两者留空则退化为模型字段单 checkpoint 模式（云端 MiniMax 无需填写）。</span>
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
import { Plus, Pencil, Trash2, Loader2, Bot, Sparkles, Monitor, RefreshCw, Server, Play, Download, ScanSearch, X, HardDrive, FolderOpen, Check, Info } from 'lucide-vue-next'
import BaseSelect from '~/components/BaseSelect.vue'
import { toast } from 'vue-sonner'
import { aiConfigAPI, aiProvidersAPI, traceAPI, localModelsAPI } from '~/composables/useApi'
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
const cfgForm = reactive({ name: '', provider: '', api_key: '', base_url: '', modelStr: '', service_type: 'text', priority: 0, negative_prompt: '', checkpoint_fl2va: '', checkpoint_ref2va: '' })
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
  Object.assign(cfgForm, { name: '', provider: '', api_key: '', base_url: '', modelStr: '', service_type: t, priority: 0, negative_prompt: '', checkpoint_fl2va: '', checkpoint_ref2va: '' })
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
    checkpoint_fl2va: c.checkpoint_map?.fl2va || '',
    checkpoint_ref2va: c.checkpoint_map?.ref2va || '',
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

function buildCheckpointMap(): Record<string, string> | null | undefined {
  if (cfgForm.service_type !== 'video') return undefined
  const fl2va = (cfgForm.checkpoint_fl2va || '').trim()
  const ref2va = (cfgForm.checkpoint_ref2va || '').trim()
  const map: Record<string, string> = {}
  if (fl2va) map.fl2va = fl2va
  if (ref2va) map.ref2va = ref2va
  return Object.keys(map).length ? map : null
}

async function saveCfg() {
  if (!cfgForm.provider) { toast.warning('选择服务商'); return }
  const models = cfgForm.modelStr.split(',').map(s => s.trim()).filter(Boolean)

  // 保存前校验：平台支持列举时检测模型是否存在，缺失则红字预警确认
  if (models.length && !(await ensureModelsValidated(models))) return

  const checkpoint_map = buildCheckpointMap()
  try {
    if (cfgEditId.value) await aiConfigAPI.update(cfgEditId.value, { name: cfgForm.name, provider: cfgForm.provider, api_key: cfgForm.api_key, base_url: cfgForm.base_url, model: models, priority: cfgForm.priority, negative_prompt: cfgForm.negative_prompt, checkpoint_map })
    else await aiConfigAPI.create({ service_type: cfgForm.service_type, provider: cfgForm.provider, name: cfgForm.name || `${cfgForm.provider}-${cfgForm.service_type}`, api_key: cfgForm.api_key, base_url: cfgForm.base_url, model: models, priority: cfgForm.priority, negative_prompt: cfgForm.negative_prompt, checkpoint_map })
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

// ===== 本地服务运行时健康检查 =====
const runtimeHealth = ref<any>(null)
const runtimeBusy = ref(false)

async function refreshRuntimeHealth() {
  runtimeBusy.value = true
  try { runtimeHealth.value = await aiConfigAPI.runtimeHealth() } catch (e: any) { toast.error(e.message) } finally { runtimeBusy.value = false }
}

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

async function deleteOllamaModel(model: string) {
  if (!(await confirm({ message: `确认删除 Ollama 模型 ${model}？此操作不可撤销。`, danger: true }))) return
  ollamaBusy.value = true
  try {
    await aiConfigAPI.ollamaDelete(model)
    toast.success(`已删除模型 ${model}`)
    await refreshOllama()
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

// ===== 本地模型扫描与注册 =====
const scanLoading = ref(false)
const scanResult = ref<any>(null)
const scanError = ref('')
const scanProgress = ref<any>(null)
const scanFilter = ref('all')
const selectedPaths = ref<Set<string>>(new Set())
const autoRegister = ref(false)
const selectedDrive = ref('')
const drives = ref<any[]>([])
const modelDirInput = ref('')
const modelDir = ref('')
const customRootInput = ref('')
const customRoots = ref<string[]>([])
const scanTaskId = ref('')
const scanTimer = ref<any>(null)
const registerLoading = ref(false)
const registerResult = ref<any>(null)
const deletingPath = ref('')
const hfRepo = ref('')
const hfFiles = ref<any[]>([])
const hfSelectedFiles = ref<string[]>([])
const hfListing = ref(false)
const hfDownloading = ref(false)
const hfProgress = ref<any>(null)
const hfPercent = computed(() => {
  const p = hfProgress.value
  if (!p || !p.overallTotal) return 0
  return Math.min(100, Math.round((p.overall / p.overallTotal) * 100))
})
const hfProgressText = computed(() => {
  const p = hfProgress.value
  if (!p) return ''
  if (p.status === 'done') return `下载完成，共 ${formatSize(p.total)}`
  if (p.status === 'error') return `下载失败：${p.error || ''}`
  return `下载中 ${formatSize(p.overall)} / ${formatSize(p.overallTotal)}${p.file ? ` · ${p.file}` : ''}`
})

const HF_SOURCES = [
  { value: 'hf_mirror', label: 'hf-mirror（国内）' },
  { value: 'hf', label: 'Hugging Face（境外）' },
  { value: 'modelscope', label: 'ModelScope（国内）' },
]
const hfSource = ref(localStorage.getItem('darme.hfSource') || 'hf_mirror')
watch(hfSource, (v) => localStorage.setItem('darme.hfSource', v))

const KIND_META: Record<string, { label: string; groups: string[] }> = {
  text: { label: '文本', groups: ['llm', 'text'] },
  image: { label: '图像', groups: ['image', 'diffusion', 'checkpoint'] },
  video: { label: '视频', groups: ['video', 'h3', 'minimax'] },
  audio: { label: '音频', groups: ['audio', 'voice', 'tts'] },
  unknown: { label: '其他', groups: [] },
}
const ROLE_LABEL: Record<string, string> = {
  checkpoint: '基础模型',
  lora: 'LoRA 适配器',
  vae: 'VAE 编码器',
  embedder: '文本编码器',
  standalone: '独立权重',
  unknown: '未知',
}

const scanProgressPercent = computed(() => {
  if (!scanProgress.value) return 0
  const p = scanProgress.value
  if (!p.total) return 0
  return Math.min(100, Math.round((p.scannedFiles / p.total) * 100))
})

const scanModels = computed<any[]>(() => {
  const all: any[] = scanResult.value?.models || []
  if (scanFilter.value === 'all') return all
  return all.filter((m: any) => (KIND_META[m.kind] ? m.kind : 'unknown') === scanFilter.value)
})
const registerableModels = computed(() => scanModels.value.filter((m) => isRegisterable(m)))
const selectedCount = computed(() => selectedPaths.value.size)
const allSelected = computed(() => registerableModels.value.length > 0 && selectedPaths.value.size === registerableModels.value.length)

function kindOf(m: any): string {
  return KIND_META[m.kind] ? m.kind : 'unknown'
}
function isRegisterable(m: any): boolean {
  return m.role === 'standalone' && m.suggested?.callable !== false
}
function toggleSelect(path: string) {
  const next = new Set(selectedPaths.value)
  if (next.has(path)) next.delete(path)
  else next.add(path)
  selectedPaths.value = next
}
function toggleSelectAll() {
  if (allSelected.value) selectedPaths.value = new Set()
  else selectedPaths.value = new Set(registerableModels.value.map((m) => m.path))
}
async function loadDrives() {
  try { drives.value = (await localModelsAPI.drives())?.drives || [] } catch { /* ignore */ }
}
async function loadLocalRoots() {
  try {
    const res = await localModelsAPI.roots()
    const md = res?.paths?.models_dir
    if (md) { modelDir.value = md; modelDirInput.value = md }
    customRoots.value = (res?.extra || []).filter(Boolean)
  } catch { /* ignore */ }
}
async function saveModelDir() {
  const p = modelDirInput.value.trim().replace(/[\\/]+$/, '')
  if (!p) { toast.warning('路径不能为空'); return }
  try {
    await localModelsAPI.savePaths({ models_dir: p })
    modelDir.value = p
    toast.success('已保存模型存储目录')
    await loadLocalRoots()
  } catch (e: any) { toast.error(e.message) }
}
function addCustomRoot() {
  const p = customRootInput.value.trim().replace(/[\\/]+$/, '')
  if (!p) return
  if (customRoots.value.includes(p)) { toast.warning('该目录已在列表中'); return }
  customRoots.value.push(p)
  customRootInput.value = ''
  saveCustomRoots()
}
async function removeCustomRoot(r: string) {
  customRoots.value = customRoots.value.filter((x) => x !== r)
  await saveCustomRoots()
}
async function saveCustomRoots() {
  try { await localModelsAPI.savePaths({ roots: customRoots.value }) } catch (e: any) { toast.error(e.message) }
}
async function scanLocalModels() {
  scanLoading.value = true
  scanError.value = ''
  scanResult.value = null
  scanFilter.value = 'all'
  selectedPaths.value = new Set()
  registerResult.value = null
  if (scanTimer.value) clearInterval(scanTimer.value)
  const roots: string[] = []
  if (selectedDrive.value) roots.push(selectedDrive.value)
  if (modelDir.value) roots.push(modelDir.value)
  if (customRoots.value.length) roots.push(...customRoots.value)
  try {
    const res = await localModelsAPI.scanAsync(roots.length ? { roots } : {})
    scanTaskId.value = res?.taskId || ''
    if (!scanTaskId.value) throw new Error('后端未返回任务 ID')
    scanTimer.value = setInterval(pollScan, 1500)
  } catch (e: any) {
    scanError.value = e.message
    scanLoading.value = false
  }
}
async function pollScan() {
  if (!scanTaskId.value) return
  try {
    const s = await localModelsAPI.scanStatus(scanTaskId.value)
    const p = s?.progress
    if (p) {
      scanProgress.value = p
      if (p.done) {
        if (scanTimer.value) clearInterval(scanTimer.value)
        scanLoading.value = false
        if (p.cancelled) { toast('扫描已取消'); scanProgress.value = null; return }
        if (p.error) { scanError.value = p.error; scanProgress.value = null; return }
        scanResult.value = p.result
        scanProgress.value = null
        if (autoRegister.value && registerableModels.value.length) await registerSelected()
      }
    }
  } catch { /* 任务可能已过期，忽略 */ }
}
async function cancelScan() {
  if (scanTaskId.value) {
    try { await localModelsAPI.scanCancel(scanTaskId.value) } catch { /* ignore */ }
  }
  scanTaskId.value = ''
  if (scanTimer.value) clearInterval(scanTimer.value)
  scanLoading.value = false
  scanProgress.value = null
  toast('已取消扫描')
}
async function registerSelected() {
  const models = scanModels.value.filter((m) => selectedPaths.value.has(m.path))
  if (!models.length) return
  registerLoading.value = true
  try {
    const res = await localModelsAPI.register(models)
    registerResult.value = res
    const ok = (res?.created?.length || 0) + (res?.updated?.length || 0)
    if (ok) toast.success(`已注册 ${ok} 个模型`)
    if (res?.skipped?.length) toast.warning(`跳过 ${res.skipped.length} 个`)
    await loadLocalConfigs()
    selectedPaths.value = new Set()
    scanResult.value = null
  } catch (e: any) { toast.error(e.message) } finally { registerLoading.value = false }
}
function isInsideModelDir(m: any): boolean {
  return !!modelDir.value && m.path.toLowerCase().startsWith(modelDir.value.toLowerCase().replace(/[\\/]+$/, ''))
}
async function deleteModelFile(m: any) {
  if (!(await confirm({ message: `确认删除本地文件 ${m.path}？此操作不可撤销。`, danger: true }))) return
  deletingPath.value = m.path
  try {
    await localModelsAPI.delFiles([m.path])
    toast.success('已删除')
    if (scanResult.value?.models) scanResult.value = { ...scanResult.value, models: scanResult.value.models.filter((x: any) => x.path !== m.path) }
    selectedPaths.value.delete(m.path)
  } catch (e: any) { toast.error(e.message) } finally { deletingPath.value = '' }
}

// ===== Hugging Face / ModelScope 模型下载 =====
async function listHfFiles() {
  const repo = hfRepo.value.trim()
  if (!repo) return
  hfListing.value = true
  hfFiles.value = []
  hfSelectedFiles.value = []
  try {
    const res = await localModelsAPI.hfFiles(repo, undefined, hfSource.value)
    hfFiles.value = res?.files || []
    if (!hfFiles.value.length && res?.note) toast(res.note)
  } catch (e: any) { toast.error(e.message) } finally { hfListing.value = false }
}
function toggleHfFile(name: string) {
  if (hfSelectedFiles.value.includes(name)) hfSelectedFiles.value = hfSelectedFiles.value.filter((x) => x !== name)
  else hfSelectedFiles.value = [...hfSelectedFiles.value, name]
}
function toggleAllHf() {
  if (hfSelectedFiles.value.length === hfFiles.value.length) hfSelectedFiles.value = []
  else hfSelectedFiles.value = hfFiles.value.map((f) => f.name)
}
async function startHfDownload() {
  const repo = hfRepo.value.trim()
  const files = [...hfSelectedFiles.value]
  if (!repo || !files.length) return
  hfDownloading.value = true
  hfProgress.value = null
  try {
    await localModelsAPI.hfDownload(repo, files, '', hfSource.value, (ev) => {
      hfProgress.value = ev
      if (ev?.status === 'error') toast.error(ev.message || '下载失败')
    })
    toast.success('下载完成')
    hfProgress.value = null
    await loadLocalRoots()
  } catch (e: any) { toast.error(e.message) } finally { hfDownloading.value = false }
}
function formatSize(bytes?: number): string {
  if (!bytes || bytes <= 0) return '-'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = bytes
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}

onMounted(() => { loadCfgs(); loadLocalConfigs(); loadGpuStatus(); refreshOllama(); loadProviders(); loadTokenStats(); refreshRuntimeHealth(); loadDrives(); loadLocalRoots() })
onUnmounted(() => { if (scanTimer.value) clearInterval(scanTimer.value) })
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

/* ===== 本地服务状态 ===== */
.runtime-panel {
  margin-top: 16px;
  padding: 16px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.035);
}
.runtime-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.runtime-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.07);
  background: rgba(255, 255, 255, 0.03);
}
.runtime-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--danger);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--danger) 18%, transparent);
}
.runtime-dot.on {
  background: var(--success, #22c55e);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--success, #22c55e) 18%, transparent);
}
.runtime-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.runtime-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.runtime-sub {
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ===== 本地模型扫描与注册 ===== */
.scan-panel {
  margin-top: 16px;
  padding: 16px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.035);
}
.scan-progress { margin-top: 12px; }
.scan-progress-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}
.scan-progress-label b { color: var(--text); }
.scan-progress-dir {
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 11px;
  color: var(--muted);
  opacity: 0.7;
}
.scan-desc {
  font-size: 12px;
  line-height: 1.6;
  color: var(--muted);
  margin: 10px 0 0;
}
.scan-roots {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}
.scan-roots-label {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.scan-roots-hint {
  font-size: 11px;
  font-weight: 400;
  color: var(--muted);
  opacity: 0.75;
}
.scan-roots-input-row {
  display: flex;
  gap: 8px;
}
.scan-roots-input-row .input {
  flex: 1;
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 12px;
}
.scan-drive-select {
  width: 150px;
  font-size: 12px;
}
.scan-auto-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--muted);
  cursor: pointer;
  user-select: none;
}
.scan-auto-toggle input {
  accent-color: var(--accent);
}
.scan-roots-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.scan-root-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px 4px 7px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  background: rgba(255, 255, 255, 0.04);
  font-size: 12px;
  color: var(--text);
}
.scan-root-chip-icon { color: var(--muted); flex-shrink: 0; }
.scan-root-chip-path {
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 11px;
}
.scan-root-chip-x {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1px;
  border: none;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  border-radius: 4px;
}
.scan-root-chip-x:hover { color: var(--danger); background: rgba(255, 0, 0, 0.08); }

/* HF 下载 */
.hf-download {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  padding: 12px;
  border-radius: 10px;
  border: 1px dashed rgba(255, 255, 255, 0.1);
}
.hf-source-row { display: flex; gap: 6px; flex-wrap: wrap; }
.hf-source-btn {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: transparent;
  font-size: 11px;
  color: var(--muted);
  cursor: pointer;
  transition: all 0.15s;
}
.hf-source-btn:hover { color: var(--text); border-color: var(--border); }
.hf-source-btn.active {
  color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}
.hf-file-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 220px;
  overflow-y: auto;
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 8px;
  padding: 4px;
  margin-top: 4px;
}
.hf-file-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text);
}
.hf-file-row:hover { background: rgba(255, 255, 255, 0.04); }
.hf-file-name { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hf-file-size { font-size: 11px; color: var(--muted); flex-shrink: 0; }
.hf-progress { margin-top: 10px; }
.hf-progress-track {
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.07);
  overflow: hidden;
}
.hf-progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 60%, #fff));
  transition: width 0.3s;
}
.hf-progress-text { font-size: 11px; color: var(--muted); margin-top: 4px; }

.scan-error {
  margin-top: 12px;
  font-size: 12px;
  color: var(--danger);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--danger) 25%, transparent);
  padding: 8px 10px;
  border-radius: 8px;
}
.scan-summary {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 14px;
  font-size: 12px;
  color: var(--muted);
}
.scan-summary-total b { color: var(--text); }
.kind-tag-text { --tag-c: #38bdf8; }
.kind-tag-image { --tag-c: #a78bfa; }
.kind-tag-video { --tag-c: #f472b6; }
.kind-tag-audio { --tag-c: #34d399; }
.kind-tag-unknown { --tag-c: #94a3b8; }
.scan-summary .tag.kind-tag-text,
.scan-summary .tag.kind-tag-image,
.scan-summary .tag.kind-tag-video,
.scan-summary .tag.kind-tag-audio,
.scan-summary .tag.kind-tag-unknown,
.scan-item-meta .tag.kind-tag-text,
.scan-item-meta .tag.kind-tag-image,
.scan-item-meta .tag.kind-tag-video,
.scan-item-meta .tag.kind-tag-audio,
.scan-item-meta .tag.kind-tag-unknown {
  color: var(--tag-c);
  border-color: color-mix(in srgb, var(--tag-c) 35%, transparent);
  background: color-mix(in srgb, var(--tag-c) 10%, transparent);
}
.scan-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.scan-filters { display: flex; gap: 4px; flex-wrap: wrap; }
.scan-filter {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: transparent;
  font-size: 11px;
  color: var(--muted);
  cursor: pointer;
  transition: all 0.15s;
}
.scan-filter:hover { color: var(--text); }
.scan-filter.active {
  color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}
.scan-toolbar-actions { display: flex; gap: 8px; }
.scan-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
}
.scan-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.07);
  background: rgba(255, 255, 255, 0.03);
  cursor: pointer;
}
.scan-item:hover { border-color: rgba(255, 255, 255, 0.14); }
.scan-item.disabled { opacity: 0.45; cursor: default; }
.scan-check { accent-color: var(--accent); margin-top: 3px; flex-shrink: 0; }
.scan-item-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.scan-item-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.scan-item-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 11px;
}
.scan-item-path {
  color: var(--muted);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.scan-item-note {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: #fbbf24;
  opacity: 0.85;
}
.scan-item-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  flex-shrink: 0;
}
.scan-item-size { font-size: 11px; color: var(--muted); font-family: var(--font-mono, ui-monospace, monospace); }
.scan-item-runtime { font-size: 10px; color: var(--muted); opacity: 0.7; }
.scan-item-del { padding: 4px; }
.scan-empty {
  margin-top: 14px;
  text-align: center;
  font-size: 12px;
  color: var(--muted);
  padding: 20px 0;
}
.scan-skipped {
  margin-top: 14px;
  font-size: 12px;
  color: var(--muted);
  border: 1px dashed rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  padding: 10px 12px;
}
.scan-skipped-title { font-weight: 600; color: var(--text); margin-bottom: 4px; }
.scan-skipped-item { padding: 2px 0; }
</style>

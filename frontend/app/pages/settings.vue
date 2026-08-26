<template>
  <div class="settings-layout">
    <aside class="settings-nav">
      <div class="nav-group">
        <div class="nav-group-label">基础</div>
        <button v-for="t in baseTabs" :key="t.id" :class="['nav-item', { active: tab === t.id }]" @click="tab = t.id">
          <component :is="t.icon" :size="14" />
          {{ t.label }}
        </button>
      </div>
      <div class="nav-advanced">
        <label class="advanced-toggle">
          <span>Agent 高级配置</span>
          <input type="checkbox" v-model="showAdvanced" />
          <span class="advanced-slider"></span>
        </label>
        <p class="advanced-note">仅展开 Agent 配置与 Skills。工作台功能和分镜字段保持默认可见。</p>
      </div>
      <div v-if="showAdvanced" class="nav-group">
        <div class="nav-group-label">高级</div>
        <button v-for="t in advancedTabs" :key="t.id" :class="['nav-item', { active: tab === t.id }]" @click="tab = t.id">
          <component :is="t.icon" :size="14" />
          {{ t.label }}
        </button>
      </div>
    </aside>

    <div class="settings-content">

      <!-- ===== AI 服务配置 ===== -->
      <div v-if="tab === 'ai'" class="settings-scroll">
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

      <!-- ===== Agent 配置 ===== -->
      <div v-else-if="tab === 'agents'" class="settings-scroll">
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
          <h2 class="settings-title">Agent 配置</h2>
          <p class="settings-desc">高级区只保留 Agent 运行配置。这里可以调整模型、提示词和参数，保存后立即生效。</p>
        </div>
        <div class="agent-list">
          <div v-for="a in agentDefs" :key="a.type" class="card agent-card">
            <div class="agent-card-head" @click="toggleAgentEdit(a.type)">
              <div class="agent-type-badge">{{ a.icon }}</div>
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:14px">{{ a.label }}</div>
                <div class="dim" style="font-size:12px">{{ a.type }}</div>
              </div>
              <span v-if="getAgentCfg(a.type)" class="tag tag-success">已配置</span>
              <span v-else class="tag">默认</span>
              <ChevronDown :size="14" :style="{ transform: editingAgent === a.type ? 'rotate(180deg)' : '', transition: '0.2s' }" />
            </div>
            <div v-if="editingAgent === a.type" class="agent-card-body">
              <label class="field">
                <span class="field-label">模型 <span class="dim">(留空使用 AI 服务默认)</span></span>
                <BaseSelect v-model="agentForm.model" :options="textModelSelectOptions" placeholder="— 使用 AI 服务默认 —" searchable />
              </label>
              <div class="field-row">
                <label class="field">
                  <span class="field-label">Temperature</span>
                  <input v-model.number="agentForm.temperature" class="input" type="number" min="0" max="2" step="0.1" />
                </label>
                <label class="field">
                  <span class="field-label">Max Tokens</span>
                  <input v-model.number="agentForm.max_tokens" class="input" type="number" min="100" max="32000" />
                </label>
              </div>
              <label class="field">
                <span class="field-label">System Prompt</span>
                <textarea v-model="agentForm.system_prompt" class="textarea" rows="12" placeholder="Agent 系统提示词..." />
              </label>

              <!-- ===== Skill 绑定面板 ===== -->
              <div class="skill-bind-panel">
                <div class="skill-bind-head">
                  <span class="skill-bind-title">
                    <FileText :size="13" /> 绑定 Skills
                    <span class="dim" style="font-weight:400;font-size:11px">（仅已启用的会注入 Prompt）</span>
                  </span>
                  <button class="btn btn-ghost btn-sm" @click="resetAgentSkills(a.type)">恢复默认</button>
                </div>
                <p v-if="!agentSkillBindings.length" class="dim" style="font-size:11px;padding:8px 0">
                  暂无可用 Skill。请在「Skills」Tab 中创建。
                </p>
                <div v-else class="skill-bind-list">
                  <div
                    v-for="(binding, idx) in agentSkillBindings"
                    :key="binding.id"
                    :class="['skill-bind-item', { disabled: !binding.enabled, 'drag-over': skillDragOverIdx === idx }]"
                    :draggable="true"
                    @dragstart="onSkillDragStart(idx, $event)"
                    @dragover.prevent="onSkillDragOver(idx)"
                    @drop.prevent="onSkillDrop(idx)"
                    @dragend="onSkillDragEnd"
                  >
                    <span class="skill-bind-drag" title="拖拽调整优先级">&#x2261;</span>
                    <label class="skill-bind-toggle">
                      <input type="checkbox" v-model="binding.enabled" />
                      <span></span>
                    </label>
                    <div class="skill-bind-info">
                      <span class="skill-bind-name">{{ getSkillName(binding.id) || binding.id }}</span>
                      <span class="dim" style="font-size:10px">{{ binding.id }}</span>
                    </div>
                    <span class="skill-bind-priority">#{{ idx + 1 }}</span>
                  </div>
                </div>
                <p class="dim" style="font-size:10px;margin-top:6px">
                  拖拽可调整优先级，保存后按列表顺序决定 Skill 加载顺序，仅 enabled=true 的 Skill 会在运行时加载。
                </p>
              </div>

              <div class="agent-card-foot">
                <button class="btn btn-ghost btn-sm" @click="resetAgentPrompt(a.type)">恢复默认 Prompt</button>
                <span v-if="agentSaved === a.type" class="tag tag-success" style="margin-left:8px">
                  <Check :size="10" /> 已保存
                </span>
                <button class="btn btn-primary btn-sm ml-auto" :disabled="agentSaving" @click="saveAgentCfg(a.type)">
                  <Loader2 v-if="agentSaving" :size="12" class="animate-spin" />
                  保存
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ===== Skills 编辑 ===== -->
      <div v-else-if="tab === 'skills'" class="skills-layout">
        <!-- Agent 左侧列表 -->
        <aside class="skills-agent-list">
          <div class="skills-agent-title">Agent 列表</div>
          <button
            v-for="a in agentDefs"
            :key="a.type"
            :class="['skills-agent-item', { active: selectedAgent === a.type }]"
            @click="selectAgent(a.type)"
          >
            <span class="agent-type-badge">{{ a.icon }}</span>
            <span class="skills-agent-label">{{ a.label }}</span>
            <span v-if="agentSkillCount(a.type) > 0" class="skill-count-badge">{{ agentSkillCount(a.type) }}</span>
          </button>
        </aside>

        <!-- Skill 管理右侧主区域 -->
        <div class="settings-scroll skills-main">
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
            <div style="display:flex;align-items:center;gap:10px">
              <span class="agent-type-badge" style="width:32px;height:32px;font-size:16px">{{ selectedAgentIcon }}</span>
              <div>
                <h2 class="settings-title" style="margin:0">{{ selectedAgentLabel }}</h2>
                <div class="dim" style="font-size:12px">{{ selectedAgentType }} — Skills</div>
              </div>
            </div>
            <p class="settings-desc" style="margin-top:10px">Skills 仅作为 Agent 的高级提示词层使用，不影响工作台常规功能入口。</p>
            <button class="btn btn-primary btn-sm" @click="startAddSkill">
              <Plus :size="13" /> 新增 Skill
            </button>
          </div>

          <!-- 无 skill 提示 -->
          <div v-if="!currentSkills.length" class="step-empty" style="padding:48px 24px">
            <div class="empty-visual">
              <FileText :size="28" />
            </div>
            <div class="empty-title">暂无 Skill</div>
            <div class="empty-desc">点击右上角「新增 Skill」创建第一个提示词文件</div>
          </div>

          <!-- Skill 列表 -->
          <div class="skill-list" v-else>
            <div v-for="s in currentSkills" :key="s.id" class="card skill-card">
              <div class="skill-card-head" @click="toggleSkillEdit(s.id)">
                <FileText :size="14" style="color:var(--accent);flex-shrink:0" />
                <div style="flex:1;min-width:0">
                  <div style="font-weight:600;font-size:13px">{{ s.name }}</div>
                  <div class="dim" style="font-size:11px">{{ s.description }}</div>
                </div>
                <button class="btn btn-ghost btn-icon" style="margin-right:4px" @click.stop="deleteSkill(s.id)">
                  <Trash2 :size="13" />
                </button>
                <ChevronDown :size="14" :style="{ transform: editingSkill === s.id ? 'rotate(180deg)' : '', transition: '0.2s' }" />
              </div>
              <div v-if="editingSkill === s.id" class="skill-card-body">
                <textarea
                  v-model="skillContent"
                  class="textarea mono"
                  rows="20"
                  style="font-size:12px;line-height:1.6"
                  placeholder="编写 SKILL.md 内容..."
                />
                <div class="skill-card-foot">
                  <span class="dim" style="font-size:11px">skills/{{ selectedAgentType }}/{{ s.id }}/SKILL.md</span>
                  <span v-if="skillSaved === s.id" class="tag tag-success" style="margin-left:8px">
                    <Check :size="10" /> 已保存
                  </span>
                  <button class="btn btn-primary btn-sm ml-auto" :disabled="skillSaving" @click="saveSkill(s.id)">
                    <Loader2 v-if="skillSaving" :size="12" class="animate-spin" />
                    保存
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ===== 生成历史 ===== -->
      <div v-else-if="tab === 'history'" class="settings-scroll">
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
          <h2 class="settings-title">生成历史</h2>
          <p class="settings-desc">汇总全部图片与视频生成记录，含模型、状态、耗时与提示词。失败项可展开查看具体原因。</p>
        </div>

        <div class="history-toolbar">
          <div class="history-filters">
            <button v-for="f in historyFilters" :key="f.value" :class="['chip', { active: historyFilter === f.value }]" @click="historyFilter = f.value">
              {{ f.label }}
            </button>
          </div>
          <button class="btn btn-ghost btn-icon" :disabled="historyLoading" title="刷新" @click="loadGenerations">
            <Loader2 v-if="historyLoading" :size="13" class="animate-spin" />
            <RefreshCw v-else :size="13" />
          </button>
        </div>

        <div class="history-stats">
          <div class="history-stat">
            <div class="history-stat-num">{{ historyStats.total }}</div>
            <div class="history-stat-label">总记录</div>
          </div>
          <div class="history-stat ok">
            <div class="history-stat-num">{{ historyStats.success }}</div>
            <div class="history-stat-label">成功</div>
          </div>
          <div class="history-stat warn">
            <div class="history-stat-num">{{ historyStats.processing }}</div>
            <div class="history-stat-label">处理中</div>
          </div>
          <div class="history-stat err">
            <div class="history-stat-num">{{ historyStats.failed }}</div>
            <div class="history-stat-label">失败</div>
          </div>
        </div>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Generation Logs</div>
              <div class="setup-title">生成记录</div>
            </div>
          </div>
          <div v-if="historyLoading" class="config-empty">正在加载生成记录…</div>
          <div v-else-if="!filteredGenerations.length" class="config-empty">暂无生成记录，运行图片 / 视频生成后自动记录。</div>
          <ul v-else class="history-list">
            <li v-for="g in filteredGenerations" :key="`${g.type}-${g.id}`" class="history-item">
              <button class="history-item-head" @click="toggleHistory(g)">
                <span :class="['history-type', g.type]">
                  <ImageIcon v-if="g.type === 'image'" :size="14" />
                  <Film v-else :size="14" />
                </span>
                <span class="history-item-main">
                  <span class="history-item-title">{{ g.type === 'image' ? '图片生成' : '视频生成' }}</span>
                  <span class="history-item-meta">
                    <span class="mono">{{ g.model || '—' }}</span>
                    <span class="dot">·</span>
                    <span>{{ g.provider || '—' }}</span>
                  </span>
                </span>
                <span :class="['status-badge', statusClass(g.status)]">{{ statusLabel(g.status) }}</span>
                <span class="history-item-time">
                  <Clock :size="12" />
                  {{ fmtTime(g.createdAt) }}
                </span>
                <ChevronDown :size="14" :class="['history-chevron', { open: historyExpandedId === `${g.type}-${g.id}` }]" />
              </button>
              <div v-if="historyExpandedId === `${g.type}-${g.id}`" class="history-item-body">
                <div class="history-detail-row"><span class="history-detail-label">模型</span><span class="mono">{{ g.model || '—' }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">服务商</span><span>{{ g.provider || '—' }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">状态</span><span :class="['status-badge', statusClass(g.status)]">{{ statusLabel(g.status) }}</span></div>
                <div class="history-detail-row"><span class="history-detail-label">耗时</span><span>{{ fmtElapsed(g.elapsedMs) }}</span></div>
                <div v-if="g.duration != null" class="history-detail-row"><span class="history-detail-label">时长</span><span>{{ g.duration }}s</span></div>
                <div v-if="g.taskId" class="history-detail-row"><span class="history-detail-label">任务 ID</span><span class="mono">{{ g.taskId }}</span></div>
                <div v-if="g.prompt" class="history-detail-block">
                  <span class="history-detail-label">提示词</span>
                  <p class="history-prompt">{{ g.prompt }}</p>
                </div>
                <div v-if="g.errorMsg" class="history-detail-block error">
                  <span class="history-detail-label">失败原因</span>
                  <p class="history-prompt">{{ g.errorMsg }}</p>
                </div>
                <div v-if="g.url" class="history-detail-block">
                  <span class="history-detail-label">产物</span>
                  <a :href="g.url" target="_blank" rel="noopener" class="history-link">打开产物</a>
                </div>
              </div>
            </li>
          </ul>
        </section>
      </div>

      <!-- ===== 数据存储 ===== -->
      <div v-else-if="tab === 'storage'" class="settings-scroll">
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
          <h2 class="settings-title">数据存储</h2>
          <p class="settings-desc">统一管理数据库与图片、视频、音频等生成文件。切换目录时自动迁移旧数据（旧目录保留作安全备份）。</p>
        </div>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Current Storage</div>
              <div class="setup-title">当前存储位置</div>
            </div>
            <button class="btn btn-ghost btn-icon" :disabled="storageLoading" title="刷新" @click="loadStorageInfo">
              <Loader2 v-if="storageLoading" :size="13" class="animate-spin" />
              <RefreshCw v-else :size="13" />
            </button>
          </div>
          <div v-if="storageInfo" class="storage-grid">
            <div class="storage-row">
              <span class="storage-icon"><Database :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">数据根目录</div>
                <div class="mono storage-path">{{ storageInfo.dataRoot }}</div>
              </div>
            </div>
            <div class="storage-row">
              <span class="storage-icon"><FileText :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">数据库文件</div>
                <div class="mono storage-path">{{ storageInfo.dbPath }}</div>
                <div class="dim storage-sub">{{ storageInfo.dbExists ? `已创建 · ${fmtBytes(storageInfo.dbSizeBytes)}` : '尚未创建（首次写入时自动创建）' }}</div>
              </div>
            </div>
            <div class="storage-row">
              <span class="storage-icon"><FolderOpen :size="16" /></span>
              <div class="storage-meta">
                <div class="storage-label">生成文件目录（图片 / 视频 / 音频）</div>
                <div class="mono storage-path">{{ storageInfo.storagePath }}</div>
                <div class="dim storage-sub">{{ storageInfo.storageExists ? `已占用 ${fmtBytes(storageInfo.storageSizeBytes)}` : '尚未创建（首次生成时自动创建）' }}</div>
              </div>
            </div>
          </div>
          <p v-else class="config-empty">正在加载存储信息…</p>
        </section>

        <section class="setup-panel card">
          <div class="setup-panel-head">
            <div>
              <div class="setup-kicker">Change Directory</div>
              <div class="setup-title">切换到其他目录</div>
              <div class="setup-desc">填写电脑上的目标目录绝对路径，例如 <span class="mono">D:\drama-data</span>。</div>
            </div>
          </div>
          <div class="storage-form">
            <label class="field">
              <span class="field-label">目标目录路径</span>
              <input v-model="newDataRoot" class="input" placeholder="如 D:\drama-data 或 /home/user/drama-data" />
            </label>
            <label class="storage-check">
              <input type="checkbox" v-model="migrateData" />
              自动迁移旧数据（复制数据库与生成文件，旧目录保留）
            </label>
            <div class="storage-actions">
              <button class="btn btn-primary" :disabled="storageChanging || !newDataRoot.trim()" @click="changeDataRoot">
                <Loader2 v-if="storageChanging" :size="14" class="animate-spin" />
                <HardDrive v-else :size="14" />
                切换并迁移
              </button>
            </div>
            <p class="dim storage-note">切换后当前运行中的实例会立即指向新目录，重启后依然生效。迁移采用复制方式，旧目录不会删除，可随时切回。</p>
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

    <!-- Add Skill Dialog -->
    <div v-if="addSkillDialog" class="overlay" @click.self="addSkillDialog = false">
      <form class="modal card" @submit.prevent="confirmAddSkill">
        <h2 class="modal-title">新增 Skill — {{ selectedAgentLabel }}</h2>
        <label class="field">
          <span class="field-label">Skill 目录名 <span class="dim">(英文，唯一)</span></span>
          <input v-model="newSkillForm.id" class="input" placeholder="如 custom-extraction" />
        </label>
        <label class="field">
          <span class="field-label">名称</span>
          <input v-model="newSkillForm.name" class="input" placeholder="如 自定义提取规则" />
        </label>
        <label class="field">
          <span class="field-label">描述</span>
          <input v-model="newSkillForm.description" class="input" placeholder="简短描述此 Skill 的用途" />
        </label>
        <div class="modal-actions">
          <button type="button" class="btn" @click="addSkillDialog = false">取消</button>
          <button type="submit" class="btn btn-primary" :disabled="!newSkillForm.id">创建</button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Plus, Pencil, Trash2, FileText, ChevronDown, Check, Loader2, Bot, Cpu, Sparkles, Monitor, RefreshCw, Server, HardDrive, Database, FolderOpen, History, Image as ImageIcon, Film, Clock, Play, Download } from 'lucide-vue-next'
import BaseSelect from '~/components/BaseSelect.vue'
import { toast } from 'vue-sonner'
import { aiConfigAPI, agentConfigAPI, skillsAPI, aiProvidersAPI, traceAPI, storageAPI, generationsAPI, type StorageInfo, type GenerationRecord } from '~/composables/useApi'
import brandLogo from '~/assets/brand-logo.svg'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()

const showBrandImage = ref(true)
const tab = ref('ai')
const showAdvanced = ref(false)
const baseTabs = [
  { id: 'ai', label: 'AI 服务', icon: Cpu },
  { id: 'history', label: '生成历史', icon: History },
  { id: 'storage', label: '数据存储', icon: HardDrive },
]
const advancedTabs = [
  { id: 'agents', label: 'Agent 配置', icon: Bot },
  { id: 'skills', label: 'Skills', icon: FileText },
]
watch(showAdvanced, (v) => {
  if (!v && !baseTabs.some((t) => t.id === tab.value)) tab.value = 'ai'
})

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
    await loadAgents()
    presetDialog.value = false
    toast.success('推荐配置与默认 Agent LLM 已写入')
  } catch (e) {
    toast.error(e.message)
  }
}

// ===== Agent Configs =====
const agentCfgs = ref([])
const editingAgent = ref(null)
const agentSaving = ref(false)
const agentSaved = ref(null)
const agentForm = reactive({ model: '', temperature: 0.7, max_tokens: 4096, system_prompt: '' })
const agentSkillBindings = ref([])   // 当前编辑中的 Agent 的 Skill 绑定 [{ id, enabled, priority }]
const availableSkills = ref([])      // 全局可用 Skill 列表 { id, name, description }[]
const skillDragFrom = ref(null)      // 拖拽起始索引
const skillDragOverIdx = ref(null)   // 当前悬停索引

// 默认 Skill 映射（与后端 AGENT_SKILL_MAP 保持一致）
const DEFAULT_AGENT_SKILLS = {
  script_rewriter: ['script_rewriter'],
  extractor: ['extractor'],
  storyboard_breaker: ['storyboard_breaker', 'extractor'],
  voice_assigner: ['voice_assigner'],
  grid_prompt_generator: ['grid_prompt_generator'],
}

const agentDefs = [
  { type: 'script_rewriter', label: '剧本改写', icon: '📝' },
  { type: 'extractor', label: '角色场景提取', icon: '🔍' },
  { type: 'storyboard_breaker', label: '分镜拆解', icon: '🎬' },
  { type: 'voice_assigner', label: '音色分配', icon: '🎙' },
  { type: 'grid_prompt_generator', label: '图片提示词生成', icon: '🖼' },
]

const defaultPrompts = {
  script_rewriter: `你是专业编剧，擅长将小说改编为短剧剧本。

工作流程：
1. 调用 read_episode_script 读取原始内容
2. 根据读取到的内容，自己进行改写（输出格式化剧本格式）
3. 调用 save_script 保存改写后的完整剧本

格式化剧本格式：
- 场景头：## S编号 | 内景/外景 · 地点 | 时间段
- 动作描写：自然段落，不包含镜头语言
- 对白：角色名：（状态/表情）台词内容
- 每个场景 30-60 秒内容`,
  extractor: `你是制片助理，擅长从剧本中提取角色和场景信息，并在提取时与项目已有数据进行智能去重。

工作流程：
1. 调用 read_script_for_extraction 读取格式化剧本
2. 调用 read_existing_characters 读取项目中已存在的角色列表（用于去重）
3. 调用 read_existing_scenes 读取项目中已存在的场景列表（用于去重）
4. 分析剧本内容，提取所有角色信息
5. 对每个角色：若同名已存在则合并更新，若不存在则新增
6. 调用 save_dedup_characters 保存角色（去重合并，自动处理新增和更新）
7. 分析剧本内容，提取所有场景信息
8. 对每个场景：若同地点+时间段已存在则复用，若不存在则新增
9. 调用 save_dedup_scenes 保存场景（去重合并，自动处理新增和复用）

去重规则：
- 角色：按名字精确匹配，同名保留现有（合并信息）
- 场景：按【地点+时间段】精确匹配；同地点不同时段视为新场景

提取要求：
- 角色要包含完整的外貌特征描述（发型、服装、体态等）
- 场景要包含光线、色调、氛围等视觉信息
- 不要遗漏任何有台词或重要动作的角色`,
  storyboard_breaker: `你是资深影视分镜师，擅长将剧本拆解为分镜方案。

工作流程：
1. 调用 read_storyboard_context 读取剧本、角色列表、场景列表
2. 将剧本拆解为镜头序列（每个镜头 10-15 秒）
3. 为每个镜头生成视频提示词（video_prompt）
4. 调用 save_storyboards 保存所有分镜`,
  voice_assigner: `你是配音导演，擅长为角色选择合适的音色。

工作流程：
1. 调用 list_voices 获取可用音色列表
2. 调用 get_characters 获取所有角色信息
3. 根据每个角色的性别、性格、年龄、角色定位，选择最匹配的音色
4. 对每个角色调用 assign_voice 分配音色，并说明选择理由

注意：每个角色都必须分配音色，不要遗漏。`,
  grid_prompt_generator: `你是专业的 AI 图像提示词工程师，擅长为角色、场景和宫格图生成高质量的英文提示词。

你将收到用户的请求，告知要生成哪种类型的提示词：
- "角色" → 生成角色图片提示词
- "场景" → 生成场景图片提示词
- "宫格" → 生成宫格图提示词

## 角色图片提示词

工作流程：
1. 调用 read_characters 读取所有角色信息
2. 根据角色外貌特征（appearance）、性格（personality）、定位（role）生成英文提示词
3. 提示词结构：[外貌描述]，[性格/气质]，[角色定位]，[电影感]，[高质量]，[无文字水印]

## 场景图片提示词

工作流程：
1. 调用 read_scenes 读取所有场景信息
2. 根据场景地点（location）、时间段（time）、已有描述（prompt）生成英文提示词
3. 提示词结构：[地点]，[时间/光线/氛围]，[已有描述]，[电影感场景]，[高质量]，[无文字水印]

## 宫格图提示词（参考 skills/grid-image-generator/SKILL.md）

工作流程：
1. 调用 read_shots_for_grid 读取选中镜头的详细信息
2. 根据 mode 调用 generate_grid_prompt：
   - first_frame 模式：每格=一个镜头的首帧，NxN 风格统一
   - first_last 模式：每个镜头占2格（左首右尾），同一行风格连续
   - multi_ref 模式：所有格子都是同一镜头的不同参考角度
3. 返回 grid_prompt（整体提示词）和 cell_prompts（每格提示词）

提示词规范：
- 使用英文提示词
- 必须包含 "consistent art style" 保持风格统一
- 必须包含 "cinematic quality"
- 避免出现文字或水印`,
}

function getAgentCfg(type) {
  return agentCfgs.value.find(a => a.agent_type === type)
}

const textModelGroups = computed(() => {
  return cfgs.value
    .filter(c => c.service_type === 'text' && c.is_active && c.api_key)
    .map(c => ({
      label: `${c.provider} — ${c.name}`,
      models: Array.isArray(c.model) ? c.model : (c.model ? [c.model] : []),
    }))
    .filter(g => g.models.length > 0)
})

const textModelSelectOptions = computed(() =>
  textModelGroups.value.map(g => ({
    label: g.label,
    options: g.models.map(m => ({ label: m, value: m })),
  }))
)

async function loadAgents() {
  try { agentCfgs.value = await agentConfigAPI.list() }
  catch (e) { toast.error(e.message) }
}

function toggleAgentEdit(type) {
  if (editingAgent.value === type) { editingAgent.value = null; agentSkillBindings.value = []; return }
  const cfg = getAgentCfg(type)
  agentForm.model = cfg?.model || ''
  agentForm.temperature = cfg?.temperature ?? 0.7
  agentForm.max_tokens = cfg?.max_tokens ?? 4096
  agentForm.system_prompt = cfg?.system_prompt || defaultPrompts[type] || ''
  // 加载 Skill 绑定配置
  if (cfg?.skills) {
    try { agentSkillBindings.value = JSON.parse(cfg.skills) } catch { agentSkillBindings.value = [] }
  } else {
    const defaults = DEFAULT_AGENT_SKILLS[type] || []
    agentSkillBindings.value = defaults.map((id, i) => ({ id, enabled: true, priority: i + 1 }))
  }
  agentSaved.value = null
  editingAgent.value = type
}

function resetAgentPrompt(type) {
  agentForm.system_prompt = defaultPrompts[type] || ''
  toast.info('已恢复默认提示词，点击保存生效')
}

function resetAgentSkills(type) {
  const defaults = DEFAULT_AGENT_SKILLS[type] || []
  agentSkillBindings.value = defaults.map((id, i) => ({ id, enabled: true, priority: i + 1 }))
  toast.info('已恢复默认 Skill 绑定，点击保存生效')
}

function getSkillName(skillId) {
  return availableSkills.value.find(s => s.id === skillId)?.name
}

function onSkillDragStart(idx, e) {
  skillDragFrom.value = idx
  if (e?.dataTransfer) {
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', String(idx))
  }
}
function onSkillDragOver(idx) {
  skillDragOverIdx.value = idx
}
function onSkillDrop(idx) {
  const from = skillDragFrom.value
  if (from == null || from === idx) { skillDragOverIdx.value = null; return }
  const list = [...agentSkillBindings.value]
  const [moved] = list.splice(from, 1)
  list.splice(idx, 0, moved)
  agentSkillBindings.value = list
  skillDragOverIdx.value = null
}
function onSkillDragEnd() {
  skillDragFrom.value = null
  skillDragOverIdx.value = null
}

async function saveAgentCfg(type) {
  agentSaving.value = true
  agentSaved.value = null
  try {
    const existing = getAgentCfg(type)
    // 按当前列表顺序重写 priority，确保拖拽排序后顺序正确落库
    const orderedSkills = agentSkillBindings.value.map((b, i) => ({ ...b, priority: i + 1 }))
    const data = {
      agent_type: type,
      name: agentDefs.find(a => a.type === type)?.label || type,
      model: agentForm.model,
      temperature: agentForm.temperature,
      max_tokens: agentForm.max_tokens,
      system_prompt: agentForm.system_prompt,
      skills: orderedSkills.length ? JSON.stringify(orderedSkills) : null,
    }
    if (existing) {
      await agentConfigAPI.update(existing.id, data)
    } else {
      await agentConfigAPI.create(data)
    }
    await loadAgents()
    agentSaved.value = type
    toast.success(`${agentDefs.find(a => a.type === type)?.label} 配置已保存`)
    setTimeout(() => { if (agentSaved.value === type) agentSaved.value = null }, 3000)
  } catch (e) {
    toast.error(e.message)
  } finally {
    agentSaving.value = false
  }
}

// ===== Skills =====
const selectedAgent = ref('script_rewriter')
const allSkills = ref([])   // { id, name, description }[]
const editingSkill = ref(null)
const skillContent = ref('')
const skillSaving = ref(false)
const skillSaved = ref(null)
const addSkillDialog = ref(false)
const newSkillForm = reactive({ id: '', name: '', description: '' })

const selectedAgentType = computed(() => selectedAgent.value)
const selectedAgentLabel = computed(() => agentDefs.find(a => a.type === selectedAgent.value)?.label || '')
const selectedAgentIcon = computed(() => agentDefs.find(a => a.type === selectedAgent.value)?.icon || '')

function agentSkillCount(type) {
  return allSkills.value.filter(s => s.id === type || s.id.startsWith(type + '/')).length
}

const currentSkills = computed(() =>
  allSkills.value.filter(s => s.id === selectedAgent.value || s.id.startsWith(selectedAgent.value + '/'))
)

async function loadAllSkills() {
  try { allSkills.value = await skillsAPI.list() }
  catch (e) { toast.error(e.message) }
}

/** 加载全局可用 Skill 列表（用于 Agent 配置中的 Skill 绑定面板） */
async function loadAvailableSkills() {
  try { availableSkills.value = await skillsAPI.list() }
  catch (e) { /* 静默失败，面板显示空状态 */ }
}

async function selectAgent(type) {
  selectedAgent.value = type
  editingSkill.value = null
}

function startAddSkill() {
  newSkillForm.id = ''
  newSkillForm.name = ''
  newSkillForm.description = ''
  addSkillDialog.value = true
}

async function confirmAddSkill() {
  if (!newSkillForm.id) return
  const skillId = `${selectedAgent.value}/${newSkillForm.id}`
  try {
    await skillsAPI.create({ id: skillId, name: newSkillForm.name, description: newSkillForm.description })
    addSkillDialog.value = false
    await loadAllSkills()
    toast.success('Skill 创建成功')
  } catch (e) {
    toast.error(e.message)
  }
}

async function deleteSkill(id) {
  if (!(await confirm({ message: `确定删除 Skill「${id}」？`, danger: true }))) return
  try {
    await skillsAPI.del(id)
    if (editingSkill.value === id) editingSkill.value = null
    await loadAllSkills()
    toast.success('已删除')
  } catch (e) {
    toast.error(e.message)
  }
}

async function toggleSkillEdit(id) {
  if (editingSkill.value === id) { editingSkill.value = null; return }
  try {
    const res = await skillsAPI.get(id)
    skillContent.value = res.content
    skillSaved.value = null
    editingSkill.value = id
  } catch (e) { toast.error(e.message) }
}

async function saveSkill(id) {
  skillSaving.value = true
  skillSaved.value = null
  try {
    await skillsAPI.update(id, skillContent.value)
    await loadAllSkills()
    skillSaved.value = id
    toast.success(`已保存`)
    setTimeout(() => { if (skillSaved.value === id) skillSaved.value = null }, 3000)
  } catch (e) {
    toast.error(e.message)
  } finally {
    skillSaving.value = false
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

// ===== 生成历史 =====
const generations = ref<GenerationRecord[]>([])
const historyLoading = ref(false)
const historyFilter = ref<'all' | 'image' | 'video'>('all')
const historyExpandedId = ref<string | null>(null)

const historyFilters = [
  { value: 'all', label: '全部' },
  { value: 'image', label: '图片' },
  { value: 'video', label: '视频' },
] as const

const filteredGenerations = computed(() => {
  if (historyFilter.value === 'all') return generations.value
  return generations.value.filter((g) => g.type === historyFilter.value)
})

const historyStats = computed(() => {
  const list = generations.value
  const total = list.length
  const success = list.filter((g) => isSuccessStatus(g.status)).length
  const failed = list.filter((g) => isFailedStatus(g.status)).length
  return { total, success, failed, processing: total - success - failed }
})

function isSuccessStatus(s: string): boolean {
  return ['completed', 'succeeded', 'success', 'done'].includes((s || '').toLowerCase())
}
function isFailedStatus(s: string): boolean {
  return ['failed', 'error', 'cancelled', 'canceled'].includes((s || '').toLowerCase())
}
function statusLabel(s: string): string {
  if (isSuccessStatus(s)) return '成功'
  if (isFailedStatus(s)) return '失败'
  const low = (s || '').toLowerCase()
  if (['pending', 'queued', 'processing', 'running', 'generating'].includes(low)) return '处理中'
  return s || '未知'
}
function statusClass(s: string): string {
  if (isSuccessStatus(s)) return 'ok'
  if (isFailedStatus(s)) return 'err'
  return 'warn'
}
function fmtTime(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
function fmtElapsed(ms?: number | null): string {
  if (ms == null) return '—'
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  const totalSec = Math.round(ms / 1000)
  return `${Math.floor(totalSec / 60)}m ${totalSec % 60}s`
}
function toggleHistory(g: GenerationRecord) {
  const key = `${g.type}-${g.id}`
  historyExpandedId.value = historyExpandedId.value === key ? null : key
}
async function loadGenerations() {
  historyLoading.value = true
  try {
    generations.value = await generationsAPI.list({ limit: 200 })
  } catch (e: any) {
    toast.error(e.message || '加载生成历史失败')
  } finally {
    historyLoading.value = false
  }
}

// ===== 数据存储 =====
const storageInfo = ref<StorageInfo | null>(null)
const storageLoading = ref(false)
const storageChanging = ref(false)
const newDataRoot = ref('')
const migrateData = ref(true)

function fmtBytes(n: number | null | undefined): string {
  if (n == null || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`
}

async function loadStorageInfo() {
  storageLoading.value = true
  try { storageInfo.value = await storageAPI.info() }
  catch (e: any) { toast.error(e.message || '加载存储信息失败') }
  finally { storageLoading.value = false }
}

async function changeDataRoot() {
  const target = newDataRoot.value.trim()
  if (!target) { toast.error('请填写目标目录路径'); return }
  storageChanging.value = true
  try {
    storageInfo.value = await storageAPI.change(target, migrateData.value)
    newDataRoot.value = ''
    toast.success(migrateData.value ? '目录已切换，旧数据已自动迁移（旧目录保留）' : '目录已切换')
    loadTokenStats()
  } catch (e: any) {
    toast.error(e.message || '切换目录失败')
  } finally { storageChanging.value = false }
}

onMounted(() => { loadCfgs(); loadAgents(); loadAllSkills(); loadAvailableSkills(); loadLocalConfigs(); loadGpuStatus(); refreshOllama(); loadProviders(); loadTokenStats(); loadStorageInfo(); loadGenerations() })
</script>

<style scoped>
.settings-layout { display: flex; height: 100%; background: var(--bg-base); }

.settings-nav {
  width: 220px; flex-shrink: 0; padding: 16px 10px; border-right: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 14px; background: var(--bg-1);
}
.nav-group { display: flex; flex-direction: column; gap: 4px; }
.nav-group-label {
  font-size: 10px; font-weight: 700; color: var(--text-3);
  letter-spacing: 0.12em; text-transform: uppercase; padding: 0 10px 4px;
}
.nav-item {
  display: flex; align-items: center; gap: 8px; padding: 9px 12px; font-size: 13px;
  border: none; background: none; color: var(--text-2); cursor: pointer;
  border-radius: var(--radius); transition: all 0.12s; text-align: left; width: 100%;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-0); }
.nav-item.active { background: var(--accent-bg); color: var(--accent-text); font-weight: 600; box-shadow: var(--shadow-card); }
.nav-advanced {
  padding: 12px 8px;
  border-top: 1px solid rgba(27, 41, 64, 0.08);
  border-bottom: 1px solid rgba(27, 41, 64, 0.08);
}
.advanced-toggle {
  display: grid; grid-template-columns: 1fr auto auto; align-items: center; gap: 10px;
  font-size: 12px; color: var(--text-2);
}
.advanced-toggle input { display: none; }
.advanced-slider {
  position: relative; width: 38px; height: 22px; border-radius: 999px;
  background: rgba(27, 41, 64, 0.12); transition: background 0.18s ease;
}
.advanced-slider::after {
  content: ''; position: absolute; top: 3px; left: 3px; width: 16px; height: 16px;
  border-radius: 50%; background: #fff; box-shadow: 0 2px 6px rgba(18, 24, 38, 0.18); transition: transform 0.18s ease;
}
.advanced-toggle input:checked + .advanced-slider { background: var(--accent); }
.advanced-toggle input:checked + .advanced-slider::after { transform: translateX(16px); }
.advanced-note {
  margin: 8px 0 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--text-3);
}

.settings-content { flex: 1; overflow: hidden; }
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

/* Agent */
.agent-list { display: flex; flex-direction: column; gap: 8px; }
.agent-card { overflow: hidden; }
.agent-card-head { display: flex; align-items: center; gap: 10px; padding: 14px 16px; cursor: pointer; transition: background 0.1s; }
.agent-card-head:hover { background: var(--bg-hover); }
.agent-type-badge { width: 36px; height: 36px; border-radius: var(--radius); background: var(--accent-bg); color: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.agent-card-body { padding: 0 16px 16px; display: flex; flex-direction: column; gap: 12px; border-top: 1px solid var(--border); padding-top: 16px; }
.agent-card-foot { display: flex; align-items: center; gap: 8px; padding-top: 8px; }

/* Skill 绑定面板 */
.skill-bind-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: rgba(244,248,255,0.6);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.skill-bind-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.skill-bind-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-1);
  display: flex;
  align-items: center;
  gap: 5px;
}
.skill-bind-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.skill-bind-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.72);
  transition: opacity 0.15s, background 0.15s, box-shadow 0.15s, transform 0.15s;
}
.skill-bind-item.disabled {
  opacity: 0.48;
  background: rgba(245,245,245,0.9);
}
.skill-bind-item.drag-over {
  background: var(--accent-bg);
  box-shadow: inset 0 -2px 0 var(--accent);
}
.skill-bind-drag {
  color: var(--text-3);
  cursor: grab;
  font-size: 16px;
  user-select: none;
  flex-shrink: 0;
}
.skill-bind-drag:active { cursor: grabbing; }
.skill-bind-toggle {
  position: relative;
  width: 28px;
  height: 16px;
  cursor: pointer;
  flex-shrink: 0;
}
.skill-bind-toggle input { opacity: 0; width: 0; height: 0; }
.skill-bind-toggle span {
  position: absolute; inset: 0; background: var(--bg-3); border-radius: 99px; transition: 0.2s;
}
.skill-bind-toggle span::before {
  content: ''; position: absolute; width: 12px; height: 12px; left: 2px; bottom: 2px;
  background: #fff; border-radius: 50%; transition: transform 0.2s; box-shadow: var(--shadow-sm);
}
.skill-bind-toggle input:checked + span { background: var(--accent); }
.skill-bind-toggle input:checked + span::before { transform: translateX(12px); }
.skill-bind-info {
  flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px;
}
.skill-bind-name {
  font-size: 12px; font-weight: 500; color: var(--text-1);
}
.skill-bind-priority {
  font-size: 10px; font-weight: 700; font-family: var(--font-mono);
  color: var(--text-3); flex-shrink: 0;
}

/* Skills 布局 */
.skills-layout { display: flex; height: 100%; overflow: hidden; }
.skills-agent-list {
  width: 200px; flex-shrink: 0; border-right: 1px solid var(--border);
  background: var(--bg-1); display: flex; flex-direction: column;
  overflow-y: auto;
}
.skills-agent-title {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-3); padding: 14px 14px 8px;
}
.skills-agent-item {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 14px; font-size: 13px; cursor: pointer;
  border: none; background: none; color: var(--text-2);
  transition: all 0.12s; width: 100%; text-align: left;
  border-radius: 0;
}
.skills-agent-item:hover { background: var(--bg-hover); color: var(--text-0); }
.skills-agent-item.active { background: var(--accent-bg); color: var(--accent-text); font-weight: 600; }
.skills-agent-label { flex: 1; }
.skill-count-badge {
  font-size: 10px; font-weight: 700; font-family: var(--font-mono);
  background: var(--accent-bg); color: var(--accent-text);
  padding: 1px 5px; border-radius: 99px;
}
.skills-agent-item.active .skill-count-badge { background: rgba(255,255,255,0.2); color: inherit; }
.skills-main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
.skills-main .settings-scroll { max-width: 900px; }

/* Skill */
.skill-list { display: flex; flex-direction: column; gap: 8px; }
.skill-card { overflow: hidden; }
.skill-card-head { display: flex; align-items: center; gap: 10px; padding: 12px 16px; cursor: pointer; transition: background 0.1s; }
.skill-card-head:hover { background: var(--bg-hover); }
.skill-card-body { padding: 0 16px 16px; display: flex; flex-direction: column; gap: 10px; border-top: 1px solid var(--border); padding-top: 12px; }
.skill-card-foot { display: flex; align-items: center; gap: 8px; }

/* Shared */
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--text-1); }
.field-hint { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

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
.preset-grid-form .field-hint a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}
.preset-grid-form .field-hint a:hover {
  text-decoration: underline;
}

@media (max-width: 900px) {
  .preset-grid,
  .preset-grid.compact {
    grid-template-columns: 1fr;
  }
}

/* ===== 数据存储 ===== */
.storage-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 14px;
}
.storage-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 14px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.storage-icon {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: var(--bg-1);
  color: var(--accent);
}
.storage-meta { min-width: 0; flex: 1; }
.storage-label { font-size: 11px; color: var(--text-3); margin-bottom: 2px; }
.storage-path {
  font-size: 12px;
  color: var(--text-1);
  word-break: break-all;
  line-height: 1.5;
}
.storage-sub { font-size: 11px; margin-top: 3px; }
.storage-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
}
.storage-check {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-2);
  cursor: pointer;
  user-select: none;
}
.storage-check input {
  width: 15px;
  height: 15px;
  accent-color: var(--accent);
  cursor: pointer;
}
.storage-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.storage-note { font-size: 11px; line-height: 1.6; }

/* ===== 生成历史 ===== */
.history-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.history-filters { display: flex; gap: 8px; }
.chip {
  padding: 6px 14px;
  font-size: 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-1);
  color: var(--text-2);
  cursor: pointer;
  transition: all 0.14s;
}
.chip:hover { color: var(--text-0); border-color: var(--text-3); }
.chip.active {
  background: var(--accent-bg);
  border-color: transparent;
  color: var(--accent-text);
  font-weight: 600;
}
.history-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 18px;
}
.history-stat {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.history-stat-num { font-size: 20px; font-weight: 700; font-family: var(--font-display); color: var(--text-0); }
.history-stat-label { font-size: 12px; color: var(--text-3); margin-top: 4px; }
.history-stat.ok .history-stat-num { color: #2f9e63; }
.history-stat.warn .history-stat-num { color: #c9973f; }
.history-stat.err .history-stat-num { color: #c95844; }

.history-list { display: flex; flex-direction: column; gap: 8px; margin: 0; padding: 0; list-style: none; }
.history-item {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-2);
  overflow: hidden;
}
.history-item-head {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 12px 14px;
  border: none;
  background: none;
  cursor: pointer;
  text-align: left;
  transition: background 0.14s;
}
.history-item-head:hover { background: var(--bg-hover); }
.history-type {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.history-type.image { background: rgba(13,148,136,0.14); color: #0d9488; }
.history-type.video { background: rgba(150, 92, 235, 0.16); color: #965ceb; }
.history-item-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.history-item-title { font-size: 13px; font-weight: 600; color: var(--text-0); }
.history-item-meta { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-3); }
.history-item-meta .dot { color: var(--text-3); }
.history-item-time {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-3);
  white-space: nowrap;
}
.history-chevron { color: var(--text-3); transition: transform 0.18s; flex-shrink: 0; }
.history-chevron.open { transform: rotate(180deg); }

.status-badge {
  flex-shrink: 0;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 999px;
  border: 1px solid transparent;
  white-space: nowrap;
}
.status-badge.ok { background: rgba(47, 158, 99, 0.12); color: #2f9e63; }
.status-badge.warn { background: rgba(201, 151, 63, 0.14); color: #c9973f; }
.status-badge.err { background: rgba(201, 88, 68, 0.12); color: #c95844; }

.history-item-body {
  padding: 12px 14px;
  border-top: 1px solid var(--border);
  background: var(--bg-1);
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.history-detail-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  font-size: 12px;
  color: var(--text-1);
}
.history-detail-label {
  flex-shrink: 0;
  width: 64px;
  color: var(--text-3);
  font-size: 12px;
}
.history-detail-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.history-detail-block .history-detail-label { width: auto; }
.history-prompt {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-2);
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  word-break: break-word;
  white-space: pre-wrap;
}
.history-detail-block.error .history-detail-label { color: #c95844; }
.history-detail-block.error .history-prompt { color: #c95844; border-color: rgba(201, 88, 68, 0.25); }
.history-link {
  font-size: 12px;
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}
.history-link:hover { text-decoration: underline; }
</style>

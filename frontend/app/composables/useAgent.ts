import { toast } from 'vue-sonner'
import { api } from './useApi'

// 各 agent 的预估耗时（毫秒），用于显示进度文案
const AGENT_PROGRESS_LABELS: Record<string, { label: string; estimate: number }> = {
  script_rewriter: { label: '改写剧本', estimate: 60_000 },
  extractor: { label: '提取角色场景', estimate: 90_000 },
  storyboard_breaker: { label: '拆解分镜', estimate: 120_000 },
  voice_assigner: { label: '分配音色', estimate: 45_000 },
  grid_prompt_generator: { label: '生成提示词', estimate: 60_000 },
}

// 超时上限：10 分钟（agent 最多 20 步，单步 30 秒以内）
const MAX_TIMEOUT = 10 * 60 * 1000

// 只读工具的前缀（`read_script_for_extraction` / `list_voices` / `get_characters` …）
const READ_ONLY_PREFIXES = ['read_', 'list_', 'get_']

function isReadOnlyTool(name: string): boolean {
  return READ_ONLY_PREFIXES.some((prefix) => name.startsWith(prefix))
}

/** 一次 agent 运行的结果摘要 —— 后端 `data` 里的原样内容 + 一句人话结论。 */
export interface AgentOutput {
  /** agent 类型（extractor / script_rewriter …） */
  type: string
  /** 中文名，用于标题 */
  label: string
  /** 模型最后一段话（后端 `data.text`）—— 这是「有返回但没展示」的那部分内容 */
  text: string
  /** 本次实际调用过的工具名（`toolName` 的平铺列表，便于界面直接渲染） */
  toolNames: string[]
  /**
   * 是否做过「非只读」动作（读了/列了之外，真的写了东西）。
   *
   * ⚠️ 为什么用这个判据：agent 的职责是「读数据 → 调 save_* 落库」，而模型有可能只读了
   * 剧本就开始输出分析（实测 2026-10-28：extractor 只调了 `read_script_for_extraction`，
   * 然后返回一段「故事亮点分析」）。这时数据库一行没变，前端若还弹「完成！」就是在骗人。
   * 工具命名契约是 `read_*` / `list_*` / `get_*` 只读，写入是 `save_*` / `assign_voice` /
   * `generate_grid_prompt` 这些 —— 用「只读前缀」判定比枚举每个 agent 的写入工具更耐漂移
   * （把 `list_voices` 当成写动作会让 voice_assigner 继续谎报完成）。
   */
  acted: boolean
  /** 本次耗时（秒） */
  seconds: number
}

interface LastRun {
  type: string
  msg: string
  dramaId: number
  episodeId: number
  onDone?: () => void
}

export function useAgent() {
  const running = ref(false)
  const runningType = ref<string | null>(null)
  const elapsed = ref(0)
  const progressText = ref('')
  /** 最近一次运行的结果（含模型原话）—— 由界面决定怎么展示 */
  const lastOutput = ref<AgentOutput | null>(null)

  let toastId: string | number | undefined
  let timer: ReturnType<typeof setInterval> | undefined
  let abortCtrl: AbortController | undefined
  let lastRun: LastRun | undefined

  async function run(type: string, msg: string, dramaId: number, episodeId: number, onDone?: () => void) {
    if (running.value) { toast.warning('操作执行中'); return }

    const meta = AGENT_PROGRESS_LABELS[type] || { label: type, estimate: 60_000 }
    lastRun = { type, msg, dramaId, episodeId, onDone }

    running.value = true
    runningType.value = type
    elapsed.value = 0
    progressText.value = `${meta.label}中...`
    lastOutput.value = null

    const startTime = performance.now()

    // 用 message toast，不依赖 update API（vue-sonner 不支持 update）
    // 进度通过 ref 暴露给组件渲染（不用 toast 显示动态秒数）
    toastId = toast(`${meta.label}进行中...`, { duration: MAX_TIMEOUT })

    // 进度刷新（仅更新 ref，不调用 toast API）
    timer = setInterval(() => {
      elapsed.value = Math.round((performance.now() - startTime) / 1000)
      const pct = Math.min(95, Math.round((elapsed.value * 1000 / meta.estimate) * 100))
      progressText.value = `${meta.label}中... ${elapsed.value}s (${pct}%)`
    }, 1000)

    // AbortController 长超时保护
    abortCtrl = new AbortController()
    const timeoutId = setTimeout(() => abortCtrl?.abort(), MAX_TIMEOUT)

    try {
      // 直接用 fetch（绕过 api.post 的默认行为），传入 signal
      const resp = await fetch(`/api/v1/agent/${type}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, drama_id: dramaId, episode_id: episodeId }),
        signal: abortCtrl.signal,
      })

      const json = await resp.json()
      const seconds = Math.round((performance.now() - startTime) / 1000)

      // 关闭进度 toast
      if (toastId !== undefined) toast.dismiss(toastId)

      if (!resp.ok || (json.code && json.code >= 400)) {
        const errMsg = json.message || `${resp.status}`
        console.error(`[Agent] ${type} failed (${seconds}s):`, errMsg)
        toast.error(`${meta.label}失败：${errMsg}（用时 ${seconds}s）`)
        return
      }

      // 后端把结果放在 data 里：{ type: 'done', text, toolCalls, toolResults, usage }
      const data = json.data || {}
      const toolCalls: any[] = Array.isArray(data.toolCalls) ? data.toolCalls : []
      const toolNames = toolCalls
        .map((call) => String(call?.toolName || ''))
        .filter(Boolean)
      const text = typeof data.text === 'string' ? data.text.trim() : ''
      const acted = toolNames.some((name) => !isReadOnlyTool(name))

      lastOutput.value = { type, label: meta.label, text, toolNames, acted, seconds }

      if (acted) {
        toast.success(`${meta.label}完成！用时 ${seconds}s${toolNames.length ? ` · ${toolNames.length} 步工具调用` : ''}`)
      } else {
        // 只读了数据就收工 ⇒ 库里什么都没变。以前这里照样弹「完成！」并把 text 丢掉，
        // 用户看到的就是「有返回、界面没变化」。现在明说，并把模型原话交给界面展示。
        console.warn(`[Agent] ${type} 未落库（无写入工具调用），text=${text.length} 字`)
        toast.warning(`${meta.label}没有保存任何结果：AI 只读取了数据就结束（用时 ${seconds}s）`, {
          duration: 8000,
        })
      }

      // 刷新是无害的：作过保存就该看到新数据，没作保存只是回到旧数据（不改口）
      onDone?.()
    } catch (err: any) {
      const seconds = Math.round((performance.now() - startTime) / 1000)
      if (toastId !== undefined) toast.dismiss(toastId)

      let errMsg = err.message || '未知错误'
      if (err.name === 'AbortError') {
        errMsg = `超时（${Math.round(MAX_TIMEOUT / 1000)}s），可能是 AI 模型响应过慢或卡住，请检查后端日志或重试`
      }

      console.error(`[Agent] ${type} error (${seconds}s):`, err)
      toast.error(`${meta.label}失败：${errMsg}（用时 ${seconds}s）`)
    } finally {
      clearTimeout(timeoutId)
      if (timer) { clearInterval(timer); timer = undefined }
      running.value = false
      runningType.value = null
      progressText.value = ''
      abortCtrl = undefined
    }
  }

  /** 按上次的入参重跑一次（输出面板里的「重试」） */
  function retry() {
    if (!lastRun) return
    const { type, msg, dramaId, episodeId, onDone } = lastRun
    return run(type, msg, dramaId, episodeId, onDone)
  }

  return { running, runningType, elapsed, progressText, lastOutput, run, retry }
}

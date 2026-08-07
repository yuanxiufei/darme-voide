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

export function useAgent() {
  const running = ref(false)
  const runningType = ref<string | null>(null)
  const elapsed = ref(0)
  const progressText = ref('')
  let toastId: string | number | undefined
  let timer: ReturnType<typeof setInterval> | undefined
  let abortCtrl: AbortController | undefined

  async function run(type: string, msg: string, dramaId: number, episodeId: number, onDone?: () => void) {
    if (running.value) { toast.warning('操作执行中'); return }

    const meta = AGENT_PROGRESS_LABELS[type] || { label: type, estimate: 60_000 }

    running.value = true
    runningType.value = type
    elapsed.value = 0
    progressText.value = `${meta.label}中...`

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

      // 成功
      const toolCount = json.data?.toolCalls?.length || 0
      toast.success(`${meta.label}完成！用时 ${seconds}s${toolCount ? ` · ${toolCount} 步工具调用` : ''}`)
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

  return { running, runningType, elapsed, progressText, run }
}

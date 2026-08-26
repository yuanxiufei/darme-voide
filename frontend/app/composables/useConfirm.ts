import { reactive } from 'vue'

export interface ConfirmOptions {
  title?: string
  message: string
  confirmText?: string
  cancelText?: string
  /** 危险操作（删除等）：确认按钮显示为红色 */
  danger?: boolean
}

const state = reactive({
  visible: false,
  title: '',
  message: '',
  confirmText: '确定',
  cancelText: '取消',
  danger: false,
})

let _resolve: ((v: boolean) => void) | null = null

/**
 * 全局统一确认对话框（单例）。
 * 用法：`const { confirm } = useConfirm(); if (!(await confirm({ message: '确认删除？', danger: true }))) return`
 */
export function useConfirm() {
  function confirm(opts: string | ConfirmOptions): Promise<boolean> {
    const o: ConfirmOptions = typeof opts === 'string' ? { message: opts } : opts
    state.title = o.title || '确认操作'
    state.message = o.message
    state.confirmText = o.confirmText || '确定'
    state.cancelText = o.cancelText || '取消'
    state.danger = !!o.danger
    state.visible = true
    return new Promise<boolean>((resolve) => { _resolve = resolve })
  }

  function settle(v: boolean) {
    state.visible = false
    _resolve?.(v)
    _resolve = null
  }

  return { confirm, settle, state }
}

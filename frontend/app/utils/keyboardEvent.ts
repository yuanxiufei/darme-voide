/**
 * 键盘事件的**输入法守卫**（移植自上游 `lib/keyboard-event.ts` 的口径，MIT；按 Vue 重写 ✓）。
 *
 * ## 为什么必须有这一层
 *
 * 中文/日文输入法下，**按回车是在"确认候选词"**，浏览器仍会派发 `keydown Enter` ✗。
 * 于是「输入标签后回车」「重命名后回车」「输预设名回车保存」这些框，会在用户**选词**
 * 的那一下被误触发：标签被提前提交成半个词、重命名直接关掉、预设存下一个错名 ✗。
 *
 * 判据三条**必须都看** ✓：
 * 1. `event.isComposing`（标准字段 ✓）；
 * 2. `keyCode === 229`（Safari/旧 Chromium 在组合态只给 229 ✓，`key` 会是 `"Process"`）；
 * 3. React 合成事件的 `nativeEvent`（本仓是 Vue 用不到，但保留以兼容移植口径 ✓）。
 *
 * ⚠️ 为什么挂在 **`keydown`** 而不是 `keyup`：`keyup` 时组合已经结束（`isComposing` 已变 false ✗），
 * 那时再判就**判不出来**了 —— 用 `keyup.enter` 的写法天生躲不开这个坑 ✓。
 */

/** 结构化的键事件（原生 `KeyboardEvent` 天然满足 ✓） */
export interface KeyboardEventLike {
  key?: string
  isComposing?: boolean
  keyCode?: number
  which?: number
  shiftKey?: boolean
  ctrlKey?: boolean
  metaKey?: boolean
  nativeEvent?: { isComposing?: boolean; keyCode?: number; which?: number }
}

/** 输入法正在组合（此时回车属于输入法，不属于页面 ✓） */
export function isImeComposing(event: KeyboardEventLike): boolean {
  const native = event.nativeEvent
  return Boolean(
    event.isComposing
      || native?.isComposing
      || event.keyCode === 229
      || event.which === 229
      || native?.keyCode === 229
      || native?.which === 229,
  )
}

/** 干净的裸回车：`Enter` 且不带 Shift/Ctrl/Meta，且不在组合态 ✓ */
export function isPlainEnterKey(event: KeyboardEventLike): boolean {
  return event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.metaKey && !isImeComposing(event)
}

/**
 * 模板里直接用：`@keydown.enter="guardEnter($event, save)"` ✓
 *
 * 只在「确认是裸回车」时才 `preventDefault()` **并**执行 ✓。
 * ⚠️ 别在模板上写 `.prevent`（修饰符无条件阻止默认行为，组合态会把输入法的确认也挡掉 ✗）。
 */
export function guardEnter(event: KeyboardEventLike, handler: () => void): boolean {
  if (!isPlainEnterKey(event)) return false
  const native = event as unknown as { preventDefault?: () => void }
  native.preventDefault?.()
  handler()
  return true
}

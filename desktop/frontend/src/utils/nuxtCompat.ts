/**
 * 桌面端对 Nuxt 两个全局 API 的**同名兼容实现**（本工程自有，不来自 frontend/）。
 *
 * ## 它现在就在干活，不是「预留件」
 *
 * 桌面端 `navigateTo(...)` 有 6 个文件 12 处、`useState(...)` 有 2 处，
 * **全部零 import** —— 靠 `vite.config.ts` 的 `AutoImport` 注入（`dirs` 含
 * `src/utils`）。所以：本文件被删、或 `dirs` 里少了 `src/utils`，症状都是
 * 「点一下没反应」，而 `vite build` 不报错（它只是注入不到）。
 *
 * ⚠️ `vue-tsc` 也不能指望：`src/auto-imports.d.ts` 是 unplugin 的生成物、被
 * `.gitignore` 忽略，本机留着它（**陈旧也照样声明**）⇒ typecheck 一路绿，
 * 只有干净克隆（没跑过 dev/build）才会报。实测把它移走后 typecheck 报
 * **481 处 `Cannot find name`**（`ref`/`computed`/`navigateTo` 全在内）。
 * 所以守卫必须是脚本级的：`scripts/check-isolation.mjs` 的
 * B2（实现齐全）+ B3（`dirs` 覆盖）+ B4（`imports`/`dts`）+ H（`vueTemplate`）。
 *
 * ## 为什么做同名兼容，而不是逐个改调用点
 *
 * 逐个改的话：`navigateTo` 返回 Promise 不能直接用于模板事件、`useState` 的
 * 第二参数是工厂函数（不是初值），每处都要判断语义 ⇒ 出错概率远高于写一个等价物。
 * 同名的好处是：**调用点零改动**，两端 `Onboarding.vue` / `layouts/default.vue`
 * 因此能保持逐字一致（`check:parity` 的判据 B）。
 *
 * ## 只提供这两件
 *
 * 没有 `useCookie` —— 两端都没有调用点，凭想象补一个等价物属于猜测语义。
 *
 * ## 与 Nuxt 语义的对齐点
 *
 * - `navigateTo(to)` 支持 `'/path'`、`'/?a=1'`、`{ path, query }`
 * - `useState(key, init)` 是**按 key 共享的进程级单例**，`init` 只在首次求值
 *   （Nuxt 里 SSR/CSR 各一份；桌面端只有一份，正是想要的行为）
 */
import { ref, type Ref } from 'vue'
import router from '../router'

type NavigateTarget =
  | string
  | { path?: string; name?: string; query?: Record<string, string>; params?: Record<string, string> }

/** 等价于 Nuxt 的 `navigateTo`：内部就是 `router.push`，额外容忍 `'/path?k=v'` 写法 */
export function navigateTo(to: NavigateTarget): Promise<unknown> {
  if (typeof to === 'string') {
    const idx = to.indexOf('?')
    if (idx === -1) return router.push(to)
    const path = to.slice(0, idx)
    const query = Object.fromEntries(new URLSearchParams(to.slice(idx + 1)))
    return router.push({ path, query })
  }
  return router.push(to)
}

const registry = new Map<string, Ref<unknown>>()

/** 等价于 Nuxt 的 `useState`：同 key 返回同一个 ref，`init` 仅首次执行 */
export function useState<T>(key: string, init: () => T): Ref<T> {
  const hit = registry.get(key)
  if (hit) return hit as Ref<T>
  const created = ref(init()) as Ref<T>
  registry.set(key, created as Ref<unknown>)
  return created
}

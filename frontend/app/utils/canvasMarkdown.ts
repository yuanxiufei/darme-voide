/**
 * 极简 Markdown 渲染（**不引依赖**，也**不信任输入**）✓。
 *
 * 为什么自己写而不是装 `markdown-it`：本仓前端只有 5 个依赖，为一张便利贴级别的
 * 渲染拉进一个解析器不划算；更关键的是**安全** —— 自己写能把「先转义、再替换」
 * 这条顺序钉死在一个文件里，装库则要额外配 sanitize，配错就是 XSS ✗。
 *
 * ## 两条铁律
 *
 * 1. **先 `escapeHtml` 再拼标签** —— 顺序反了，用户写的 `<img onerror>` 会原样进 DOM ✗
 * 2. **链接只放行 `http(s)/mailto/#` 和相对路径** —— 不拦 `javascript:` 等于送一个
 *    「点一下就执行」的入口 ✗（拦下来只留文字，不给链接）
 */

/** HTML 转义（**任何拼进 innerHTML 的用户文本都必须先过这一步**） */
export function escapeHtml(text: string): string {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/** 只放行安全协议；不安全的返回 `null`（调用方据此只留文字） */
export function safeUrl(raw: string): string | null {
  const url = String(raw ?? '').trim()
  if (!url) return null
  if (/^(https?:\/\/|mailto:|#|\/)/i.test(url)) return url
  return null
}

function inline(text: string): string {
  // `code` 先抽走：否则里面的 `*` / `[` 会被后面的行内规则吃掉 ✗
  const codes: string[] = []
  let out = text.replace(/`([^`]+)`/g, (_match, code: string) => {
    codes.push(`<code>${code}</code>`)
    return `\u0000${codes.length - 1}\u0000`
  })
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label: string, href: string) => {
    const url = safeUrl(href)
    // 不安全的 URL：**只留文字**。留个死链比留个能执行的链接好 ✓
    if (!url) return label
    const external = /^https?:\/\//i.test(url)
    return `<a href="${url}"${external ? ' target="_blank" rel="noopener noreferrer"' : ''}>${label}</a>`
  })
  return out.replace(/\u0000(\d+)\u0000/g, (_match, index: string) => codes[Number(index)] ?? '')
}

/**
 * 渲染成 HTML 字符串（调用方用 `v-html`）。
 *
 * 支持的语法就这几样：标题、有序/无序列表、围栏代码块、引用、分隔线、粗体、斜体、
 * 行内代码、链接。**够用就行** —— 少一个特性只会显示成纯文本，不会渲染成错的 ✓。
 */
export function renderMarkdown(source: string): string {
  const text = escapeHtml(String(source ?? '')).replace(/\r\n?/g, '\n')
  const lines = text.split('\n')
  const out: string[] = []
  let list: 'ul' | 'ol' | null = null

  const closeList = () => {
    if (list) {
      out.push(`</${list}>`)
      list = null
    }
  }

  let i = 0
  while (i < lines.length) {
    const line = lines[i]

    if (/^\s*```/.test(line)) {
      closeList()
      const body: string[] = []
      i += 1
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        body.push(lines[i])
        i += 1
      }
      i += 1 // 跳过收尾围栏（没闭合就吃到文末，不报错 ✓）
      out.push(`<pre><code>${body.join('\n')}</code></pre>`)
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      closeList()
      const level = heading[1].length
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`)
      i += 1
      continue
    }

    if (/^(?:---+|\*\*\*+|___+)\s*$/.test(line.trim())) {
      closeList()
      out.push('<hr />')
      i += 1
      continue
    }

    // 引用：正文已经过 escapeHtml ⇒ `>` 变成 `&gt;`，这里按转义后的形式匹配 ✓
    const quote = /^&gt;\s?(.*)$/.exec(line)
    if (quote) {
      closeList()
      out.push(`<blockquote>${inline(quote[1])}</blockquote>`)
      i += 1
      continue
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line)
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (bullet || ordered) {
      const want: 'ul' | 'ol' = bullet ? 'ul' : 'ol'
      if (list !== want) {
        closeList()
        out.push(`<${want}>`)
        list = want
      }
      out.push(`<li>${inline((bullet ?? ordered)![1])}</li>`)
      i += 1
      continue
    }

    if (!line.trim()) {
      closeList()
      i += 1
      continue
    }

    closeList()
    out.push(`<p>${inline(line)}</p>`)
    i += 1
  }

  closeList()
  return out.join('')
}

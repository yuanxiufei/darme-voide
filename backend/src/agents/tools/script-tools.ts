/**
 * 剧本改写 Agent 工具
 * 工厂函数模式 — 注入 episodeId，工具不再需要 LLM 传递 ID
 */
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { db, schema } from '../../db/index.js'
import { eq } from 'drizzle-orm'
import { now } from '../../utils/response.js'
import { sliceLongText } from '../../utils/text-slice.js'

export function createScriptTools(episodeId: number) {
  const readEpisodeScript = createTool({
    id: 'read_episode_script',
    description: 'Read the script content of the current episode.',
    inputSchema: z.object({}),
    execute: async () => {
      const [ep] = db.select().from(schema.episodes)
        .where(eq(schema.episodes.id, episodeId)).all()
      if (!ep) return { error: `Episode not found (id=${episodeId})` }
      const content = ep.content || ep.scriptContent
      if (!content) return { error: `Episode has no content (id=${episodeId})` }
      const sliced = sliceLongText(content)
      return {
        content: sliced.text,
        word_count: content.length,
        episode_id: episodeId,
        truncated: sliced.truncated,
        total_chars: sliced.total_chars,
      }
    },
  })

  const rewriteToScreenplay = createTool({
    id: 'rewrite_to_screenplay',
    description: 'Read the original content for AI rewriting. Returns the source text with formatting instructions.',
    inputSchema: z.object({
      instructions: z.string().optional().describe('Additional rewrite instructions'),
    }),
    execute: async ({ instructions }) => {
      const [ep] = db.select().from(schema.episodes)
        .where(eq(schema.episodes.id, episodeId)).all()
      if (!ep) return { error: `Episode not found` }
      const source = ep.content || ep.scriptContent
      if (!source) return { error: `Episode has no content to rewrite` }
      const sliced = sliceLongText(source)

      return {
        source_content: sliced.text,
        truncated: sliced.truncated,
        total_chars: sliced.total_chars,
        instruction: `请将以下内容改写为格式化剧本。

输出硬性要求：
- 只输出剧本正文，禁止输出任何分析、评论、解读、剧情预测或总结。
- 对白必须逐角色分行：每个角色说的话单独占一行，一人一句换行，禁止把两个及以上人物的对话写在同一段。
- 对白用第一人称直接引语，禁止第三人称转述（如"小雪说……""他说道……"）。

说话人划分铁律（谁说的话，最高优先级）：
- 每一句台词都必须清楚标注是谁说的，禁止多人台词混在一段。
- 一句一行、一行一人。
- 叙述体拆解归因：原文是叙述体（如"何长青叹了口气说……林雪抬头道……"）时，必须拆成两行各自归到正确角色名下，绝不能整段照抄。
- 归因准确：根据上下文判断说话人，不能张冠李戴。
- 示例：【错误】"何长青叹了口气，说自己也不知道。林雪轻声说别担心。"【正确】"何长青：（叹气）我也不知道。"换行"林雪：（轻声）别担心，我们一起想办法。"

格式规范：
- 场景头：## S编号 | 内景/外景 · 地点 | 时间段
- 动作描写：自然段落，不包含镜头语言
- 对白：角色名：（状态/表情）台词内容
- 每个场景 30-60 秒内容

${instructions || ''}

【原始内容】
${sliced.text}`,
      }
    },
  })

  const saveScript = createTool({
    id: 'save_script',
    description: 'Save the rewritten screenplay content to the current episode.',
    inputSchema: z.object({
      content: z.string().describe('The formatted screenplay content to save'),
    }),
    execute: async ({ content }) => {
      db.update(schema.episodes)
        .set({ scriptContent: content, updatedAt: now() })
        .where(eq(schema.episodes.id, episodeId))
        .run()
      // 剧本内容指纹门禁：改写落库后重算指纹
      await import('../../services/script-fingerprint.js').then(m => m.refreshEpisodeScriptHash(episodeId))
      return { message: `Script saved`, word_count: content.length }
    },
  })

  return { readEpisodeScript, rewriteToScreenplay, saveScript }
}

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
import { SCREENPLAY_FORMAT_RULES } from '../../shared/prompt-blocks.js'

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

${SCREENPLAY_FORMAT_RULES}

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

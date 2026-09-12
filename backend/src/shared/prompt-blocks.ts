/**
 * 可复用提示词规范块（Prompt Blocks）
 * ============================================================
 * 用途：当同一套规范需要出现在多个位置时（Agent 的 system prompt、工具返回值的
 * instruction、skill 文档），**只在本文件定义一次**，各处引用，杜绝副本漂移。
 *
 * 动因：格式化剧本规范此前在 3 处各存一份 ——
 *   1. `agents/index.ts` DEFAULT_PROMPTS.script_rewriter
 *   2. `agents/tools/script-tools.ts` rewrite_to_screenplay 的 instruction
 *   3. `skills/script_rewriter/SKILL.md`
 * 三份内容高度重合且已出现表述差异（示例、措辞、工具名不一致）→ 统一收口到此处。
 *
 * 约定：
 *   - 命名 `<领域>_<用途>`，如 SCREENPLAY_FORMAT_RULES；
 *   - 块内只写**跨场景稳定**的规范（格式/铁律/范例）；
 *     一次性或场景专属的措辞留在调用处；
 *   - 在注释里标注「引用方」，便于改动前评估影响面。
 * ============================================================
 */

/**
 * 格式化剧本规范（硬性要求 + 说话人铁律 + 格式 + 对标示范）
 *
 * 引用方：`DEFAULT_PROMPTS.script_rewriter`、`script-tools.rewrite_to_screenplay`
 */
export const SCREENPLAY_FORMAT_RULES = `输出硬性要求（必须严格遵守）：
1. 最终输出只能是「格式化剧本正文」，禁止输出任何分析、评论、解读、剧情预测、总结或客套话。绝不要出现"人物性格剖析""情节张力""后续发展预测"之类的内容。
2. 对白必须逐角色分行：每个角色说的话单独占一行，格式为「角色名：（状态/表情）台词内容」。一人一句就换一行，禁止把两个及以上人物的对话写在同一段里。
3. 对白一律用第一人称直接引语，禁止用第三人称转述（例如禁止"小雪说……""他说道……""云姐提醒……"这类写法），必须是角色亲口说出的台词。
4. 旁白、内心独白同样单独成行，用「旁白：内容」或「角色名（内心）：内容」标注。

说话人划分铁律（谁说的话，最高优先级）：
- 每一句台词都必须清楚标注是谁说的，绝不能出现"这句话不知道是谁说的"或多人台词混在一段的情况。
- 一句一行、一行一人：一行只允许一个角色说话。
- 叙述体拆解归因：原文是叙述体（如"何长青叹了口气说……林雪抬头道……"）时，必须拆成两行，各自归到正确角色名下，绝不能整段照抄。
- 归因准确：根据上下文判断说话人，不能张冠李戴；同一角色连续说多句也每行都标名字。
- 示例：【错误】"何长青叹了口气，说自己也不知道。林雪轻声说别担心。"【正确】"何长青：（叹气）我也不知道。"换行"林雪：（轻声）别担心，我们一起想办法。"

格式化剧本格式：
- 场景头：## S编号 | 内景/外景 · 地点 | 时间段（编号 S01/S02 递增）
- 动作描写：用（...）括号包裹，成段或独占一行，不包含镜头语言
- 对白：角色名：（状态/表情）台词内容，一人一行
- 旁白/内心独白单独成行：旁白：内容
- 每个场景 30-60 秒内容

示范（格式与质量对标）：
## S01 | 外景 · 东家城堡花园广场 | 白天

（一辆超豪华轿车驶入城堡般的巨大花园，在喷泉环绕的广场前停下。）

云姐：（笑靥如花，迎上前）小雪，一路辛苦啦。
小雪：（热情拥抱）云姐，我可想死你了！

（小雪目光扫过广场，落在喷泉旁盘坐的少年凌云身上。）

小雪：（眨着灵动的眼睛，好奇）云姐，他是谁呀？怎么坐在你家门外，太阳这么大，不怕晒黑么？
云姐：（压低声音）他就说了句——「在下凌云，前来拜会！」

## S02 | 外景 · 东家城堡花园广场 | 白天

小雪：（跃跃欲试）那简单！让本小姐试他一试就知道。
云姐：（连忙阻止）小雪，别！大长老吩咐过，不让我们打扰他。`

/**
 * 角色立绘提示词结构骨架
 *
 * 引用方：`DEFAULT_PROMPTS.extractor`
 * 说明：原为 `skills/grid_prompt_generator/reference/character-prompt.md`，
 * 因加载器只读 SKILL.md、reference/ 对 Agent 不可达，故上收到此处统一注入。
 */
export const IMAGE_PROMPT_TEMPLATE_CHARACTER = `结构骨架（英文，单段）：
A [gender] [age] character, [name], [body type], [facial features]. [hair description]. [clothing details]. [pose and expression]. Background: [simple/gradient].
Style: [art style], high quality, detailed, character concept art.

示例：
A young woman in her early 20s, Li Mei, slender build, delicate oval face with bright almond eyes. Long black hair flowing over shoulders with subtle waves. Wearing a vintage blue qipao with floral embroidery. Standing confidently with a slight smile, one hand on hip. Background: soft gradient.
Style: cinematic anime, high quality, detailed, character concept art.`

/**
 * 场景背景图提示词结构骨架
 *
 * 引用方：`DEFAULT_PROMPTS.extractor`
 */
export const IMAGE_PROMPT_TEMPLATE_SCENE = `结构骨架（英文，单段，纯背景无人）：
A cinematic [style] pure background scene depicting [location] at [time]. The scene shows [environment details, architecture, objects, lighting]. No characters, no people, no figures.
Style: [art style], rich details, high quality, atmospheric lighting. Mood: [mood description].

示例：
A cinematic anime-style pure background scene depicting a traditional Japanese courtyard at dusk. The scene shows wooden corridors surrounding a zen garden with raked white gravel, a single cherry blossom tree with petals falling, stone lanterns casting warm light, sliding shoji doors partially open. No characters.
Style: ghibli, rich details, high quality, warm golden hour lighting. Mood: peaceful, nostalgic, serene.`

/**
 * 单镜画面提示词结构骨架（分镜首帧/主图用）
 *
 * 引用方：`DEFAULT_PROMPTS.grid_prompt_generator`（宫格）与分镜静帧链路
 */
export const IMAGE_PROMPT_TEMPLATE_SHOT = `结构骨架（英文，多行）：
[Shot type] shot, [camera angle], [art style].
[Character(s) description and action].
[Environment and setting].
[Lighting and atmosphere].
Style: cinematic, high quality, [additional style tags].

示例：
Medium close-up shot, slightly low angle, cinematic anime style. A young man with messy dark hair grips a rusty wrench tightly, his face illuminated by a single overhead fluorescent light, sweat visible on his forehead, determined expression. Dimly lit repair shop interior with tool boards and scattered parts in background. Harsh overhead lighting creating strong shadows, blue-grey color palette.
Style: cinematic, high quality, dramatic lighting, film grain.`

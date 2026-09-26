import json, collections, statistics, re, sys

# ⚠️ 2026-09-25 统一成 `reconfigure` ✓：原写法 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`
#    **丢掉原 wrapper** ✗ ⇒ 与调用方（`run_all.py` / `check_all.py` 等）的缓冲与刷新顺序不一致 ✓✗，
#    而且守卫 `check_cli_encoding.py` 认不出它 ⇒ 会被误报成「裸跑会崩」✓✗（本次就是它报的 ✓）。
#    语义等价 ✓ 但更稳：保留原 wrapper 与 line buffering ✓，还能顺带设 `errors="replace"` ✓
#    （原写法遇 locale 外字符是**抛异常** ✗ 而这里不会 ✗）。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os as _os, pathlib as _pathlib
P = (_os.environ.get('SEEDANCE2_CORPUS')
     or str(_pathlib.Path(__file__).resolve().parents[4]  # 本文件在 backend-py/app/scripts/corpus/ 下（2026-09-15 从 scripts/corpus/ 搬来）  # depth-adjusted-to-app
            / 'data' / 'prompt-corpus' / 'seedance2' / 'metadata.jsonl'))

# 按需开关（**不设 = 输出与历史逐字一致**）：
#   SEEDANCE2_LIMIT=N   只收集前 N 条有效记录（调试 / 抽样时不必全量解析 37 MB）
LIMIT = int(_os.environ.get('SEEDANCE2_LIMIT') or 0)


# Chinese prompt = i18n.zh.p ; fall back to raw_p
def zh_of(r):
    i = r.get('i18n') or {}
    z = i.get('zh')
    if isinstance(z, dict) and isinstance(z.get('p'), str) and z['p'].strip():
        return z['p']
    rp = r.get('raw_p')
    return rp if isinstance(rp, str) and rp.strip() else None


# 边读边投影：**不再保留原始对象树**（实测内存峰值 75.4 MB → 14.1 MB），
# 因为后续统计只用到下面这 6 个字段。语义与原「先 recs 再投影」完全等价。
rows = []
with open(P, encoding='utf-8') as f:
    for line in f:
        if LIMIT and len(rows) >= LIMIT:
            break
        if not line.strip():
            continue
        r = json.loads(line)
        p = zh_of(r)
        if not p:
            continue
        z = (r.get('i18n') or {}).get('zh') or {}
        rows.append({
            'cat': r.get('category'),
            'p': p,
            't': z.get('t'),
            'tags': z.get('tags') or [],
            'dur': (r.get('spec') or {}).get('duration'),
            'model': (r.get('model_info') or {}).get('name'),
        })

print('rows with Chinese prompt =', len(rows))
lens = [len(x['p']) for x in rows]
print('zh len min/median/mean/max =', min(lens), int(statistics.median(lens)),
      int(statistics.mean(lens)), max(lens))
cjk = sum(1 for x in rows if re.search(r'[\u4e00-\u9fff]', x['p']))
print(f'actually contains CJK: {cjk} ({cjk*100//len(rows)}%)')
print()

# ---------- 1. 时间码分段 ----------
pats = {
    '分段标题 X-Y秒：': r'(?m)^\s*\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?\s*秒?\s*[：:|]',
    '文内 X-Y秒': r'\d+\s*[-–—~]\s*\d+\s*秒',
    '镜头N / 分镜N': r'(?:镜头|分镜|画面|第)\s*\d+',
    '⏱️ 标记': r'⏱',
    '段落标题【】': r'【[^】]{2,12}】',
    '章节 Chapter': r'(?i)chapter\s*\d+',
    '图片引用 图N/ImageN': r'(?:图|图片|Image)\s*\d+',
}
print('=== 结构特征命中率 ===')
for name, pat in pats.items():
    rx = re.compile(pat)
    c = sum(1 for x in rows if rx.search(x['p']))
    print(f'  {name:22s} {c:5d}  ({c*100//len(rows)}%)')
print()

# ---------- 2. 关键词家族命中率 ----------
groups = {
    '镜头运动': ['推近', '推进', '拉远', '拉近', '跟随', '手持', '环绕', '摇镜', '平移', '升降',
                 '变焦', '旋转', '俯冲', '仰拍', '俯拍', '慢镜头', '升格', '降格', '定格', '一镜到底'],
    '景别': ['极致特写', '大特写', '特写', '近景', '中景', '全景', '远景', '广角', '超广角', '微距', '中近景'],
    '光线': ['逆光', '侧光', '顶光', '底光', '柔光', '硬光', '暖光', '冷光', '暖金', '金色阳光',
             '蓝调', '霓虹', '低调', '高调', '轮廓光', '斑驳'],
    '画质收口': ['8K', '4K', '胶片颗粒', '浅景深', '虚化', '电影级', '超写实', '锐利', 'HDR', '景深'],
    '一致性': ['一致', '无变形', '无漂移', '无伪影', '百分百还原', '保持不', '全程保持', '身份', '杜绝美化', '同一人物'],
    '负面约束': ['无水印', '无字幕', '无文字', '不要文字', '禁止', '避免', '无机器人感', '无夸张', '不出现'],
    '音频/对白': ['音效', '对白', '同期声', '配乐', '旁白', '低语', '环境音', 'BGM', '口型', '唇部同步', '呼吸'],
    '表演细节': ['微表情', '瞳孔', '喉结', '呼吸', '颤抖', '睫毛', '嘴角', '眼眶', '泪', '肢体'],
    '时长': ['15秒', '10秒', '5秒', '30秒', '60秒', '3秒'],
}
print('=== 关键词家族命中率（含该家族任一词的提示词占比）===')
for g, ws in groups.items():
    hits = [w for w in ws if any(w in x['p'] for x in rows)]
    c = sum(1 for x in rows if any(w in x['p'] for w in ws))
    print(f'  {g:10s} {c:5d} ({c*100//len(rows):2d}%)  命中词: {", ".join(hits)}')
print()

# ---------- 3. 单词频次（各家族 top 词） ----------
print('=== 各家族词频 TOP ===')
for g, ws in groups.items():
    cnt = []
    for w in ws:
        c = sum(x['p'].count(w) for x in rows)
        if c:
            cnt.append((c, w))
    cnt.sort(reverse=True)
    print(f'  [{g}] ' + '  '.join(f'{w}={c}' for c, w in cnt[:12]))
print()

# ---------- 4. tags 频次 ----------
tc = collections.Counter()
for x in rows:
    tg = x['tags']
    if isinstance(tg, list):
        for t in tg:
            if isinstance(t, str) and t.strip():
                tc[t.strip()] += 1
print('=== zh tags 总数 =', len(tc), ' TOP60 ===')
print('  ', '  '.join(f'{t}({c})' for t, c in tc.most_common(60)))
print()

# ---------- 5. 分类 x 时长 ----------
print('=== category x 数量 ===')
print('  ', collections.Counter(x['cat'] for x in rows).most_common())
print()
print('=== model ===')
print('  ', collections.Counter(x['model'] for x in rows).most_common())
print()

# ---------- 6. 有效时长分布 ----------
ds = [x['dur'] for x in rows if isinstance(x['dur'], (int, float)) and 0 < x['dur'] <= 300]
print(f'=== 有效 duration (0<d<=300) n={len(ds)} ===')
if ds:
    b = collections.Counter()
    for d in ds:
        b[int(d // 5) * 5] += 1
    for k in sorted(b):
        print(f'  {k:3d}-{k+4:3d}s : {b[k]}')
print()

# ---------- 7. 一条 Entertainment 中文样例 ----------
print('=== 样例：Entertainment 中等长度中文提示词 ===')
mid = [x for x in rows if x['cat'] == 'Entertainment' and 700 <= len(x['p']) <= 1100]
if mid:
    s = mid[len(mid) // 2]
    print('title =', s['t'])
    print('tags  =', s['tags'])
    print('len   =', len(s['p']))
    print('-' * 60)
    print(s['p'])

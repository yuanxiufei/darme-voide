import json, collections, statistics, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os as _os, pathlib as _pathlib
P = (_os.environ.get('SEEDANCE2_CORPUS')
     or str(_pathlib.Path(__file__).resolve().parents[2]
            / 'data' / 'prompt-corpus' / 'seedance2' / 'metadata.jsonl'))
# 按需开关（**不设 = 输出与历史逐字一致**）：SEEDANCE2_LIMIT=N 只读前 N 条。
# 本脚本做 i18n / spec 深挖，要保留原始对象（含 recs[0] 样本），故不做投影。
LIMIT = int(_os.environ.get('SEEDANCE2_LIMIT') or 0)

recs = []
with open(P, encoding='utf-8') as f:
    for l in f:
        if LIMIT and len(recs) >= LIMIT:
            break
        if l.strip():
            recs.append(json.loads(l))

print('TOTAL =', len(recs))
print()

# ---- i18n languages ----
langs = collections.Counter()
for r in recs:
    for k in (r.get('i18n') or {}):
        langs[k] += 1
print('i18n langs:', langs.most_common())
print()

# ---- raw_p ----
raw_nonempty = sum(1 for r in recs if isinstance(r.get('raw_p'), str) and r['raw_p'].strip())
print('raw_p non-empty:', raw_nonempty)
print('raw_p sample  :', json.dumps(recs[0].get('raw_p'), ensure_ascii=False)[:1200])
print()

for lg in (r.get('i18n') or {}).keys():
    v = recs[0]['i18n'][lg]
    print(f'i18n.{lg} keys =', list(v.keys()) if isinstance(v, dict) else type(v))
    print(f'i18n.{lg} sample =', json.dumps(v, ensure_ascii=False)[:900])
    print()

# ---- category / platform / featured ----
print('category :', collections.Counter(r.get('category') for r in recs).most_common())
print()
print('platform :', collections.Counter(r.get('platform') for r in recs).most_common(20))
print()
print('featured :', collections.Counter(r.get('is_featured') for r in recs).most_common())
print()

# ---- spec ----
def nums(key):
    out = []
    for r in recs:
        s = r.get('spec')
        if isinstance(s, dict):
            v = s.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
    return out

for k in ('duration', 'width', 'height', 'ratio'):
    a = nums(k)
    if a:
        print(f'spec.{k}: n={len(a)} min={min(a)} median={statistics.median(a)} max={max(a)}')
print()
print('safety_rating:', collections.Counter(
    (r.get('spec') or {}).get('safety_rating') for r in recs).most_common())
print()

# ---- model_info ----
print('model_info:', collections.Counter(
    json.dumps(r.get('model_info'), ensure_ascii=False) for r in recs).most_common(5))
print()

# ---- prompt text stats (use en.p, fall back to any lang) ----
def prompt_of(r):
    i = r.get('i18n') or {}
    for lg in ('en', 'zh', 'ja'):
        if isinstance(i.get(lg), dict) and isinstance(i[lg].get('p'), str) and i[lg]['p'].strip():
            return i[lg]['p']
    for v in i.values():
        if isinstance(v, dict) and isinstance(v.get('p'), str) and v['p'].strip():
            return v['p']
    return None

ps = [prompt_of(r) for r in recs]
ps = [p for p in ps if p]
print('prompts found:', len(ps))
lens = [len(p) for p in ps]
print('len min/median/mean/max =', min(lens), int(statistics.median(lens)),
      int(statistics.mean(lens)), max(lens))
zh = sum(1 for p in ps if re.search(r'[\u4e00-\u9fff]', p))
print(f'contains-CJK: {zh} ({zh*100//len(ps)}%)')
print()

# ---- time-code segmentation detection ----
TC = re.compile(r'(?m)^\s*\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?\s*(?:seconds?|s\b|秒)', re.I)
TC2 = re.compile(r'\d+\s*[-–—]\s*\d+\s*(?:seconds?|秒)', re.I)
seg = sum(1 for p in ps if TC.search(p) or TC2.search(p))
print(f'time-code segmented prompts: {seg} ({seg*100//len(ps)}%)')
print()

# show one full prompt to inspect structure
longest = max(ps, key=len)
print('=== LONGEST PROMPT (len=%d) ===' % len(longest))
print(longest[:4000])

import json, collections, statistics, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os as _os, pathlib as _pathlib
P = (_os.environ.get('SEEDANCE2_CORPUS')
     or str(_pathlib.Path(__file__).resolve().parents[2]
            / 'data' / 'prompt-corpus' / 'seedance2' / 'metadata.jsonl'))

recs = []
with open(P, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print('TOTAL =', len(recs))
print()
print('--- key frequencies ---')
kc = collections.Counter(k for r in recs for k in r)
for k, v in kc.most_common():
    print(f'  {k:24s} {v}')
print()
print('--- sample record (full, truncated) ---')
print(json.dumps(recs[0], ensure_ascii=False, indent=2)[:3000])
print()

# locate prompt-ish field
cand = [k for k in kc if re.search(r'prompt|text|desc|caption', k, re.I)]
print('prompt-ish fields =', cand)
print()


def analyze(field):
    vals = [r.get(field) for r in recs]
    vals = [v for v in vals if isinstance(v, str) and v.strip()]
    if not vals:
        print(f'[{field}] no string values')
        return
    lens = [len(v) for v in vals]
    print(f'--- field "{field}": n={len(vals)} ---')
    print('  len  min/median/mean/max =', min(lens), int(statistics.median(lens)),
          int(statistics.mean(lens)), max(lens))
    zh = sum(1 for v in vals if re.search(r'[\u4e00-\u9fff]', v))
    print(f'  contains-CJK = {zh} ({zh*100//len(vals)}%)')
    enonly = sum(1 for v in vals if not re.search(r'[\u4e00-\u9fff]', v))
    print(f'  pure-non-CJK = {enonly} ({enonly*100//len(vals)}%)')
    # length buckets
    buckets = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 10**9)]
    print('  length buckets:')
    for lo, hi in buckets:
        c = sum(1 for l in lens if lo <= l < hi)
        print(f'    {lo:5d}-{hi if hi<10**9 else "inf":>5}: {c}')
    print()


for c in cand:
    analyze(c)

# also analyze any nested/metadata structures
print('--- nested structures ---')
for k in kc:
    v = recs[0].get(k)
    if isinstance(v, (dict, list)):
        print(f'  {k}: {type(v).__name__} =', json.dumps(v, ensure_ascii=False)[:300])

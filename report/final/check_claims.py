"""
Verifies main.tex against claims.py and writes claims_ledger.md.   Usage: python report/final/check_claims.py
Exit code 1 if any tag is unregistered or any registered string is missing from its file.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from claims import CLAIMS

norm = lambda t: re.sub(r'\s+', ' ', t)
tex = open(os.path.join(HERE, 'main.tex')).read()
used = {}
for m in re.finditer(r'\\src\{([^}]*)\}', tex):
    for t in (x.strip() for x in m.group(1).split(',')):
        used[t] = used.get(t, 0) + 1
problems, rows = [], []
for t in sorted(used):
    if t not in CLAIMS:
        problems.append(f'UNREGISTERED tag {t}'); continue
    for path, strings in CLAIMS[t]:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            problems.append(f'{t}: missing file {path}'); rows.append((t, path, '-', 'MISSING FILE')); continue
        text = norm(open(full, errors='replace').read())
        for st in strings:
            ok = norm(st) in text
            rows.append((t, path, st, 'verified' if ok else 'NOT FOUND'))
            if not ok:
                problems.append(f'{t}: "{st}" not found in {path}')
unused = sorted(set(CLAIMS) - set(used))
with open(os.path.join(HERE, 'claims_ledger.md'), 'w') as f:
    f.write('# Claims ledger\n\nEvery `\\src{tag}` in `main.tex` maps to the files and exact strings below. '
            '"verified" = the string occurs verbatim (whitespace-normalised) in the file.\n\n| tag | uses | source file | string | status |\n|---|---|---|---|---|\n')
    for t, path, st, status in rows:
        f.write(f'| {t} | {used[t]} | `{path}` | {st.replace("|", "/")} | {status} |\n')
n_ok = sum(1 for r in rows if r[3] == 'verified')
print(f'{len(used)} tags used ({sum(used.values())} citations); {len(rows)} string checks, {n_ok} verified; {len(unused)} registered but unused')
for p in problems: print('PROBLEM:', p)
if unused: print('unused tags:', ' '.join(unused))
sys.exit(1 if problems else 0)

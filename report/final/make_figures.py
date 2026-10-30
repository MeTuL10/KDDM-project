"""
Builds the new figures for the final report from logged numbers (no retyped experiment values where a log exists).
Run from anywhere:  python report/final/make_figures.py
  fig_leaderboard.pdf : public leaderboard score of every submission we made, by track
  fig_learning.pdf    : (a) learning curve, (b) cost of older training data  [expt-5/outputs/learning_curve.log]
  fig_errors.pdf      : (a) out-of-fold RMSE by flat type (Metul), (b) where the squared error sits (Aayush, root pipeline)
"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'figures')
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
C = {'anchor': '#1f5fa8', 'history': '#d1691b', 'auto': '#2a8a5e', 'root': '#777777'}

# ---- leaderboard (public score; sources: expt-5/PROGRESS_REPORT.md leaderboard log, report/beina/model_experiment.md 6.4,
#      report/punyaa/README.md headline results, expt-2/PROGRESS_REPORT.md, progress-report.md, Metul's score as supplied) ----
rows = [  # (label, score, family)
    ('Root pipeline, tuned (A)', 451.42, 'root'),
    ('V1: tri-model, no block encodings (A)', 443.000, 'anchor'),
    ('V3: deeper trees (A)', 434.053, 'anchor'),
    ('V2 improved: premium + bag (A)', 434.02878, 'anchor'),
    ('V2 honest anchor = FINAL (A)', 433.65480, 'anchor'),
    ('V2 CatBoost-heavy weights (A)', 433.63052, 'anchor'),
    ('V2 (A)', 433.60708, 'anchor'),
    ('Honest anchor blend (B)', 433.89670, 'anchor'),
    ('Published-median blend (B)', 433.36655, 'anchor'),
    ('Rent-history blend v2 (P)', 434.81, 'history'),
    ('Rent-history blend v3 (P)', 434.70, 'history'),
    ('AutoGluon, 2 residual layers (M)', 436.28615, 'auto'),
]
rows.sort(key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(3.45, 2.85))
for i, (lab, sc, fam) in enumerate(rows):
    ax.plot([sc], [i], 'o', color=C[fam], ms=4.5, mec='black', mew=0.4 if 'FINAL' in lab else 0, zorder=3)
    ax.text(sc + (0.55 if sc < 440 else -0.55), i, f'{sc:.2f}', va='center', ha='left' if sc < 440 else 'right', fontsize=6)
ax.set_ylim(-1.3, len(rows) - 0.5); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
ax.set_xlim(431.5, 463); ax.set_xlabel('Public leaderboard RMSE (SGD)')
ax.axvspan(433.3, 436.4, color='#e8e8e8', zorder=0)
ax.text(434.85, -0.95, 'plateau', ha='center', fontsize=6, color='#555555')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=C[k], ms=4, label=l) for k, l in
                   [('anchor', 'anchor + block encodings'), ('history', 'causal rent history'), ('auto', 'AutoGluon stack'), ('root', 'root pipeline')]],
          loc='center right', bbox_to_anchor=(1.0, 0.52), fontsize=5.5, frameon=False, handletextpad=0.2, borderaxespad=0.2)
ax.grid(axis='x', lw=0.3, alpha=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'fig_leaderboard.pdf')); plt.close(fig)

# ---- learning curve / recency (parsed from the log) ----
lg = open(os.path.join(ROOT, 'expt-5', 'outputs', 'learning_curve.log')).read()
curve = [(int(n.replace(',', '')), float(r)) for n, r in re.findall(r'%\s+([\d,]+) rows\s+RMSE\s+([\d.]+)', lg)]
rec = [(int(g), float(c)) for g, c in re.findall(r'gap\s+(\d+) mo.*?cost of being older:\s+([+-][\d.]+)', lg)]
assert len(curve) == 5 and len(rec) == 6, (curve, rec)
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={'width_ratios': [1.15, 1]})
a.plot([n / 1000 for n, _ in curve], [r for _, r in curve], 'o-', color=C['anchor'], ms=3.2, lw=1)
a.set_xscale('log'); a.minorticks_off(); a.set_xticks([13, 33, 66, 133]); a.set_xticklabels(['13', '33', '66', '133'])
a.set_xlabel('training rows (thousands)', fontsize=6.5); a.set_ylabel('RMSE'); a.set_title('(a) learning curve', fontsize=7)
b.bar([str(g) for g, _ in rec], [c for _, c in rec], color=[C['history'] if c > 0 else '#999999' for _, c in rec], width=0.65)
b.axhline(0, color='black', lw=0.5); b.set_xlabel('months older', fontsize=6.5); b.set_ylabel('RMSE cost', fontsize=6.5)
b.set_title('(b) cost of older data', fontsize=7)
for ax_ in (a, b):
    ax_.grid(axis='y', lw=0.3, alpha=0.6); ax_.set_axisbelow(True)
    for s in ('top', 'right'): ax_.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'fig_learning.pdf')); plt.close(fig)
print('figures written:', os.listdir(OUT))

# ---- error concentration ----
# (a) report/metul/eda.md section H: out-of-fold RMSE of executive 720, five-room 653, three-room 542
seg = [('3-room', 542), ('5-room', 653), ('executive', 720)]
# (b) expt-5/PROGRESS_REPORT.md "Error decomposition": 1,096 of 16,961 rows (rent <= 2,100) with RMSE 1,090.19; rest RMSE 369.94
n_low, n_all, rm_low, rm_rest = 1096, 16961, 1090.19, 369.94
share_rows = n_low / n_all
mse_low, mse_rest = share_rows * rm_low ** 2, (1 - share_rows) * rm_rest ** 2
share_err = mse_low / (mse_low + mse_rest)
assert abs((mse_low + mse_rest) ** 0.5 - 452.56) < 0.05, 'decomposition must reproduce the overall RMSE 452.56'
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={'width_ratios': [1, 1]})
a.bar([n for n, _ in seg], [v for _, v in seg], color=[C['anchor'], C['history'], C['history']], width=0.6)
for i, (_, v) in enumerate(seg): a.text(i, v + 8, str(v), ha='center', fontsize=6)
a.set_ylim(0, 820); a.set_ylabel('out-of-fold RMSE', fontsize=6.5); a.set_title('(a) by flat type (M)', fontsize=7)
b.bar(['rows', 'squared\nerror'], [100 * share_rows, 100 * share_err], color=['#999999', C['history']], width=0.55)
for i, v in enumerate([100 * share_rows, 100 * share_err]): b.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=6)
b.set_ylim(0, 48); b.set_ylabel('share of validation set (%)', fontsize=6.5); b.set_title('(b) rent $\\leq$ 2,100 (A)', fontsize=7)
for ax_ in (a, b):
    ax_.grid(axis='y', lw=0.3, alpha=0.6); ax_.set_axisbelow(True)
    for sp_ in ('top', 'right'): ax_.spines[sp_].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'fig_errors.pdf')); plt.close(fig)
print(f'error concentration: {100 * share_rows:.2f}% of rows carry {100 * share_err:.1f}% of squared error')

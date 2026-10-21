"""Build figures (figures/*.pdf) and LaTeX tables (results/tab_*.tex) from results/*. Prints quoted statistics."""
import os, sys, json, math
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'figure.dpi': 200, 'font.family': 'serif'})
S = json.load(open('results/bench_summary.json'))
POLS = ['Search', 'RAG', 'Mandate', 'Mandate+Esc', 'Rules', 'AlwaysEsc']
def f(m, h, pct=True):
    return f"{100*m:.1f}$\\pm${100*h:.1f}" if pct else f"{m:.2f}$\\pm${h:.2f}"
L = ['\\begin{tabular}{lcccccc}', '\\toprule',
     'Policy & Wrong authority & Unsupported closure & Escalation rate & Esc.\\ sensitivity & Esc.\\ precision & Unnecessary esc.\\\\', '\\midrule']
for p in POLS:
    s = S[p]
    esc = s['escalation_rate'][0] > 0
    L.append(f"{p} & {f(*s['wrong_authority'])} & {f(*s['unsupported_closure'])} & {f(*s['escalation_rate'])} & "
             + (f"{f(*s['esc_sensitivity'])} & {f(*s['esc_precision'])} & {f(*s['unnecessary_esc'])}" if esc else '-- & -- & --') + '\\\\')
L += ['\\bottomrule', '\\end{tabular}']
open('results/tab_bench.tex', 'w').write('\n'.join(L) + '\n')

# sensitivity figure
ds = pd.read_csv('results/bench_sensitivity.csv'); ds = ds[ds['family'].isna()]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
for a, (k, lab) in zip(ax, [('cfg_p_register_stale', 'Register staleness'), ('cfg_p_contested', 'Contested-mandate prevalence'),
                            ('cfg_detect', 'Check detection probability')]):
    d = ds[ds[k].notna()]
    for p, mk in zip(['RAG', 'Mandate', 'Mandate+Esc', 'Rules'], 'osD^'):
        g = d[d.policy == p].groupby(k)['unsupported_closure'].mean()
        a.plot(g.index, 100 * g.values, marker=mk, ms=3, label=p)
    a.set_xlabel(lab); a.grid(alpha=.3)
ax[0].set_ylabel('Unsupported closure (%)'); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig('figures/sensitivity.pdf'); plt.close()

# capacity figure
d = pd.read_csv('results/des_runs.csv')
base = d[d.burst.isna() & d.absence.isna() & d.priority_escalations.isna()]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for a, pooled in zip(ax, [False, True]):
    for p, mk in zip(['RAG', 'Mandate+Esc', 'Rules', 'AlwaysEsc'], 'oDs^'):
        g = base[(base.policy == p) & (base.pooled == pooled)].groupby('arrival_per_hour')['p95_hours'].mean()
        a.plot(g.index, g.values, marker=mk, ms=3, label=p)
    a.set_yscale('log'); a.set_xlabel('Arrivals per working hour'); a.grid(alpha=.3, which='both')
    a.set_title('Pooled reviewers' if pooled else 'Departmental queues')
ax[0].set_ylabel('95th-pct. time to closure (h)'); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig('figures/capacity.pdf'); plt.close()

# stress table at default load (6/h, departmental)
rows = []
def agg(sub, m):
    x = sub[m].values; return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x))
T = ['\\begin{tabular}{llccc}', '\\toprule', 'Condition & Policy & Median (h) & 95th pct.\\ (h) & SLA $>$40\\,h (\\%)\\\\', '\\midrule']
conds = [('Default', base[(base.arrival_per_hour == 6) & (base.pooled == False)]),
         ('Demand bursts', d[d.burst == True]), ('Reviewer absence', d[d.absence == True]),
         ('Escalation priority', d[d.priority_escalations == True])]
for cn, sub in conds:
    for i, p in enumerate(['RAG', 'Mandate+Esc', 'Rules', 'AlwaysEsc']):
        s = sub[sub.policy == p]
        m1 = agg(s, 'median_hours'); m2 = agg(s, 'p95_hours'); m3 = agg(s, 'sla_violation')
        T.append(f"{cn if i == 0 else ''} & {p} & {m1[0]:.2f}$\\pm${m1[1]:.2f} & {m2[0]:.2f}$\\pm${m2[1]:.2f} & {100*m3[0]:.1f}$\\pm${100*m3[1]:.1f}\\\\")
    T.append('\\midrule' if cn != 'Escalation priority' else '\\bottomrule')
T.append('\\end{tabular}')
open('results/tab_stress.tex', 'w').write('\n'.join(T) + '\n')
print(json.dumps({p: {k: round(v[0], 4) for k, v in S[p].items()} for p in POLS}, indent=0))
print('family ranges', json.dumps(S['family_range']))
print(json.load(open('results/des_validation.json')))
for p in ['Mandate+Esc', 'AlwaysEsc', 'Rules']:
    for pooled in [False, True]:
        g = base[(base.policy == p) & (base.pooled == pooled)].groupby('arrival_per_hour')[['p95_hours', 'sla_violation']].mean()
        print(p, 'pooled' if pooled else 'dept', g.round(2).to_dict())
for k in ['cfg_p_register_stale', 'cfg_p_contested', 'cfg_detect']:
    dd = ds[ds[k].notna()]
    print(k, dd.groupby(['policy', k])[['unsupported_closure', 'wrong_authority', 'escalation_rate']].mean().round(3).loc[['Mandate', 'Mandate+Esc']].to_dict())

"""Reproduce every number, table and figure of the MandateSim analysis.
Usage: python scripts/run_all.py   (about a few minutes on a laptop CPU)
Outputs: results/*.json, results/*.csv, figures/*.pdf, results/latex_tables.tex
"""
import sys, os, json, math, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd
from mandatesim.bench import make_cases, evaluate, POLICIES, FAMILIES, DEFAULT
from mandatesim.des import simulate, DES_DEFAULT
from multiprocessing import Pool

SEEDS = list(range(20)); N = 2000
POLS = list(POLICIES)

def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = 1.96 * x.std(ddof=1) / math.sqrt(len(x)); return m, h

# ---------------------------------------------------------- 1. benchmark: default setting, 20 seeds
def bench_job(args):
    seed, over = args
    cases, P = make_cases(N, seed=seed, **over)
    rows = []
    for p in POLS:
        r, _ = evaluate(cases, p, seed, P); r.update(policy=p, seed=seed, **{f'cfg_{k}': v for k, v in over.items()})
        rows.append(r)
        for fam in FAMILIES:   # per case family
            sub = [c for c in cases if c.family == fam]
            if sub:
                rf, _ = evaluate(sub, p, seed, P); rows.append({**rf, 'policy': p, 'seed': seed, 'family': fam})
    return rows

def main():
    os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
    with Pool(4) as pool:
        rows = sum(pool.map(bench_job, [(s, {}) for s in SEEDS]), [])
        df = pd.DataFrame(rows); df.to_csv('results/bench_default.csv', index=False)
        # sensitivity: register staleness and contested mandates
        sens = []
        for key_, vals in [('p_register_stale', [0.0, 0.05, 0.1, 0.2, 0.3]), ('p_contested', [0.0, 0.04, 0.08, 0.16, 0.24]),
                           ('detect', [0.6, 0.75, 0.9, 0.99])]:
            for v in vals:
                sens += sum(pool.map(bench_job, [(s, {key_: v}) for s in SEEDS[:10]]), [])
        ds = pd.DataFrame(sens); ds.to_csv('results/bench_sensitivity.csv', index=False)
    S = {}
    main_rows = df[df['family'].isna()] if 'family' in df else df
    for p in POLS:
        d = main_rows[main_rows.policy == p]
        S[p] = {m: ci(d[m]) for m in ['wrong_authority', 'unsupported_closure', 'escalation_rate', 'esc_sensitivity',
                                         'esc_precision', 'unnecessary_esc', 'misrouted_esc']}
    fam_rows = df[df['family'].notna()]
    S['family_range'] = {p: {m: [float(fam_rows[fam_rows.policy == p].groupby('family')[m].mean().min()),
                                  float(fam_rows[fam_rows.policy == p].groupby('family')[m].mean().max())]
                              for m in ['unsupported_closure', 'wrong_authority']} for p in POLS}
    json.dump(S, open('results/bench_summary.json', 'w'), indent=1)

    # ---------------------------------------------------------- 2. DES validation against Erlang-C (M/M/c)
    from mandatesim.bench import Case
    val = []
    for lam, mu_min, c in [(4.0, 10.0, 1), (5.0, 20.0, 2)]:
        dummy = [Case('x', 'Planning', 'Planning', False, [], 1, False, 'Planning')]
        dec = [('answer', 1, 'Planning')]
        Ws = []
        for s in range(10):
            r = simulate(dec, dummy, seed=s, arrival_per_hour=lam, hours=4000, signoff_mean_min=mu_min, cv='exp',
                         reviewers_per_dept=c, pooled=False)
            Ws.append(r['median_hours'])
        # Erlang-C mean sojourn
        a = lam * mu_min / 60; rho = a / c
        pc = (a ** c / math.factorial(c) / (1 - rho)) / (sum(a ** k / math.factorial(k) for k in range(c)) + a ** c / math.factorial(c) / (1 - rho))
        Wq = pc / (c / (mu_min / 60) - lam)
        # mean sojourn by simulation
        sims = []
        for s in range(10):
            import mandatesim.des as D
            sims.append(_mean_sojourn(dec, dummy, s, lam, mu_min, c))
        val.append(dict(lam=lam, mean_service_min=mu_min, servers=c, rho=rho, erlang_mean_sojourn_h=Wq + mu_min / 60,
                        sim_mean_sojourn_h=float(np.mean(sims)), sim_ci=float(1.96 * np.std(sims, ddof=1) / np.sqrt(len(sims)))))
    json.dump(val, open('results/des_validation.json', 'w'), indent=1)

    # ---------------------------------------------------------- 3. capacity simulation per policy
    cases, P = make_cases(N, seed=0)
    decs = {p: evaluate(cases, p, 0, P)[1] for p in POLS}
    jobs = []
    for p in ['RAG', 'Mandate', 'Mandate+Esc', 'Rules', 'AlwaysEsc']:
        for lam in [3, 4, 5, 6, 7, 8, 10]:
            for pooled in [False, True]:
                for s in range(10):
                    jobs.append((p, s, dict(arrival_per_hour=lam, pooled=pooled)))
        for stress in ['burst', 'absence', 'priority_escalations']:
            for s in range(10):
                jobs.append((p, s, {stress: True}))
    with Pool(4) as pool:
        out = pool.map(_des_job, [(decs[p], cases, p, s, o) for p, s, o in jobs])
    dd = pd.DataFrame(out); dd.to_csv('results/des_runs.csv', index=False)

def _mean_sojourn(dec, dummy, s, lam, mu_min, c):
    import mandatesim.des as D
    import heapq
    # re-run and compute mean sojourn (simulate returns median/p95; compute mean via small wrapper)
    rng = np.random.default_rng(s)
    r = D.simulate(dec, dummy, seed=s, arrival_per_hour=lam, hours=4000, signoff_mean_min=mu_min, cv='exp',
                   reviewers_per_dept=c, return_mean=True)
    return r['mean_hours']

def _des_job(a):
    dec, cases, p, s, o = a
    r = simulate(dec, cases, seed=s, **o); r.update(policy=p, seed=s, **{k: v for k, v in o.items()})
    return r

if __name__ == '__main__':
    main()

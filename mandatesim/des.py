"""Discrete-event simulation of human review capacity for AI-drafted municipal responses.

Cases arrive (Poisson, optionally with bursts). A routing policy decides per case: 'answer' -> short sign-off
review by the owning department; 'escalate' -> full review by the destination department (or governance forum).
Reviewer pools are either departmental (one queue per department) or pooled (one shared queue). Service times
are lognormal. Optional reviewer absence. Escalated cases may be transferred once when misrouted.
"""
import heapq, numpy as np
from .bench import DEPTS, FORUM

DES_DEFAULT = dict(arrival_per_hour=6.0, hours=8 * 20 * 6, reviewers_per_dept=1, forum_reviewers=1,
                   signoff_mean_min=6.0, escalation_mean_min=45.0, cv=0.8, transfer_penalty_min=20.0,
                   pooled=False, burst=False, absence=False, sla_hours=40.0, priority_escalations=False, return_mean=False)


def lognorm(rng, mean, cv):
    if cv == 'exp':
        return rng.exponential(mean)
    s2 = np.log(1 + cv ** 2); mu = np.log(mean) - s2 / 2
    return rng.lognormal(mu, np.sqrt(s2))


def simulate(decisions, cases, seed=0, **over):
    """decisions: list of (action, value, dest) aligned with cases (cycled if fewer than arrivals)."""
    P = {**DES_DEFAULT, **over}; rng = np.random.default_rng(seed)
    T = P['hours'] * 60.0
    pools = (['ALL'] if P['pooled'] else DEPTS) + [FORUM]
    cap = {p: (P['reviewers_per_dept'] * len(DEPTS) if p == 'ALL' else P['reviewers_per_dept']) for p in pools}
    cap[FORUM] = P['forum_reviewers']
    busy = {p: 0 for p in pools}; queue = {p: [] for p in pools}
    ev = []; t = 0.0; k = 0; seq = 0
    # arrivals
    while True:
        rate = P['arrival_per_hour'] / 60.0
        if P['burst'] and (int(t // (60 * 8 * 5)) % 4 == 3):     # every 4th week: doubled demand
            rate *= 2
        t += rng.exponential(1 / rate)
        if t > T: break
        heapq.heappush(ev, (t, seq, 'arr', k)); seq += 1; k += 1
    n_arr = k; rec = {}
    def pool_of(dest):
        return FORUM if dest == FORUM else ('ALL' if P['pooled'] else dest)
    WEEK = 60 * 8 * 5
    def capacity(p, now):
        c = cap[p]
        if P['absence'] and (now // WEEK) % 3 == 1:
            if p == 'ALL':
                c -= 1
            elif p == DEPTS[int(now // WEEK) % len(DEPTS)]:
                c = max(0, c - 1)
        return c
    def start(p, now):
        while queue[p] and busy[p] < capacity(p, now):
            _, _, j, st = heapq.heappop(queue[p]); busy[p] += 1
            heapq.heappush(ev, (now + st, next_seq(), 'dep', (j, p)))
    cnt = [seq]
    def next_seq():
        cnt[0] += 1; return cnt[0]
    while ev:
        now, _, typ, x = heapq.heappop(ev)
        if typ == 'arr':
            j = x; ci = j % len(cases); a, v, dest = decisions[ci]
            c = cases[ci]
            if a == 'answer':
                st = lognorm(rng, P['signoff_mean_min'], P['cv']); dst = dest; prio = 1
            else:
                st = lognorm(rng, P['escalation_mean_min'], P['cv']); dst = dest; prio = 0 if P['priority_escalations'] else 1
                if dest != c.true_dest:  # misrouted escalation needs a transfer step
                    st += P['transfer_penalty_min']
            p = pool_of(dst)
            rec[j] = dict(arr=now, esc=(a != 'answer'), pool=p)
            heapq.heappush(queue[p], (prio, now, j, st)); start(p, now)
        else:
            j, p = x; busy[p] -= 1; rec[j]['done'] = now; start(p, now)
        for q in pools:
            start(q, now)
    done = [r for r in rec.values() if 'done' in r]
    W = np.array([(r['done'] - r['arr']) / 60 for r in done])
    We = np.array([(r['done'] - r['arr']) / 60 for r in done if r['esc']])
    backlog = n_arr - len(done)
    # conservation check: every arrival is either done or still queued/in service
    in_sys = sum(len(q) for q in queue.values()) + sum(busy.values())
    assert backlog == in_sys, (backlog, in_sys)
    return dict(n_arrivals=n_arr, completed=len(done), backlog_end=backlog,
                median_hours=float(np.median(W)) if len(W) else np.nan,
                p95_hours=float(np.percentile(W, 95)) if len(W) else np.nan,
                p95_esc_hours=float(np.percentile(We, 95)) if len(We) else np.nan,
                sla_violation=float(np.mean(W > P['sla_hours'])) if len(W) else np.nan,
                unfinished_share=backlog / max(1, n_arr),
                mean_hours=float(np.mean(W)) if len(W) else np.nan,
                share_escalated=float(np.mean([r['esc'] for r in rec.values()])))

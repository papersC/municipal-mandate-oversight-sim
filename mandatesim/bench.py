"""Synthetic municipal conflict-episode generator and routing policies.

All cases are SYNTHETIC. Ground truth (true owner, correct guidance, whether the case is resolvable without a
human decision) is known by construction. Parameters are explicit assumptions, varied in sensitivity analyses.
"""
import numpy as np
from dataclasses import dataclass, field

DEPTS = ['Planning', 'Environment', 'Finance', 'CustomerService', 'Transport', 'Legal']
FORUM = 'GovernanceForum'
FAMILIES = ['permit', 'complaint', 'fee', 'waste', 'roadworks']  # case families (strata)

DEFAULT = dict(n_claim_types=40, p_register_stale=0.10, p_contested=0.08, p_owner_missing=0.10,
               p_owner_stale=0.10, p_disagree=0.35, p_unsupported=0.05, detect=0.90, false_flag=0.03,
               keyword_acc=0.70)


@dataclass
class Case:
    family: str
    true_owner: str
    register_owner: str
    contested: bool
    claims: list            # list of dicts: dept, value, fresh, supported
    correct_value: int
    should_escalate: bool
    true_dest: str


def make_world(rng, P):
    owners = rng.choice(DEPTS, P['n_claim_types'])
    fam = rng.choice(FAMILIES, P['n_claim_types'])
    reg = owners.copy()
    stale = rng.random(P['n_claim_types']) < P['p_register_stale']
    for i in np.where(stale)[0]:
        reg[i] = rng.choice([d for d in DEPTS if d != owners[i]])
    return owners, reg, fam


def make_case(rng, world, P):
    owners, reg, fam = world
    k = int(rng.integers(len(owners)))
    owner = owners[k]; contested = rng.random() < P['p_contested']
    correct = 1
    claims = []
    owner_present = rng.random() >= P['p_owner_missing']
    owner_fresh = rng.random() >= P['p_owner_stale']
    if owner_present:
        claims.append(dict(dept=owner, value=correct if owner_fresh else 0, fresh=owner_fresh,
                           supported=rng.random() >= P['p_unsupported']))
    others = rng.choice([d for d in DEPTS if d != owner], int(rng.integers(1, 3)), replace=False)
    for d in others:
        v = 0 if rng.random() < P['p_disagree'] else correct
        claims.append(dict(dept=d, value=v, fresh=rng.random() > 0.1, supported=rng.random() >= P['p_unsupported']))
    rng.shuffle(claims)
    resolvable = owner_present and owner_fresh and not contested and claims and \
        [c for c in claims if c['dept'] == owner][0]['supported']
    dest = FORUM if contested else owner
    return Case(fam[k], owner, reg[k], contested, claims, correct, not resolvable, dest)


def make_cases(n, seed=0, **over):
    P = {**DEFAULT, **over}; rng = np.random.default_rng(seed)
    w = make_world(rng, P)
    return [make_case(rng, w, P) for _ in range(n)], P


# ------------------------------------------------------------------ policies -> (action, value, attributed/dest)
def pol_search(c, rng, P):
    """Ordinary search: staff/agent takes the first retrieved departmental source."""
    cl = c.claims[0]; return ('answer', cl['value'], cl['dept'])


def pol_rag(c, rng, P):
    """Retrieval-augmented drafting without mandates: majority value across departmental claims (consensus)."""
    vals = [cl['value'] for cl in c.claims]
    v = max(set(vals), key=lambda x: (vals.count(x), rng.random()))
    d = [cl['dept'] for cl in c.claims if cl['value'] == v][0]
    return ('answer', v, d)


def pol_mandate(c, rng, P):
    """Mandate visibility: answer with the register owner's claim if present, otherwise consensus."""
    m = [cl for cl in c.claims if cl['dept'] == c.register_owner]
    if m: return ('answer', m[0]['value'], c.register_owner)
    return pol_rag(c, rng, P)


def _flag(rng, truth, P):
    return (rng.random() < P['detect']) if truth else (rng.random() < P['false_flag'])


def pol_mandate_esc(c, rng, P):
    """Mandate-aware drafting with structured escalation: checks owner presence, freshness, support and disputes."""
    m = [cl for cl in c.claims if cl['dept'] == c.register_owner]
    disputed = _flag(rng, c.contested, P)
    stale = _flag(rng, bool(m) and not m[0]['fresh'], P)
    unsup = _flag(rng, bool(m) and not m[0]['supported'], P)
    if disputed: return ('escalate', None, FORUM)
    if not m or stale or unsup: return ('escalate', None, c.register_owner)
    return ('answer', m[0]['value'], c.register_owner)


def pol_rules(c, rng, P):
    """Rule-based routing without mandates: escalate on any disagreement to a keyword-routed department."""
    vals = {cl['value'] for cl in c.claims}
    if len(vals) > 1:
        dest = c.true_owner if rng.random() < P['keyword_acc'] else rng.choice([d for d in DEPTS if d != c.true_owner])
        return ('escalate', None, dest)
    return ('answer', c.claims[0]['value'], c.claims[0]['dept'])


def pol_always(c, rng, P):
    return ('escalate', None, FORUM if _flag(rng, c.contested, P) else c.register_owner)


POLICIES = {'Search': pol_search, 'RAG': pol_rag, 'Mandate': pol_mandate, 'Mandate+Esc': pol_mandate_esc,
            'Rules': pol_rules, 'AlwaysEsc': pol_always}


def evaluate(cases, policy, seed=0, P=None):
    rng = np.random.default_rng(seed + 1000); P = P or DEFAULT
    n = len(cases); out = []
    for c in cases:
        a, v, d = POLICIES[policy](c, rng, P)
        out.append((a, v, d))
    A = np.array([o[0] == 'answer' for o in out]); E = ~A
    should = np.array([c.should_escalate for c in cases])
    wrong_attr = np.array([(o[2] != c.true_owner) if o[0] == 'answer' else (o[2] != c.true_dest) for o, c in zip(out, cases)])
    unsupported = np.array([o[0] == 'answer' and (c.should_escalate or o[1] != c.correct_value) for o, c in zip(out, cases)])
    r = dict(wrong_authority=wrong_attr.mean(), unsupported_closure=unsupported.mean(),
             escalation_rate=E.mean(),
             esc_sensitivity=(E & should).sum() / max(1, should.sum()),
             esc_precision=(E & should).sum() / max(1, E.sum()),
             unnecessary_esc=(E & ~should).sum() / max(1, (~should).sum()),
             misrouted_esc=(E & wrong_attr).sum() / max(1, E.sum()))
    return {k: float(v) for k, v in r.items()}, out

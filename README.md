# municipal-mandate-oversight-sim

Code and data for the article **"Who Gets to Decide? Departmental Authority and Ethical Oversight in Municipal
Agentic AI"** (N. Ababneh and K. Almi'ani), submitted to *AI and Ethics* (Springer Nature).

The repository contains

* `mandatesim/bench.py` – synthetic generator of municipal conflict episodes with known ground truth (true owning
  department, correct guidance, need for a human decision) and the six compared routing policies
  (`Search`, `RAG`, `Mandate`, `Mandate+Esc`, `Rules`, `AlwaysEsc`) plus outcome measures.
* `mandatesim/des.py` – discrete-event simulator of departmental / pooled human review queues
  (Poisson arrivals, lognormal service, transfers for misrouted escalations, demand bursts, reviewer absence,
  escalation priority) with a case-conservation check.
* `scripts/run_all.py` – runs every experiment (benchmark: 20 seeds x 2,000 episodes; sensitivity analyses;
  Erlang-C validation of the simulator; capacity sweeps and stress tests) and writes `results/`.
* `scripts/make_outputs.py` – builds the LaTeX tables (`results/tab_*.tex`) and figures (`figures/*.pdf`) used in
  the article and prints every number quoted in the text.
* `results/` – the raw outputs used in the article (CSV/JSON).

**All data are synthetic.** No municipal records, personal data or human participants are involved. Every
parameter (see `DEFAULT` in `bench.py` and `DES_DEFAULT` in `des.py`) is an explicit assumption.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py        # ~30 s on 4 CPU cores
python scripts/make_outputs.py   # tables, figures, quoted statistics
```

Tested with Python 3.13, NumPy 2.5, pandas 3.0, matplotlib 3.11. Results are deterministic for fixed seeds.

## License
MIT (code). See `LICENSE`.

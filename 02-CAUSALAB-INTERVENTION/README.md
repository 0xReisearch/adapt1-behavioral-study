# CausaLab intervention replication instructions

## 1. Fixed protocol

- Public CausaLab main suite at benchmark revision `42ba47fb`
- Five suites: 3, 4, 5, 6, and 7 nodes
- 50 graphs per suite, 250 graphs total
- Primary run seed: 1
- Fresh-state audit seed: 3
- `PYTHONHASHSEED=0`
- Graph-size intervention budgets: 8, 12, 16, 20, and 24
- A fresh Domain for every graph
- One held-out prediction after the intervention budget is exhausted
- Hidden target frequency and graph edges read only after prediction submission
- No LLM calls

The hosted Adapt-1 API must contain Core revision `2eec3252464da075484263b2967901eef402b6b5` or a compatible later revision. The API must expose Domain creation, event ingestion, and query routes without rewriting the submitted payloads.

## 2. Install the public benchmark

```bash
git clone https://github.com/DylanZSZ/CausaLab-Benchmark.git
cd CausaLab-Benchmark
git checkout 42ba47fb

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r agents/requirements.txt
python -m pip install -e .
python -m pip install -r /absolute/path/to/02-CAUSALAB-INTERVENTION/requirements.txt
```

Set paths and credentials. Supply your own API origin and API key:

```bash
export CAUSALAB_BENCHMARK_ROOT="$PWD"
export CAUSALAB_EXPERIMENT_ROOT='/absolute/path/to/02-CAUSALAB-INTERVENTION'
export NEUROADAPT_API_URL='https://rei-neuroadapt-api.reilabs.org/api/v1'
export NEUROADAPT_API_KEY='<your-api-key>'
export RUN_ID="causalab-$(date -u +%Y%m%dT%H%M%SZ)"
```

Do not place credentials in files or command-line arguments.

## 3. Run a smoke test

```bash
PYTHONHASHSEED=0 PYTHONPATH="$CAUSALAB_BENCHMARK_ROOT:$CAUSALAB_EXPERIMENT_ROOT" \
  "$CAUSALAB_BENCHMARK_ROOT/.venv/bin/python" \
  "$CAUSALAB_EXPERIMENT_ROOT/run_causalab_production.py" \
    --benchmark-root "$CAUSALAB_BENCHMARK_ROOT" \
    --run-id "$RUN_ID-smoke" \
    --suite 3nodes \
    --seed 1 \
    --limit 1 \
    --api-timeout 120 \
    --output "$CAUSALAB_EXPERIMENT_ROOT/results/$RUN_ID/smoke"
```

The smoke run must finish with a numeric target prediction and a nonempty causal graph. The output directory contains `summary.json` only.

## 4. Run the primary 250-graph replication

```bash
for suite in 3nodes 4nodes 5nodes 6nodes 7nodes; do
  PYTHONHASHSEED=0 PYTHONPATH="$CAUSALAB_BENCHMARK_ROOT:$CAUSALAB_EXPERIMENT_ROOT" \
    "$CAUSALAB_BENCHMARK_ROOT/.venv/bin/python" \
    "$CAUSALAB_EXPERIMENT_ROOT/run_causalab_production.py" \
      --benchmark-root "$CAUSALAB_BENCHMARK_ROOT" \
      --run-id "$RUN_ID-seed1" \
      --suite "$suite" \
      --seed 1 \
      --api-timeout 120 \
      --output "$CAUSALAB_EXPERIMENT_ROOT/results/$RUN_ID/seed1/$suite"
done
```

## 5. Run the fresh-state audit

Use the identical files and settings with seed 3 and a new run identity:

```bash
for suite in 3nodes 4nodes 5nodes 6nodes 7nodes; do
  PYTHONHASHSEED=0 PYTHONPATH="$CAUSALAB_BENCHMARK_ROOT:$CAUSALAB_EXPERIMENT_ROOT" \
    "$CAUSALAB_BENCHMARK_ROOT/.venv/bin/python" \
    "$CAUSALAB_EXPERIMENT_ROOT/run_causalab_production.py" \
      --benchmark-root "$CAUSALAB_BENCHMARK_ROOT" \
      --run-id "$RUN_ID-seed3" \
      --suite "$suite" \
      --seed 3 \
      --api-timeout 120 \
      --output "$CAUSALAB_EXPERIMENT_ROOT/results/$RUN_ID/seed3/$suite"
done
```

Do not change the runner, Domain declaration, intervention values, or Core version between the two runs.

## 6. Check the scores

Compare each generated `summary.json` with `EXPECTED_SCORES.json`.

Primary seed-1 macro scores across 250 graphs:

| Task accuracy | All-edge precision | All-edge recall | All-edge F1 | Mean directed SHD | Frequency-edge F1 | Frequency-weight F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 97.2% | 0.888 | 0.989 | 0.931 | 1.356 | 1.000 | 1.000 |

Fresh seed-3 audit macro scores:

| Task accuracy | Numeric within 5 Hz | All-edge F1 | Mean directed SHD |
|---:|---:|---:|---:|
| 98.4% | 100% | 0.929 | 1.352 |

Keep `PYTHONHASHSEED=0`. The simulator uses Python's randomized `hash()` for part of its generated numeric state.

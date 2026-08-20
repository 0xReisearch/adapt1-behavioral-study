# Reproduction

## Pinned sources

- Neuroadapt repository: `https://github.com/00unitrei/neuroadapt-api`
- Neuroadapt commit: `1059933df90bcb1d5848cf6adfda419accc8f3f0`
- POPGym repository: `https://github.com/proroklab/popgym`
- POPGym commit: `410d5aa626dae8024f498354d8781a0d1870c399`

The harness calls Core directly. It does not use an LLM, an external solver,
pretrained task state, or state retained from another run.

## Environment

Use Python 3.12.

```bash
git clone https://github.com/00unitrei/neuroadapt-api.git
cd neuroadapt-api
git checkout 1059933df90bcb1d5848cf6adfda419accc8f3f0

git clone https://github.com/proroklab/popgym.git ../popgym
git -C ../popgym checkout 410d5aa626dae8024f498354d8781a0d1870c399

python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -e ../popgym
pip install torch
```

Copy `harness/run_repeat_previous.py` into the installed Neuroadapt checkout or
call it through its absolute path.

The public harness writes summary results only. It fails if Core does not return
a legal selected policy or a decision ID. It does not implement a client-side
policy fallback.

## RepeatPreviousEasy

Run every variant for each seed tuple in this table:

| Index | Training seed start | Evaluation seed start | Learner seed |
|---:|---:|---:|---:|
| 0 | 700 | 1700 | 65000 |
| 1 | 800 | 1800 | 65001 |
| 2 | 900 | 1900 | 65002 |
| 3 | 1000 | 2000 | 65003 |
| 4 | 1100 | 2100 | 65004 |

Example:

```bash
python harness/run_repeat_previous.py \
  --environment easy \
  --variant tcp \
  --maximum-lag 16 \
  --episodes 10 \
  --seed-start 700 \
  --eval-episodes 16 \
  --eval-seed-start 1700 \
  --learner-seed 65000 \
  --output reproduced/easy-tcp-seed-0.json
```

Use `memoryless`, `tcp`, and `tcp_cup` as the three `--variant` values.

## RepeatPreviousMedium

Run both `tcp` and `tcp_cup` for the five tuples below. Only seed 0 includes the
frozen evaluation reported in the article.

| Index | Training seed start | Evaluation seed start | Learner seed | Frozen episodes |
|---:|---:|---:|---:|---:|
| 0 | 2700 | 3700 | 66000 | 16 |
| 1 | 2800 | - | 66001 | 0 |
| 2 | 2900 | - | 66002 | 0 |
| 3 | 3000 | - | 66003 | 0 |
| 4 | 3100 | - | 66004 | 0 |

Example:

```bash
python harness/run_repeat_previous.py \
  --environment medium \
  --variant tcp_cup \
  --maximum-lag 64 \
  --episodes 10 \
  --seed-start 2700 \
  --eval-episodes 16 \
  --eval-seed-start 3700 \
  --learner-seed 66000 \
  --output reproduced/medium-tcp-cup-seed-0.json
```

For seeds 1 through 4, pass `--eval-episodes 0`. Their evaluation seed argument
is ignored.

## Validity checks

Every frozen result must satisfy:

```text
protocol.evaluation_mode == "frozen_no_feedback"
protocol.future_observation_supplied == false
protocol.target_lag_declared == false
protocol.forced_coverage == false
frozen_state_unchanged == true
frozen_state_before == frozen_state_after
```

The evaluator knows the public benchmark target rule so that it can calculate
accuracy and probability loss after the action is sealed. This target is never
added to the Domain context or sent before the decision.

## Wisconsin

Wisconsin reproduction is intentionally unavailable in this package. The exact
four-condition runner, twelve-feature manifest, component-predictor
construction, and thirty per-order outputs were absent from the supplied
artifacts. `results/wisconsin/STATUS.md` lists what must be recovered.

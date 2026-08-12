# Symbolic Alchemy sequential replication instructions

## 1. Fixed protocol

- Python 3.11 or 3.12
- Official `dm_alchemy` source at commit `68a26254b5c0f15e84fa0c15d66bf0c626ede8e0`
- Official fixed Symbolic Alchemy evaluation bank
- Bank entries 0 through 25 in their published order
- 26 episodes
- 200 native environment steps per episode
- Ten source trials inside each episode
- `observe_used=True`
- `end_trial_action=False`
- One empty Adapt-1 Domain for the complete 26-episode series
- A fresh session identifier for each 200-step episode
- No Domain wipe between episodes
- No session reset between the ten trials inside an episode
- Thompson exploration through the Domain policy configuration
- No task pretraining, demonstrations, expert actions, action masks, or external solver
- No LLM calls
- Primary score: mean raw environment return across the 26 episodes

The task return is stochastic. The recorded 26-episode mean is `252.00`. Use the descriptive mean-return range `233.54` to `270.46` for a clean reproduction check. This range is a screening range from the recorded evaluation. It is not a multi-seed confidence interval.

## 2. Install the public benchmark

```bash
git clone https://github.com/google-deepmind/dm_alchemy.git
cd dm_alchemy
git checkout 68a26254b5c0f15e84fa0c15d66bf0c626ede8e0

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install .
python -m pip install 'dm-env==1.6' 'requests==2.34.2'
```

Keep the source checkout fixed for the full run. Do not edit the chemistry bank or item bank.

## 3. Configure API access

Use an Adapt-1 API build that accepts the sequential and bound-transition fields in `HARNESS_AND_DOMAIN.md`. Domain creation must return the same values without rewriting them.

Set credentials through environment variables:

```bash
export DM_ALCHEMY_ROOT='/absolute/path/to/dm_alchemy'
export NEUROADAPT_API_URL='https://rei-neuroadapt-api.reilabs.org/api/v1'
export NEUROADAPT_API_KEY='<your-api-key>'
export RUN_ID="symbolic-alchemy-$(date -u +%Y%m%dT%H%M%SZ)"
```

Do not store the API key, bearer headers, owner identity, request identifiers, or API responses in the repository.

## 4. Build the runner

Implement one local runner from `HARNESS_AND_DOMAIN.md`. The runner must:

1. Create one fresh Domain before episode 0.
2. Load official fixed-bank entries 0 through 25.
3. Create one fresh session identifier per episode.
4. Map only the public 39-field symbolic observation and the public episode-start flag into each query.
5. Execute the policy selected by Adapt-1 without application-side ranking or correction.
6. Submit the exact pre-action context, executed action, raw reward, normalized reward, public next state, and terminal flag as feedback.
7. Execute native action 0 without a query after all three stones have been used for the current trial tail.
8. Keep Domain learner state across all 26 episodes.
9. Write one aggregate summary. Do not write HTTP traces, decision payloads, task states, per-step ledgers, or per-episode learner snapshots.

The recommended command surface is:

```bash
python run_symbolic_alchemy.py \
  --benchmark-root "$DM_ALCHEMY_ROOT" \
  --episode-offset 0 \
  --episodes 26 \
  --steps 200 \
  --skip-inactive-tail \
  --output aggregate-summary.json
```

The runner source is intentionally excluded from this instructions-only package.

## 5. Run a smoke test

Use a separate fresh Domain and output directory:

```bash
python run_symbolic_alchemy.py \
  --benchmark-root "$DM_ALCHEMY_ROOT" \
  --episode-offset 0 \
  --episodes 1 \
  --steps 200 \
  --skip-inactive-tail \
  --output smoke-summary.json
```

Confirm that the run completes 200 native steps, admits feedback once for each queried action, and exposes no hidden task fields. Delete the smoke-test Domain. Do not continue the official series from smoke-test state.

## 6. Run the 26-episode replication

Start again with a new empty Domain. Process all episodes in one uninterrupted series when possible. If transport recovery is required, preserve the same Domain and continue only from a verified aggregate checkpoint. Never replay an ambiguous feedback mutation.

Compare the aggregate result with `EXPECTED_SCORES.json`.

| Measurement | Recorded reference | Reproduction check |
|---|---:|---:|
| Episodes | 26 | exactly 26 |
| Native steps | 5,200 | exactly 5,200 |
| Mean raw return | 252.00 | 233.54 to 270.46 |
| Total raw return | 6,552 | stochastic |
| First 13 mean | 233.54 | descriptive only |
| Last 13 mean | 270.46 | descriptive only |
| Missing feedback admissions | 0 | exactly 0 |
| LLM calls | 0 | exactly 0 |

Action sampling and online fitting make the return nondeterministic. Use the range check for the mean. Keep the protocol invariants exact.

## 7. Public references

- DeepMind Alchemy source: https://github.com/google-deepmind/dm_alchemy
- Wang et al., *Alchemy: A structured task distribution for meta-reinforcement learning*: https://arxiv.org/abs/2102.02926

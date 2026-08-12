# Trading Simulation

## 1. Scope

This is a small-scale online adaptation simulation with 30 deterministic market scenarios and 120 decisions per scenario. It uses one pass through each scenario. Every scenario starts with a fresh Adapt-1 Domain, so no learned state crosses scenario boundaries.

The learner receives the current public numeric features and predicts the next price delta for each listed asset. The controller liquidates existing positions, then allocates available cash to the asset with the largest positive predicted relative gain. The next market state is revealed after the decision and ingested as the learning outcome.

The sanitized dataset uses only neutral asset labels `S0` through `S4` and feature labels `F0` onward. It contains current feature values, current prices, and next prices. It contains no prior actions, predictions, model responses, credentials, hidden answers, source-specific labels, or execution histories.

The reference run used Adapt-1 Core revision `57edf6944e47f05dc7f1690901f430eb79e1d5f2` and made zero LLM calls.

## 2. Install

From this experiment directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the hosted Adapt-1 API origin and your API key:

```bash
export NEUROADAPT_API_URL='https://rei-neuroadapt-api.reilabs.org/api/v1'
export NEUROADAPT_API_KEY='<your-api-key>'
export RUN_ID="trading-small-$(date -u +%Y%m%dT%H%M%SZ)"
```

Do not place the API key in the runner, command-line arguments, result file, or dataset.

## 3. Run one scenario

```bash
python run_experiment.py \
  --dataset data/market_scenarios.json \
  --limit 1 \
  --run-id "$RUN_ID-smoke" \
  --output smoke-summary.json
```

The command must create one fresh Domain, complete 120 decisions, delete the Domain, and write `smoke-summary.json`.

## 4. Run all 30 scenarios

```bash
python run_experiment.py \
  --dataset data/market_scenarios.json \
  --run-id "$RUN_ID" \
  --output full-summary.json
```

Run the scenarios sequentially. Do not reuse a Domain between scenarios. Do not ingest an outcome before its corresponding query.

## 5. Check the scores

Compare `full-summary.json` with `EXPECTED_SCORES.json`.

| Measurement | Recorded value |
|---|---:|
| Scenarios | 30 |
| Decisions per scenario | 120 |
| Mean profit | +167.8914% |
| Minimum scenario profit | +44.6578% |
| Maximum scenario profit | +358.6647% |
| Steps 0 to 19 prediction MAE | 0.051661 |
| Steps 100 to 119 prediction MAE | 0.006406 |
| Relative MAE reduction | 87.60% |
| LLM calls | 0 |

Profit is measured as `(final portfolio value - initial cash) / initial cash`. Prediction MAE is computed per scenario, then averaged across scenarios. The early window covers steps 0 through 19. The final window covers steps 100 through 119.

This result is a one-pass small-scale simulation. Keep that label when presenting or comparing it.

# Trading Simulation harness and Domain build guide

## 1. Harness boundary

The harness exposes a deterministic sequence of public numeric features and prices. Adapt-1 predicts the next price delta. A fixed controller converts that prediction into a legal portfolio update. The next prices are revealed after the decision and become the learning outcome.

The harness owns portfolio accounting, action execution, scenario reset, and profit scoring. Adapt-1 owns the numeric transition estimate. No language model, news model, pretrained perception model, or future-value lookup participates.

## 2. Sanitized dataset contract

`data/market_scenarios.json` is an array of scenarios:

```json
{
  "scenario_id": "scenario-1",
  "initial_cash": 50000.0,
  "steps": [
    {
      "features": {"F0": 0.03, "F1": -0.10},
      "prices": {"S0": 48.80, "S1": 76.57},
      "next_prices": {"S0": 48.77, "S1": 76.80}
    }
  ]
}
```

At decision time, expose `features` and `prices`. Keep `next_prices` unavailable until the action has been selected and applied.

The packaged scenarios contain 30 tasks with 120 steps each. Feature and asset counts are discovered from each scenario. The neutral names are part of the public experiment contract and must remain stable within a scenario.

## 3. Design the Domain

Create one fresh Domain per scenario.

### Event and query mapping

- Event type: `market_transition`
- Query inputs: `context.values.features.<feature>`
- Event inputs: `values.features.<feature>`
- Numeric targets: `values.delta.<asset>`
- Target value: `round(next_price, 2) - round(current_price, 2)`

### Exact learning configuration

```json
{
  "enabled": true,
  "transition": {
    "enabled": true,
    "event_types": ["market_transition"],
    "required_support": 2,
    "neighbors": 128,
    "max_distance": 1.0,
    "max_samples": 512,
    "numeric_model": "linear",
    "numeric_ridge": 1e-10,
    "numeric_min_skill": -1.0
  }
}
```

Construct `input_paths` as `values.features.<feature>` and construct one numeric target for every `values.delta.<asset>` path found in the scenario.

The Domain declaration contains no portfolio rule, future price, scenario score, or scenario-specific coefficient.

## 4. Build the prequential harness loop

For each scenario:

1. Create a unique Domain and session.
2. Set cash to `initial_cash` and every position to zero.
3. Read current features and current prices.
4. Query Adapt-1 before ingesting the current step outcome.
5. Read predictions from `transition_prediction.path_values`.
6. Apply the fixed controller described below.
7. Reveal `next_prices`.
8. Compute rounded next-price deltas.
9. Ingest one `market_transition` event with current features and revealed deltas.
10. Repeat for 120 steps.
11. Mark the portfolio at the final next prices.
12. Compute scenario profit rate.
13. Delete the Domain.

Never query after ingesting the outcome for the same step. Never reuse a Domain across scenarios.

## 5. Build the fixed portfolio controller

1. If Adapt-1 abstains, liquidate any existing position and hold cash.
2. For each available asset prediction, compute `predicted_delta / max(0.1, round(current_price, 2))`.
3. Select the asset with the largest value.
4. If its predicted delta is nonpositive, liquidate and hold cash.
5. Otherwise liquidate every existing position at the exact current prices.
6. Calculate the requested quantity as:

```text
max(0, floor(available_cash_after_sale / max(0.1, round(selected_price, 2))) - 1)
```

7. Execute the purchase at the exact current price only when its exact cost does not exceed cash.
8. Carry the resulting cash and integer positions into the next step.

The one-unit reserve and exact-cost check are required for score compatibility.

## 6. Build the score calculation

For each scenario:

```text
profit_rate = (final_portfolio_value - initial_cash) / initial_cash
```

Report the mean, minimum, and maximum profit rate across the 30 scenarios.

For prediction adaptation, compute absolute error for every available asset prediction. Compute a mean error within each scenario for steps 0 through 19 and steps 100 through 119. Average those scenario means across all scenarios. Then compute:

```text
relative_reduction = (early_mae - final_mae) / early_mae
```

## 7. Isolation checks

- 30 scenarios completed
- 120 decisions per scenario
- query occurs before outcome ingestion at every step
- fresh Domain and session per scenario
- Domain deleted after each scenario
- no scenario output is added to the next scenario's context
- result file contains aggregate scores only
- LLM call count remains zero

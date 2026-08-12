# CausaLab harness and Domain build guide

## 1. Harness boundary

The harness uses the public simulator action interface for navigation, dialog selection, numeric interventions, and final submission. Adapt-1 receives only public initial observations, public states returned after legal interventions, and the held-out reactor state without its target frequency.

The benchmark graph record is used to instantiate the simulator. Do not send that record, its edges, coefficients, or target frequency to Adapt-1. Read hidden truth only after the prediction has been submitted through the reactor dialog.

`run_causalab_frozen.py` contains the environment adapter and Domain logic. `run_causalab_production.py` transports the same Domain calls through the hosted API.

## 2. Discover the task surface

Build the adapter from public observations at runtime:

1. Parse the two initial numeric observations from the public task description.
2. Identify the single declared frequency field by its public label.
3. Discover the nearest public property manipulator and crystal reactor objects.
4. Open the manipulator dialog through legal environment actions.
5. Parse the baseline state from the dialog.
6. Discover the controllable fields from the dialog options labeled `Adjust`.
7. Read the remaining intervention budget from the public dialog.

Do not hard-code property names, graph edges, coefficients, target values, or graph-specific rules.

## 3. Design the Domain

Create a fresh Domain and owner identity for every graph. Disable autonomous training outside the declared event path.

### Event types

- `observation`: a public complete numeric state
- `intervention_result`: the before state, requested intervention, resulting state, and a Boolean changed map

### Inputs and targets

- Inputs: `values.state.<controllable-variable>` for every public controllable field
- Numeric target: `values.state.<frequency-field>`
- Causal variables: one before and after path pair for every public observable field
- Intervention identity: `values.intervention.target`

### Exact learning configuration

```json
{
  "transition": {
    "enabled": true,
    "event_types": ["observation", "intervention_result"],
    "required_support": 2,
    "neighbors": 8,
    "max_distance": 1.0,
    "max_samples": 256,
    "numeric_model": "auto",
    "numeric_ridge": 1e-10,
    "numeric_min_skill": 0.05,
    "causal_graph": {
      "enabled": true,
      "event_types": ["intervention_result"],
      "intervention_path": "values.intervention.target",
      "minimum_effect": 1e-8,
      "minimum_change_fraction": 0.5,
      "polynomial_degree": 1,
      "max_parents": 6,
      "complexity_penalty": 1e-4
    }
  }
}
```

Populate `input_paths`, `targets`, and causal variable path pairs from the public fields discovered for the current graph.

## 4. Build the intervention harness

1. Ingest the two public initial observations.
2. Ingest the public manipulator baseline.
3. Use generic intervention candidates `10`, `90`, `30`, and `70` in the public 0 to 100 control range.
4. Sort candidates by distance from the current baseline, farthest first.
5. Set `values_per_property = budget // controllable_property_count`.
6. Rotate the candidate order by property index so two properties do not always receive the same value at the same design step.
7. Interleave properties. At each design step, intervene once on each controllable field.
8. Record `before`, `state`, `intervention.target`, `intervention.requested_value`, and the per-field changed map.
9. Add an `intervention_changed` relation from the controlled field to each other field that changed.
10. Query Adapt-1 after each result so the current causal hypothesis can accompany subsequent public actions.
11. Use the full legal budget.

The action payload may carry a running public-data summary and the current learned hypothesis because the benchmark action schema accepts those fields. Never place hidden truth in that payload.

## 5. Build held-out prediction and submission

1. Exit the manipulator dialog.
2. Navigate to the reactor through the public action API.
3. Open the reactor dialog.
4. Parse the held-out state and remove the frequency field.
5. Query the Domain once with `context.values.state` set to that held-out state.
6. Require a numeric value at `transition_prediction.path_values.values.state.<frequency-field>`.
7. Freeze that value before any hidden read.
8. Submit it through the public reactor dialog.
9. Read completion status, target frequency, and graph edges after submission.
10. Compute task accuracy, absolute numeric error, edge precision and recall, edge F1, directed SHD, root F1, frequency-edge F1, and frequency-weight F1.

## 6. Isolation and reproducibility checks

- set `PYTHONHASHSEED=0` before process start
- create a unique Domain and session for every graph
- reuse no events or learned state across graphs
- preserve the intervention plan between seed 1 and seed 3
- keep the benchmark at revision `42ba47fb`
- keep Core at the required compatible revision
- confirm every suite contains 50 graphs
- confirm hidden truth is read after submission for every graph
- confirm the generated output contains aggregate summaries only

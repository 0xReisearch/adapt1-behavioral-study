# Symbolic Alchemy harness and Domain build guide

## 1. Harness boundary

The official environment owns task generation, hidden chemistry, native state transitions, trial resets, rewards, termination, and the 39-value public symbolic observation.

The harness owns fixed-bank loading, public field naming, API transport, execution of the selected native action, feedback order, inactive-tail handling, session boundaries, and aggregate scoring.

Adapt-1 owns action selection, online state updates, transition-effect induction, uncertainty handling, planning, and policy ranking. The harness must not add an action mask, rule-based veto, external planner, score correction, or fallback heuristic.

Never send any of these fields to Adapt-1:

- episode chemistry identity
- latent stone coordinates
- latent potion types
- potion or stone maps
- graph structure
- bottleneck or rotation state
- legal-action mask
- oracle or expert action
- future observation or reward
- evaluator answer

The environment can use hidden fields internally to execute the official task. Keep them outside query and feedback payloads.

## 2. Load the fixed evaluation series

Use the official source helper to load the fixed chemistry and item bank at `chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries`. Select entries 0 through 25 in order. For each entry, instantiate the symbolic environment with:

```text
observe_used = true
end_trial_action = false
```

Each entry produces one 200-step episode. The environment manages ten trial boundaries inside those 200 steps.

Create one Domain before episode 0. Keep it for all 26 episodes. Create a fresh session identifier at each 200-step episode boundary. Keep that session identifier constant through the episode's ten trials. A new session isolates episode-local trajectory state while the Domain learner version and admitted sample count continue to increase.

## 3. Map the public observation

The learner-visible vector has exactly 39 values.

### Stone slots

For stone slot `s` in `0..2`, map indices `5*s` through `5*s+4`:

| Offset | Field name | Public value |
|---:|---|---|
| 0 | `stone_slot_s_perceived_axis_0` | raw perceived coordinate |
| 1 | `stone_slot_s_perceived_axis_1` | raw perceived coordinate |
| 2 | `stone_slot_s_perceived_axis_2` | raw perceived coordinate |
| 3 | `stone_slot_s_visible_value_indicator` | public stone reward divided by 3 |
| 4 | `stone_slot_s_is_used` | 0 for present, 1 for used or empty |

Use `2.0` for every empty stone coordinate and empty visible-value field, as supplied by the public interface.

### Potion slots

For potion slot `p` in `0..11`, map indices `15 + 2*p` and `16 + 2*p`:

| Offset | Field name | Public value |
|---:|---|---|
| 0 | `potion_slot_p_perceived_type_indicator` | `perceived_potion.index / 3 - 1` |
| 1 | `potion_slot_p_is_used` | 0 for present, 1 for used or empty |

Use `1.0` for an empty potion type field, as supplied by the public interface.

Add `episode_start=1` only for the first timestep returned by the episode reset. Use `episode_start=0` on later timesteps. Keep episode and trial numbers out of the learner-visible context.

Reject a timestep if the public vector width differs from 39.

## 4. Declare the 40 native actions

Create one policy hypothesis for every native action ID from 0 through 39. Use relation `select_symbolic_slot_action` and weight `1.0` for every hypothesis.

For each hypothesis, use name `native-action-NN`, policy `action_NN`, the action feature values below, and prediction text `Execute the declared native slot action.`

Generate the action grammar with these formulas:

| Action | Native ID | Policy feature values |
|---|---:|---|
| no-op | `0` | `operation=no_op`, `stone_slot=-1`, `potion_slot=-1` |
| place stone `s` in cauldron | `1 + 13*s` | `operation=place_in_cauldron`, `stone_slot=s`, `potion_slot=-1` |
| apply potion `p` to stone `s` | `2 + 13*s + p` | `operation=apply_potion`, `stone_slot=s`, `potion_slot=p` |

Use policy names `action_00` through `action_39`. Preserve the native ID bijection. Query all 40 policies on every active step.

When all three public stone-used flags exceed `0.5`, execute native action 0 for the remaining inactive trial tail. Do not query Adapt-1 and do not submit feedback for those forced public no-ops.

## 5. Build the Domain declaration

Create the Domain with `POST /api/v1/domains`. Supply a fresh `domain_id`, a fresh owner `session_id`, the 40 generated hypotheses, `query_templates.feedback_outcomes=["native_reward"]`, the schema below, and the learning configuration below.

Use these schema fields:

```json
{
  "entities": ["stone_slot", "potion_slot", "cauldron", "native_slot_action"],
  "relations": ["select_symbolic_slot_action"],
  "event_types": ["native_action_outcome"],
  "constraints": {
    "episode_start": {"enum": [0, 1]},
    "native_action_id": {"enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]},
    "stone_slot": {"enum": [-1, 0, 1, 2]},
    "potion_slot": {"enum": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    "native_reward_raw": {"between": [-3, 15]},
    "policy_reward_normalized": {"between": [0, 1]}
  }
}
```

Add all 39 public field names from section 3, plus `episode_start`, `native_reward_raw`, and `policy_reward_normalized`, to the Domain signal list.

Use the following learning configuration:

```json
{
  "enabled": true,
  "credit_assignment": {
    "mode": "eligibility_trace",
    "delay_path": "values.delay_steps",
    "discount": 0.95,
    "minimum_weight": 0.05,
    "neutral_reward": 0.16666666666666666
  },
  "policy": {
    "abstain_on_ood": false,
    "action_transfer_strength": 0.0,
    "exploration_mode": "thompson",
    "exploration_strength": 1.0,
    "min_context_observations": 4,
    "transfer_strength": 0.15
  },
  "sequential": {
    "enabled": true,
    "discount": 0.95,
    "n_step": 5,
    "episode_path": "metadata.episode_id",
    "step_path": "metadata.step",
    "next_context_path": "values.next_state",
    "reward_path": "values.step_reward",
    "terminal_path": "values.terminal",
    "minimum_validation_skill": 0.05,
    "bound_transition": {
      "enabled": true,
      "state_selector_feature": "stone_slot",
      "state_features": {
        "available": "values.stone_slot_{selector}_is_used",
        "axis_0": "values.stone_slot_{selector}_perceived_axis_0",
        "axis_1": "values.stone_slot_{selector}_perceived_axis_1",
        "axis_2": "values.stone_slot_{selector}_perceived_axis_2",
        "objective": "values.stone_slot_{selector}_visible_value_indicator"
      },
      "state_bounds": {
        "available": [-0.01, 0.01],
        "axis_0": [-1.01, 1.01],
        "axis_1": [-1.01, 1.01],
        "axis_2": [-1.01, 1.01],
        "objective": [-1.01, 1.01]
      },
      "action_selector_feature": "potion_slot",
      "action_features": {
        "tool_type": "values.potion_slot_{selector}_perceived_type_indicator",
        "tool_used": "values.potion_slot_{selector}_is_used"
      },
      "action_bounds": {"tool_used": [-0.01, 0.01]},
      "signature_features": ["operation"],
      "objective_feature": "objective",
      "episode_start_path": "values.episode_start",
      "objective_min": -1.0,
      "objective_max": 1.0,
      "transition_values": {"operation": ["apply_potion"]},
      "terminal_values": {"operation": ["place_in_cauldron"]},
      "terminal_reward_map": {
        "-1.0": 0.0,
        "-0.3333333333333333": 0.1111111111111111,
        "0.3333333333333333": 0.2222222222222222,
        "1.0": 1.0
      },
      "idle_values": {"operation": ["no_op"]},
      "idle_reward": 0.16666666666666666,
      "unresolved_entity_reward": 0.16666666666666666,
      "minimum_support": 1,
      "model_weight": 1.0,
      "novelty_bonus": 0.35,
      "transition_cost": 0.01,
      "planning_horizon": 15,
      "planning_beam_width": 256,
      "planning_minimum_coverage": 0.6666666666666666,
      "planning_improvement_margin": 0.0,
      "effect_features": ["axis_0", "axis_1", "axis_2"],
      "effect_catalog": [
        {"axis_0": 2.0}, {"axis_0": -2.0},
        {"axis_1": 2.0}, {"axis_1": -2.0},
        {"axis_2": 2.0}, {"axis_2": -2.0}
      ],
      "effect_assignment": "bijective",
      "reversible_effects": true,
      "topology_model": "factorized_conjunctive",
      "topology_feature_values": {
        "axis_0": [-1.0, 1.0],
        "axis_1": [-1.0, 1.0],
        "axis_2": [-1.0, 1.0]
      },
      "topology_require_connected": true,
      "topology_complexity_prior": [],
      "empirical_topology_inference": false,
      "belief_planning_horizon": 8,
      "belief_max_worlds": 256,
      "belief_minimum_evidence": 0,
      "belief_planning_mode": "qmdp",
      "belief_exact_world_min": 20000,
      "belief_exact_world_limit": 40000,
      "belief_exact_horizon": 3,
      "belief_qmdp_max_worlds": 64,
      "belief_beam_width": 32,
      "belief_minimum_value": 0.0,
      "belief_information_weight": 0.0,
      "effect_tolerance": 0.01,
      "effect_uncertainty_weight": 0.35,
      "objective_model": "linear",
      "objective_predictors": ["axis_0", "axis_1", "axis_2"],
      "objective_intercept": false,
      "objective_hypotheses": [
        {"coefficients": {"axis_0": 0.3333333333333333, "axis_1": 0.3333333333333333, "axis_2": 0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": 0.3333333333333333, "axis_1": 0.3333333333333333, "axis_2": -0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": 0.3333333333333333, "axis_1": -0.3333333333333333, "axis_2": 0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": 0.3333333333333333, "axis_1": -0.3333333333333333, "axis_2": -0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": -0.3333333333333333, "axis_1": 0.3333333333333333, "axis_2": 0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": -0.3333333333333333, "axis_1": 0.3333333333333333, "axis_2": -0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": -0.3333333333333333, "axis_1": -0.3333333333333333, "axis_2": 0.3333333333333333}, "intercept": 0.0},
        {"coefficients": {"axis_0": -0.3333333333333333, "axis_1": -0.3333333333333333, "axis_2": -0.3333333333333333}, "intercept": 0.0}
      ]
    }
  },
  "latent_belief": {"historical_regime_weight": 1.0},
  "context": {"feature_paths": [], "event_types": [], "max_samples": 4096},
  "training": {
    "enabled": true,
    "min_samples": 24,
    "retrain_interval": 4096,
    "dimensions": 96,
    "epochs": 25,
    "learning_rate": 0.04
  }
}
```

This declaration supplies the public task ontology. It defines the possible signed-axis effects and objective family. It contains no episode-specific mapping, chemistry identity, latent graph, or future result.

After Domain creation, compare every returned sequential and latent-belief field with the submitted declaration. Stop if the API changes or removes a field.

## 6. Query one active step

Send this request to `POST /api/v1/domains/{domain_id}/query` before the environment action:

```json
{
  "session_id": "<current-episode-session>",
  "question": "Choose one native slot action for the current public state.",
  "relation": "select_symbolic_slot_action",
  "selection_mode": "auto",
  "allow_exploration": true,
  "update_memory_state": false,
  "top_k": 40,
  "return_fields": ["selection", "decision_id", "learning_state", "ranked_hypotheses"],
  "context": "<exact public pre-action context>"
}
```

Use the explicit selected policy from the response. If the response has no explicit selected marker, accept `ranked_hypotheses[0]` only when all 40 legal policies have numeric selection scores and the first score is strictly greater than the second. Stop on an unresolved tie or incomplete ranking.

Map the selected policy directly to its native action ID. Execute that ID once.

## 7. Submit one feedback event

Read the public next observation, raw `dm_env` reward, and terminal flag after execution. Normalize the policy reward with:

```text
policy_reward = (native_reward_raw + 3) / 18
```

Send one execution feedback request to `POST /api/v1/domains/{domain_id}/feedback`:

```json
{
  "session_id": "<current-episode-session>",
  "decision_id": "<query decision identifier when returned>",
  "feedback_kind": "execution",
  "outcome": "native_reward",
  "relation": "select_symbolic_slot_action",
  "policy": "<executed policy>",
  "context": "<exact public pre-action context>",
  "values": {
    "reward": "<normalized policy reward>",
    "step_reward": "<normalized policy reward>",
    "native_reward_raw": "<exact raw environment reward>",
    "native_action_id": "<executed native action ID>",
    "next_state": "<exact public post-action context>",
    "terminal": "<exact environment terminal flag>",
    "delay_steps": 0
  },
  "metadata": {
    "episode_id": "<stable opaque identifier for this 200-step episode>",
    "step": "<global step index>",
    "episode_step": "<0 through 199>"
  }
}
```

Bind feedback to the exact pre-action context, decision, policy, and executed action. Submit it once. If a mutation response is ambiguous, stop and reconcile the exposed learner version before any continuation. Never replay an ambiguous mutation without an idempotency guarantee.

## 8. Execute the series

Use this order for every episode:

1. Create a fresh episode session.
2. Reset the fixed-bank environment entry.
3. Map the 39 public values and set `episode_start`.
4. If all three stones are used, execute native action 0 and continue.
5. Query Adapt-1.
6. Execute the selected native action.
7. Read the public next observation, reward, and terminal flag.
8. Submit one bound feedback event.
9. Advance the global and episode step counters.
10. Continue until 200 native steps are complete.
11. Record the episode raw return in process memory for aggregate scoring.
12. Start the next episode with a new session and the same Domain.

After episode 25, write one aggregate summary with episode count, native step count, total return, mean return, first-half mean, second-half mean, feedback-gap count, and LLM-call count. Do not serialize action histories, observations, response bodies, decision identifiers, learner snapshots, task chemistry, or per-episode returns in the publishable output.

## 9. Reproduction checks

- fixed source commit and fixed-bank order
- entries 0 through 25 processed once
- 200 native steps per episode
- 5,200 native steps total
- one empty Domain at the start
- one Domain across the complete series
- fresh session per episode
- session held across all ten trials in an episode
- 39 public observation values plus `episode_start`
- all 40 native actions available on active queries
- no hidden task field in query or feedback
- no application-side action selection
- query before action and feedback after action
- exact raw reward used for scoring
- normalized reward used only for policy feedback
- no missing feedback admission for a queried action
- no query or feedback during an inactive trial tail
- zero LLM calls
- aggregate output only

Results are stochastic. A clean 26-episode mean should be compared with the descriptive range `233.54` to `270.46`. The recorded mean was `252.00`. An exact equality check is invalid for this protocol.

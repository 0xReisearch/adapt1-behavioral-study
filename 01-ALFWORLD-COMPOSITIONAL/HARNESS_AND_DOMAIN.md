# ALFWorld harness and Domain build guide

## 1. Harness boundary

The harness connects the official text environment to two Adapt-1 components:

- `CandidatePolicyLearner` ranks the legal actions supplied by the environment.
- `EpisodicPlanMemory` reconstructs compositional plan stages from successful training episodes.

The environment remains responsible for legal action generation, state transitions, termination, reward, and success scoring. During evaluation, the harness never reads the expert plan.

The included implementation is divided as follows:

| File | Responsibility |
|---|---|
| `harness/config_tw.yaml` | Official ALFWorld paths, TextWorld mode, six task families, 50-step limit, and admissible action surface |
| `harness/domain.json` | Stable learner contract for context, candidate, history, novelty, and training boundaries |
| `harness/run_neuroadapt.py` | Feature adapter, training collection, policy ranking, compositional plan ranking, environment execution, and scoring |
| `harness/run_multiseed.py` | Shared training collection, five independent policy fits, ID/OOD evaluation, and aggregate statistics |
| `core/plan_memory.py` | Compositional plan representation and retrieval used by the evaluation harness |

## 2. Design the Domain

Use one episode-local `TextActionDomain` for each active game. Reset it at every environment reset.

### State and context

Build the query context from information available before action selection:

- normalized step index: `step / 50`
- normalized legal candidate count: `candidate_count / 32`
- whether the observation changed
- normalized goal length
- signed-hash features for goal tokens and goal bigrams
- signed-hash features for observation tokens and observation bigrams
- signed-hash features from prior action and observation tokens
- action-head bigrams and trigrams from recent actions
- the last four actions
- the last two observations

Use deterministic signed token hashing. The complete representation uses 768 dimensions. Keep the hashing procedure and bucket counts unchanged across collection and evaluation.

### Candidate representation

For every legal action, compute:

- overlap with goal tokens
- coverage of goal tokens
- overlap with current observation tokens
- overlap with episode history
- exact action repeat count
- argument visit count
- equality with the previous action and the action two steps back
- count for the same observation-action pair
- maximum similarity to the four most recent actions
- normalized token count
- whether the primary argument occurs in the goal
- whether the primary argument occurs in the observation
- signed-hash features for action tokens, action bigrams, and action head

The environment supplies the candidate set. The harness does not generate free-form actions.

### Learning declaration

Keep the included `harness/domain.json` unchanged for a comparable run. Its key settings are:

```json
{
  "domain_id": "interactive_candidate_policy",
  "metric": "binary_episode_success",
  "action_budget": 50,
  "learning": {
    "kind": "candidate_policy",
    "discount": 0.97,
    "dimensions": 768,
    "hidden_dimensions": 256,
    "reference_actions": "training_split_only",
    "evaluation_expert_access": false
  }
}
```

### Compositional plan state

Build plan state only from successful official training episodes. Preserve entity type and numbered instance identity separately. Represent repeated goals as separate obligations. Track entity progress through the induced source, intermediate, and destination stages.

At evaluation time, start a plan session from the current goal. The session may rank legal actions, score milestone progress, estimate goal bindings, and advance after the chosen legal action. It receives no evaluation label or expert action.

## 3. Build the training harness

1. Initialize `AlfredTWEnv` with split `train` and the included configuration.
2. Reset a batch and extract the templated goal from each initial observation.
3. Create one `TextActionDomain` per game.
4. Read admissible commands from `infos["admissible_commands"]`.
5. Read the official expert action only for the training split.
6. Compute the pre-action context and features for every legal candidate.
7. Call `CandidatePolicyLearner.observe` with the expert action marked as a reference action.
8. Execute the expert action in the official environment.
9. Add the reward, terminal flag, and successful episode plan to the current training collection.
10. Update episode-local history with the executed action and resulting observation.
11. Stop at termination or 50 decisions.
12. Continue until 3,520 training games have been collected.

Collect the shared training set once with seed 42. Copy the resulting decision objects into five newly initialized learners. Fit each learner for 45 epochs under seeds 0 through 4.

## 4. Build the frozen evaluation harness

1. Select `eval_in_distribution` for ID or `eval_out_of_distribution` for OOD.
2. Reset the official environment and create fresh episode-local state.
3. Read the current observation, goal, and admissible commands.
4. Compute the context and candidate features.
5. Rank candidates with the fitted candidate policy.
6. Rank the same legal candidates with the goal-conditioned plan session.
7. Compute milestone, goal-binding, novelty, grounded-goal, and consistency terms.
8. Combine ranks with the fixed weights below.
9. Execute the highest-ranked legal action.
10. Update only episode-local action and plan progress.
11. Do not call `observe`, refit the policy, ingest success labels, or read expert actions.
12. Score success after environment termination.

Use these fixed evaluation weights:

| Component | Weight |
|---|---:|
| Candidate policy rank | 0.15 |
| Plan rank | 0.05 |
| Milestone score | 0.50 |
| Goal overlap and coverage | 0.05 |
| Novelty | 0.05 |
| Grounded goal score | 0.20 |

Apply the included repeat and consistency penalties after the weighted sum. Preserve the deterministic candidate-name tie break.

## 5. Split isolation checks

Before accepting a result, confirm all of the following:

- training reads only the official training split
- evaluation loads only the fitted learner and training-derived plan state
- expert actions are unavailable in both evaluation splits
- evaluation rewards and success labels are used only for reporting
- a fresh `TextActionDomain` is created for every game
- ID contains 140 games per seed
- OOD contains 134 games per seed
- every run covers all six task families
- macro success is the unweighted mean of the six family success rates

The source harness writes local diagnostic output while running. None of that prior output is included in this package.

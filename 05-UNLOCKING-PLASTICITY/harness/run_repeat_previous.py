from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from neuroadapt import NeuroadaptEngine
from neuroadapt.domain import DomainRegistry
from popgym.envs.repeat_previous import (
    RepeatPreviousEasy,
    RepeatPreviousMedium,
)


POLICIES = tuple(f"emit_{value}" for value in range(4))


def domain_learning(
    *, temporal: bool, cup: bool, maximum_lag: int = 16
) -> dict[str, Any]:
    posterior: dict[str, Any] = {
        "enabled": True,
        "scope": "relation",
        "learning_rate": 2.0,
        "forgetting_factor": 0.985,
        "minimum_feedback": 3,
        "minimum_support": 2.0,
        "variance_floor": 0.0025,
        "complexity_penalty": 0.01,
        "information_gain_weight": 0.0,
    }
    if cup:
        posterior["plasticity"] = {
            "enabled": True,
            "forgetting_factor": 0.99,
            "maximum_order": 2,
            "minimum_support": 16,
            "birth_threshold": 0.0025,
            "inhibition_threshold": 0.0025,
            "confidence_z": 1.644854,
            "complexity_penalty": 0.001,
            "weight_temperature": 0.025,
            "maximum_structures": 128,
            "maximum_pending": 256,
            "maximum_active": 4,
            "quantization_bins": 5,
            "prior_strength": 1.0,
            "loss": "log",
        }
    return {
        "enabled": True,
        "context": {
            "feature_paths": ["values.observation"],
            "event_types": [],
            "max_samples": 4096,
        },
        "temporal_context": {
            "enabled": temporal,
            "episode_path": "metadata.episode_id",
            "step_path": "metadata.step",
            "input_paths": ["values.observation"],
            "maximum_lag": maximum_lag,
        },
        "reward": {
            "aggregation": "weighted_mean",
            "components": [
                {
                    "field": "values.correct",
                    "goal": "maximize",
                    "min": 0.0,
                    "max": 1.0,
                    "weight": 1.0,
                    "required": True,
                }
            ],
        },
        "policy": {
            "transfer_strength": 0.0,
            "action_transfer_strength": 0.0,
            "min_context_observations": 1,
            "model_max_weight": 0.9,
            "model_ood_threshold": 1.0,
            "ood_confidence_penalty": 0.0,
            "exploration_mode": "exploit",
            "exploration_strength": 0.0,
            "abstain_on_ood": False,
        },
        "posterior": posterior,
        "latent_belief": {"enabled": False},
        "training": {
            "enabled": True,
            "min_samples": 64,
            "retrain_interval": 128,
            "dimensions": 256,
            "epochs": 75,
            "learning_rate": 0.02,
        },
    }


def choose_policy(response: dict[str, Any]) -> str:
    selected = (response.get("selection") or {}).get("selected_policy")
    if selected in POLICIES:
        return str(selected)
    raise RuntimeError("Domain returned no legal selected policy")


def normalized_candidate_probabilities(
    ranked_hypotheses: list[dict[str, Any]],
) -> list[float]:
    by_policy = {
        str(row.get("policy")): float(row.get("selection_score") or 0.0)
        for row in ranked_hypotheses
        if row.get("policy") in POLICIES
    }
    if set(by_policy) != set(POLICIES):
        raise RuntimeError("Domain did not return all four legal candidates")
    clipped = [max(0.0, by_policy[policy]) for policy in POLICIES]
    denominator = sum(clipped)
    if denominator <= 1e-12:
        return [0.25] * 4
    return [value / denominator for value in clipped]


def state_fingerprint(learner: Any) -> dict[str, Any]:
    posterior = learner.posterior_adjudicator.export_state()
    return {
        "learner_version": learner.version,
        "sample_count": len(learner.samples),
        "model_version": learner.model.version if learner.model is not None else None,
        "model_trained_samples": (
            learner.model.trained_samples if learner.model is not None else None
        ),
        "posterior_sha256": hashlib.sha256(
            json.dumps(posterior, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def await_final_training(registry: DomainRegistry, key: tuple[str, str, str]) -> None:
    definition = registry.domains[(key[0], key[1])]
    definition.learning["training"]["enabled"] = False
    deadline = time.time() + 600.0
    while time.time() < deadline:
        learner = registry.contextual_learners[key]
        future = registry._policy_training_futures.get(key)
        if future is not None and not future.done():
            time.sleep(0.1)
            continue
        if learner.model is not None and learner.training_status.get("status") != "running":
            return
        if learner.training_status.get("status") == "failed":
            raise RuntimeError(f"policy training failed: {learner.training_status}")
        time.sleep(0.1)
    raise TimeoutError("final policy model did not finish within 600 seconds")


def run(
    *,
    variant: str,
    episodes: int,
    seed_start: int,
    learner_seed: int,
    eval_episodes: int = 0,
    eval_seed_start: int = 1700,
    environment_name: str = "easy",
    maximum_lag: int = 16,
) -> dict[str, Any]:
    temporal = variant != "memoryless"
    cup = variant == "tcp_cup"
    owner = f"repeat-previous-{variant}-{learner_seed}"
    domain_id = f"repeat-previous-{environment_name}-{variant}-{learner_seed}"
    engine = NeuroadaptEngine(autonomous_training=False)
    registry = DomainRegistry(engine)
    registry.create_domain(
        owner_session_id=owner,
        domain_id=domain_id,
        description=(
            "Select one value from the legal output alphabet using the current "
            "public observation and online execution feedback."
        ),
        schema={
            "entities": ["public_observation", "legal_output"],
            "relations": ["selects_output"],
            "signals": ["observation", "execution_correct"],
            "event_types": ["execution_feedback"],
            "constraints": {
                "observation": {"enum": [0, 1, 2, 3]},
                "legal_output": {"enum": [0, 1, 2, 3]},
            },
        },
        hypotheses=[
            {
                "name": f"legal-output-{value}",
                "relation": "selects_output",
                "policy": f"emit_{value}",
                "policy_features": {"output_value": value},
                "predicts": ["A legal output value is selected."],
                "weight": 1.0,
            }
            for value in range(4)
        ],
        query_templates={
            "feedback_outcomes": ["correct", "incorrect", "unscored"]
        },
        learning=domain_learning(
            temporal=temporal,
            cup=cup,
            maximum_lag=maximum_lag,
        ),
    )

    episode_returns: list[float] = []
    evaluation_returns: list[float] = []
    action_counts: Counter[str] = Counter()
    cup_applied = 0
    source_counts: Counter[str] = Counter()
    frozen_log_losses: list[float] = []
    frozen_brier_scores: list[float] = []
    frozen_correct: list[bool] = []
    global_step = 0
    started = time.time()

    episode_specs = [
        ("train", index, seed)
        for index, seed in enumerate(range(seed_start, seed_start + episodes), start=1)
    ] + [
        ("frozen_evaluation", index, seed)
        for index, seed in enumerate(
            range(eval_seed_start, eval_seed_start + eval_episodes),
            start=1,
        )
    ]
    learner_key = (owner, domain_id, owner)
    frozen_before: dict[str, Any] | None = None
    environments = {
        "easy": (RepeatPreviousEasy, 4),
        "medium": (RepeatPreviousMedium, 32),
    }
    environment_type, target_history_length = environments[environment_name]
    for phase, episode_index, seed in episode_specs:
        if phase == "frozen_evaluation" and frozen_before is None:
            await_final_training(registry, learner_key)
            frozen_before = state_fingerprint(registry.contextual_learners[learner_key])
        environment = environment_type()
        observation, _ = environment.reset(seed=seed)
        observation_history: list[int] = []
        episode_return = 0.0
        step = 1
        while True:
            observation_history.append(int(observation))
            context = {
                "values": {"observation": int(observation)},
                "metadata": {
                    "episode_id": f"repeat-previous-{seed}",
                    "step": step,
                },
            }
            response = registry.query(
                domain_id,
                owner_session_id=owner,
                session_id=owner,
                question="Select one legal output value.",
                top_k=4,
                return_fields=["selection", "decision_id", "ranked_hypotheses"],
                relation="selects_output",
                context=context,
                selection_mode="exploit",
                allow_exploration=phase == "train",
            )
            if not response.get("decision_id"):
                raise RuntimeError("Domain query returned no decision_id")
            ranked_hypotheses = response.get("ranked_hypotheses") or []
            probabilities = normalized_candidate_probabilities(ranked_hypotheses)
            policy = choose_policy(response)
            action = int(policy.removeprefix("emit_"))
            selected_row = next(
                row for row in ranked_hypotheses if row.get("policy") == policy
            )
            diagnostics = selected_row.get("contextual_policy") or {}
            adjudication = diagnostics.get("posterior_adjudication") or {}
            cup_diagnostics = adjudication.get("counterfactual_utility_plasticity") or {}
            cup_applied += int(bool(cup_diagnostics.get("applied")))
            for source in adjudication.get("sources") or []:
                source_counts[str(source.get("source") or "unknown")] += 1

            next_observation, native_reward, terminated, truncated, _ = environment.step(action)
            native_reward = float(native_reward)
            measured = native_reward != 0.0
            task_success = 1.0 if native_reward > 0.0 else 0.0 if native_reward < 0.0 else 0.5
            outcome = "correct" if native_reward > 0.0 else "incorrect" if native_reward < 0.0 else "unscored"
            if measured:
                target = observation_history[-target_history_length]
                if (action == target) != (native_reward > 0.0):
                    raise RuntimeError("Evaluator target and environment reward disagree")
                if phase == "frozen_evaluation":
                    probability = max(probabilities[target], 1e-12)
                    frozen_log_losses.append(-math.log(probability))
                    frozen_brier_scores.append(
                        sum(
                            (probabilities[index] - (1.0 if index == target else 0.0))
                            ** 2
                            for index in range(4)
                        )
                    )
                    frozen_correct.append(action == target)
            update: dict[str, Any] = {}
            if phase == "train":
                update = registry.feedback(
                    domain_id,
                    owner_session_id=owner,
                    session_id=owner,
                    outcome=outcome,
                    feedback_kind="execution",
                    decision_id=response["decision_id"],
                    relation="selects_output",
                    policy=policy,
                    context=context,
                    values={"correct": task_success, "native_reward": native_reward},
                    metadata={
                        "episode_id": f"repeat-previous-{seed}",
                        "step": step,
                        "global_step": global_step,
                    },
                )
            action_counts[policy] += 1
            episode_return += native_reward
            global_step += 1
            observation = next_observation
            step += 1
            if terminated or truncated:
                break
        if phase == "train":
            episode_returns.append(episode_return)
        else:
            evaluation_returns.append(episode_return)
        print(
            f"variant={variant} phase={phase} episode={episode_index}/"
            f"{episodes if phase == 'train' else eval_episodes} seed={seed} "
            f"return={episode_return:.6f}",
            flush=True,
        )

    if frozen_before is None:
        await_final_training(registry, learner_key)
    learner = registry.contextual_learners[(owner, domain_id, owner)]
    frozen_after = state_fingerprint(learner) if frozen_before is not None else None
    plasticity = learner.posterior_adjudicator.utility_plasticity
    return {
        "benchmark": f"POPGym RepeatPrevious{environment_name.title()}",
        "source_commit": "410d5aa626dae8024f498354d8781a0d1870c399",
        "variant": variant,
        "protocol": {
            "episodes": episodes,
            "seed_start": seed_start,
            "learner_seed": learner_seed,
            "evaluation_episodes": eval_episodes,
            "evaluation_seed_start": eval_seed_start if eval_episodes else None,
            "prediction": "prequential",
            "evaluation_mode": "frozen_no_feedback",
            "prior_task_state": "empty",
            "relation_count": 1,
            "equal_action_priors": True,
            "forced_coverage": False,
            "target_lag_declared": False,
            "future_observation_supplied": False,
            "maximum_lag": maximum_lag,
            "evaluator_target_history_length": target_history_length,
        },
        "interactions": global_step,
        "episode_returns": episode_returns,
        "mean_return": statistics.fmean(episode_returns),
        "first_episode": episode_returns[0],
        "warm_mean": statistics.fmean(episode_returns[1:]) if episodes > 1 else None,
        "final_10_mean": statistics.fmean(episode_returns[-10:]) if episodes >= 10 else None,
        "frozen_evaluation_returns": evaluation_returns,
        "frozen_evaluation_mean": (
            statistics.fmean(evaluation_returns) if evaluation_returns else None
        ),
        "frozen_action_accuracy": (
            statistics.fmean(frozen_correct) if frozen_correct else None
        ),
        "frozen_nll": (
            statistics.fmean(frozen_log_losses) if frozen_log_losses else None
        ),
        "frozen_brier": (
            statistics.fmean(frozen_brier_scores) if frozen_brier_scores else None
        ),
        "frozen_state_before": frozen_before,
        "frozen_state_after": frozen_after,
        "frozen_state_unchanged": (
            frozen_before == frozen_after if frozen_before is not None else None
        ),
        "action_counts": dict(action_counts),
        "posterior_source_counts": dict(source_counts),
        "cup_applied_decisions": cup_applied,
        "cup_structures": plasticity.structures("selects_output", active_only=True),
        "learner": {
            "version": learner.version,
            "sample_count": len(learner.samples),
            "model_type": learner.model.model_type if learner.model is not None else None,
            "training_status": learner.training_status,
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("memoryless", "tcp", "tcp_cup"),
        default="tcp_cup",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-seed-start", type=int, default=1700)
    parser.add_argument("--learner-seed", type=int, default=65000)
    parser.add_argument(
        "--environment",
        choices=("easy", "medium"),
        default="easy",
    )
    parser.add_argument("--maximum-lag", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        variant=args.variant,
        episodes=args.episodes,
        seed_start=args.seed_start,
        learner_seed=args.learner_seed,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        environment_name=args.environment,
        maximum_lag=args.maximum_lag,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

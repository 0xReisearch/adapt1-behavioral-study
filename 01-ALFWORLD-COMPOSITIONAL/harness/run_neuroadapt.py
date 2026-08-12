from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from neuroadapt.candidate_policy import CandidatePolicyLearner
from neuroadapt.plan_memory import EpisodicPlan, EpisodicPlanMemory, PlanStep


TOKEN = re.compile(r"[a-z0-9]+")
TASK = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def hash_features(prefix: str, values: list[str], buckets: int = 128) -> dict[str, float]:
    output: dict[str, float] = defaultdict(float)
    for value in values:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=9).digest()
        bucket = int.from_bytes(digest[:8], "big") % buckets
        sign = 1.0 if digest[8] & 1 else -1.0
        output[f"{prefix}_{bucket:03d}"] += sign
    return dict(output)


def ngrams(values: list[str], size: int) -> list[str]:
    return ["_".join(values[index : index + size]) for index in range(max(0, len(values) - size + 1))]


def entity_references(action: str) -> tuple[str, ...]:
    values = tokens(action)
    return tuple(
        f"{values[index - 1]} {value}"
        for index, value in enumerate(values)
        if index > 0 and value.isdigit()
    )


class TextActionDomain:
    def __init__(self, goal: str) -> None:
        self.goal = goal
        self.goal_tokens = tokens(goal)
        self.action_counts: Counter[str] = Counter()
        self.argument_counts: Counter[str] = Counter()
        self.state_action_counts: Counter[tuple[str, str]] = Counter()
        self.history_tokens: Counter[str] = Counter()
        self.recent_actions: list[str] = []
        self.recent_observations: list[str] = []
        self.previous_observation = ""
        self.current_observation = ""

    def context(self, observation: str, step: int, candidates: list[str]) -> dict[str, Any]:
        observation_tokens = tokens(observation)
        values: dict[str, Any] = {
            "step": float(step) / 50.0,
            "candidate_count": float(len(candidates)) / 32.0,
            "observation_changed": float(observation != self.previous_observation),
            "goal_length": float(len(self.goal_tokens)) / 16.0,
        }
        values.update(hash_features("goal", self.goal_tokens + ngrams(self.goal_tokens, 2)))
        values.update(hash_features("observation", observation_tokens + ngrams(observation_tokens, 2)))
        values.update(hash_features("history", list(self.history_tokens.elements())[-128:]))
        recent_heads = [tokens(action)[0] for action in self.recent_actions if tokens(action)]
        values.update(hash_features("action_sequence", ngrams(recent_heads, 2) + ngrams(recent_heads, 3)))
        for offset, action in enumerate(reversed(self.recent_actions[-4:])):
            action_tokens = tokens(action)
            values.update(hash_features(f"recent_action_{offset}", action_tokens + ngrams(action_tokens, 2), 64))
        for offset, recent in enumerate(reversed(self.recent_observations[-2:])):
            recent_tokens = tokens(recent)
            values.update(hash_features(f"recent_observation_{offset}", recent_tokens, 64))
        return values

    def candidates(self, observation: str, candidates: list[str]) -> dict[str, dict[str, Any]]:
        self.current_observation = observation
        observation_set = set(tokens(observation))
        goal_set = set(self.goal_tokens)
        history_set = set(self.history_tokens)
        output: dict[str, dict[str, Any]] = {}
        for action in candidates:
            action_tokens = tokens(action)
            action_set = set(action_tokens)
            head = action_tokens[0] if action_tokens else ""
            primary_argument = next(
                (value for value in action_tokens[1:] if not value.isdigit()), ""
            )
            argument = " ".join(action_tokens[1:])
            recent_token_sets = [set(tokens(value)) for value in self.recent_actions[-4:]]
            features: dict[str, Any] = {
                "goal_overlap": len(action_set & goal_set) / max(1, len(action_set)),
                "goal_coverage": len(action_set & goal_set) / max(1, len(goal_set)),
                "observation_overlap": len(action_set & observation_set) / max(1, len(action_set)),
                "history_overlap": len(action_set & history_set) / max(1, len(action_set)),
                "exact_repeat_count": float(self.action_counts[action]) / 10.0,
                "argument_visit_count": float(self.argument_counts[argument]) / 10.0,
                "same_as_previous": float(bool(self.recent_actions and action == self.recent_actions[-1])),
                "same_as_two_back": float(len(self.recent_actions) >= 2 and action == self.recent_actions[-2]),
                "state_action_repeat_count": float(
                    self.state_action_counts[(observation, action)]
                ),
                "recent_action_similarity": max(
                    (len(action_set & recent) / max(1, len(action_set | recent)) for recent in recent_token_sets),
                    default=0.0,
                ),
                "token_count": len(action_tokens) / 10.0,
                "head_position": 1.0,
                "primary_argument_goal": float(primary_argument in goal_set),
                "primary_argument_observation": float(primary_argument in observation_set),
            }
            features.update(hash_features("action", action_tokens + ngrams(action_tokens, 2)))
            features.update(hash_features("head", [head], buckets=32))
            output[action] = features
        return output

    def update(self, action: str, observation: str) -> None:
        action_tokens = tokens(action)
        argument = " ".join(action_tokens[1:])
        self.action_counts[action] += 1
        self.argument_counts[argument] += 1
        self.state_action_counts[(self.current_observation, action)] += 1
        self.history_tokens.update(action_tokens)
        self.history_tokens.update(tokens(observation))
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-8:]
        self.recent_observations.append(observation)
        self.recent_observations = self.recent_observations[-4:]
        self.previous_observation = observation


def load_config(config_path: Path, *, train_games: int = -1, eval_games: int = -1) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset"]["num_train_games"] = int(train_games)
    config["dataset"]["num_eval_games"] = int(eval_games)
    config["general"]["training_method"] = "dagger"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = 50
    return config


def extract_goal(observation: str) -> str:
    match = TASK.search(observation)
    if not match:
        raise RuntimeError("task statement missing from initial observation")
    return match.group(1).strip()


def task_type(gamefile: str) -> str:
    return next(
        (
            value
            for part in Path(gamefile).parts
            for value in TASK_TYPES.values()
            if part.startswith(value + "-")
        ),
        "unknown",
    )


def load_plan_memory(trace_path: Path) -> EpisodicPlanMemory:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            episodes[int(row["episode"])].append(row)
    plans = []
    for episode, rows in episodes.items():
        rows.sort(key=lambda row: int(row["step"]))
        if not rows or not bool(rows[-1].get("won")):
            continue
        plans.append(EpisodicPlan(
            plan_id=f"reference-{episode}",
            goal=str(rows[0]["goal"]),
            steps=tuple(
                PlanStep(observation=str(row["observation"]), action=str(row["selected"]))
                for row in rows
            ),
            utility=1.0,
        ))
    return EpisodicPlanMemory(plans)


def collect_reference(
    learner: CandidatePolicyLearner,
    *,
    config_path: Path,
    episodes: int,
    trace_path: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = load_config(config_path, train_games=episodes)
    if episodes % batch_size:
        raise ValueError("training episodes must be divisible by batch size")
    environment = AlfredTWEnv(config, "train").init_env(batch_size)
    completed = 0
    decisions = 0
    successes = 0
    started = time.time()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        while completed < episodes:
            observations, infos = environment.reset()
            goals = [extract_goal(observation) for observation in observations]
            domains = [TextActionDomain(goal) for goal in goals]
            gamefiles = list(infos["extra.gamefile"])
            active = [True] * batch_size
            won = [False] * batch_size
            for step in range(50):
                actions: list[str] = []
                decision_rows: list[tuple[dict[str, Any], dict[str, dict[str, Any]], str] | None] = []
                for index in range(batch_size):
                    candidates = list(infos["admissible_commands"][index])
                    if not active[index]:
                        actions.append("look" if "look" in candidates else candidates[0])
                        decision_rows.append(None)
                        continue
                    expert = infos["extra.expert_plan"][index][0]
                    if expert not in candidates:
                        expert = "look" if "look" in candidates else candidates[0]
                    actions.append(expert)
                    decision_rows.append((
                        domains[index].context(observations[index], step, candidates),
                        domains[index].candidates(observations[index], candidates),
                        expert,
                    ))
                next_observations, rewards, dones, next_infos = environment.step(actions)
                for index, decision_row in enumerate(decision_rows):
                    if decision_row is None:
                        continue
                    context, candidate_features, expert = decision_row
                    won[index] = bool(next_infos["won"][index])
                    learner.observe(
                        episode_id=f"train-{completed + index}",
                        step=step,
                        context=context,
                        candidates=candidate_features,
                        selected=expert,
                        reward=float(rewards[index]),
                        terminal=bool(dones[index]),
                        reference_action=True,
                    )
                    trace.write(json.dumps({
                        "episode": completed + index,
                        "step": step,
                        "gamefile": gamefiles[index],
                        "goal": goals[index],
                        "observation": observations[index],
                        "candidate_count": len(candidate_features),
                        "selected": expert,
                        "reward": float(rewards[index]),
                        "terminal": bool(dones[index]),
                        "won": won[index],
                    }) + "\n")
                    decisions += 1
                    domains[index].update(expert, next_observations[index])
                    if dones[index]:
                        active[index] = False
                observations, infos = next_observations, next_infos
                if not any(active):
                    break
            successes += sum(map(int, won))
            completed += batch_size
            if completed % 100 == 0:
                print(f"reference {completed}/{episodes}: decisions={decisions} success={successes}", flush=True)
    environment.close()
    return {
        "episodes": completed,
        "decisions": decisions,
        "successes": successes,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def evaluate(
    learner: CandidatePolicyLearner,
    *,
    plan_memory: EpisodicPlanMemory | None,
    plan_weight: float,
    milestone_weight: float,
    goal_weight: float,
    novelty_weight: float,
    grounded_weight: float,
    config_path: Path,
    split: str,
    episodes: int,
    trace_path: Path,
    batch_size: int = 20,
) -> dict[str, Any]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    train_eval = "eval_in_distribution" if split == "id" else "eval_out_of_distribution"
    config = load_config(config_path, eval_games=episodes)
    environment_builder = AlfredTWEnv(config, train_eval)
    total = min(episodes, environment_builder.num_games) if episodes > 0 else environment_builder.num_games
    if total % batch_size:
        raise ValueError("evaluation episodes must be divisible by batch size")
    environment = environment_builder.init_env(batch_size)
    completed = 0
    successes = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "episodes": 0})
    started = time.time()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        while completed < total:
            observations, infos = environment.reset()
            goals = [extract_goal(observation) for observation in observations]
            domains = [TextActionDomain(goal) for goal in goals]
            plan_sessions = [plan_memory.session(goal) if plan_memory is not None else None for goal in goals]
            completed_entities = [set() for _ in goals]
            bound_destinations: list[str | None] = [None for _ in goals]
            gamefiles = list(infos["extra.gamefile"])
            categories = [task_type(gamefile) for gamefile in gamefiles]
            active = [True] * batch_size
            won = [False] * batch_size
            for step in range(50):
                actions: list[str] = []
                rankings: list[list[dict[str, Any]] | None] = []
                for index in range(batch_size):
                    candidates = list(infos["admissible_commands"][index])
                    if not active[index]:
                        actions.append("look" if "look" in candidates else candidates[0])
                        rankings.append(None)
                        continue
                    context = domains[index].context(observations[index], step, candidates)
                    candidate_features = domains[index].candidates(observations[index], candidates)
                    model_ranking = learner.rank(context=context, candidates=candidate_features)
                    model_scores = {str(row["candidate"]): float(row["score"]) for row in model_ranking}
                    plan_session = plan_sessions[index]
                    plan_scores = (
                        {
                            str(row["candidate"]): float(row["score"])
                            for row in plan_session.rank(
                                observation=observations[index],
                                candidates=candidates,
                            )
                        }
                        if plan_session is not None
                        else {}
                    )
                    milestone_scores = (
                        {
                            str(row["candidate"]): float(row["score"])
                            for row in plan_session.milestone_rank(candidates=candidates)
                        }
                        if plan_session is not None
                        else {}
                    )
                    binding_scores = (
                        {
                            candidate: plan_session.goal_binding_score(candidate)
                            for candidate in candidates
                        }
                        if plan_session is not None
                        else {candidate: 1.0 for candidate in candidates}
                    )
                    binding_probabilities = (
                        {
                            candidate: plan_session.goal_binding_probability(candidate)
                            for candidate in candidates
                        }
                        if plan_session is not None
                        else {candidate: 0.0 for candidate in candidates}
                    )
                    grounded_values = {
                        candidate: (
                            float(candidate_features[candidate]["primary_argument_goal"])
                            * float(candidate_features[candidate]["primary_argument_observation"])
                            * binding_probabilities[candidate]
                            * milestone_scores.get(candidate, 0.0)
                        )
                        for candidate in candidates
                    }
                    grounded_scores = grounded_values
                    consistency_scores: dict[str, float] = {}
                    for candidate in candidates:
                        references = entity_references(candidate)
                        consistent = 1.0
                        if (
                            references
                            and references[0] in completed_entities[index]
                            and binding_probabilities[candidate] >= 0.55
                        ):
                            consistent = 0.02
                        if (
                            len(references) >= 2
                            and bound_destinations[index] is not None
                            and plan_session is not None
                            and plan_session.completion_probability(candidate) >= 0.5
                            and references[-1] != bound_destinations[index]
                        ):
                            consistent = 0.02
                        consistency_scores[candidate] = consistent
                    denominator = max(1, len(candidates) - 1)
                    model_rank_scores = {
                        str(row["candidate"]): 1.0 - index / denominator
                        for index, row in enumerate(model_ranking)
                    }
                    plan_order = sorted(
                        candidates,
                        key=lambda candidate: (plan_scores.get(candidate, 0.0), candidate),
                        reverse=True,
                    )
                    plan_rank_scores = {
                        candidate: 1.0 - index / denominator
                        for index, candidate in enumerate(plan_order)
                    }
                    model_weight = max(
                        0.0,
                        1.0
                        - plan_weight
                        - milestone_weight
                        - goal_weight
                        - novelty_weight
                        - grounded_weight,
                    )
                    ranking = [
                        {
                            "candidate": candidate,
                            "score": (0.25 + 0.75 * binding_scores[candidate]) * (
                                model_weight * model_rank_scores[candidate]
                                + plan_weight * plan_rank_scores[candidate]
                                + milestone_weight * milestone_scores.get(candidate, 0.0)
                                + goal_weight * (
                                    0.7 * float(candidate_features[candidate]["goal_overlap"])
                                    + 0.3 * float(candidate_features[candidate]["goal_coverage"])
                                )
                                + novelty_weight * math.exp(
                                    -10.0 * float(candidate_features[candidate]["exact_repeat_count"])
                                )
                                + grounded_weight * grounded_scores[candidate]
                            ) * math.exp(
                                -3.0
                                * float(candidate_features[candidate]["state_action_repeat_count"])
                            ) * consistency_scores[candidate],
                            "model_score": model_scores[candidate],
                            "plan_score": plan_scores.get(candidate, 0.0),
                            "milestone_score": milestone_scores.get(candidate, 0.0),
                            "goal_binding_score": binding_scores[candidate],
                            "goal_binding_probability": binding_probabilities[candidate],
                            "grounded_goal_score": grounded_scores[candidate],
                            "consistency_score": consistency_scores[candidate],
                            "model_rank": model_rank_scores[candidate],
                            "plan_rank": plan_rank_scores[candidate],
                            "goal_score": (
                                0.7 * float(candidate_features[candidate]["goal_overlap"])
                                + 0.3 * float(candidate_features[candidate]["goal_coverage"])
                            ),
                            "novelty_score": math.exp(
                                -10.0 * float(candidate_features[candidate]["exact_repeat_count"])
                            ),
                        }
                        for candidate in candidates
                    ]
                    ranking.sort(key=lambda row: (row["score"], row["candidate"]), reverse=True)
                    actions.append(str(ranking[0]["candidate"]))
                    rankings.append(ranking)
                next_observations, rewards, dones, next_infos = environment.step(actions)
                for index, ranking in enumerate(rankings):
                    if ranking is None:
                        continue
                    won[index] = bool(next_infos["won"][index])
                    trace.write(json.dumps({
                        "episode": completed + index,
                        "step": step,
                        "gamefile": gamefiles[index],
                        "task_type": categories[index],
                        "goal": goals[index],
                        "observation": observations[index],
                        "candidate_count": len(infos["admissible_commands"][index]),
                        "selected": actions[index],
                        "selected_score": ranking[0]["score"],
                        "top5": ranking[:5],
                        "reward": float(rewards[index]),
                        "terminal": bool(dones[index]),
                        "won": won[index],
                    }) + "\n")
                    domains[index].update(actions[index], next_observations[index])
                    if plan_sessions[index] is not None:
                        plan_sessions[index].advance(actions[index])
                        references = entity_references(actions[index])
                        if (
                            references
                            and plan_sessions[index].completion_probability(actions[index]) >= 0.5
                        ):
                            completed_entities[index].add(references[0])
                            if len(references) >= 2 and bound_destinations[index] is None:
                                bound_destinations[index] = references[-1]
                    if dones[index]:
                        active[index] = False
                observations, infos = next_observations, next_infos
                if not any(active):
                    break
            successes += sum(map(int, won))
            for index, category in enumerate(categories):
                by_type[category]["episodes"] += 1
                by_type[category]["successes"] += int(won[index])
            completed += batch_size
            print(f"evaluation {completed}/{total}: success={successes} ({successes/completed:.3%})", flush=True)
    environment.close()
    return {
        "split": split,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(1, completed),
        "by_task_type": {
            key: {**value, "success_rate": value["successes"] / max(1, value["episodes"])}
            for key, value in sorted(by_type.items())
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "expert_available_during_evaluation": False,
        "episodic_plan_memory": {
            "enabled": plan_memory is not None,
            "plan_count": len(plan_memory.plans) if plan_memory is not None else 0,
            "weight": plan_weight,
            "milestone_weight": milestone_weight,
            "goal_weight": goal_weight,
            "novelty_weight": novelty_weight,
            "grounded_weight": grounded_weight,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=140)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--plan-trace", type=Path)
    parser.add_argument("--plan-weight", type=float, default=0.05)
    parser.add_argument("--milestone-weight", type=float, default=0.5)
    parser.add_argument("--goal-weight", type=float, default=0.05)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--grounded-weight", type=float, default=0.2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    domain_config = json.loads((Path(__file__).parent / "domain.json").read_text(encoding="utf-8"))
    learning = domain_config["learning"]
    if args.model:
        learner = CandidatePolicyLearner.from_state(json.loads(args.model.read_text(encoding="utf-8")))
        reference_report = None
        plan_memory = load_plan_memory(args.plan_trace) if args.plan_trace else None
    else:
        learner = CandidatePolicyLearner(
            dimensions=int(learning["dimensions"]),
            hidden_dimensions=int(learning["hidden_dimensions"]),
            discount=float(learning["discount"]),
        )
        reference_report = collect_reference(
            learner,
            config_path=args.config,
            episodes=args.train_episodes,
            trace_path=args.output / "training_trace.jsonl",
            batch_size=args.train_batch_size,
        )
        plan_memory = load_plan_memory(args.output / "training_trace.jsonl")
        training_report = learner.train(epochs=args.epochs, device="auto", negative_limit=24)
        (args.output / "model.json").write_text(
            json.dumps(learner.export_state(), indent=2), encoding="utf-8"
        )
        print(json.dumps({"reference": reference_report, "training": training_report}, indent=2), flush=True)
    evaluation_report = evaluate(
        learner,
        plan_memory=plan_memory,
        plan_weight=max(0.0, min(1.0, float(args.plan_weight))),
        milestone_weight=max(0.0, min(1.0, float(args.milestone_weight))),
        goal_weight=max(0.0, min(1.0, float(args.goal_weight))),
        novelty_weight=max(0.0, min(1.0, float(args.novelty_weight))),
        grounded_weight=max(0.0, min(1.0, float(args.grounded_weight))),
        config_path=args.config,
        split=args.split,
        episodes=args.eval_episodes,
        trace_path=args.output / f"evaluation_{args.split}_trace.jsonl",
        batch_size=args.eval_batch_size,
    )
    family_rates = [
        float(row["success_rate"])
        for row in evaluation_report["by_task_type"].values()
    ]
    macro_success = sum(family_rates) / len(family_rates)
    evaluation_report["macro_success_rate"] = macro_success
    target = 0.859 if args.split == "ood" else 0.852
    report = {
        "domain": domain_config,
        "initialization": {
            "model": str(args.model) if args.model else None,
            "plan_trace": str(args.plan_trace) if args.plan_trace else None,
            "mode": "loaded_state" if args.model else "random_initialization",
            "plan_memory": "loaded_state" if args.plan_trace else "current_run_training_trace",
        },
        "reference_collection": reference_report,
        "training": learner.training_report,
        "evaluation": evaluation_report,
        "published_target": {
            "system": "MemHarness",
            "success_rate": target,
            "aggregation": "six_family_macro",
        },
        "beats_published_target": macro_success > target,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

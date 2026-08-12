#!/usr/bin/env python3
"""Train multiple clean candidate-policy seeds on one fixed ALFWorld corpus."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from neuroadapt.candidate_policy import CandidatePolicyLearner

from run_neuroadapt import collect_reference, evaluate, load_plan_memory


TASK_FAMILIES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def macro_success(report: dict[str, Any]) -> float:
    by_type = report["by_task_type"]
    missing = sorted(set(TASK_FAMILIES) - set(by_type))
    if missing:
        raise RuntimeError(f"Missing task families: {missing}")
    return statistics.fmean(float(by_type[name]["success_rate"]) for name in TASK_FAMILIES)


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # Two-sided 95% Student-t critical values for the supported small seed counts.
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(values), 1.96)
    half_width = t95 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--collection-seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=3520)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--id-batch-size", type=int, default=20)
    parser.add_argument("--ood-batch-size", type=int, default=67)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--plan-weight", type=float, default=0.05)
    parser.add_argument("--milestone-weight", type=float, default=0.50)
    parser.add_argument("--goal-weight", type=float, default=0.05)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--grounded-weight", type=float, default=0.20)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise SystemExit("Provide at least two unique seeds")
    if 140 % args.id_batch_size:
        raise SystemExit("ID batch size must divide 140")
    if 134 % args.ood_batch_size:
        raise SystemExit("OOD batch size must divide 134")

    args.output.mkdir(parents=True, exist_ok=True)
    domain = json.loads((Path(__file__).parent / "domain.json").read_text())
    learning = domain["learning"]
    run_config = {
        "core_revision": "83b10c1c12da636b4ee3f4b8bee2a71f76f26c83",
        "seeds": seeds,
        "collection_seed": args.collection_seed,
        "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size,
        "id_episodes": 140,
        "id_batch_size": args.id_batch_size,
        "ood_episodes": 134,
        "ood_batch_size": args.ood_batch_size,
        "epochs": args.epochs,
        "weights": {
            "plan": args.plan_weight,
            "milestone": args.milestone_weight,
            "goal": args.goal_weight,
            "novelty": args.novelty_weight,
            "grounded": args.grounded_weight,
        },
        "information_boundary": {
            "reference_actions": "official training split only",
            "evaluation_expert_actions": False,
            "prior_model": None,
            "prior_plan_trace": None,
            "shared_across_seeds": "current-run reference decisions and plan memory",
        },
        "primary_metric": "six-family macro success rate",
        "secondary_metric": "episode micro success rate",
        "published_targets": {"id_macro": 0.852, "ood_macro": 0.859},
    }
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    seed_process(args.collection_seed)
    reference_learner = CandidatePolicyLearner(
        dimensions=int(learning["dimensions"]),
        hidden_dimensions=int(learning["hidden_dimensions"]),
        discount=float(learning["discount"]),
    )
    training_trace = args.output / "training_trace.jsonl"
    reference_report = collect_reference(
        reference_learner,
        config_path=args.config,
        episodes=args.train_episodes,
        trace_path=training_trace,
        batch_size=args.train_batch_size,
    )
    plan_memory = load_plan_memory(training_trace)
    reference_audit = {
        **reference_report,
        "decision_objects": len(reference_learner.decisions),
        "plan_count": len(plan_memory.plans),
        "collection_seed": args.collection_seed,
    }
    (args.output / "reference_report.json").write_text(json.dumps(reference_audit, indent=2) + "\n")

    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"=== learner seed {seed} ===", flush=True)
        started = time.time()
        seed_process(seed)
        learner = CandidatePolicyLearner(
            dimensions=int(learning["dimensions"]),
            hidden_dimensions=int(learning["hidden_dimensions"]),
            discount=float(learning["discount"]),
        )
        # Decisions are immutable after collection. A shallow list copy gives every
        # seed exactly the same examples while keeping model state independent.
        learner.decisions = list(reference_learner.decisions)
        training_report = learner.train(
            epochs=args.epochs,
            seed=seed,
            device="auto",
            negative_limit=24,
        )

        seed_dir = args.output / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "model.json").write_text(json.dumps(learner.export_state(), indent=2) + "\n")

        common = {
            "learner": learner,
            "plan_memory": plan_memory,
            "plan_weight": args.plan_weight,
            "milestone_weight": args.milestone_weight,
            "goal_weight": args.goal_weight,
            "novelty_weight": args.novelty_weight,
            "grounded_weight": args.grounded_weight,
            "config_path": args.config,
        }
        id_report = evaluate(
            **common,
            split="id",
            episodes=140,
            trace_path=seed_dir / "evaluation_id_trace.jsonl",
            batch_size=args.id_batch_size,
        )
        ood_report = evaluate(
            **common,
            split="ood",
            episodes=134,
            trace_path=seed_dir / "evaluation_ood_trace.jsonl",
            batch_size=args.ood_batch_size,
        )
        id_report["macro_success_rate"] = macro_success(id_report)
        ood_report["macro_success_rate"] = macro_success(ood_report)
        result = {
            "seed": seed,
            "initialization": {
                "mode": "random_initialization",
                "prior_model": None,
                "prior_plan_trace": None,
                "training_corpus": "shared_current_run_reference_collection",
            },
            "training": training_report,
            "id": id_report,
            "ood": ood_report,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (seed_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        seed_results.append(result)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "id_micro": id_report["success_rate"],
                    "id_macro": id_report["macro_success_rate"],
                    "ood_micro": ood_report["success_rate"],
                    "ood_macro": ood_report["macro_success_rate"],
                }
            ),
            flush=True,
        )
        del learner
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    aggregate = {
        "protocol": run_config,
        "reference_collection": reference_audit,
        "seed_results": seed_results,
        "summary": {
            "id_macro": summarize([row["id"]["macro_success_rate"] for row in seed_results]),
            "id_micro": summarize([row["id"]["success_rate"] for row in seed_results]),
            "ood_macro": summarize([row["ood"]["macro_success_rate"] for row in seed_results]),
            "ood_micro": summarize([row["ood"]["success_rate"] for row in seed_results]),
        },
    }
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

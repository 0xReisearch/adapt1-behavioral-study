from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError((actual, expected))


def contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(item, forbidden) for item in value)
    return False


def without_domain_id(value: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(value))
    clone.pop("domain_id")
    return clone


def verify_artifact_scope() -> None:
    if any(path.is_dir() for path in ROOT.rglob("traces")):
        raise AssertionError("trace directory is present")
    for path in ROOT.rglob("*.json"):
        value = load_json(path)
        if contains_key(value, "trace"):
            raise AssertionError(f"trace key found in {path.relative_to(ROOT)}")
    result_paths = list((ROOT / "results").rglob("*"))
    if any("hard" in path.name.lower() for path in result_paths):
        raise AssertionError("RepeatPreviousHard artifact is present")


def verify_domains() -> None:
    domain_root = ROOT / "domains" / "resolved"
    easy_memoryless = load_json(domain_root / "easy_memoryless.json")
    easy_tcp = load_json(domain_root / "easy_tcp.json")
    easy_tcp_cup = load_json(domain_root / "easy_tcp_cup.json")
    medium_tcp = load_json(domain_root / "medium_tcp.json")
    medium_tcp_cup = load_json(domain_root / "medium_tcp_cup.json")

    assert easy_memoryless["learning"]["temporal_context"]["enabled"] is False
    assert easy_tcp["learning"]["temporal_context"]["enabled"] is True
    assert easy_tcp_cup["learning"]["temporal_context"]["enabled"] is True
    assert easy_tcp["learning"]["temporal_context"]["maximum_lag"] == 16
    assert medium_tcp["learning"]["temporal_context"]["maximum_lag"] == 64
    assert "plasticity" not in easy_tcp["learning"]["posterior"]
    assert easy_tcp_cup["learning"]["posterior"]["plasticity"]["enabled"] is True
    assert "plasticity" not in medium_tcp["learning"]["posterior"]
    assert medium_tcp_cup["learning"]["posterior"]["plasticity"]["enabled"] is True

    memoryless_base = without_domain_id(easy_memoryless)
    tcp_base = without_domain_id(easy_tcp)
    memoryless_base["learning"]["temporal_context"]["enabled"] = True
    assert memoryless_base == tcp_base

    easy_cup_base = without_domain_id(easy_tcp_cup)
    easy_cup_base["learning"]["posterior"].pop("plasticity")
    assert easy_cup_base == tcp_base

    medium_base = without_domain_id(medium_tcp)
    medium_cup_base = without_domain_id(medium_tcp_cup)
    medium_cup_base["learning"]["posterior"].pop("plasticity")
    assert medium_cup_base == medium_base


def verify_easy() -> None:
    root = ROOT / "results" / "repeat_previous_easy"
    runs = sorted((root / "runs").glob("*.json"))
    assert len(runs) == 15
    for path in runs:
        run = load_json(path)
        assert run["frozen_state_unchanged"] is True
        assert run["frozen_state_before"] == run["frozen_state_after"]
        assert run["protocol"]["target_lag_declared"] is False
        assert run["protocol"]["future_observation_supplied"] is False
        assert run["protocol"]["forced_coverage"] is False

    with (root / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in ("memoryless", "tcp", "tcp_cup")
    }
    aggregate = load_json(root / "aggregate.json")
    for variant, variant_rows in by_variant.items():
        assert len(variant_rows) == 5
        returns = [float(row["frozen_return"]) for row in variant_rows]
        accuracies = [float(row["frozen_action_accuracy"]) for row in variant_rows]
        nlls = [float(row["frozen_nll"]) for row in variant_rows]
        briers = [float(row["frozen_brier"]) for row in variant_rows]
        reported = aggregate["variants"][variant]
        close(statistics.fmean(returns), reported["frozen_return_mean"])
        close(statistics.stdev(returns), reported["frozen_return_sample_sd_across_seeds"])
        close(statistics.fmean(accuracies), reported["frozen_action_accuracy_mean"])
        close(statistics.fmean(nlls), reported["frozen_nll_mean"])
        close(statistics.fmean(briers), reported["frozen_brier_mean"])

    close(aggregate["variants"]["memoryless"]["frozen_return_mean"], -0.5109375)
    close(aggregate["variants"]["memoryless"]["frozen_action_accuracy_mean"], 0.24453125)
    close(aggregate["variants"]["tcp"]["frozen_return_mean"], 1.0)
    close(aggregate["variants"]["tcp"]["frozen_action_accuracy_mean"], 1.0)
    close(aggregate["variants"]["tcp_cup"]["frozen_return_mean"], 1.0)


def verify_medium() -> None:
    root = ROOT / "results" / "repeat_previous_medium"
    runs = sorted((root / "runs").glob("*.json"))
    assert len(runs) == 10
    with (root / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    aggregate = load_json(root / "aggregate.json")
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in ("tcp", "tcp_cup")
    }
    for variant, variant_rows in by_variant.items():
        values = [float(row["mean_learning_curve_return"]) for row in variant_rows]
        reported = aggregate["variants"][variant]
        close(statistics.fmean(values), reported["mean_learning_curve_return"])
        close(statistics.stdev(values), reported["sample_sd_across_seeds"])
    close(aggregate["variants"]["tcp"]["mean_learning_curve_return"], 0.2838888888888884)
    close(aggregate["variants"]["tcp_cup"]["mean_learning_curve_return"], 0.3166666666666661)
    close(aggregate["tcp_cup_minus_tcp"]["mean_absolute_gain"], 0.032777777777777704)
    assert aggregate["tcp_cup_minus_tcp"]["paired_seed_wins"] == 4
    lower, upper = aggregate["tcp_cup_minus_tcp"]["paired_95_percent_t_interval"]
    assert lower < 0.0 < upper

    structures = load_json(root / "final_cup_structures.json")
    assert len(structures) == 5
    seed_0 = structures[0]["structures"]
    seed_3 = structures[3]["structures"]
    assert any(
        row["participants"] == ["contextual_memory", "learned_model"]
        and row["status"] == "inhibited"
        for row in seed_0
    )
    assert any(
        row["participants"] == ["contextual_memory", "learned_model"]
        and row["status"] == "consolidated"
        for row in seed_3
    )


def verify_wisconsin_status() -> None:
    reported = load_json(ROOT / "results" / "wisconsin" / "reported_control_ladder.json")
    assert reported["reproduction_status"] == "aggregate_only"
    assert reported["order_count"] == 30
    close(reported["conditions"]["cup"]["log_loss_mean"], 0.3087)
    close(reported["conditions"]["fixed_uniform"]["log_loss_mean"], 0.3288)
    assert not (ROOT / "harness" / "run_wisconsin_ladder.py").exists()


def main() -> None:
    verify_artifact_scope()
    verify_domains()
    verify_easy()
    verify_medium()
    verify_wisconsin_status()
    print("verified: domains, detailed results, freeze checks, topology, and no traces")


if __name__ == "__main__":
    main()

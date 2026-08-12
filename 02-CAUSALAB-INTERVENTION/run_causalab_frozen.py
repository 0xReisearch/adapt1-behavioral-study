#!/usr/bin/env python3
"""Frozen CausaLab evaluation through the public DiscoveryWorld action API."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from causalab_reeval.metrics import (
    compute_directed_shd,
    compute_edge_metrics,
    compute_frequency_weight_metrics,
    extract_root_nodes_from_edges,
    extract_true_root_nodes_for_frequency,
    frequency_parent_edges,
    prf,
)
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from neuroadapt import NeuroadaptEngine
from neuroadapt.domain import DomainRegistry


NUMBER = re.compile(r"^\s*[-*]?\s*([^:]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:Hz)?\s*$", re.I)
OBSERVATION = re.compile(r"Observation\s+\d+\s*:\s*(\{[^\n]+\})")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _canonical(label: str, candidates: Iterable[str]) -> str | None:
    label_key = _normalized(label).removesuffix("hz")
    rows = sorted(set(candidates))
    exact = [name for name in rows if _normalized(name) == label_key]
    if exact:
        return exact[0]
    compatible = [
        name
        for name in rows
        if _normalized(name).startswith(label_key) or label_key.startswith(_normalized(name))
    ]
    return compatible[0] if len(compatible) == 1 else None


def _initial_observations(task_description: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw in OBSERVATION.findall(task_description):
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict) and parsed:
            rows.append({str(key): float(value) for key, value in parsed.items()})
    return rows


def _dialog_state(dialog_text: str, candidates: Iterable[str]) -> dict[str, float]:
    state: dict[str, float] = {}
    for line in dialog_text.splitlines():
        match = NUMBER.match(line)
        if not match:
            continue
        label, value = match.groups()
        name = _canonical(label.strip(), candidates)
        if name is not None:
            state[name] = float(value)
    return state


def _nearby_named(observation: dict[str, Any], prefix: str) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for rows in observation["ui"]["nearbyObjects"]["objects"].values():
        objects.extend(rows)
    matches = [row for row in objects if str(row.get("name", "")).startswith(prefix)]
    if not matches:
        raise RuntimeError(f"public observation did not expose {prefix!r}")
    return sorted(matches, key=lambda row: (row.get("distance", 999), row["uuid"]))[0]


def _act(
    env: DiscoveryWorldAPI,
    execution: dict[str, Any],
    *,
    trace: list[dict[str, Any]],
    phase: str,
    past_data: list[dict[str, Any]],
    hypothesis: dict[str, Any],
    experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "memory": f"phase={phase}; observations={len(past_data)}",
        "thought": f"Execute the next declared {phase} operation from current evidence.",
        "past_data": past_data,
        "hypothesis": hypothesis,
        "experiment": experiment or {},
        **execution,
    }
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = env.performAgentAction(0, payload)
        env.tick()
        observation = env.getAgentObservation(0)
    trace.append(
        {
            "step": len(trace) + 1,
            "phase": phase,
            "action": payload,
            "result": result,
            "last_action_message": observation["ui"].get("lastActionMessage", ""),
            "dialog": observation["ui"].get("dialog_box", {}),
            "task_progress": observation["ui"].get("taskProgress", []),
        }
    )
    if not result.get("success"):
        raise RuntimeError(f"public action failed: {execution}: {result}")
    return observation


def _planned_values(baseline: float) -> list[float]:
    candidates = [10.0, 90.0, 30.0, 70.0]
    return sorted(candidates, key=lambda value: (abs(value - baseline), value), reverse=True)


def _changed(before: dict[str, float], after: dict[str, float]) -> dict[str, bool]:
    return {
        name: not math.isclose(after[name], before[name], abs_tol=1e-8, rel_tol=1e-8)
        for name in sorted(set(before) & set(after))
    }


def _domain(
    controllable: list[str],
    observable: list[str],
    *,
    frequency: str,
    owner: str,
) -> DomainRegistry:
    inputs = [f"values.state.{name}" for name in controllable]
    causal_variables = [
        {
            "name": name,
            "before_path": f"values.before.{name}",
            "after_path": f"values.state.{name}",
        }
        for name in observable
    ]
    engine = NeuroadaptEngine(autonomous_training=False)
    registry = DomainRegistry(engine)
    registry.create_domain(
        owner_session_id=owner,
        domain_id="causal-discovery",
        schema={"event_types": ["observation", "intervention_result"]},
        hypotheses=[],
        learning={
            "transition": {
                "enabled": True,
                "event_types": ["observation", "intervention_result"],
                "input_paths": inputs,
                "targets": [{"path": f"values.state.{frequency}", "type": "number"}],
                "required_support": 2,
                "neighbors": 8,
                "max_distance": 1.0,
                "max_samples": 256,
                "numeric_model": "auto",
                "numeric_ridge": 1e-10,
                "numeric_min_skill": 0.05,
                "causal_graph": {
                    "enabled": True,
                    "event_types": ["intervention_result"],
                    "intervention_path": "values.intervention.target",
                    "variables": causal_variables,
                    "minimum_effect": 1e-8,
                    "minimum_change_fraction": 0.5,
                    "polynomial_degree": 1,
                    "max_parents": 6,
                    "complexity_penalty": 1e-4,
                },
            }
        },
    )
    return registry


def _hypothesis(query: dict[str, Any], frequency: str) -> dict[str, Any]:
    transition = query.get("transition_prediction") or {}
    graph = transition.get("causal_graph") or {}
    edges = [
        {"from": row["from"], "to": row["to"]}
        for row in graph.get("edges") or []
    ]
    equation = next(
        (
            row
            for row in graph.get("equations") or []
            if row.get("target") == frequency and row.get("status") == "selected"
        ),
        {},
    )
    coefficients: dict[str, float] = {}
    values = list(equation.get("coefficients") or [])
    if values:
        coefficients["base"] = float(values[0])
        for index, parent in enumerate(equation.get("parents") or [], start=1):
            if index < len(values):
                coefficients[f"c_{parent}"] = float(values[index])
    terms = " + ".join(f"c_{name}*{name}" for name in equation.get("parents") or [])
    return {
        "edges": edges,
        "freq_equation": f"{frequency} = base" + (f" + {terms}" if terms else ""),
        "coefficients": coefficients,
    }


def _query(
    registry: DomainRegistry,
    owner: str,
    state: dict[str, float],
) -> dict[str, Any]:
    return registry.query(
        "causal-discovery",
        owner_session_id=owner,
        session_id=owner,
        question="Predict the held-out outcome and causal structure",
        context={"values": {"state": state}},
        update_memory_state=False,
    )


def _load_environment(
    record: dict[str, Any], *, seed: int, thread_id: int
) -> tuple[DiscoveryWorldAPI, Path, dict[str, Any]]:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with handle:
        json.dump(record, handle)
    config_path = Path(handle.name)
    os.environ["CAUSAL_GRAPH_CONFIG"] = str(config_path)
    os.environ["ENV_SEED"] = str(seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = DiscoveryWorldAPI(threadID=thread_id)
        loaded = env.loadScenario(
            "Reactor Lab Causal", "Causal", randomSeed=seed, numUserAgents=1
        )
        observation = env.getAgentObservation(0)
    if not loaded:
        raise RuntimeError("CausaLab failed to load the graph scenario")
    return env, config_path, observation


def run_graph(record: dict[str, Any], *, seed: int, graph_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    env, config_path, observation = _load_environment(
        record, seed=seed, thread_id=900000 + graph_index
    )
    action_trace: list[dict[str, Any]] = []
    try:
        task_description = observation["ui"]["taskProgress"][0]["description"]
        initial = _initial_observations(task_description)
        if len(initial) != 2:
            raise RuntimeError(f"expected two public initial observations, got {len(initial)}")
        candidates = sorted(initial[0])
        frequency_candidates = [name for name in candidates if "freq" in name.lower()]
        if len(frequency_candidates) != 1:
            raise RuntimeError("could not identify the declared frequency target")
        frequency = frequency_candidates[0]
        observable = candidates

        manipulator = _nearby_named(observation, "property manipulator")
        reactor = _nearby_named(observation, "crystal reactor")
        past_data: list[dict[str, Any]] = [
            {
                "id": f"initial-{index}",
                "props": {key: value for key, value in row.items() if key != frequency},
                "freq": row[frequency],
            }
            for index, row in enumerate(initial, start=1)
        ]
        hypothesis: dict[str, Any] = {"edges": [], "freq_equation": "", "coefficients": {}}
        observation = _act(
            env,
            {"action": "TELEPORT_TO_OBJECT", "arg1": manipulator["uuid"]},
            trace=action_trace,
            phase="navigation",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        observation = _act(
            env,
            {"action": "TALK", "arg1": manipulator["uuid"]},
            trace=action_trace,
            phase="causal_discovery",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        dialog = observation["ui"]["dialog_box"]
        baseline = _dialog_state(dialog.get("dialogIn", ""), observable)
        options = dialog.get("dialogOptions") or {}
        option_to_property: dict[int, str] = {}
        exit_option: int | None = None
        for raw_index, label in options.items():
            index = int(raw_index)
            if str(label).lower().startswith("adjust "):
                name = _canonical(str(label)[len("Adjust ") :], observable)
                if name and name != frequency:
                    option_to_property[index] = name
            elif "exit" in str(label).lower():
                exit_option = index
        controllable = sorted(option_to_property.values())
        if set(baseline) != set(observable) or not controllable:
            raise RuntimeError("public manipulator dialog did not expose the declared state")

        owner = f"causalab-frozen-{record['graph_id']}-seed-{seed}"
        registry = _domain(controllable, observable, frequency=frequency, owner=owner)
        for index, state in enumerate(initial, start=1):
            registry.ingest_event(
                "causal-discovery",
                owner_session_id=owner,
                session_id=owner,
                event_type="observation",
                values={"state": state},
                metadata={"source": "public_initial_observation", "index": index},
            )
        registry.ingest_event(
            "causal-discovery",
            owner_session_id=owner,
            session_id=owner,
            event_type="observation",
            values={"state": baseline},
            metadata={"source": "public_manipulator_baseline"},
        )
        past_data.append(
            {
                "id": "live-0",
                "props": {key: value for key, value in baseline.items() if key != frequency},
                "freq": baseline[frequency],
            }
        )

        budget_match = re.search(r"Remaining:\s*(\d+)", dialog.get("dialogIn", ""))
        if not budget_match:
            raise RuntimeError("public dialog did not expose the intervention budget")
        budget = int(budget_match.group(1))
        values_per_property = max(1, budget // len(controllable))
        plans: dict[str, list[float]] = {}
        for property_index, name in enumerate(controllable):
            values = _planned_values(baseline[name])[:values_per_property]
            rotation = property_index % len(values)
            plans[name] = values[rotation:] + values[:rotation]
        property_to_option = {name: index for index, name in option_to_property.items()}

        current = baseline
        trial = 0
        for plan_index in range(values_per_property):
            for name in controllable:
                requested = plans[name][plan_index]
                trial += 1
                experiment = {"target_prop": name, "target_value": requested}
                observation = _act(
                    env,
                    {"chosen_dialog_option_int": property_to_option[name]},
                    trace=action_trace,
                    phase="causal_discovery",
                    past_data=past_data,
                    hypothesis=hypothesis,
                    experiment=experiment,
                )
                before = dict(current)
                observation = _act(
                    env,
                    {"value": requested},
                    trace=action_trace,
                    phase="causal_discovery",
                    past_data=past_data,
                    hypothesis=hypothesis,
                    experiment=experiment,
                )
                current = _dialog_state(
                    observation["ui"]["dialog_box"].get("dialogIn", ""), observable
                )
                changed = _changed(before, current)
                registry.ingest_event(
                    "causal-discovery",
                    owner_session_id=owner,
                    session_id=owner,
                    event_type="intervention_result",
                    values={
                        "state": current,
                        "before": before,
                        "intervention": {"target": name, "requested_value": requested},
                        "changed": changed,
                    },
                    relations=[
                        {"from": name, "to": target, "type": "intervention_changed"}
                        for target, did_change in changed.items()
                        if did_change and target != name
                    ],
                    metadata={
                        "trial": trial,
                        "source": "public_action_api",
                        "interventional_design": "controlled_intervention",
                    },
                )
                past_data.append(
                    {
                        "id": f"live-{trial}",
                        "props": {key: value for key, value in current.items() if key != frequency},
                        "freq": current[frequency],
                    }
                )
                hypothesis = _hypothesis(_query(registry, owner, current), frequency)

        if exit_option is not None:
            observation = _act(
                env,
                {"chosen_dialog_option_int": exit_option},
                trace=action_trace,
                phase="causal_discovery",
                past_data=past_data,
                hypothesis=hypothesis,
            )
        observation = _act(
            env,
            {"action": "TELEPORT_TO_OBJECT", "arg1": reactor["uuid"]},
            trace=action_trace,
            phase="reactor_transfer",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        observation = _act(
            env,
            {"action": "TALK", "arg1": reactor["uuid"]},
            trace=action_trace,
            phase="reactor_transfer",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        reactor_dialog = observation["ui"]["dialog_box"]
        reactor_state = _dialog_state(reactor_dialog.get("dialogIn", ""), observable)
        reactor_state.pop(frequency, None)
        if set(reactor_state) != set(observable) - {frequency}:
            raise RuntimeError("public reactor dialog did not expose the held-out state")

        query = _query(registry, owner, reactor_state)
        transition = query["transition_prediction"]
        predicted_frequency = (transition.get("path_values") or {}).get(
            f"values.state.{frequency}"
        )
        if predicted_frequency is None:
            raise RuntimeError(f"Core abstained: {transition.get('abstention_reason')}")
        hypothesis = _hypothesis(query, frequency)

        observation = _act(
            env,
            {"chosen_dialog_option_int": 1},
            trace=action_trace,
            phase="reactor_submission",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        observation = _act(
            env,
            {"value": float(predicted_frequency)},
            trace=action_trace,
            phase="reactor_submission",
            past_data=past_data,
            hypothesis=hypothesis,
        )
        task_progress = observation["ui"]["taskProgress"][0]

        graph = transition.get("causal_graph") or {}
        predicted_edges = [
            {"from": row["from"], "to": row["to"]}
            for row in graph.get("edges") or []
        ]
        true_frequency = float(
            env.world.taskScorer.tasks[0].scoringInfo["targetFrequency"]
        )
        true_edges = list(record.get("edges") or [])
        frequency_coefficients = hypothesis["coefficients"]
        predicted_roots = extract_root_nodes_from_edges(predicted_edges)
        true_roots = extract_true_root_nodes_for_frequency(record)
        absolute_error = abs(float(predicted_frequency) - true_frequency)
        return {
            "graph_id": record["graph_id"],
            "seed": seed,
            "protocol": "public_action_api_frozen",
            "initial_observation_count": len(initial),
            "budget": budget,
            "actions": action_trace,
            "held_out": {
                "visible_state": reactor_state,
                "predicted_frequency": predicted_frequency,
                "true_frequency": true_frequency,
                "absolute_error": absolute_error,
                "task_success": bool(task_progress.get("completedSuccessfully")),
            },
            "predicted_edges": predicted_edges,
            "true_edges": true_edges,
            "edge_metrics": compute_edge_metrics(predicted_edges, true_edges),
            "directed_shd": compute_directed_shd(predicted_edges, true_edges),
            "root_metrics": prf(predicted_roots, true_roots),
            "frequency_edge_metrics": compute_edge_metrics(
                frequency_parent_edges(predicted_edges),
                frequency_parent_edges(true_edges),
            ),
            "frequency_coefficients": frequency_coefficients,
            "frequency_weight_metrics": compute_frequency_weight_metrics(
                frequency_coefficients, record, tolerance=1e-5
            ),
            "core_query": query,
            "llm_calls": 0,
            "hidden_truth_read_after_submission": True,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    finally:
        config_path.unlink(missing_ok=True)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["held_out"]["absolute_error"] for row in rows]
    return {
        "graphs": len(rows),
        "task_accuracy": _mean(float(row["held_out"]["task_success"]) for row in rows),
        "mean_absolute_error": _mean(errors),
        "all_edge_precision": _mean(row["edge_metrics"]["precision"] for row in rows),
        "all_edge_recall": _mean(row["edge_metrics"]["recall"] for row in rows),
        "all_edge_f1": _mean(row["edge_metrics"]["f1"] for row in rows),
        "mean_directed_shd": _mean(row["directed_shd"]["shd"] for row in rows),
        "root_f1": _mean(row["root_metrics"]["f1"] for row in rows),
        "frequency_edge_f1": _mean(row["frequency_edge_metrics"]["f1"] for row in rows),
        "frequency_weight_f1": _mean(
            row["frequency_weight_metrics"]["weight_f1"] for row in rows
        ),
        "mean_actions": _mean(len(row["actions"]) for row in rows),
        "mean_elapsed_seconds": _mean(row["elapsed_seconds"] for row in rows),
        "llm_calls": 0,
        "seed": rows[0]["seed"] if rows else None,
        "protocol": "public_action_api_frozen",
        "hidden_truth_read_after_submission": all(
            row["hidden_truth_read_after_submission"] for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--suite", default="3nodes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.benchmark_root / "release" / "causalab_dataset" / "data" / f"{args.suite}.jsonl"
    records = _jsonl(source)
    if args.limit > 0:
        records = records[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = run_graph(record, seed=args.seed, graph_index=index)
        rows.append(row)
        print(
            f"[{index + 1}/{len(records)}] {row['graph_id']} "
            f"success={row['held_out']['task_success']} "
            f"error={row['held_out']['absolute_error']:.6f} "
            f"edge_f1={row['edge_metrics']['f1']:.3f}"
        )
    summary = summarize(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

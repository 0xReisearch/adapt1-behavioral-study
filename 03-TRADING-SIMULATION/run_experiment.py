#!/usr/bin/env python3
"""Run the sanitized Trading Simulation through Adapt-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

import httpx


def _base_url(value: str) -> str:
    base = value.rstrip("/")
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")]
    return base


class AdaptClient:
    def __init__(self, *, api_url: str, api_key: str, timeout: float) -> None:
        self.client = httpx.Client(
            base_url=_base_url(api_url),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        session_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.client.request(
            method,
            f"/api/v1{path}",
            headers={"X-Neuroadapt-Session-Id": session_id},
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        if not response.content:
            return {}
        return response.json()

    def close(self) -> None:
        self.client.close()


def _create_domain(
    client: AdaptClient,
    *,
    session_id: str,
    domain_id: str,
    feature_names: list[str],
    asset_names: list[str],
) -> None:
    payload = {
        "domain_id": domain_id,
        "session_id": session_id,
        "description": "Learn numeric effects from observed market features to next-step price changes.",
        "schema": {"event_types": ["market_transition"]},
        "hypotheses": [],
        "learning": {
            "enabled": True,
            "transition": {
                "enabled": True,
                "event_types": ["market_transition"],
                "input_paths": [f"values.features.{name}" for name in feature_names],
                "targets": [
                    {"path": f"values.delta.{name}", "type": "number"}
                    for name in asset_names
                ],
                "required_support": 2,
                "neighbors": 128,
                "max_distance": 1.0,
                "max_samples": 512,
                "numeric_model": "linear",
                "numeric_ridge": 1e-10,
                "numeric_min_skill": -1.0,
            },
        },
    }
    client.request("POST", "/domains", session_id=session_id, payload=payload)


def _query(
    client: AdaptClient,
    *,
    session_id: str,
    domain_id: str,
    features: dict[str, float],
) -> dict[str, Any]:
    return client.request(
        "POST",
        f"/domains/{domain_id}/query",
        session_id=session_id,
        payload={
            "session_id": session_id,
            "question": "Predict the next numeric outcomes from the current observed features.",
            "context": {"values": {"features": features}},
            "update_memory_state": False,
        },
    )


def _ingest(
    client: AdaptClient,
    *,
    session_id: str,
    domain_id: str,
    step: int,
    features: dict[str, float],
    deltas: dict[str, float],
) -> None:
    client.request(
        "POST",
        f"/domains/{domain_id}/events",
        session_id=session_id,
        payload={
            "session_id": session_id,
            "event_type": "market_transition",
            "values": {"features": features, "delta": deltas},
            "metadata": {"step": step, "source": "sanitized_market_stream"},
        },
    )


def _choose_asset(
    prediction: dict[str, Any],
    prices: dict[str, float],
) -> str | None:
    paths = (prediction.get("transition_prediction") or {}).get("path_values") or {}
    expected = {
        asset: float(paths[f"values.delta.{asset}"])
        for asset in prices
        if f"values.delta.{asset}" in paths
    }
    if not expected:
        return None
    best = max(
        expected,
        key=lambda asset: expected[asset] / max(0.1, round(float(prices[asset]), 2)),
    )
    return best if expected[best] > 0 else None


def _execute(
    *,
    cash: float,
    positions: dict[str, int],
    prices: dict[str, float],
    selected_asset: str | None,
) -> tuple[float, dict[str, int]]:
    available_for_selection = round(cash, 2) + sum(
        quantity * round(float(prices[asset]), 2)
        for asset, quantity in positions.items()
    )
    for asset, quantity in list(positions.items()):
        if quantity > 0:
            cash += quantity * float(prices[asset])
            positions[asset] = 0
    if selected_asset is None:
        return cash, positions
    selected_price = float(prices[selected_asset])
    quantity = max(
        0,
        int(available_for_selection // max(0.1, round(selected_price, 2))) - 1,
    )
    cost = quantity * selected_price
    if quantity > 0 and cost <= cash:
        cash -= cost
        positions[selected_asset] += quantity
    return cash, positions


def _window_mae(errors_by_step: list[list[float]], start: int, stop: int) -> float | None:
    values = [error for row in errors_by_step[start:stop] for error in row]
    return statistics.fmean(values) if values else None


def _run_scenario(
    client: AdaptClient,
    *,
    run_id: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    identity = hashlib.sha256(f"{run_id}:{scenario_id}".encode()).hexdigest()[:20]
    session_id = f"market-{identity}"
    domain_id = f"market-{identity}"
    steps = scenario["steps"]
    feature_names = sorted({name for step in steps for name in step["features"]})
    asset_names = sorted(steps[0]["prices"])
    _create_domain(
        client,
        session_id=session_id,
        domain_id=domain_id,
        feature_names=feature_names,
        asset_names=asset_names,
    )
    cash = float(scenario["initial_cash"])
    positions = {asset: 0 for asset in asset_names}
    errors_by_step: list[list[float]] = []
    try:
        for step_index, step in enumerate(steps):
            features = {name: float(value) for name, value in step["features"].items()}
            prices = {name: float(value) for name, value in step["prices"].items()}
            next_prices = {
                name: float(value) for name, value in step["next_prices"].items()
            }
            prediction = _query(
                client,
                session_id=session_id,
                domain_id=domain_id,
                features=features,
            )
            selected_asset = _choose_asset(prediction, prices)
            cash, positions = _execute(
                cash=cash,
                positions=positions,
                prices=prices,
                selected_asset=selected_asset,
            )
            deltas = {
                asset: round(next_prices[asset], 2) - round(prices[asset], 2)
                for asset in asset_names
            }
            paths = (prediction.get("transition_prediction") or {}).get("path_values") or {}
            errors_by_step.append(
                [
                    abs(float(paths[f"values.delta.{asset}"]) - deltas[asset])
                    for asset in asset_names
                    if f"values.delta.{asset}" in paths
                ]
            )
            _ingest(
                client,
                session_id=session_id,
                domain_id=domain_id,
                step=step_index,
                features=features,
                deltas=deltas,
            )
        final_prices = {
            name: float(value) for name, value in steps[-1]["next_prices"].items()
        }
        final_value = cash + sum(
            positions[asset] * final_prices[asset] for asset in asset_names
        )
        initial_cash = float(scenario["initial_cash"])
        return {
            "scenario_id": scenario_id,
            "steps": len(steps),
            "initial_value": initial_cash,
            "final_value": final_value,
            "profit_rate": (final_value - initial_cash) / initial_cash,
            "early_prediction_mae": _window_mae(errors_by_step, 0, 20),
            "final_prediction_mae": _window_mae(errors_by_step, 100, 120),
        }
    finally:
        try:
            client.request(
                "DELETE",
                f"/domains/{domain_id}?session_id={session_id}",
                session_id=session_id,
            )
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-url", default=os.getenv("NEUROADAPT_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("NEUROADAPT_API_KEY", ""))
    parser.add_argument("--api-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.api_url:
        parser.error("set NEUROADAPT_API_URL or pass --api-url")
    if not args.api_key:
        parser.error("set NEUROADAPT_API_KEY or pass --api-key")
    scenarios = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit > 0:
        scenarios = scenarios[: args.limit]
    client = AdaptClient(api_url=args.api_url, api_key=args.api_key, timeout=args.api_timeout)
    rows: list[dict[str, Any]] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            result = _run_scenario(client, run_id=args.run_id, scenario=scenario)
            rows.append(result)
            print(
                f"[{index}/{len(scenarios)}] {result['scenario_id']} "
                f"profit={result['profit_rate']:+.2%}",
                flush=True,
            )
    finally:
        client.close()
    profit_rates = [row["profit_rate"] for row in rows]
    early = [row["early_prediction_mae"] for row in rows if row["early_prediction_mae"] is not None]
    final = [row["final_prediction_mae"] for row in rows if row["final_prediction_mae"] is not None]
    early_mae = statistics.fmean(early) if early else None
    final_mae = statistics.fmean(final) if final else None
    summary = {
        "experiment": "Trading Simulation",
        "scale": "small_scale_one_pass",
        "scenarios": len(rows),
        "steps_per_scenario": sorted({row["steps"] for row in rows}),
        "mean_profit_rate": statistics.fmean(profit_rates) if profit_rates else 0.0,
        "minimum_profit_rate": min(profit_rates, default=0.0),
        "maximum_profit_rate": max(profit_rates, default=0.0),
        "early_prediction_mae": early_mae,
        "final_prediction_mae": final_mae,
        "relative_mae_reduction": (
            (early_mae - final_mae) / early_mae
            if early_mae not in (None, 0.0) and final_mae is not None
            else None
        ),
        "fresh_domain_per_scenario": True,
        "llm_calls": 0,
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

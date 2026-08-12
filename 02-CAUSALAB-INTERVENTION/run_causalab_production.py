#!/usr/bin/env python3
"""Run the frozen CausaLab protocol against a deployed Adapt-1 Domain API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx


class _EnginePlaceholder:
    def __init__(self, **_: Any) -> None:
        pass


class _RemoteDomainRegistry:
    api_base_url = ""
    api_key = ""
    run_id = ""
    timeout = 120.0

    def __init__(self, _engine: Any) -> None:
        base = self.api_base_url.rstrip("/")
        if base.endswith("/api/v1"):
            base = base[: -len("/api/v1")]
        self.client = httpx.Client(
            base_url=base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        self.domain_id = ""
        self.session_id = ""

    def _request(
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
                f"{response.text[:2000]}"
            )
        return response.json()

    def create_domain(
        self,
        *,
        owner_session_id: str,
        domain_id: str,
        description: str = "",
        schema: dict[str, Any] | None = None,
        hypotheses: list[dict[str, Any]] | None = None,
        query_templates: dict[str, Any] | None = None,
        learning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = hashlib.sha256(
            f"{self.run_id}:{owner_session_id}:{domain_id}".encode("utf-8")
        ).hexdigest()[:16]
        self.domain_id = f"causalab-{self.run_id}-{token}"
        self.session_id = owner_session_id
        payload: dict[str, Any] = {
            "domain_id": self.domain_id,
            "session_id": owner_session_id,
            "description": description,
            "schema": schema or {},
            "hypotheses": hypotheses or [],
            "learning": learning or {},
        }
        if query_templates is not None:
            payload["query_templates"] = query_templates
        return self._request(
            "POST", "/domains", session_id=owner_session_id, payload=payload
        )

    def ingest_event(
        self,
        _domain_id: str,
        *,
        owner_session_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del owner_session_id
        return self._request(
            "POST",
            f"/domains/{self.domain_id}/events",
            session_id=session_id,
            payload={"session_id": session_id, **kwargs},
        )

    def query(
        self,
        _domain_id: str,
        *,
        owner_session_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del owner_session_id
        return self._request(
            "POST",
            f"/domains/{self.domain_id}/query",
            session_id=session_id,
            payload={"session_id": session_id, **kwargs},
        )


def _install_remote_core() -> None:
    neuroadapt = types.ModuleType("neuroadapt")
    neuroadapt.NeuroadaptEngine = _EnginePlaceholder
    domain = types.ModuleType("neuroadapt.domain")
    domain.DomainRegistry = _RemoteDomainRegistry
    sys.modules["neuroadapt"] = neuroadapt
    sys.modules["neuroadapt.domain"] = domain


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--api-base-url", default=os.getenv("NEUROADAPT_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("NEUROADAPT_API_KEY", ""))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", default="3nodes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.api_base_url:
        parser.error("set NEUROADAPT_API_URL or pass --api-base-url")
    if not args.api_key:
        parser.error("set NEUROADAPT_API_KEY or pass --api-key")

    _RemoteDomainRegistry.api_base_url = args.api_base_url
    _RemoteDomainRegistry.api_key = args.api_key
    _RemoteDomainRegistry.run_id = args.run_id
    _RemoteDomainRegistry.timeout = args.api_timeout
    _install_remote_core()

    import run_causalab_frozen as frozen

    source = (
        args.benchmark_root
        / "release"
        / "causalab_dataset"
        / "data"
        / f"{args.suite}.jsonl"
    )
    records = _jsonl(source)
    if args.limit > 0:
        records = records[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = frozen.run_graph(record, seed=args.seed, graph_index=index)
        row["core_transport"] = "production_domain_api"
        row["run_id"] = args.run_id
        rows.append(row)
        print(
            f"[{index + 1}/{len(records)}] {row['graph_id']} "
            f"success={row['held_out']['task_success']} "
            f"error={row['held_out']['absolute_error']:.6f} "
            f"edge_f1={row['edge_metrics']['f1']:.3f}",
            flush=True,
        )

    summary = frozen.summarize(rows)
    summary["core_transport"] = "production_domain_api"
    summary["run_id"] = args.run_id
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

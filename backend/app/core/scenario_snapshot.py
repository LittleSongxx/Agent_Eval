from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def build_scenario_snapshot(scenario) -> dict[str, Any]:
    """Freeze the scenario config that a task should evaluate with."""

    metrics = []
    for scenario_metric in sorted(scenario.metrics or [], key=lambda item: item.id or 0):
        metric_definition = scenario_metric.metric_definition
        if metric_definition is None:
            continue
        metrics.append(
            {
                "id": scenario_metric.id,
                "metric_definition_id": scenario_metric.metric_definition_id,
                "weight": scenario_metric.weight,
                "pass_threshold": scenario_metric.pass_threshold,
                "prompt_override": getattr(scenario_metric, "prompt_override", None),
                "metric_definition": {
                    "id": metric_definition.id,
                    "name": metric_definition.name,
                    "display_name": metric_definition.display_name,
                    "metric_type": metric_definition.metric_type,
                    "config": metric_definition.config or {},
                    "category": metric_definition.category,
                    "is_builtin": metric_definition.is_builtin,
                },
            }
        )

    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "scene_type": scenario.scene_type,
        "sample_type": scenario.sample_type,
        "is_preset": scenario.is_preset,
        "metrics": metrics,
    }


def snapshot_to_scenario_metrics(snapshot: dict[str, Any] | None) -> list[Any]:
    """Hydrate frozen metric config into lightweight objects used by the engine."""

    if not snapshot:
        return []

    hydrated = []
    for item in snapshot.get("metrics") or []:
        metric_definition = item.get("metric_definition") or {}
        hydrated.append(
            SimpleNamespace(
                id=item.get("id"),
                scenario_id=snapshot.get("id"),
                metric_definition_id=item.get("metric_definition_id"),
                weight=item.get("weight", 1.0),
                pass_threshold=item.get("pass_threshold"),
                prompt_override=item.get("prompt_override"),
                metric_definition=SimpleNamespace(
                    id=metric_definition.get("id"),
                    name=metric_definition.get("name"),
                    display_name=metric_definition.get("display_name")
                    or metric_definition.get("name")
                    or "Unknown Metric",
                    metric_type=metric_definition.get("metric_type"),
                    config=metric_definition.get("config") or {},
                    category=metric_definition.get("category"),
                    is_builtin=metric_definition.get("is_builtin", False),
                ),
            )
        )
    return hydrated

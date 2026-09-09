"""Canonical machine-readable result records shared by all benchmarks.

The schema deliberately keeps unlike resources separate: classical objective
evaluations, quantum circuit executions, final shots, synthesis time and wall
time are distinct fields and must never be added into a synthetic total.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np


SCHEMA_NAME = "temporal-antenna-benchmark"
SCHEMA_VERSION = "1.0.0"


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args):
    try:
        return subprocess.run(["git", *args], check=True, text=True,
                              capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(backend: str = "local-numpy", seeds=None) -> dict:
    """Capture reproducibility metadata at run start."""
    status = _git(["status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": _git(["rev-parse", "HEAD"]),
                "dirty": bool(status), "status_porcelain": status},
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__,
                     "scipy": _package_version("scipy"),
                     "classiq": _package_version("classiq")},
        "hardware": {"backend": backend, "platform": platform.platform(),
                     "machine": platform.machine(),
                     "processor": platform.processor(),
                     "cpu_count": os.cpu_count()},
        "seeds": dict(seeds or {}),
    }


def new_run_record(kind: str, instance: dict, config: dict,
                   backend: str = "local-numpy", seeds=None) -> dict:
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "run": {"kind": str(kind), "provenance": collect_provenance(backend, seeds),
                "config": dict(config), "status": "in_progress"},
        "instance": dict(instance),
        "methods": [],
        "artifacts": [],
    }


def method_record(name: str, family: str, objective: float, *,
                  feasible_fraction: float = 1.0, probability_of_optimum=None,
                  best_known_objective=None, trajectory=None, kpis=None,
                  classical_objective_evaluations: int = 0,
                  parameter_search_circuit_executions: int = 0,
                  final_sampling_shots: int = 0,
                  quantum_executions: int = 0,
                  wall_time_seconds: float = 0.0,
                  synthesis_time_seconds: float = 0.0,
                  resynthesis_count: int = 0, circuit=None, algorithm=None) -> dict:
    """Construct one solver row with stable names and separated resources."""
    gap = None if best_known_objective is None else float(objective - best_known_objective)
    return {
        "name": str(name), "family": str(family),
        "solution": {"objective": float(objective), "gap_to_best_known": gap,
                     "probability_of_optimum": (None if probability_of_optimum is None
                                                else float(probability_of_optimum)),
                     "feasible_fraction": float(feasible_fraction),
                     "trajectory": _jsonable(trajectory), "kpis": _jsonable(kpis or {})},
        "resources": {
            "classical_objective_evaluations": int(classical_objective_evaluations),
            "parameter_search_circuit_executions": int(parameter_search_circuit_executions),
            "final_sampling_shots": int(final_sampling_shots),
            "quantum_executions": int(quantum_executions),
            "wall_time_seconds": float(wall_time_seconds),
            "synthesis_time_seconds": float(synthesis_time_seconds),
            "resynthesis_count": int(resynthesis_count),
            "circuit": _jsonable(circuit or {"width": None, "depth": None,
                                              "gate_count": None,
                                              "two_qubit_gates": None,
                                              "three_qubit_gates": None,
                                              "pauli_terms": None}),
        },
        "algorithm": _jsonable(algorithm or {}),
    }


def validate_run_record(record: dict) -> None:
    """Raise ``ValueError`` on missing required schema fields."""
    if record.get("schema") != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ValueError("unknown or missing result schema")
    for key in ("run", "instance", "methods", "artifacts"):
        if key not in record:
            raise ValueError(f"missing top-level field: {key}")
    provenance = record["run"].get("provenance", {})
    for key in ("timestamp_utc", "git", "software", "hardware", "seeds"):
        if key not in provenance:
            raise ValueError(f"missing provenance field: {key}")
    for method in record["methods"]:
        for key in ("name", "family", "solution", "resources", "algorithm"):
            if key not in method:
                raise ValueError(f"method missing field: {key}")


def save_run_record(record: dict, path) -> Path:
    """Validate and atomically save JSON so interrupted runs stay readable."""
    validate_run_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] against the repo root and make sure it exists."""
    p = ROOT / cfg["paths"][key]
    (p.parent if key == "duckdb" else p).mkdir(parents=True, exist_ok=True)
    return p

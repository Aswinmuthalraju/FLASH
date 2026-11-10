"""Pin and record exactly what was downloaded from Hugging Face (revision sha, files, sizes, sha256).

Writes/updates data/raw/hf/MANIFEST.json. Access failures are recorded verbatim and re-raised: the download
stage stops, nothing is substituted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/raw/hf/MANIFEST.json"


def _load() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _save(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2))


def resolve_revision(repo_id: str) -> dict:
    """Commit sha + remote file list/sizes. Records the exact error on failure, then re-raises."""
    from huggingface_hub import HfApi
    m = _load()
    try:
        info = HfApi().dataset_info(repo_id, files_metadata=True)
    except Exception as e:                                  # noqa: BLE001 - record whatever went wrong
        m[repo_id] = {"status": "ACCESS_FAILED", "error_type": type(e).__name__, "error": str(e)[:500],
                      "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        _save(m)
        raise
    rec = {"status": "resolved", "revision_sha": info.sha, "last_modified": str(info.last_modified),
           "license": (info.card_data or {}).get("license") if info.card_data else None,
           "remote_files": {s.rfilename: s.size for s in (info.siblings or [])},
           "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    m[repo_id] = {**m.get(repo_id, {}), **rec}
    _save(m)
    return rec


def record_local(repo_id: str, path: Path, note: str = "") -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    m = _load()
    m.setdefault(repo_id, {}).setdefault("local_files", {})[str(path.relative_to(ROOT))] = {
        "bytes": path.stat().st_size, "sha256": h.hexdigest(), "note": note}
    m[repo_id]["status"] = "downloaded"
    _save(m)

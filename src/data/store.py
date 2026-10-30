"""Parquet + DuckDB storage helpers."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import resolve


def save_parquet(df: pd.DataFrame, cfg: dict, name: str) -> Path:
    p = resolve(cfg, "processed") / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_parquet(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg, "processed") / f"{name}.parquet")


def register_duckdb(cfg: dict, names: list[str]) -> Path:
    """(Re)create one DuckDB view per processed parquet file, so the data can be queried with SQL."""
    db = resolve(cfg, "duckdb")
    proc = resolve(cfg, "processed")
    con = duckdb.connect(str(db))
    for n in names:
        f = (proc / f"{n}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW {n} AS SELECT * FROM read_parquet('{f}')")
    con.close()
    return db


def query(cfg: dict, sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(resolve(cfg, "duckdb")), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()

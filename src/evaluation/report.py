"""Tiny helpers for writing report tables (no tabulate dependency)."""
from __future__ import annotations

import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, float):
            return "nan" if v != v else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)

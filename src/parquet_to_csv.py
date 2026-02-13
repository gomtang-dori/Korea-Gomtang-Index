# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def main():
    index_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))
    out_csv = Path(os.environ.get("CSV_PATH", "data/index_daily.csv"))

    if not index_path.exists():
        raise RuntimeError(f"Missing {index_path}. Run assemble first.")

    df = pd.read_parquet(index_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[parquet_to_csv] OK rows={len(df)} -> {out_csv}")


if __name__ == "__main__":
    main()

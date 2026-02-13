# src/merge_backfill_chunks.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main():
    chunk_root = Path("chunks")
    paths = sorted(chunk_root.rglob("*.parquet"))
    if not paths:
        raise RuntimeError("No chunk parquet found under ./chunks (download-artifact path).")

    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        if "date" not in df.columns:
            raise RuntimeError(f"missing 'date' column in chunk: {p}")
        dfs.append(df)
        print(f"[merge_chunks] loaded {p} rows={len(df)} cols={len(df.columns)}")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    out_path = Path("data/index_daily.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[merge_chunks] wrote {out_path} rows={len(merged)}")


if __name__ == "__main__":
    main()

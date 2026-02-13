# src/vkospi_fetch.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from lib.krx_dvrprod_index import KRXDrvProdIndexAPI


VKOSPI_PATH = Path("data/vkospi_level.parquet")


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def safe_to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def upsert_timeseries(old: pd.DataFrame, new: pd.DataFrame, key: str = "date") -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = safe_to_datetime(out, key)
    out = out.dropna(subset=[key]).drop_duplicates(subset=[key], keep="last").sort_values(key).reset_index(drop=True)
    return out


def main():
    ensure_dir(VKOSPI_PATH)

    api = KRXDrvProdIndexAPI.from_env()

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    # daily refresh window: 14 days (calendar) + buffer
    start = today - pd.Timedelta(days=14 + 10)

    new = api.fetch_vkospi_range(start, today, idx_nm="코스피 200 변동성지수")
    new = safe_to_datetime(new, "date")
    if not new.empty:
        new["vkospi"] = pd.to_numeric(new["vkospi"], errors="coerce")
        new = new.dropna(subset=["date", "vkospi"])

    old = pd.read_parquet(VKOSPI_PATH) if VKOSPI_PATH.exists() else pd.DataFrame()
    old = safe_to_datetime(old, "date")

    out = upsert_timeseries(old, new, "date")
    out.to_parquet(VKOSPI_PATH, index=False)

    print(f"[vkospi_fetch] OK rows(new)={len(new)} rows(total)={len(out)} -> {VKOSPI_PATH}")


if __name__ == "__main__":
    main()

# src/caches/normalize_k200_members.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

def main():
    src_path = Path(os.environ.get("K200_MEMBERS_RAW_PATH", "data/k200_members_raw.csv"))
    out_path = Path(os.environ.get("K200_MEMBERS_PATH", "data/k200_members.csv"))

    if not src_path.exists():
        raise RuntimeError(f"Missing raw members file: {src_path}")

    df = pd.read_csv(src_path, dtype=str)

    # 후보 컬럼명 대응
    code_col_candidates = ["isu_cd", "종목코드", "종목코드 ", "종목코드\t", "code", "ticker"]
    name_col_candidates = ["isu_nm", "종목명", "종목명 ", "name"]

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    code_col = pick(code_col_candidates)
    name_col = pick(name_col_candidates)

    if not code_col:
        raise RuntimeError(f"Cannot find code column in {list(df.columns)}")
    if not name_col:
        # 이름은 없어도 되지만 있으면 좋음
        df["종목명"] = ""
        name_col = "종목명"

    out = pd.DataFrame({
        "isu_cd": df[code_col].astype(str).str.strip().str.zfill(6),
        "isu_nm": df[name_col].astype(str).str.strip(),
    }).dropna()

    out = out.drop_duplicates(subset=["isu_cd"]).sort_values("isu_cd").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[normalize_k200_members] OK rows={len(out)} -> {out_path}")

if __name__ == "__main__":
    main()

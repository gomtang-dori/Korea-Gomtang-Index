#!/usr/bin/env python3
import sys, pandas as pd
import pyarrow.parquet as pq

p = sys.argv[1] if len(sys.argv) > 1 else "data/stocks/mart/panel_daily.parquet"
req = {
  "DART": ["dart_revenue","dart_operating_income","dart_net_income","dart_equity","dart_roe"],
  "FUND": ["bps","per","pbr","eps","div","dps"],
  "FLOW": ["foreign_value_net","inst_value_net","pension_value_net","fininv_value_net"],
}
pf = pq.ParquetFile(p); schema = set(pf.schema.names); want = sum(req.values(), [])
have = [c for c in want if c in schema]
df = pd.read_parquet(p, columns=have) if have else pd.DataFrame()
print(f"[panel-check] file={p}\n[panel-check] rows={pf.metadata.num_rows:,} cols={len(schema):,}")

bad = []
for g, cols in req.items():
  print(f"\n== {g} ==")
  for c in cols:
    if c not in schema:
      print(f"- {c}: MISSING"); bad.append((g,c,"MISSING")); continue
    s = df[c]; nn = int(s.notna().sum())
    allna = (nn == 0)
    print(f"- {c}: OK  nonnull={nn:,} ({(nn/len(df)*100 if len(df) else 0):.1f}%)  all_nan={allna}")
    if allna: bad.append((g,c,"ALL_NAN"))

if bad:
  print("\n[panel-check] FAIL:", bad)
  sys.exit(1)
print("\n[panel-check] PASS")

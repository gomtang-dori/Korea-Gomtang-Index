# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
import html


DOCS_DIR = Path("docs")
ANALYSIS_DIR = Path("data/analysis")


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _list_sorted(glob_pattern: str, base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob(glob_pattern), key=lambda p: p.name.lower())


def _rel_to_docs(path: Path) -> str:
    """
    return href relative to docs/ root.
    examples:
      docs/factor_corr_8Y_pearson_levels.html -> factor_corr_8Y_pearson_levels.html
      data/analysis/xxx.csv -> ../data/analysis/xxx.csv
    """
    p = path.as_posix()
    if p.startswith("docs/"):
        return p[len("docs/") :]
    return "../" + p


def _section(title: str, items_html: str) -> str:
    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      {items_html}
    </section>
    """


def _build_corr_cards() -> str:
    """
    Group by TAG (8Y / 1Y) if present in filename.
    Expected files (examples):
      docs/factor_corr_8Y_pearson_levels.html
      docs/factor_corr_8Y_pearson_levels.png
      docs/factor_corr_8Y_pearson_returns.html
      docs/factor_corr_8Y_pearson_returns.png
    """
    corr_htmls = _list_sorted("factor_corr_*.html", DOCS_DIR)
    corr_pngs = {p.stem: p for p in _list_sorted("factor_corr_*.png", DOCS_DIR)}

    if not corr_htmls:
        return "<p class='muted'>docs/factor_corr_*.html 파일이 아직 없습니다. build_8Y/build_1Y 실행 후 커밋 여부를 확인하세요.</p>"

    def tag_of(name: str) -> str:
        # factor_corr_8Y_...
        parts = name.split("_")
        if len(parts) >= 3 and parts[2] in ("8Y", "1Y"):
            return parts[2]
        if "8Y" in name:
            return "8Y"
        if "1Y" in name:
            return "1Y"
        return "ETC"

    groups: dict[str, list[Path]] = {}
    for p in corr_htmls:
        groups.setdefault(tag_of(p.name), []).append(p)

    # ensure order
    ordered_tags = [t for t in ("8Y", "1Y", "ETC") if t in groups]

    blocks = []
    for tag in ordered_tags:
        rows = []
        for h in groups[tag]:
            stem = h.stem  # without .html
            png = corr_pngs.get(stem)

            href_html = _rel_to_docs(h)
            png_html = ""
            if png is not None:
                href_png = _rel_to_docs(png)
                png_html = f"""
                  <a class="thumb" href="{html.escape(href_html)}" target="_blank" rel="noopener">
                    <img src="{html.escape(href_png)}" alt="{html.escape(png.name)}"/>
                  </a>
                  <div class="small-links">
                    <a href="{html.escape(href_png)}" target="_blank" rel="noopener">PNG 열기</a>
                  </div>
                """

            rows.append(f"""
              <div class="item">
                <div class="item-title">
                  <a href="{html.escape(href_html)}" target="_blank" rel="noopener">{html.escape(h.name)}</a>
                </div>
                {png_html}
              </div>
            """)

        blocks.append(f"""
          <div class="grid">
            {''.join(rows)}
          </div>
        """)

    return "".join(blocks)


def _build_csv_list() -> str:
    csvs = _list_sorted("*.csv", ANALYSIS_DIR)
    if not csvs:
        return "<p class='muted'>data/analysis/*.csv 파일이 아직 없습니다.</p>"

    lis = []
    for p in csvs:
        href = _rel_to_docs(p)
        lis.append(f"<li><a href='{html.escape(href)}' target='_blank' rel='noopener'>{html.escape(p.name)}</a></li>")
    return f"<ul class='csv-list'>{''.join(lis)}</ul>"


def main() -> None:
    out_path = DOCS_DIR / "corr.html"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    updated = _utc_now_str()

    corr_cards = _build_corr_cards()
    csv_list = _build_csv_list()

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GOMTANG - Correlation Links</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: #111a2e;
      --text: #e8eefc;
      --muted: #a8b3d6;
      --line: rgba(255,255,255,0.08);
      --link: #7aa7ff;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #071022 0%, #0b1220 40%, #070d18 100%);
      color: var(--text);
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 22px 16px 40px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.2px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
    }}
    a {{
      color: var(--link);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .card {{
      background: rgba(17,26,46,0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 14px;
      margin: 12px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .card h2 {{
      margin: 0 0 12px 0;
      font-size: 15px;
      color: #dbe6ff;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .item {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: rgba(0,0,0,0.10);
    }}
    .item-title {{
      font-size: 13px;
      margin-bottom: 8px;
      word-break: break-all;
    }}
    .thumb img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      display: block;
    }}
    .small-links {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .csv-list {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.8;
      font-size: 13px;
    }}
    .top-links {{
      display:flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 6px;
      font-size: 13px;
    }}
    .pill {{
      display:inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.06);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>GOMTANG · Correlation (F01~F10 + Index)</h1>
      <div class="meta">updated: {html.escape(updated)}</div>
    </header>

    <div class="top-links">
      <a class="pill" href="index.html">메인(Index)</a>
      <a class="pill" href="Korea-Gomtang-Index_8Y.html">8Y 리포트</a>
      <a class="pill" href="Korea-Gomtang-Index_1Y.html">1Y 리포트</a>
    </div>

    {_section("Heatmaps (HTML/PNG)", corr_cards)}

    {_section("CSV Downloads (data/analysis)", csv_list)}

    <section class="card">
      <h2>Notes</h2>
      <div class="muted">
        GitHub Pages는 <code>/docs</code> 폴더를 사이트 루트로 제공합니다.
        상관관계 파일은 build_8Y/build_1Y 워크플로에서 생성 후 커밋되면 자동 배포됩니다.
      </div>
    </section>

  </div>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[corr-index] OK -> {out_path}")


if __name__ == "__main__":
    main()

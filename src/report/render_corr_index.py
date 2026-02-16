# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import html


DOCS_DIR = Path("docs")
ANALYSIS_DIR = Path("data/analysis")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _rel_from_docs_root(path: Path) -> str:
    """
    GitHub Pages 소스가 /docs 이므로,
    docs 내부 파일은 '파일명' 형태로,
    docs 밖 파일은 '../' 상대경로로 링크.
    """
    p = path.as_posix()
    if p.startswith("docs/"):
        return p[len("docs/") :]
    return "../" + p


def _sorted_glob(base: Path, pattern: str) -> list[Path]:
    return sorted(base.glob(pattern), key=lambda x: x.name.lower())


def _group_tag(name: str) -> str:
    # factor_corr_8Y_... / factor_corr_1Y_...
    if "_8Y_" in name:
        return "8Y"
    if "_1Y_" in name:
        return "1Y"
    return "ETC"


def _build_heatmap_cards() -> str:
    html_files = _sorted_glob(DOCS_DIR, "factor_corr_*.html")
    png_map = {p.stem: p for p in _sorted_glob(DOCS_DIR, "factor_corr_*.png")}

    if not html_files:
        return "<p class='muted'>docs/factor_corr_*.html 파일이 아직 없습니다. build_8Y/build_1Y 실행 후 커밋 여부를 확인하세요.</p>"

    groups: dict[str, list[Path]] = {}
    for f in html_files:
        groups.setdefault(_group_tag(f.name), []).append(f)

    ordered = [t for t in ("8Y", "1Y", "ETC") if t in groups]
    blocks: list[str] = []

    for tag in ordered:
        items: list[str] = []
        for f in groups[tag]:
            png = png_map.get(f.stem)
            href_html = _rel_from_docs_root(f)

            thumb = ""
            if png:
                href_png = _rel_from_docs_root(png)
                thumb = f"""
                  <a class="thumb" href="{html.escape(href_html)}" target="_blank" rel="noopener">
                    <img src="{html.escape(href_png)}" alt="{html.escape(png.name)}"/>
                  </a>
                  <div class="small-links">
                    <a href="{html.escape(href_png)}" target="_blank" rel="noopener">PNG 열기</a>
                  </div>
                """

            items.append(f"""
              <div class="item">
                <div class="item-title">
                  <a href="{html.escape(href_html)}" target="_blank" rel="noopener">{html.escape(f.name)}</a>
                </div>
                {thumb}
              </div>
            """)

        blocks.append(f"""
          <div class="tag-title">{html.escape(tag)}</div>
          <div class="grid">{''.join(items)}</div>
        """)

    return "".join(blocks)


def _build_csv_links() -> str:
    csvs = _sorted_glob(ANALYSIS_DIR, "*.csv")
    if not csvs:
        return "<p class='muted'>data/analysis/*.csv 파일이 아직 없습니다.</p>"

    lis: list[str] = []
    for p in csvs:
        href = _rel_from_docs_root(p)
        lis.append(
            f"<li><a href='{html.escape(href)}' target='_blank' rel='noopener'>{html.escape(p.name)}</a></li>"
        )
    return f"<ul class='csv-list'>{''.join(lis)}</ul>"


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "corr.html"

    updated = _utc_now()
    heatmaps = _build_heatmap_cards()
    csv_links = _build_csv_links()

    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GOMTANG · Correlation</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: rgba(17,26,46,0.92);
      --text: #e8eefc;
      --muted: #a8b3d6;
      --line: rgba(255,255,255,0.10);
      --link: #7aa7ff;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #071022 0%, #0b1220 40%, #070d18 100%);
      color: var(--text);
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px 16px 42px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 12px;
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
    a:hover {{ text-decoration: underline; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
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
      line-height: 1.55;
    }}
    .top-links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 10px 0 2px;
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.06);
    }}
    .tag-title {{
      margin: 8px 0 8px;
      font-size: 13px;
      color: #cfe0ff;
      letter-spacing: 0.4px;
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
      border: 1px solid rgba(255,255,255,0.12);
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
      line-height: 1.85;
      font-size: 13px;
    }}
    code {{
      background: rgba(255,255,255,0.08);
      padding: 1px 6px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>GOMTANG · Correlation (F01~F10 + KOSPI/KOSDAQ/K200)</h1>
      <div class="meta">updated: {html.escape(updated)}</div>
    </header>

    <div class="top-links">
      <a class="pill" href="index.html">메인(Index)</a>
      <a class="pill" href="Korea-Gomtang-Index_8Y.html">8Y 리포트</a>
      <a class="pill" href="Korea-Gomtang-Index_1Y.html">1Y 리포트</a>
    </div>

    <section class="card">
      <h2>Heatmaps (HTML/PNG)</h2>
      {heatmaps}
    </section>

    <section class="card">
      <h2>CSV Downloads (data/analysis)</h2>
      {csv_links}
    </section>

    <section class="card">
      <h2>Notes</h2>
      <div class="muted">
        GitHub Pages는 <code>/docs</code> 폴더를 사이트 루트로 제공합니다.<br/>
        파일이 생성되었는데도 페이지에서 안 보이면, (1) 워크플로가 커밋까지 했는지, (2) Settings → Pages의 Latest deployment가 성공인지 확인하세요.
      </div>
    </section>
  </div>
</body>
</html>
"""

    out_path.write_text(page, encoding="utf-8")
    print(f"[corr-index] OK -> {out_path}")


if __name__ == "__main__":
    main()

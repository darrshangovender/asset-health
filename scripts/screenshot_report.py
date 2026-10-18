"""Render docs/report.png from docs/example-report.html.

The screenshot hides fleet-table rows whose severity is NORMAL so the alerts are
visible above the fold; the committed HTML keeps every row.

    pip install playwright && playwright install chromium
    python scripts/screenshot_report.py
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "example-report.html"
PNG = ROOT / "docs" / "report.png"

HIDE_NORMAL_ROWS = """
() => {
  const table = document.querySelector('table');
  if (!table) return 0;
  let hidden = 0;
  for (const tr of table.querySelectorAll('tbody tr, tr')) {
    const cells = [...tr.querySelectorAll('td')].map(td => td.textContent.trim());
    if (cells.includes('NORMAL')) { tr.style.display = 'none'; hidden++; }
  }
  return hidden;
}
"""


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1060, "height": 1300}, device_scale_factor=2)
        page.goto(HTML.as_uri())
        page.wait_for_timeout(300)
        hidden = page.evaluate(HIDE_NORMAL_ROWS)
        page.screenshot(path=str(PNG), clip={"x": 0, "y": 0, "width": 1060, "height": 1300})
        browser.close()
    print(f"wrote {PNG} (hid {hidden} NORMAL rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

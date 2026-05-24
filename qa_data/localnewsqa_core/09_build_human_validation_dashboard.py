#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LocalNewsQA Human Validation Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #6b7280;
      --line: #d9deea;
      --accent: #1d4ed8;
      --accent-soft: #dbeafe;
      --good: #166534;
      --good-bg: #dcfce7;
      --warn: #92400e;
      --warn-bg: #fef3c7;
      --bad: #991b1b;
      --bad-bg: #fee2e2;
      --shadow: 0 10px 30px rgba(23, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef2ff 0%, var(--bg) 200px);
      color: var(--text);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 320px minmax(420px, 1.2fr) minmax(460px, 1fr);
      gap: 18px;
      padding: 18px;
      min-height: 100vh;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-inner {{ padding: 18px; }}
    h1 {{
      font-size: 24px;
      line-height: 1.15;
      margin: 0 0 8px 0;
    }}
    h2 {{
      font-size: 14px;
      line-height: 1.2;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 0 0 12px 0;
    }}
    .lede {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      margin-bottom: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fbfcff;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .filter-group {{ margin-bottom: 14px; }}
    .filter-group label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    input[type="search"], select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      font-size: 14px;
    }}
    .list-toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .count-pill {{
      margin-left: auto;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 700;
    }}
    .row-list {{
      max-height: calc(100vh - 160px);
      overflow: auto;
      border-top: 1px solid var(--line);
    }}
    .row-item {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
      background: white;
    }}
    .row-item:hover {{ background: #f8faff; }}
    .row-item.active {{ background: #eef4ff; }}
    .row-question {{
      font-size: 14px;
      line-height: 1.45;
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: #f9fafb;
      color: var(--text);
    }}
    .chip.good {{ background: var(--good-bg); color: var(--good); border-color: #b7e4c7; }}
    .chip.warn {{ background: var(--warn-bg); color: var(--warn); border-color: #f6d37e; }}
    .chip.bad {{ background: var(--bad-bg); color: var(--bad); border-color: #f5b5b5; }}
    .detail {{
      max-height: calc(100vh - 36px);
      overflow: auto;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0 18px 0;
    }}
    .detail-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fcfdff;
    }}
    .detail-card .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .detail-card .v {{
      font-size: 14px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .section {{
      margin-bottom: 18px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .option {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 8px;
      background: white;
      line-height: 1.45;
    }}
    .option.target {{ border-color: #9dc0ff; background: #eef4ff; }}
    .option.contrast {{ border-color: #f0b1b1; background: #fff1f1; }}
    .evidence {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
      margin-bottom: 10px;
    }}
    .evidence a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .evidence a:hover {{ text-decoration: underline; }}
    .muted {{ color: var(--muted); }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .footer-note {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 12px;
      line-height: 1.5;
    }}
    @media (max-width: 1400px) {{
      .shell {{ grid-template-columns: 300px 1fr; }}
      .detail.panel {{ grid-column: 1 / -1; }}
    }}
    @media (max-width: 980px) {{
      .shell {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <div class="panel-inner">
        <h1>LocalNewsQA Human Validation</h1>
        <div class="lede">Static review dashboard for the expanded ambiguous sample. Use this view to inspect questions, evidence, and current prefills, then edit the CSV directly.</div>
        <div class="stats" id="stats"></div>

        <div class="filter-group">
          <label for="search">Search</label>
          <input id="search" type="search" placeholder="question, answer, hint, notes" />
        </div>
        <div class="filter-group">
          <label for="countryFilter">Country</label>
          <select id="countryFilter"></select>
        </div>
        <div class="filter-group">
          <label for="topicFilter">Topic</label>
          <select id="topicFilter"></select>
        </div>
        <div class="filter-group">
          <label for="prefillFilter">Prefill Status</label>
          <select id="prefillFilter">
            <option value="all">All rows</option>
            <option value="has_any_prefill">Has any prefill</option>
            <option value="needs_review">Needs review</option>
            <option value="fully_blank">Fully blank except leakage</option>
            <option value="locale_yes">Locale dependence = yes</option>
          </select>
        </div>
        <div class="filter-group">
          <label for="evidenceFilter">Evidence Coverage</label>
          <select id="evidenceFilter">
            <option value="all">All rows</option>
            <option value="both">Target and contrast evidence</option>
            <option value="target_only">Target evidence only</option>
            <option value="contrast_only">Contrast evidence only</option>
            <option value="none">No evidence captured</option>
          </select>
        </div>
        <div class="footer-note">
          Source CSV: <code>__CSV_NAME__</code><br />
          Rows are read-only here. Update the CSV in your editor after manual verification.
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-inner">
        <div class="list-toolbar">
          <h2>Rows</h2>
          <div class="count-pill" id="visibleCount"></div>
        </div>
      </div>
      <div class="row-list" id="rowList"></div>
    </section>

    <section class="panel detail" id="detailPanel">
      <div class="panel-inner" id="detailContent"></div>
    </section>
  </div>

  <script>
    const rows = __ROWS_JSON__;

    const rowList = document.getElementById('rowList');
    const detailContent = document.getElementById('detailContent');
    const visibleCount = document.getElementById('visibleCount');
    const search = document.getElementById('search');
    const countryFilter = document.getElementById('countryFilter');
    const topicFilter = document.getElementById('topicFilter');
    const prefillFilter = document.getElementById('prefillFilter');
    const evidenceFilter = document.getElementById('evidenceFilter');

    let selectedId = rows.length ? rows[0].id : null;

    function uniq(values) {{
      return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
    }}

    function fillSelect(select, values, label) {{
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = 'all';
      all.textContent = `All ${label}`;
      select.appendChild(all);
      for (const value of values) {{
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }}
    }}

    function truthy(value) {{
      return String(value || '').trim().toLowerCase();
    }}

    function hasEvidence(row, prefix) {{
      return !!String(row[`${prefix}_evidence_url`] || '').trim();
    }}

    function prefillState(row) {{
      const factual = truthy(row.judge_target_factuality);
      const locale = truthy(row.judge_locale_dependence);
      const leakage = truthy(row.judge_no_explicit_leakage);
      if (locale === 'yes') return 'locale_yes';
      if (factual || locale || leakage) return 'has_any_prefill';
      return 'fully_blank';
    }}

    function evidenceState(row) {{
      const target = hasEvidence(row, 'target');
      const contrast = hasEvidence(row, 'contrast');
      if (target && contrast) return 'both';
      if (target) return 'target_only';
      if (contrast) return 'contrast_only';
      return 'none';
    }}

    function needsReview(row) {{
      return !truthy(row.judge_target_factuality) || !truthy(row.judge_locale_dependence);
    }}

    function stats(filtered) {{
      const factual = filtered.filter(r => truthy(r.judge_target_factuality) === 'yes').length;
      const locale = filtered.filter(r => truthy(r.judge_locale_dependence) === 'yes').length;
      const leakage = filtered.filter(r => truthy(r.judge_no_explicit_leakage) === 'yes').length;
      const bothEvidence = filtered.filter(r => evidenceState(r) === 'both').length;
      return [
        ['Visible rows', filtered.length],
        ['Target factuality = yes', factual],
        ['Locale dependence = yes', locale],
        ['No explicit leakage = yes', leakage],
        ['Target + contrast evidence', bothEvidence],
        ['Needs review', filtered.filter(needsReview).length],
      ];
    }}

    function renderStats(filtered) {{
      const container = document.getElementById('stats');
      container.innerHTML = '';
      for (const [label, value] of stats(filtered)) {{
        const el = document.createElement('div');
        el.className = 'stat';
        el.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        container.appendChild(el);
      }}
    }}

    function chip(text, cls='') {{
      return `<span class="chip ${{cls}}">${{text}}</span>`;
    }}

    function escapeHtml(value) {{
      return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('\"', '&quot;');
    }}

    function parseOptions(text) {{
      return String(text || '').split('||').map(s => s.trim()).filter(Boolean);
    }}

    function filterRows() {{
      const q = search.value.trim().toLowerCase();
      return rows.filter(row => {{
        if (countryFilter.value !== 'all' && row.country !== countryFilter.value) return false;
        if (topicFilter.value !== 'all' && row.topic !== topicFilter.value) return false;

        const pf = prefillFilter.value;
        if (pf === 'has_any_prefill' && prefillState(row) === 'fully_blank') return false;
        if (pf === 'fully_blank' && prefillState(row) !== 'fully_blank') return false;
        if (pf === 'locale_yes' && truthy(row.judge_locale_dependence) !== 'yes') return false;
        if (pf === 'needs_review' && !needsReview(row)) return false;

        const ef = evidenceFilter.value;
        if (ef !== 'all' && evidenceState(row) !== ef) return false;

        if (!q) return true;
        const hay = [
          row.id, row.country, row.topic, row.question, row.options, row.target_answer,
          row.contrast_answer, row.evidence_hint, row.annotator_notes
        ].join(' ').toLowerCase();
        return hay.includes(q);
      }});
    }}

    function renderList(filtered) {{
      rowList.innerHTML = '';
      visibleCount.textContent = `${{filtered.length}} shown`;
      if (!filtered.length) {{
        rowList.innerHTML = '<div class="panel-inner empty">No rows match the current filters.</div>';
        detailContent.innerHTML = '';
        return;
      }}
      if (!filtered.some(r => r.id === selectedId)) {{
        selectedId = filtered[0].id;
      }}
      for (const row of filtered) {{
        const item = document.createElement('div');
        item.className = 'row-item' + (row.id === selectedId ? ' active' : '');
        item.innerHTML = `
          <div class="row-question">${{escapeHtml(row.question)}}</div>
          <div class="meta">
            ${{chip(row.country)}}
            ${{chip(row.topic)}}
            ${{truthy(row.judge_target_factuality)==='yes' ? chip('Target factuality yes','good') : chip('Target factuality blank','warn')}}
            ${{truthy(row.judge_locale_dependence)==='yes' ? chip('Locale yes','good') : chip('Locale blank','warn')}}
            ${{evidenceState(row)==='both' ? chip('Both evidence','good') : chip(evidenceState(row).replace('_',' '),'warn')}}
          </div>`;
        item.addEventListener('click', () => {{
          selectedId = row.id;
          render(filtered);
        }});
        rowList.appendChild(item);
      }}
    }}

    function evidenceBlock(prefix, row) {{
      const url = row[`${prefix}_evidence_url`] || '';
      const title = row[`${prefix}_evidence_title`] || '';
      const snippet = row[`${prefix}_evidence_snippet`] || '';
      const excerpt = row[`${prefix}_evidence_excerpt`] || '';
      const matchType = row[`${prefix}_match_type`] || '';
      const query = row[`${prefix}_query`] || '';
      if (!url && !title && !snippet && !excerpt && !matchType && !query) {{
        return `<div class="evidence"><div class="empty">No ${prefix} evidence captured yet.</div></div>`;
      }}
      return `
        <div class="evidence">
          <div class="meta" style="margin-bottom:8px">
            ${{chip(prefix === 'target' ? 'Target evidence' : 'Contrast evidence')}}
            ${{matchType ? chip(matchType) : ''}}
          </div>
          <div style="font-weight:700; margin-bottom:8px;">
            ${{url ? `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer">${{escapeHtml(title || url)}}</a>` : escapeHtml(title || 'Untitled evidence')}}
          </div>
          ${{query ? `<div class="muted" style="margin-bottom:8px;"><strong>Query:</strong> ${{escapeHtml(query)}}</div>` : ''}}
          ${{snippet ? `<div style="margin-bottom:8px;"><strong>Snippet:</strong> ${{escapeHtml(snippet)}}</div>` : ''}}
          ${{excerpt ? `<div><strong>Excerpt:</strong> ${{escapeHtml(excerpt)}}</div>` : ''}}
        </div>`;
    }}

    function renderDetail(row) {{
      if (!row) {{
        detailContent.innerHTML = '<div class="empty">Select a row to inspect it.</div>';
        return;
      }}
      const options = parseOptions(row.options).map(opt => {{
        const label = opt.replace(/^([A-D]:\\s*)/, '');
        let cls = '';
        if (label === row.target_answer) cls = 'target';
        else if (label === row.contrast_answer) cls = 'contrast';
        return `<div class="option ${{cls}}">${{escapeHtml(opt)}}</div>`;
      }}).join('');

      detailContent.innerHTML = `
        <h1 style="font-size:22px; margin-bottom:10px;">${{escapeHtml(row.question)}}</h1>
        <div class="meta" style="margin-bottom:12px;">
          ${{chip(row.id)}}
          ${{chip(row.country)}}
          ${{chip(row.continent)}}
          ${{chip(row.topic)}}
          ${{row.year ? chip(String(row.year)) : ''}}
        </div>

        <div class="detail-grid">
          <div class="detail-card"><div class="k">Target country</div><div class="v">${{escapeHtml(row.target_country || row.country)}}</div></div>
          <div class="detail-card"><div class="k">Contrast country</div><div class="v">${{escapeHtml(row.contrast_country)}}</div></div>
          <div class="detail-card"><div class="k">Target answer</div><div class="v">${{escapeHtml(row.target_answer)}}</div></div>
          <div class="detail-card"><div class="k">Contrast answer</div><div class="v">${{escapeHtml(row.contrast_answer)}}</div></div>
          <div class="detail-card"><div class="k">Evidence hint</div><div class="v">${{escapeHtml(row.evidence_hint) || '<span class="empty">None</span>'}}</div></div>
          <div class="detail-card"><div class="k">Annotator notes</div><div class="v">${{escapeHtml(row.annotator_notes) || '<span class="empty">None</span>'}}</div></div>
        </div>

        <div class="section">
          <div class="section-title">Current judgments</div>
          <div class="meta">
            ${{truthy(row.judge_target_factuality)==='yes' ? chip('Target factuality yes','good') : chip('Target factuality blank','warn')}}
            ${{truthy(row.judge_locale_dependence)==='yes' ? chip('Locale dependence yes','good') : chip('Locale dependence blank','warn')}}
            ${{truthy(row.judge_no_explicit_leakage)==='yes' ? chip('No explicit leakage yes','good') : chip('No explicit leakage blank','warn')}}
          </div>
        </div>

        <div class="section">
          <div class="section-title">Options</div>
          ${{options}}
        </div>

        <div class="section">
          <div class="section-title">Evidence</div>
          ${{evidenceBlock('target', row)}}
          ${{evidenceBlock('contrast', row)}}
        </div>`;
    }}

    function render(filtered) {{
      renderStats(filtered);
      renderList(filtered);
      renderDetail(filtered.find(r => r.id === selectedId));
    }}

    fillSelect(countryFilter, uniq(rows.map(r => r.country)), 'countries');
    fillSelect(topicFilter, uniq(rows.map(r => r.topic)), 'topics');

    [search, countryFilter, topicFilter, prefillFilter, evidenceFilter].forEach(el => {{
      el.addEventListener('input', () => render(filterRows()));
      el.addEventListener('change', () => render(filterRows()));
    }});

    render(filterRows());
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build a static HTML dashboard for LocalNewsQA human validation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    html_text = HTML_TEMPLATE.replace(
        "__ROWS_JSON__", json.dumps(rows, ensure_ascii=False)
    ).replace(
        "__CSV_NAME__", input_path.name
    )
    while "{{" in html_text or "}}" in html_text:
        html_text = html_text.replace("{{", "{").replace("}}", "}")

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())

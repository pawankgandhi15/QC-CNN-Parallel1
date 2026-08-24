"""
Live Notebook Execution Monitor Server
Serves a real-time monitoring dashboard for QC_CNN_Parallel_Experiments.ipynb
"""
import json
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import subprocess

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "QC_CNN_Parallel_Experiments.ipynb")
PORT = 5050
START_TIME = time.time()

def get_notebook_status():
    """Parse the notebook and return execution stats."""
    try:
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    cells = nb.get("cells", [])
    total_code = 0
    executed = 0
    errored = 0
    cell_details = []

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        total_code += 1
        outputs = cell.get("outputs", [])
        exec_count = cell.get("execution_count")
        source_lines = cell.get("source", [])
        source_preview = "".join(source_lines)[:120].replace("\n", " ")

        has_error = any(o.get("output_type") == "error" for o in outputs)
        has_output = len(outputs) > 0
        is_executed = exec_count is not None

        if is_executed:
            executed += 1
        if has_error:
            errored += 1

        # Collect last output text
        last_output = ""
        for o in outputs[-1:]:
            otype = o.get("output_type", "")
            if otype in ("stream",):
                text = o.get("text", [])
                last_output = "".join(text)[-200:] if isinstance(text, list) else str(text)[-200:]
            elif otype in ("execute_result", "display_data"):
                data = o.get("data", {})
                last_output = str(data.get("text/plain", ""))[-200:]
            elif otype == "error":
                last_output = f"ERROR: {o.get('ename', '')}: {o.get('evalue', '')}"

        cell_details.append({
            "index": total_code,
            "exec_count": exec_count,
            "source_preview": source_preview,
            "status": "error" if has_error else ("done" if is_executed else "pending"),
            "last_output": last_output.strip(),
        })

    pct = round((executed / total_code * 100) if total_code else 0, 1)
    elapsed = int(time.time() - START_TIME)
    eta = None
    if executed > 0 and executed < total_code:
        rate = elapsed / executed
        remaining = total_code - executed
        eta = int(rate * remaining)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notebook": os.path.basename(NOTEBOOK_PATH),
        "total_code_cells": total_code,
        "executed": executed,
        "errored": errored,
        "pending": total_code - executed,
        "progress_pct": pct,
        "elapsed_sec": elapsed,
        "eta_sec": eta,
        "cells": cell_details,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>📊 QC-CNN Notebook Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0a0e1a;
    --bg2: #0f1526;
    --bg3: #151d35;
    --surface: #1a2440;
    --surface2: #1e2b4a;
    --border: #2a3a60;
    --primary: #5b8dee;
    --primary2: #7ca4f4;
    --accent: #a78bfa;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --text3: #64748b;
    --glow: rgba(91,141,238,0.15);
  }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }
  body::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(91,141,238,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 80%, rgba(167,139,250,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  /* ===== HEADER ===== */
  header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,14,26,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 16px rgba(91,141,238,0.4);
  }
  .header-title { font-size: 15px; font-weight: 600; color: var(--text); }
  .header-sub { font-size: 11px; color: var(--text3); font-family: 'JetBrains Mono', monospace; margin-top: 1px; }
  .live-pill {
    display: flex; align-items: center; gap: 6px;
    background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; font-weight: 600; color: var(--success);
  }
  .live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }
  #last-update { font-size: 11px; color: var(--text3); font-family: 'JetBrains Mono', monospace; }

  /* ===== MAIN ===== */
  main { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 28px 24px; }

  /* ===== STAT CARDS ===== */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px; margin-bottom: 28px;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
  }
  .stat-card:hover { border-color: var(--primary); transform: translateY(-2px); }
  .stat-card::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, var(--glow), transparent);
    opacity: 0; transition: opacity 0.3s;
  }
  .stat-card:hover::before { opacity: 1; }
  .stat-label { font-size: 11px; font-weight: 500; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .stat-value { font-size: 30px; font-weight: 700; line-height: 1; }
  .stat-value.blue { color: var(--primary2); }
  .stat-value.green { color: var(--success); }
  .stat-value.red { color: var(--danger); }
  .stat-value.yellow { color: var(--warning); }
  .stat-value.purple { color: var(--accent); }
  .stat-sub { font-size: 11px; color: var(--text3); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

  /* ===== PROGRESS BAR ===== */
  .progress-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 24px;
  }
  .progress-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .progress-title { font-size: 13px; font-weight: 600; color: var(--text2); }
  .progress-pct { font-size: 22px; font-weight: 700; color: var(--primary2); font-family: 'JetBrains Mono', monospace; }
  .progress-track {
    height: 10px; background: var(--bg3); border-radius: 10px;
    overflow: hidden; margin-bottom: 10px;
  }
  .progress-fill {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    box-shadow: 0 0 12px rgba(91,141,238,0.5);
  }
  .progress-fill::after {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 40px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2));
    animation: shimmer 1.5s ease-in-out infinite;
  }
  @keyframes shimmer { 0%,100%{opacity:0} 50%{opacity:1} }
  .progress-meta { display: flex; gap: 20px; font-size: 11px; color: var(--text3); font-family: 'JetBrains Mono', monospace; }

  /* ===== CELL TABLE ===== */
  .cells-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }
  .cells-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .cells-title { font-size: 13px; font-weight: 600; color: var(--text2); }
  .cells-legend { display: flex; gap: 14px; font-size: 11px; color: var(--text3); }
  .legend-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  .cell-list { max-height: 440px; overflow-y: auto; }
  .cell-list::-webkit-scrollbar { width: 5px; }
  .cell-list::-webkit-scrollbar-track { background: transparent; }
  .cell-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .cell-row {
    display: grid;
    grid-template-columns: 40px 70px 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 10px 20px;
    border-bottom: 1px solid rgba(42,58,96,0.4);
    transition: background 0.2s;
  }
  .cell-row:hover { background: var(--surface2); }
  .cell-row:last-child { border-bottom: none; }
  .cell-num { font-size: 11px; font-weight: 600; color: var(--text3); font-family: 'JetBrains Mono', monospace; text-align: center; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px;
    font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .badge-done    { background: rgba(52,211,153,0.12); color: var(--success); border: 1px solid rgba(52,211,153,0.25); }
  .badge-error   { background: rgba(248,113,113,0.12); color: var(--danger);  border: 1px solid rgba(248,113,113,0.25); }
  .badge-pending { background: rgba(100,116,139,0.12); color: var(--text3);  border: 1px solid rgba(100,116,139,0.2); }
  .badge-running { background: rgba(251,191,36,0.12);  color: var(--warning); border: 1px solid rgba(251,191,36,0.25);
                   animation: badgePulse 1.2s ease-in-out infinite; }
  @keyframes badgePulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
  .cell-source { font-size: 11px; color: var(--text2); font-family: 'JetBrains Mono', monospace;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cell-exec { font-size: 10px; color: var(--text3); font-family: 'JetBrains Mono', monospace; white-space: nowrap; }

  /* ===== BOTTOM BAR ===== */
  .bottom-bar {
    margin-top: 22px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 11px; color: var(--text3);
  }
  .notebook-link {
    color: var(--primary2); text-decoration: none;
    border-bottom: 1px solid transparent; transition: border-color 0.2s;
  }
  .notebook-link:hover { border-color: var(--primary2); }

  /* ===== ERROR BANNER ===== */
  .error-banner {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 20px;
    font-size: 12px; color: var(--danger); display: none;
  }
  .error-banner.visible { display: block; }
</style>
</head>
<body>
<header>
  <div class="header-left">
    <div class="logo">⚛</div>
    <div>
      <div class="header-title">QC-CNN Notebook Monitor</div>
      <div class="header-sub" id="nb-name">QC_CNN_Parallel_Experiments.ipynb</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;">
    <span id="last-update">Fetching…</span>
    <div class="live-pill"><div class="live-dot"></div>LIVE</div>
  </div>
</header>

<main>
  <div class="error-banner" id="error-banner">⚠ Execution error detected in one or more cells. See table below.</div>

  <!-- Stat Cards -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Cells</div>
      <div class="stat-value blue" id="stat-total">—</div>
      <div class="stat-sub">code cells</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Executed</div>
      <div class="stat-value green" id="stat-done">—</div>
      <div class="stat-sub">completed</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Pending</div>
      <div class="stat-value yellow" id="stat-pending">—</div>
      <div class="stat-sub">remaining</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Errors</div>
      <div class="stat-value red" id="stat-errors">—</div>
      <div class="stat-sub">cells</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Elapsed</div>
      <div class="stat-value purple" id="stat-elapsed">—</div>
      <div class="stat-sub">hh:mm:ss</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">ETA</div>
      <div class="stat-value blue" id="stat-eta">—</div>
      <div class="stat-sub">estimated</div>
    </div>
  </div>

  <!-- Progress Bar -->
  <div class="progress-section">
    <div class="progress-header">
      <span class="progress-title">Overall Execution Progress</span>
      <span class="progress-pct" id="progress-pct">0%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progress-fill" style="width:0%"></div>
    </div>
    <div class="progress-meta">
      <span id="progress-meta-cells">— / — cells</span>
      <span id="progress-meta-time">Elapsed: —</span>
      <span id="progress-meta-eta">ETA: —</span>
    </div>
  </div>

  <!-- Cell Details -->
  <div class="cells-section">
    <div class="cells-header">
      <span class="cells-title">Cell Execution Details</span>
      <div class="cells-legend">
        <span><span class="legend-dot" style="background:var(--success)"></span>Done</span>
        <span><span class="legend-dot" style="background:var(--warning)"></span>Running</span>
        <span><span class="legend-dot" style="background:var(--text3)"></span>Pending</span>
        <span><span class="legend-dot" style="background:var(--danger)"></span>Error</span>
      </div>
    </div>
    <div class="cell-list" id="cell-list">
      <div style="padding:40px;text-align:center;color:var(--text3);">Loading cell data…</div>
    </div>
  </div>

  <div class="bottom-bar">
    <span>Auto-refreshes every 4 seconds &bull; Jupyter on <a class="notebook-link" href="http://localhost:8888/notebooks/QC_CNN_Parallel_Experiments.ipynb" target="_blank">localhost:8888</a></span>
    <span id="footer-time"></span>
  </div>
</main>

<script>
function fmtSec(s) {
  if (s == null) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return [h, m, sec].map(x => String(x).padStart(2,'0')).join(':');
}

async function refresh() {
  try {
    const res = await fetch('/status');
    const d = await res.json();

    // Cards
    document.getElementById('stat-total').textContent   = d.total_code_cells;
    document.getElementById('stat-done').textContent    = d.executed;
    document.getElementById('stat-pending').textContent = d.pending;
    document.getElementById('stat-errors').textContent  = d.errored;
    document.getElementById('stat-elapsed').textContent = fmtSec(d.elapsed_sec);
    document.getElementById('stat-eta').textContent     = fmtSec(d.eta_sec);

    // Progress
    const pct = d.progress_pct;
    document.getElementById('progress-pct').textContent = pct + '%';
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-meta-cells').textContent = d.executed + ' / ' + d.total_code_cells + ' cells';
    document.getElementById('progress-meta-time').textContent  = 'Elapsed: ' + fmtSec(d.elapsed_sec);
    document.getElementById('progress-meta-eta').textContent   = 'ETA: ' + fmtSec(d.eta_sec);

    // Error banner
    const banner = document.getElementById('error-banner');
    banner.classList.toggle('visible', d.errored > 0);

    // Timestamp
    document.getElementById('last-update').textContent = 'Updated: ' + d.timestamp;
    document.getElementById('footer-time').textContent = d.timestamp;

    // Cell list
    const list = document.getElementById('cell-list');
    const html = d.cells.map(c => {
      // Determine if this is the currently running cell
      const isRunning = (c.status === 'pending' && d.executed > 0 && c.index === d.executed + 1);
      const status = isRunning ? 'running' : c.status;
      const badgeClass = {done:'badge-done', error:'badge-error', pending:'badge-pending', running:'badge-running'}[status] || 'badge-pending';
      const icon = {done:'✓', error:'✗', pending:'○', running:'◉'}[status] || '○';
      const labelText = {done:'Done', error:'Error', pending:'Pending', running:'Running'}[status] || 'Pending';
      return `
        <div class="cell-row">
          <div class="cell-num">[${c.exec_count != null ? c.exec_count : ' '}]</div>
          <div><span class="status-badge ${badgeClass}">${icon} ${labelText}</span></div>
          <div class="cell-source" title="${c.source_preview}">${c.source_preview || '<em style=color:var(--text3)>empty</em>'}</div>
          <div class="cell-exec">#${c.index}</div>
        </div>`;
    }).join('');
    list.innerHTML = html || '<div style="padding:40px;text-align:center;color:var(--text3);">No code cells found</div>';

    // Auto-scroll to last executed cell
    const rows = list.querySelectorAll('.cell-row');
    const lastDone = [...rows].filter(r => r.querySelector('.badge-done,.badge-running')).pop();
    if (lastDone) lastDone.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch(e) {
    document.getElementById('last-update').textContent = '⚠ Server unreachable — retrying…';
  }
}

refresh();
setInterval(refresh, 4000);
</script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default access logs

    def do_GET(self):
        if self.path == "/status" or self.path.startswith("/status?"):
            data = get_notebook_status()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def run():
    server = HTTPServer(("", PORT), MonitorHandler)
    print(f"[Monitor] Running at http://localhost:{PORT}")
    print(f"[Monitor] Watching: {NOTEBOOK_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    run()

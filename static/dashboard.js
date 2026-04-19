const $ = (sel) => document.querySelector(sel);

let rateChart = null;
let levelChart = null;

async function fetchJson(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `request failed (${resp.status})`);
  }
  return resp.json();
}

function renderCards(report) {
  const anomCount = report.anomalies.length;
  const errCount = report.level_counts.ERROR || 0;
  const warnCount = report.level_counts.WARN || 0;
  const okClass = anomCount === 0 ? "ok" : "err";
  const cards = [
    { label: "Total events", value: report.total },
    { label: "Anomalies", value: anomCount, cls: okClass },
    { label: "Errors", value: errCount, cls: errCount ? "err" : "ok" },
    { label: "Warnings", value: warnCount, cls: warnCount ? "warn" : "ok" },
    { label: "Window", value: report.window_start
        ? `${report.window_start.slice(11, 19)} → ${report.window_end.slice(11, 19)}`
        : "—" },
  ];
  $("#summary").innerHTML = cards.map(c =>
    `<div class="card ${c.cls || ""}"><div class="value">${c.value}</div><div class="label">${c.label}</div></div>`
  ).join("");
}

function renderRateChart(report) {
  const labels = report.buckets.map(b => b.ts.slice(11, 16));
  const totals = report.buckets.map(b => b.total);
  const errors = report.buckets.map(b => b.errors);
  const anomalySet = new Set(report.anomalies.filter(a => a.kind === "rate_spike" || a.kind === "error_spike").map(a => a.ts));
  const anomalyPoints = report.buckets.map(b => anomalySet.has(b.ts) ? b.total : null);

  if (rateChart) rateChart.destroy();
  rateChart = new Chart($("#rate-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Events/min", data: totals, borderColor: "#38bdf8", backgroundColor: "rgba(56,189,248,0.1)", fill: true, tension: 0.2 },
        { label: "Errors/min", data: errors, borderColor: "#f87171", backgroundColor: "rgba(248,113,113,0.1)", fill: true, tension: 0.2 },
        { label: "Anomalies", data: anomalyPoints, borderColor: "#facc15", backgroundColor: "#facc15", pointRadius: 6, showLine: false, type: "scatter" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
      },
    },
  });
}

function renderLevelChart(report) {
  const entries = Object.entries(report.level_counts);
  const labels = entries.map(([k]) => k);
  const data = entries.map(([, v]) => v);
  const palette = { INFO: "#4ade80", WARN: "#fbbf24", ERROR: "#f87171", DEBUG: "#94a3b8", TRACE: "#64748b" };
  const colors = labels.map(l => palette[l] || "#60a5fa");

  if (levelChart) levelChart.destroy();
  levelChart = new Chart($("#level-chart"), {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: "#e2e8f0" } } },
    },
  });
}

function renderAnomalies(report) {
  const tbody = $("#anomaly-table tbody");
  if (!report.anomalies.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty">No anomalies detected.</td></tr>`;
    return;
  }
  tbody.innerHTML = report.anomalies.map(a =>
    `<tr>
      <td>${a.ts.slice(0, 19).replace("T", " ")}</td>
      <td class="kind-${a.kind}">${a.kind.replace("_", " ")}</td>
      <td>${a.score.toFixed(2)}</td>
      <td>${a.detail}</td>
    </tr>`
  ).join("");
}

function renderList(tableId, rows) {
  const tbody = $(`#${tableId} tbody`);
  if (!rows.length) {
    tbody.innerHTML = `<tr><td class="empty">—</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(([k, v]) =>
    `<tr><td>${k}</td><td style="text-align:right">${v}</td></tr>`
  ).join("");
}

function render(report) {
  renderCards(report);
  renderRateChart(report);
  renderLevelChart(report);
  renderAnomalies(report);
  renderList("paths-table", report.top_paths);
  renderList("ips-table", report.top_ips);
}

async function loadSample() {
  try { render(await fetchJson("/api/sample")); }
  catch (e) { alert(e.message); }
}

async function analyzeText() {
  const text = $("#log-text").value;
  if (!text.trim()) { alert("Paste some log lines first."); return; }
  try {
    render(await fetchJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }));
  } catch (e) { alert(e.message); }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  try { render(await fetchJson("/api/analyze", { method: "POST", body: fd })); }
  catch (e) { alert(e.message); }
}

$("#load-sample").addEventListener("click", loadSample);
$("#analyze-text").addEventListener("click", analyzeText);
$("#file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});

loadSample();

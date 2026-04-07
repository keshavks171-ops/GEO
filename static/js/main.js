/* main.js — Stock Predictor frontend logic */

// chart instances (kept so we can destroy before re-drawing)
let priceChart, cmChart, rocChart, fiChart;

const CHART_DEFAULTS = {
  color: "#e6edf3",
  gridColor: "rgba(48,54,61,.7)",
  font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" },
};

Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font.family = CHART_DEFAULTS.font.family;

function destroyCharts() {
  [priceChart, cmChart, rocChart, fiChart].forEach(c => c && c.destroy());
}

// ── main entry ────────────────────────────────────────────────────────────────
async function runPrediction() {
  const ticker   = document.getElementById("ticker").value.trim().toUpperCase();
  const start    = document.getElementById("start").value;
  const end      = document.getElementById("end").value;
  const n_splits = parseInt(document.getElementById("splits").value);
  const n_trees  = parseInt(document.getElementById("trees").value);

  if (!ticker) { showError("Please enter a ticker symbol."); return; }
  if (start >= end) { showError("Start date must be before end date."); return; }

  hideError();
  setLoading(true, "Downloading market data…");

  try {
    // small delay so the overlay renders before heavy work starts
    await new Promise(r => setTimeout(r, 50));
    setLoadingText("Training Random Forest… this may take 15–30 seconds.");

    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, start, end, n_splits, n_trees }),
    });

    const data = await res.json();
    if (!res.ok) { showError(data.error || "Prediction failed."); return; }

    setLoadingText("Rendering charts…");
    await new Promise(r => setTimeout(r, 30));

    renderResults(data);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    setLoading(false);
  }
}

// ── render ─────────────────────────────────────────────────────────────────────
function renderResults(d) {
  destroyCharts();

  // banner
  const dir = d.prediction.direction;
  const bannerDir = document.getElementById("banner-direction");
  bannerDir.textContent = dir === "UP" ? "📈 UP" : "📉 DOWN";
  bannerDir.className = "banner-direction " + dir.toLowerCase();

  document.getElementById("bm-confidence").textContent = pct(d.prediction.confidence);
  document.getElementById("bm-pup").textContent        = pct(d.prediction.p_up);
  document.getElementById("bm-pdown").textContent      = pct(d.prediction.p_down);
  document.getElementById("bm-rows").textContent       = d.rows.toLocaleString();

  // overall metrics
  document.getElementById("m-accuracy").textContent = d.overall.accuracy.toFixed(4);
  document.getElementById("m-auc").textContent      = d.overall.roc_auc.toFixed(4);

  // fold table
  const tbody = document.getElementById("fold-tbody");
  tbody.innerHTML = "";
  d.folds.forEach(f => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>Fold ${f.fold}</td><td>${f.accuracy.toFixed(4)}</td><td>${f.roc_auc.toFixed(4)}</td>`;
    tbody.appendChild(tr);
  });

  // charts
  drawPriceChart(d.price_history, d.ticker);
  drawCMChart(d.confusion_matrix);
  drawROCChart(d.roc_curve, d.overall.roc_auc);
  drawFIChart(d.feature_importance);

  document.getElementById("results").classList.remove("hidden");
  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── price chart ───────────────────────────────────────────────────────────────
function drawPriceChart({ dates, close }, ticker) {
  const ctx = document.getElementById("price-chart").getContext("2d");
  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [{
        label: ticker + " Close",
        data: close,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88,166,255,.08)",
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 6, maxRotation: 0 }, grid: { color: CHART_DEFAULTS.gridColor } },
        y: { grid: { color: CHART_DEFAULTS.gridColor } },
      },
    },
  });
}

// ── confusion matrix ──────────────────────────────────────────────────────────
function drawCMChart(cm) {
  // cm = [[TN, FP],[FN, TP]]
  const labels = ["Predicted DOWN", "Predicted UP"];
  const ctx = document.getElementById("cm-chart").getContext("2d");

  // build a simple 2×2 bubble-style chart using a custom bar approach
  // We'll use a matrix-style chart via stacked bar with annotations manually
  // Simpler: render as a grouped bar showing TN, FP, FN, TP
  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];

  cmChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Actual DOWN", "Actual UP"],
      datasets: [
        {
          label: "Predicted DOWN",
          data: [tn, fn],
          backgroundColor: ["rgba(88,166,255,.7)", "rgba(248,81,73,.5)"],
          borderWidth: 0,
        },
        {
          label: "Predicted UP",
          data: [fp, tp],
          backgroundColor: ["rgba(248,81,73,.5)", "rgba(63,185,80,.7)"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()}`,
          },
        },
      },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.gridColor } },
        y: { grid: { color: CHART_DEFAULTS.gridColor } },
      },
    },
  });
}

// ── ROC curve ─────────────────────────────────────────────────────────────────
function drawROCChart({ fpr, tpr }, auc) {
  const ctx = document.getElementById("roc-chart").getContext("2d");
  rocChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: fpr.map(v => v.toFixed(3)),
      datasets: [
        {
          label: `ROC (AUC = ${auc.toFixed(4)})`,
          data: tpr,
          borderColor: "#58a6ff",
          backgroundColor: "rgba(88,166,255,.1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.1,
        },
        {
          label: "Random",
          data: fpr,
          borderColor: "rgba(139,148,158,.5)",
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: {
          title: { display: true, text: "False Positive Rate" },
          grid: { color: CHART_DEFAULTS.gridColor },
          ticks: { maxTicksLimit: 6 },
        },
        y: {
          title: { display: true, text: "True Positive Rate" },
          grid: { color: CHART_DEFAULTS.gridColor },
        },
      },
    },
  });
}

// ── feature importances ───────────────────────────────────────────────────────
function drawFIChart(features) {
  const labels = features.map(f => f.feature).reverse();
  const values = features.map(f => f.importance).reverse();
  const ctx = document.getElementById("fi-chart").getContext("2d");

  fiChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance",
        data: values,
        backgroundColor: "rgba(63,185,80,.7)",
        borderWidth: 0,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.gridColor } },
        y: { grid: { color: CHART_DEFAULTS.gridColor }, ticks: { font: { size: 11 } } },
      },
    },
  });
}

// ── helpers ───────────────────────────────────────────────────────────────────
function pct(v) { return (v * 100).toFixed(1) + "%"; }

function setLoading(on, text = "") {
  const overlay = document.getElementById("loading-overlay");
  const btn = document.getElementById("run-btn");
  const spinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");

  if (on) {
    overlay.classList.remove("hidden");
    btn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Running…";
    if (text) document.getElementById("loading-text").textContent = text;
  } else {
    overlay.classList.add("hidden");
    btn.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "🚀 Run Prediction";
  }
}

function setLoadingText(text) {
  document.getElementById("loading-text").textContent = text;
}

function showError(msg) {
  const el = document.getElementById("error-msg");
  el.textContent = "⚠️ " + msg;
  el.classList.remove("hidden");
}
function hideError() {
  document.getElementById("error-msg").classList.add("hidden");
}

// allow Enter key in ticker input
document.getElementById("ticker").addEventListener("keydown", e => {
  if (e.key === "Enter") runPrediction();
});

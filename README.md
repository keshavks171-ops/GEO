# GEO — Log Analysis Dashboard

A small Flask app that parses log files and surfaces anomalies in real time.

## Features

- **Parsers** for Apache/nginx Combined Log Format and generic syslog-style lines (`<iso-ts> LEVEL message`).
- **Anomaly detection** with three independent signals:
  - Request-rate spikes (z-score over per-minute buckets)
  - Error-rate spikes (fraction of 5xx / `ERROR` lines per bucket)
  - Rare paths / statuses (low-frequency entries that produced errors)
- **Dashboard** with per-minute rate chart, log-level breakdown, anomaly table, and top paths / IPs.
- Upload a log file, paste raw lines, or click **Load sample** to try it on a synthetic dataset that contains a traffic spike, an error burst, and rare-path errors.

## Running

```bash
pip install -r requirements.txt
python sample_logs/generate.py   # regenerate the demo log (optional)
python app.py                    # serves on http://localhost:5000
```

## Tests

```bash
python -m unittest
```

## Layout

```
app.py                  Flask server + JSON API (/api/sample, /api/analyze)
analyzer.py             Log parsing + anomaly detection (pure Python, no deps)
templates/dashboard.html
static/dashboard.{js,css}
sample_logs/generate.py Synthetic log generator with embedded anomalies
tests/test_analyzer.py  Unit tests
```

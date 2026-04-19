"""Flask dashboard for log analysis with anomaly detection."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from analyzer import detect, parse_lines

SAMPLE_LOG = Path(__file__).parent / "sample_logs" / "access.log"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _analyze_text(text: str) -> dict:
    entries = parse_lines(text.splitlines())
    return detect(entries).to_dict() | {"parsed": len(entries)}


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/sample")
def api_sample():
    if not SAMPLE_LOG.exists():
        return jsonify(error="sample log missing"), 404
    return jsonify(_analyze_text(SAMPLE_LOG.read_text()))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if request.files.get("file"):
        text = request.files["file"].read().decode("utf-8", errors="replace")
    else:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
    if not text.strip():
        return jsonify(error="no log content provided"), 400
    return jsonify(_analyze_text(text))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

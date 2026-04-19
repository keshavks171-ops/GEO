"""Log parsing and anomaly detection.

Supports two common formats out of the box:
  - Combined/Common Log Format (Apache/nginx access logs)
  - Generic syslog-style lines:  "<ISO timestamp> <LEVEL> <message>"

Anomaly detection uses three independent signals:
  1. Request-rate spikes     (z-score over per-minute buckets)
  2. Error-rate spikes       (fraction of 5xx / ERROR lines per bucket)
  3. Rare status/endpoint    (low-frequency entries within the window)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Iterable


_CLF_RE = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<path>\S+)\s+\S+"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<agent>[^"]*)")?'
)

_SYSLOG_RE = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+'
    r'(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\s+'
    r'(?P<message>.*)'
)

_CLF_TS_FMT = "%d/%b/%Y:%H:%M:%S %z"


@dataclass
class LogEntry:
    ts: datetime
    level: str                # INFO / WARN / ERROR / ... (derived for access logs)
    source: str               # "access" | "app"
    ip: str | None = None
    method: str | None = None
    path: str | None = None
    status: int | None = None
    size: int | None = None
    message: str | None = None
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


def _parse_clf_ts(raw: str) -> datetime:
    return datetime.strptime(raw, _CLF_TS_FMT).astimezone(timezone.utc)


def _parse_iso_ts(raw: str) -> datetime:
    raw = raw.replace(",", ".").replace(" ", "T")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _status_level(status: int) -> str:
    if status >= 500:
        return "ERROR"
    if status >= 400:
        return "WARN"
    return "INFO"


def parse_line(line: str) -> LogEntry | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None

    m = _CLF_RE.match(line)
    if m:
        try:
            status = int(m.group("status"))
            size_raw = m.group("size")
            size = int(size_raw) if size_raw.isdigit() else None
            return LogEntry(
                ts=_parse_clf_ts(m.group("ts")),
                level=_status_level(status),
                source="access",
                ip=m.group("ip"),
                method=m.group("method"),
                path=m.group("path"),
                status=status,
                size=size,
                raw=line,
            )
        except (ValueError, KeyError):
            return None

    m = _SYSLOG_RE.match(line)
    if m:
        try:
            level = m.group("level").upper()
            if level == "WARNING":
                level = "WARN"
            if level in ("CRITICAL", "FATAL"):
                level = "ERROR"
            return LogEntry(
                ts=_parse_iso_ts(m.group("ts")),
                level=level,
                source="app",
                message=m.group("message"),
                raw=line,
            )
        except ValueError:
            return None

    return None


def parse_lines(lines: Iterable[str]) -> list[LogEntry]:
    out: list[LogEntry] = []
    for line in lines:
        entry = parse_line(line)
        if entry is not None:
            out.append(entry)
    out.sort(key=lambda e: e.ts)
    return out


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #

@dataclass
class Bucket:
    ts: str           # ISO, minute-aligned
    total: int = 0
    errors: int = 0
    warns: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0


@dataclass
class Anomaly:
    ts: str
    kind: str         # "rate_spike" | "error_spike" | "rare_path" | "rare_status"
    score: float
    detail: str


@dataclass
class Report:
    total: int
    window_start: str | None
    window_end: str | None
    buckets: list[Bucket] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    level_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    top_paths: list[tuple[str, int]] = field(default_factory=list)
    top_ips: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "buckets": [asdict(b) | {"error_rate": b.error_rate} for b in self.buckets],
            "anomalies": [asdict(a) for a in self.anomalies],
            "level_counts": self.level_counts,
            "status_counts": self.status_counts,
            "top_paths": self.top_paths,
            "top_ips": self.top_ips,
        }


def _bucket_key(ts: datetime) -> str:
    return ts.replace(second=0, microsecond=0).isoformat()


def _zscore(value: float, series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    mu = mean(series)
    sigma = pstdev(series)
    if sigma == 0:
        return 0.0
    return (value - mu) / sigma


def detect(entries: list[LogEntry], *, z_threshold: float = 3.0,
           error_rate_threshold: float = 0.25, rare_quantile: float = 0.01) -> Report:
    if not entries:
        return Report(total=0, window_start=None, window_end=None)

    buckets: dict[str, Bucket] = {}
    level_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    ip_counts: Counter[str] = Counter()

    for e in entries:
        key = _bucket_key(e.ts)
        b = buckets.setdefault(key, Bucket(ts=key))
        b.total += 1
        if e.level == "ERROR":
            b.errors += 1
        elif e.level == "WARN":
            b.warns += 1
        level_counts[e.level] += 1
        if e.status is not None:
            status_counts[str(e.status)] += 1
        if e.path:
            path_counts[e.path] += 1
        if e.ip:
            ip_counts[e.ip] += 1

    ordered = sorted(buckets.values(), key=lambda b: b.ts)
    totals = [b.total for b in ordered]
    err_rates = [b.error_rate for b in ordered]

    anomalies: list[Anomaly] = []

    for b in ordered:
        z = _zscore(b.total, totals)
        if z >= z_threshold:
            anomalies.append(Anomaly(
                ts=b.ts, kind="rate_spike", score=round(z, 2),
                detail=f"{b.total} events in one minute (z={z:.2f})",
            ))
        if b.total >= 10 and b.error_rate >= error_rate_threshold:
            ez = _zscore(b.error_rate, err_rates)
            anomalies.append(Anomaly(
                ts=b.ts, kind="error_spike",
                score=round(max(ez, b.error_rate * 10), 2),
                detail=f"error rate {b.error_rate:.0%} ({b.errors}/{b.total})",
            ))

    # Rare path / status detection: flag entries whose key occurs in the
    # bottom `rare_quantile` share of traffic, AND at least one occurrence
    # returned an error. This catches "one-off 500s" worth investigating.
    total_events = sum(totals)
    if total_events:
        min_count = max(1, math.ceil(total_events * rare_quantile))
        error_paths = {e.path for e in entries if e.path and e.level == "ERROR"}
        for path in error_paths:
            if path_counts[path] <= min_count:
                last = max(e.ts for e in entries if e.path == path)
                anomalies.append(Anomaly(
                    ts=_bucket_key(last), kind="rare_path", score=1.0,
                    detail=f"error on rarely-seen path {path} (seen {path_counts[path]}x)",
                ))
        for status, count in status_counts.items():
            if status.startswith("5") and count <= min_count:
                anomalies.append(Anomaly(
                    ts=ordered[-1].ts, kind="rare_status", score=1.0,
                    detail=f"rare status {status} seen {count}x",
                ))

    anomalies.sort(key=lambda a: (a.ts, -a.score))

    return Report(
        total=len(entries),
        window_start=entries[0].ts.isoformat(),
        window_end=entries[-1].ts.isoformat(),
        buckets=ordered,
        anomalies=anomalies,
        level_counts=dict(level_counts),
        status_counts=dict(status_counts),
        top_paths=path_counts.most_common(10),
        top_ips=ip_counts.most_common(10),
    )

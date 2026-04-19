"""Generate a synthetic access.log with embedded anomalies for the demo."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(42)

PATHS_NORMAL = [
    ("GET", "/", 200),
    ("GET", "/about", 200),
    ("GET", "/api/products", 200),
    ("GET", "/api/products/42", 200),
    ("POST", "/api/cart", 200),
    ("GET", "/static/app.js", 200),
    ("GET", "/static/style.css", 200),
    ("GET", "/api/user", 200),
    ("GET", "/health", 200),
    ("POST", "/api/login", 200),
    ("GET", "/missing", 404),
]
PATHS_RARE = [("GET", "/admin/debug", 500), ("GET", "/api/internal/dump", 500)]
IPS = ["10.0.0.5", "10.0.0.12", "10.0.0.34", "192.168.1.10", "203.0.113.7", "198.51.100.4"]
AGENTS = ['"Mozilla/5.0"', '"curl/8.1.2"']


def fmt(ts: datetime, ip: str, method: str, path: str, status: int, size: int) -> str:
    return (
        f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S %z")}] '
        f'"{method} {path} HTTP/1.1" {status} {size} '
        f'"-" {random.choice(AGENTS)}'
    )


def main() -> None:
    out = Path(__file__).parent / "access.log"
    lines: list[str] = []

    start = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    # 30 minutes of normal traffic: ~20 req/min.
    for minute in range(30):
        rate = 20
        # A traffic spike at minute 15 (~220 req/min).
        if minute == 15:
            rate = 220
        # Error burst at minute 22 (40% 5xx).
        err_frac = 0.40 if minute == 22 else 0.02

        for _ in range(rate):
            method, path, status = random.choice(PATHS_NORMAL)
            if random.random() < err_frac:
                status = 500
            offset = random.random() * 60
            ts = start + timedelta(minutes=minute, seconds=offset)
            lines.append(fmt(ts, random.choice(IPS), method, path, status, random.randint(200, 8000)))

    # Rare path errors at minute 27.
    for method, path, status in PATHS_RARE:
        ts = start + timedelta(minutes=27, seconds=random.random() * 60)
        lines.append(fmt(ts, "203.0.113.99", method, path, status, 512))

    lines.sort()  # approximate chronological order
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()

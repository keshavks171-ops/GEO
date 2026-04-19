"""Unit tests for analyzer.py. Run with `python -m unittest`."""

from __future__ import annotations

import unittest
from pathlib import Path

from analyzer import detect, parse_line, parse_lines


class ParseLineTests(unittest.TestCase):
    def test_combined_log_format(self) -> None:
        line = (
            '10.0.0.1 - - [19/Apr/2026:12:00:01 +0000] '
            '"GET /api/x HTTP/1.1" 200 1234 "-" "curl/8"'
        )
        e = parse_line(line)
        assert e is not None
        self.assertEqual(e.source, "access")
        self.assertEqual(e.ip, "10.0.0.1")
        self.assertEqual(e.method, "GET")
        self.assertEqual(e.path, "/api/x")
        self.assertEqual(e.status, 200)
        self.assertEqual(e.level, "INFO")

    def test_5xx_becomes_error_level(self) -> None:
        line = '1.1.1.1 - - [19/Apr/2026:12:00:01 +0000] "GET / HTTP/1.1" 503 0'
        e = parse_line(line)
        assert e is not None
        self.assertEqual(e.level, "ERROR")

    def test_syslog_format(self) -> None:
        e = parse_line("2026-04-19T12:00:01Z ERROR database down")
        assert e is not None
        self.assertEqual(e.source, "app")
        self.assertEqual(e.level, "ERROR")
        self.assertEqual(e.message, "database down")

    def test_warning_normalizes_to_warn(self) -> None:
        e = parse_line("2026-04-19T12:00:01Z WARNING slow query")
        assert e is not None
        self.assertEqual(e.level, "WARN")

    def test_unparsable_returns_none(self) -> None:
        self.assertIsNone(parse_line("this is not a log line"))
        self.assertIsNone(parse_line(""))


class DetectTests(unittest.TestCase):
    def test_empty_report(self) -> None:
        r = detect([])
        self.assertEqual(r.total, 0)
        self.assertEqual(r.anomalies, [])

    def test_rate_spike_detected(self) -> None:
        lines: list[str] = []
        # 10 minutes at ~5 req/min, then 1 minute at 120 req/min.
        for minute in range(10):
            for second in range(5):
                lines.append(
                    f'1.1.1.1 - - [19/Apr/2026:12:{minute:02d}:{second:02d} +0000] '
                    f'"GET / HTTP/1.1" 200 100'
                )
        for second in range(120):
            lines.append(
                f'1.1.1.1 - - [19/Apr/2026:12:11:{second % 60:02d} +0000] '
                f'"GET / HTTP/1.1" 200 100'
            )
        r = detect(parse_lines(lines))
        kinds = {a.kind for a in r.anomalies}
        self.assertIn("rate_spike", kinds)

    def test_error_spike_detected(self) -> None:
        lines: list[str] = []
        for minute in range(5):
            for second in range(20):
                lines.append(
                    f'1.1.1.1 - - [19/Apr/2026:12:{minute:02d}:{second:02d} +0000] '
                    f'"GET / HTTP/1.1" 200 100'
                )
        # Bad minute: 15 errors out of 20.
        for second in range(20):
            status = 500 if second < 15 else 200
            lines.append(
                f'1.1.1.1 - - [19/Apr/2026:12:06:{second:02d} +0000] '
                f'"GET / HTTP/1.1" {status} 100'
            )
        r = detect(parse_lines(lines))
        kinds = {a.kind for a in r.anomalies}
        self.assertIn("error_spike", kinds)


class SampleLogTests(unittest.TestCase):
    def test_sample_log_has_anomalies(self) -> None:
        path = Path(__file__).resolve().parent.parent / "sample_logs" / "access.log"
        if not path.exists():
            self.skipTest("sample log not generated")
        entries = parse_lines(path.read_text().splitlines())
        self.assertGreater(len(entries), 100)
        r = detect(entries)
        self.assertGreater(len(r.anomalies), 0)


if __name__ == "__main__":
    unittest.main()

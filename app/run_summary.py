"""Run summaries and end-to-end verification of a demo run.

The producer and consumer each write a JSON summary of what *they* observed:

* Producer summary -> broker delivery confirmations (acknowledgements).
* Consumer summary -> messages actually read back from the topic.

`verify_end_to_end` compares the two. Only when both sides agree do we call
the run end-to-end verified. Run as a CLI to print a PASS/FAIL report:

    python -m app.run_summary --producer runs/p.json --consumer runs/c.json --expected 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SummaryError(ValueError):
    """Raised when a summary file is missing, unreadable or incomplete."""


@dataclass(frozen=True)
class VerificationCheck:
    """One comparison between an expected and an observed value."""

    name: str
    expected: object
    actual: object

    @property
    def passed(self) -> bool:
        """True when the observed value equals the expected value."""
        return self.expected == self.actual


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a summary as pretty-printed JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_summary(path: Path) -> dict[str, Any]:
    """Load a summary JSON file.

    Raises:
        SummaryError: If the file is missing or does not contain a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SummaryError(f"summary file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SummaryError(f"summary file is not valid JSON: {path}: {error}") from error
    if not isinstance(data, dict):
        raise SummaryError(f"summary file must contain a JSON object: {path}")
    return data


def _require(summary: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in summary:
        raise SummaryError(f"{source} summary is missing '{key}'")
    return summary[key]


def verify_end_to_end(
    producer_summary: Mapping[str, Any],
    consumer_summary: Mapping[str, Any],
    expected_count: int,
) -> list[VerificationCheck]:
    """Compare producer delivery stats with consumer receipt stats.

    Raises:
        SummaryError: If a required key is missing from either summary.
    """

    def produced(key: str) -> Any:
        return _require(producer_summary, key, "producer")

    def consumed(key: str) -> Any:
        return _require(consumer_summary, key, "consumer")

    delivered = produced("delivered")
    return [
        VerificationCheck("producer: delivered == expected", expected_count, delivered),
        VerificationCheck("producer: delivery failures", 0, produced("failed")),
        VerificationCheck("producer: unconfirmed after flush", 0, produced("not_confirmed")),
        VerificationCheck("consumer: received == delivered", delivered, consumed("received")),
        VerificationCheck(
            "consumer: schema-valid == received", consumed("received"), consumed("valid")
        ),
        VerificationCheck("consumer: malformed records", 0, consumed("malformed")),
        VerificationCheck("consumer: duplicate event IDs", 0, consumed("duplicates")),
        VerificationCheck(
            "consumer: unique event IDs == delivered", delivered, consumed("unique_event_ids")
        ),
        VerificationCheck("consumer: Kafka errors", 0, consumed("kafka_errors")),
        VerificationCheck(
            "per-partition counts match", produced("per_partition"), consumed("per_partition")
        ),
    ]


def format_report(checks: Sequence[VerificationCheck]) -> str:
    """Render checks as a human-readable report ending in PASS or FAIL."""
    lines = ["End-to-end verification", "-" * 72]
    for check in checks:
        status = "ok  " if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name:<42} expected={check.expected} actual={check.actual}")
    lines.append("-" * 72)
    lines.append("RESULT: PASS" if all(check.passed for check in checks) else "RESULT: FAIL")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a producer/consumer demo run.")
    parser.add_argument("--producer", type=Path, required=True, help="Producer summary JSON")
    parser.add_argument("--consumer", type=Path, required=True, help="Consumer summary JSON")
    parser.add_argument("--expected", type=int, required=True, help="Expected event count")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 on PASS, 1 on FAIL, 2 on unusable input."""
    args = _parse_args(argv)
    try:
        checks = verify_end_to_end(
            load_summary(args.producer), load_summary(args.consumer), args.expected
        )
    except SummaryError as error:
        print(f"RESULT: FAIL ({error})", file=sys.stderr)
        return 2
    # The report is the CLI's primary output, so it goes to stdout.
    print(format_report(checks))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())

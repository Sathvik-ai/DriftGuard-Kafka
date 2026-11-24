"""Tests for run summaries and end-to-end verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.run_summary import (
    SummaryError,
    format_report,
    load_summary,
    main,
    verify_end_to_end,
    write_summary,
)


def producer_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "delivered": 1000,
        "failed": 0,
        "not_confirmed": 0,
        "per_partition": {"0": 300, "1": 350, "2": 350},
    }
    summary.update(overrides)
    return summary


def consumer_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "received": 1000,
        "valid": 1000,
        "malformed": 0,
        "duplicates": 0,
        "unique_event_ids": 1000,
        "kafka_errors": 0,
        "per_partition": {"0": 300, "1": 350, "2": 350},
    }
    summary.update(overrides)
    return summary


class TestVerifyEndToEnd:
    def test_matching_runs_pass_every_check(self) -> None:
        checks = verify_end_to_end(producer_summary(), consumer_summary(), 1000)
        assert all(check.passed for check in checks)
        assert format_report(checks).endswith("RESULT: PASS")

    @pytest.mark.parametrize(
        ("producer_overrides", "consumer_overrides"),
        [
            ({"delivered": 999}, {}),
            ({"failed": 1}, {}),
            ({"not_confirmed": 2}, {}),
            ({}, {"received": 999, "valid": 999, "unique_event_ids": 999}),
            ({}, {"malformed": 1, "valid": 999}),
            ({}, {"duplicates": 1, "unique_event_ids": 999}),
            ({}, {"kafka_errors": 1}),
            ({}, {"per_partition": {"0": 1000}}),
        ],
    )
    def test_any_mismatch_fails(
        self, producer_overrides: dict[str, Any], consumer_overrides: dict[str, Any]
    ) -> None:
        checks = verify_end_to_end(
            producer_summary(**producer_overrides), consumer_summary(**consumer_overrides), 1000
        )
        assert not all(check.passed for check in checks)
        assert format_report(checks).endswith("RESULT: FAIL")

    def test_missing_key_raises(self) -> None:
        incomplete = producer_summary()
        del incomplete["failed"]
        with pytest.raises(SummaryError, match="failed"):
            verify_end_to_end(incomplete, consumer_summary(), 1000)


class TestSummaryFiles:
    def test_write_then_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "summary.json"
        write_summary(path, {"delivered": 5})
        assert load_summary(path) == {"delivered": 5}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SummaryError, match="not found"):
            load_summary(tmp_path / "absent.json")

    @pytest.mark.parametrize("content", ["{broken", "[1, 2]"])
    def test_invalid_content_raises(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "bad.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(SummaryError):
            load_summary(path)


class TestCli:
    def test_exit_codes_reflect_result(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        producer_path, consumer_path = tmp_path / "p.json", tmp_path / "c.json"
        write_summary(producer_path, producer_summary())
        write_summary(consumer_path, consumer_summary())
        args = ["--producer", str(producer_path), "--consumer", str(consumer_path)]

        assert main([*args, "--expected", "1000"]) == 0
        assert "RESULT: PASS" in capsys.readouterr().out
        assert main([*args, "--expected", "1001"]) == 1
        assert "RESULT: FAIL" in capsys.readouterr().out

    def test_unreadable_input_returns_2(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "missing.json")
        assert main(["--producer", missing, "--consumer", missing, "--expected", "1"]) == 2

from typing import Any

import pytest

from rechunk_lambda_benchmarks.cost import (
    GB_SECOND_USD,
    REQUEST_USD,
    S3_GET_USD,
    S3_PUT_USD,
    actual,
)


def test_actual():
    joined: list[dict[str, Any]] = [
        {"billed_ms": 1000, "memory_size_bytes": 1024_000_000},
        {"billed_ms": 3000, "memory_size_bytes": 1024_000_000},
        {"billed_ms": None, "memory_size_bytes": 1024_000_000},
    ]
    spend = actual(joined)
    assert spend is not None
    assert spend["invocations"] == 2
    assert spend["gb_seconds"] == pytest.approx(4.0)
    assert spend["total_usd"] == pytest.approx(4.0 * GB_SECOND_USD + 2 * REQUEST_USD)


def test_actual_without_billing_data():
    assert actual([{"billed_ms": None, "memory_size_bytes": None}]) is None


def test_actual_with_s3_requests():
    joined = [
        {"billed_ms": 1000, "memory_size_bytes": 1024_000_000, "s3_gets": 2, "s3_puts": 1.5},
        {"billed_ms": 1000, "memory_size_bytes": 1024_000_000, "s3_gets": 2, "s3_puts": 1.5},
    ]
    spend = actual(joined)
    assert spend is not None
    assert spend["s3_recorded"]
    assert (spend["gets"], spend["puts"]) == (4, 3)
    assert spend["s3_usd"] == pytest.approx(4 * S3_GET_USD + 3 * S3_PUT_USD)
    assert spend["total_usd"] == pytest.approx(
        2.0 * GB_SECOND_USD + 2 * REQUEST_USD + spend["s3_usd"]
    )

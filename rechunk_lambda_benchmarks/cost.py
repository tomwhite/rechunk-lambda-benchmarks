"""
What a run cost, priced from the CloudWatch REPORT lines `cloudwatch.py` joins.
"""

from typing import Any

# arm64 list prices, standard US and EU regions
PRICING_AS_OF = "2026-09-08"
GB_SECOND_USD = 0.0000133334
REQUEST_USD = 0.20 / 1_000_000
S3_GET_USD = 0.0004 / 1000
S3_PUT_USD = 0.005 / 1000

# Logs Insights reports @memorySize in decimal MB, but Lambda bills with
# 1 GB = 1024 MB: a 1769 MB function is 1.7275 GB, not 1.769.
MEGABYTE = 1_000_000
MEGABYTES_PER_GIGABYTE = 1024


def gigabytes(memory_size_bytes: float) -> float:
    """A function's memory size in the gigabytes Lambda bills in."""
    return memory_size_bytes / MEGABYTE / MEGABYTES_PER_GIGABYTE


def actual(joined: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Cost of the invocations in `joined`, or None if the REPORT lines
    carried no billing data.

    Counts only the invocations joined here: a coordinator's own cost,
    and any invocation whose REPORT line was missed, are not included.
    S3 requests are the per-task counts the CLI stamps from the plan, so
    exclude retries and metadata; `s3_recorded` is False for stats files
    written without them.
    """
    billable = [
        j
        for j in joined
        if j.get("billed_ms") is not None and j.get("memory_size_bytes") is not None
    ]
    if not billable:
        return None

    gb_seconds = sum(gigabytes(j["memory_size_bytes"]) * j["billed_ms"] / 1000 for j in billable)
    compute_usd = gb_seconds * GB_SECOND_USD
    request_usd = len(billable) * REQUEST_USD
    s3_recorded = all("s3_gets" in j for j in billable)
    gets = round(sum(j.get("s3_gets", 0) for j in billable))
    puts = round(sum(j.get("s3_puts", 0) for j in billable))
    s3_usd = gets * S3_GET_USD + puts * S3_PUT_USD
    return {
        "invocations": len(billable),
        "gb_seconds": gb_seconds,
        "compute_usd": compute_usd,
        "request_usd": request_usd,
        "s3_recorded": s3_recorded,
        "gets": gets,
        "puts": puts,
        "s3_usd": s3_usd,
        "total_usd": compute_usd + request_usd + s3_usd,
    }


def summarize_actual(cost: dict[str, Any]) -> str:
    """The cost lines for a run's report."""
    lambda_usd = cost["compute_usd"] + cost["request_usd"]
    if cost["s3_recorded"]:
        s3 = f"${cost['s3_usd']:.4f} ({cost['gets']:,} GETs, {cost['puts']:,} PUTs)"
    else:
        s3 = "not recorded in this stats file, so not in the total"
    return "\n".join(
        [
            f"cost:           ${cost['total_usd']:.4f} total (approximate)",
            f"                  lambda  ${lambda_usd:.4f} "
            f"({cost['gb_seconds']:.1f} GB-s, {cost['invocations']:,} requests)",
            f"                  S3      {s3}",
            f"                (arm64 and S3 Standard list prices as of {PRICING_AS_OF};",
            "                 excludes the free tier, retries and metadata requests)",
        ]
    )

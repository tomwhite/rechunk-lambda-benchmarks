"""
Server-side timing for a run: the other half of `map_lambda.stats`.

Lambda writes a REPORT line for every invocation, with the real
execution duration, the init duration of a cold start, and the memory
used. `req_id`, which `write_jsonl` records, is the join key.

No single entry point, so a command can report between the steps:

    stats = map_lambda.stats.read_jsonl(path)
    windows = query_windows(stats)
    reports, truncated = query_all_reports(boto3.client("logs"), group, windows)
    joined = join(stats, reports)
    print(summarize(joined, missing(stats, reports)))

What the resulting split means:

    init          cold-start setup, separate from `execution`, so a
                  cold invocation costs both.
    execution     the handler running, storage round trips included.
    dispatch      `invoke()` being called to the handler starting:
                  connection queueing and setup, and Lambda's own queue.
    return        the handler finishing to `invoke()` returning.

`return` is measured from @timestamp, when Lambda wrote the REPORT
line, so it crosses a clock boundary: a few milliseconds either way
means zero.
"""

import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from map_lambda.stats import Stats, distribution, percentile

# @initDuration is absent on warm invocations, which is how a cold start is detected.
REPORT_QUERY = """
fields @timestamp, @requestId, @duration, @billedDuration, @initDuration
| fields @maxMemoryUsed, @memorySize
| filter @type = "REPORT"
| limit 10000
"""

# UTC with no zone marker, which Python would parse as local time.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# The REPORT line is written after the invocation returns, on Lambda's clock.
WINDOW_PADDING_SECONDS = 60

# Logs Insights' row cap per query, as in REPORT_QUERY's `limit`. It
# truncates silently, and the subset left is not a random one.
REPORT_LIMIT = 10_000

# Well under REPORT_LIMIT: a window also holds REPORT lines from other
# runs and from retried attempts.
SLICE_RECORDS = 5_000

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 120.0

# @memorySize and @maxMemoryUsed are in decimal megabytes.
MEGABYTE = 1_000_000


def log_group_name(function_name: str) -> str:
    """Where Lambda writes a function's logs, by convention."""
    return f"/aws/lambda/{function_name}"


def time_window(stats: list[Stats], padding: float = WINDOW_PADDING_SECONDS) -> tuple[int, int]:
    """
    The epoch-second window to search, from a run's client-side stats,
    padded because the client's and Lambda's clocks only roughly agree.
    """
    start = min(s["invoke_start"] for s in stats)
    end = max(s["invoke_end"] for s in stats)
    return int(start - padding), int(end + padding)


def parse_results(results: Iterable[list[dict[str, str]]]) -> dict[str, dict[str, float]]:
    """
    Turn Logs Insights' list-of-field-dicts rows into `{req_id: report}`
    of floats, with @timestamp as an epoch second.
    """
    reports = {}
    for row in results:
        fields = {field["field"]: field["value"] for field in row}
        req_id = fields.pop("@requestId", None)
        if req_id is None:
            continue
        stamp = fields.pop("@timestamp", None)
        report = {
            name: float(value)
            for name, value in fields.items()
            if name.startswith("@") and name != "@ptr"
        }
        if stamp is not None:
            report["@timestamp"] = (
                datetime.strptime(stamp, TIMESTAMP_FORMAT).replace(tzinfo=UTC).timestamp()
            )
        reports[req_id] = report
    return reports


def query_windows(
    stats: list[Stats],
    padding: float = WINDOW_PADDING_SECONDS,
    slice_records: int = SLICE_RECORDS,
) -> list[tuple[int, int]]:
    """
    Contiguous windows covering the run's padded window, one per
    `slice_records` invocations. Results are merged before joining, so a
    REPORT line clock skew moves into the next window is still found.
    """
    start, end = time_window(stats, padding)
    ends = sorted(s["invoke_end"] for s in stats)
    cuts = (int(ends[i]) for i in range(slice_records, len(ends), slice_records))
    bounds = sorted({start, end, *(c for c in cuts if start < c < end)})
    if len(bounds) < 2:
        return [(start, end)]
    return list(pairwise(bounds))


def query_reports(
    logs: Any,
    log_group: str,
    start: int,
    end: int,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, dict[str, float]]:
    """
    Run REPORT_QUERY over `log_group` and wait for it, returning
    `{req_id: report}` for every invocation in the window. Not filtered
    on this run's request ids: that would not raise the row limit.
    """
    query_id = logs.start_query(
        logGroupName=log_group,
        startTime=start,
        endTime=end,
        queryString=REPORT_QUERY,
    )["queryId"]

    deadline = time.monotonic() + timeout
    while True:
        response = logs.get_query_results(queryId=query_id)
        status = response["status"]
        if status == "Complete":
            return parse_results(response["results"])
        if status in ("Failed", "Cancelled", "Timeout"):
            raise RuntimeError(f"CloudWatch Logs Insights query {status.lower()} ({query_id})")
        if time.monotonic() > deadline:
            logs.stop_query(queryId=query_id)
            raise RuntimeError(
                f"CloudWatch Logs Insights query still {status.lower()} after {timeout:.0f}s "
                f"({query_id}) - raise --timeout"
            )
        time.sleep(poll_interval)


def query_all_reports(
    logs: Any,
    log_group: str,
    windows: list[tuple[int, int]],
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    on_window: Callable[[int, int, int], None] | None = None,
) -> tuple[dict[str, dict[str, float]], int]:
    """
    Run `query_reports` over each window and merge by request id,
    returning the reports and how many windows came back full - which
    were truncated, and the caller must say so.
    """
    reports: dict[str, dict[str, float]] = {}
    truncated = 0
    for start, end in windows:
        found = query_reports(
            logs, log_group, start, end, poll_interval=poll_interval, timeout=timeout
        )
        if len(found) >= REPORT_LIMIT:
            truncated += 1
        reports.update(found)
        if on_window is not None:
            on_window(start, end, len(found))
    return reports, truncated


def join(stats: list[Stats], reports: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """
    Match each client-side record to its REPORT line, splitting the
    client-observed duration into dispatch, init, execution and return.

    The four sum to `client_ms` by construction: `dispatch_ms` is what
    is left, so clock skew lands there rather than in a gap. Without
    @timestamp both are None and only `unaccounted_ms` is available.

    Invocations with no REPORT line are dropped; see `missing`.
    """
    joined = []
    for s in stats:
        report = reports.get(s["req_id"])
        if report is None:
            continue
        client_ms = (s["invoke_end"] - s["invoke_start"]) * 1000
        init_ms = report.get("@initDuration")
        execution_ms = report["@duration"]
        unaccounted_ms = client_ms - execution_ms - (init_ms or 0.0)
        report_end = report.get("@timestamp")
        return_ms = None if report_end is None else (s["invoke_end"] - report_end) * 1000
        joined.append(
            {
                **s,
                "client_ms": client_ms,
                "init_ms": init_ms,
                "execution_ms": execution_ms,
                "billed_ms": report.get("@billedDuration"),
                "max_memory_bytes": report.get("@maxMemoryUsed"),
                "memory_size_bytes": report.get("@memorySize"),
                "unaccounted_ms": unaccounted_ms,
                "return_ms": return_ms,
                "dispatch_ms": None if return_ms is None else unaccounted_ms - return_ms,
            }
        )
    return joined


def missing(stats: list[Stats], reports: dict[str, dict[str, float]]) -> list[str]:
    """The request ids with no REPORT line in the window."""
    return [s["req_id"] for s in stats if s["req_id"] not in reports]


def summarize(joined: list[dict[str, Any]], not_found: list[str]) -> str:
    """Where a run's time actually went, per invocation."""
    if not joined:
        return (
            "no REPORT lines matched this run's request ids - the logs may not have "
            "been ingested yet (wait a few seconds and retry), or the function name "
            "may be wrong"
        )

    lines = [f"matched:        {len(joined)} invocations"]
    if not_found:
        lines.append(
            f"                ({len(not_found)} had no REPORT line - logs still being "
            f"ingested, or from a run outside the queried window)"
        )

    cold = [j for j in joined if j["init_ms"] is not None]
    share = 100 * len(cold) / len(joined)
    lines.append(f"cold starts:    {len(cold)} of {len(joined)} ({share:.0f}%)")
    if cold:
        lines.append(distribution("init:", [j["init_ms"] for j in cold]))
    lines.append(distribution("execution:", [j["execution_ms"] for j in joined]))

    split = [j for j in joined if j["dispatch_ms"] is not None]
    if len(split) == len(joined):
        lines.append(distribution("dispatch:", [j["dispatch_ms"] for j in joined]))
        lines.append(distribution("return:", [j["return_ms"] for j in joined]))
        lines.append(
            "                (dispatch: pool queueing, connection setup and Lambda's own "
            "queue before the handler runs. return: the response coming back)"
        )
    else:
        lines.append(distribution("unaccounted:", [j["unaccounted_ms"] for j in joined]))
        lines.append(
            "                (queueing on the client, network, and Lambda dispatch - "
            "not the function's own time)"
        )

    memory = [j["max_memory_bytes"] for j in joined if j["max_memory_bytes"] is not None]
    sizes = {j["memory_size_bytes"] for j in joined if j["memory_size_bytes"] is not None}
    if memory:
        peak = f"{max(memory) / MEGABYTE:.0f} MB peak"
        configured = f" of {max(sizes) / MEGABYTE:.0f} MB configured" if sizes else ""
        lines.append(f"memory:         {peak}{configured}")

    billed = [j["billed_ms"] for j in joined if j["billed_ms"] is not None]
    if billed:
        lines.append(f"billed:         {sum(billed) / 1000:.1f}s across {len(billed)} invocations")

    return "\n".join(lines)


def slowest(joined: list[dict[str, Any]], n: int = 5) -> str:
    """
    The worst client-observed invocations, with their split - the thing
    to look at when the max is far from the p90.

    Named by `index`, the dispatch position map-lambda records: a chunk
    coordinate would need the run's shape and chunking, which a stats
    file does not carry.
    """
    worst = sorted(joined, key=lambda j: j["client_ms"], reverse=True)[:n]
    lines = [f"slowest {len(worst)} invocations:"]
    for j in worst:
        init = f"init {j['init_ms']:.0f}ms, " if j["init_ms"] is not None else ""
        if j["dispatch_ms"] is None:
            rest = f"unaccounted {j['unaccounted_ms']:.0f}ms"
        else:
            rest = f"dispatch {j['dispatch_ms']:.0f}ms, return {j['return_ms']:.0f}ms"
        lines.append(
            f"  {j['req_id']}  index {j['index']}  {j['client_ms']:.0f}ms "
            f"= {init}execution {j['execution_ms']:.0f}ms, {rest}"
        )
    return "\n".join(lines)


QUEUEING_SPREAD_LIMIT = 0.5


def queueing_dominates(joined: list[dict[str, Any]]) -> bool:
    """
    Whether waiting, rather than running, is what separates one
    invocation's duration from another's.

    What breaks a cold-versus-warm comparison is *spread* in the time
    spent outside the function, not its size: an overhead every
    invocation pays equally cancels out of the comparison, while one
    that some pay and others do not swamps it. So this is the gap
    between median and near-floor unaccounted time, against a typical
    duration.

    The floor is the 5th percentile rather than the minimum, and clamped
    at zero: `dispatch_ms` carries the clock skew between a coordinator
    and Lambda's REPORT line, so at any scale some invocations report a
    little negative and the minimum is one of them.
    """
    if not joined:
        return False
    unaccounted = sorted(j["unaccounted_ms"] for j in joined)
    floor = max(percentile(unaccounted, 0.05), 0.0)
    spread = percentile(unaccounted, 0.5) - floor
    typical = percentile(sorted(j["client_ms"] for j in joined), 0.5)
    return typical > 0 and spread > QUEUEING_SPREAD_LIMIT * typical


def cold_start_penalty(joined: list[dict[str, Any]]) -> str | None:
    """
    Whether cold starts are actually what's slow, comparing the median
    client-observed duration of cold and warm invocations.

    Suppressed when queueing dominates, because the comparison then
    inverts: cold starts cluster at the front of a run, where queue
    waits are shortest, so cold looks faster than warm.
    """
    cold = sorted(j["client_ms"] for j in joined if j["init_ms"] is not None)
    warm = sorted(j["client_ms"] for j in joined if j["init_ms"] is None)
    if not cold or not warm:
        return None
    if queueing_dominates(joined):
        return (
            "cold vs warm:   not comparable - most of each duration is queueing "
            "for a connection, not the invocation (raise --concurrency to compare)"
        )
    cold_p50, warm_p50 = percentile(cold, 0.5), percentile(warm, 0.5)
    return (
        f"cold vs warm:   p50 {cold_p50:.0f}ms cold, {warm_p50:.0f}ms warm "
        f"({cold_p50 - warm_p50:+.0f}ms)"
    )

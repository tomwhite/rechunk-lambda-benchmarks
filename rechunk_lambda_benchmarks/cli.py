import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import boto3
import click
import numpy as np
import zarr
from botocore.exceptions import BotoCoreError, ClientError
from map_lambda import DEFAULT_CONCURRENCY, LOCAL_ENDPOINT_URL, Lambdas
from map_lambda import stats as stats_module
from map_lambda import timeline as timeline_module
from zarr.errors import BaseZarrError

from rechunk_lambda_benchmarks import chunks as chunks_module
from rechunk_lambda_benchmarks import cloudwatch, cost
from rechunk_lambda_benchmarks import hashed_integers as hashed_integers_module
from rechunk_lambda_benchmarks import plan as plan_module
from rechunk_lambda_benchmarks import rechunk as rechunk_module
from rechunk_lambda_benchmarks.store import path_to_store, store_url


class ShapeType(click.ParamType):
    """A comma-separated list of integers, e.g. `100,100`."""

    name = "shape"

    def convert(self, value: Any, param: Any, ctx: Any) -> tuple[int, ...]:
        if isinstance(value, tuple):
            return value
        try:
            return tuple(int(part) for part in value.split(","))
        except ValueError:
            self.fail(f"{value!r} is not a comma-separated list of integers", param, ctx)


SHAPE = ShapeType()


def first_line(e: BaseException) -> str:
    return str(e).strip().splitlines()[0]


def open_array(store: str, name: str | None = None) -> Any:
    """Open an existing array, reporting a missing/unreadable store cleanly."""
    try:
        return zarr.open_array(store=zarr.storage.ObjectStore(path_to_store(store)), zarr_format=3)
    except Exception as e:
        raise click.ClickException(
            f"could not open a Zarr array at {name or store}: {first_line(e)}"
        ) from e


def resolve_coordinators(coordinators: int | None) -> "int | Literal['auto']":
    return "auto" if coordinators is None else coordinators


def resolve_endpoint_url(local: bool, endpoint_url: str | None) -> str | None:
    if local and endpoint_url is not None:
        raise click.UsageError("--local and --endpoint-url are mutually exclusive")
    return LOCAL_ENDPOINT_URL if local else endpoint_url


def collect_stats(stats: bool, stats_path: str | None) -> list[Any] | None:
    """The list to collect per-invocation timings into, or None for no timing."""
    return [] if stats or stats_path else None


def batching(
    shape: tuple[int, ...], chunks: tuple[int, ...], per_invocation: int
) -> tuple[int, str, str | None]:
    """`(n_tasks, unit, detail)` describing a run of `per_invocation` chunks at a time."""
    n_chunks = chunks_module.chunk_count(shape, chunks)
    if per_invocation == 1:
        return n_chunks, "chunks", None
    n_tasks = -(-n_chunks // per_invocation)
    return n_tasks, "invocations", f"{n_chunks} chunks, {per_invocation} at a time"


def run(
    build_coro: Callable[[Callable[[int], None]], Any],
    *,
    n_tasks: int,
    unit: str,
    collected: list[Any] | None = None,
    stats_path: str | None = None,
    concurrency: int | None = None,
    detail: str | None = None,
    s3_requests: list[tuple[float, float]] | None = None,
) -> Any:
    """
    Run the coroutine `build_coro` returns, reporting how long it took
    and, with `collected`, the client-side timing summary.

    `build_coro` takes map-lambda's `on_progress`, so this owns the bar's
    lifetime and closes it when a task raises. The bar goes to stderr,
    leaving stdout for the summaries.

    Failed tasks, unreachable or forbidden Lambdas and stores in the
    wrong state surface as messages, not tracebacks; a failed task's
    carries the request id to find in CloudWatch.

    `s3_requests` is the (GETs, PUTs) per task of each `for_each` call,
    stamped on the records for `cloudwatch-report` to price.
    """
    what = f"{n_tasks} {unit}" + (f" ({detail})" if detail else "")
    start = time.monotonic()
    try:
        with click.progressbar(length=n_tasks, label=f"dispatching {what}", file=sys.stderr) as bar:
            result = asyncio.run(build_coro(bar.update))
    except (RuntimeError, BotoCoreError, ClientError, BaseZarrError) as e:
        raise click.ClickException(first_line(e)) from e
    elapsed = time.monotonic() - start
    click.echo(f"done: {what} in {elapsed:.2f}s ({n_tasks / elapsed:.1f}/s)")
    if collected is not None:
        click.echo(stats_module.summarize_invocations(collected, elapsed, concurrency))
        if s3_requests is not None:
            for record in collected:
                record["s3_gets"], record["s3_puts"] = s3_requests[record["call"]]
        if stats_path is not None:
            stats_module.write_jsonl(collected, stats_path)
            click.echo(f"wrote {len(collected)} invocation records to {stats_path}")
    return result


endpoint_url_option = click.option(
    "--endpoint-url",
    default=None,
    help=("Lambda endpoint URL. Omit to use your normal AWS credentials and region."),
)

local_option = click.option(
    "--local",
    "local",
    is_flag=True,
    help=f"Shorthand for --endpoint-url {LOCAL_ENDPOINT_URL} (cargo lambda watch).",
)

coordinators_option = click.option(
    "--coordinators",
    type=click.IntRange(min=1),
    default=None,
    help=("Dispatch through this many coordinator lambdas, splitting --concurrency between them."),
)

concurrency_option = click.option(
    "--concurrency",
    type=int,
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Maximum invocations on the wire at once.",
)

retries_option = click.option(
    "--retries",
    default=2,
    show_default=True,
    help=("Retries per task after a transport failure. A task that raised is not retried."),
)

stats_option = click.option(
    "--stats",
    is_flag=True,
    help="Time each invocation and print a client-side summary.",
)

stats_path_option = click.option(
    "--stats-file",
    "stats_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help=(
        "Also write the per-invocation timings, with their req_ids, as "
        "JSON lines to this path (implies --stats)."
    ),
)

chunks_per_invocation_option = click.option(
    "--chunks-per-invocation",
    type=click.IntRange(min=1),
    default=1,
    help=("Chunks to send in one invocation."),
)


@click.group()
@click.version_option(package_name="rechunke-lambda-benchmarks")
def cli() -> None:
    """A CLI for rechunker lambda benchmarks."""


@cli.command("hashed-integers")
@click.argument("store")
@click.option("--shape", type=SHAPE, required=True, help="Array shape, e.g. 100,100.")
@click.option("--chunks", type=SHAPE, required=True, help="Chunk shape, e.g. 10,10.")
@endpoint_url_option
@local_option
@coordinators_option
@concurrency_option
@retries_option
@stats_option
@stats_path_option
@chunks_per_invocation_option
def hashed_integers_command(
    store: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    endpoint_url: str | None,
    local: bool,
    coordinators: int | None,
    concurrency: int,
    retries: int,
    stats: bool,
    stats_path: str | None,
    chunks_per_invocation: int,
) -> None:
    """
    Create a uint32 array of hashed values at STORE.
    """
    endpoint_url = resolve_endpoint_url(local, endpoint_url)
    # resolved here too, so the message below names what was written
    store = store_url(store, mkdir=True)
    collected = collect_stats(stats, stats_path)
    n_tasks, unit, detail = batching(shape, chunks, chunks_per_invocation)
    coordinator_stats: list[dict[str, Any]] = []

    async def go(on_done: Callable[[int], None]) -> Any:
        async with Lambdas(
            endpoint_url=endpoint_url,
            coordinators=resolve_coordinators(coordinators),
            concurrency=concurrency,
            retries=retries,
            on_progress=on_done,
            coordinator_stats=coordinator_stats,
            invocation_stats=collected,
        ) as lambdas:
            return await hashed_integers_module.hashed_integers(
                lambdas, store, shape, chunks=chunks, chunks_per_invocation=chunks_per_invocation
            )

    array = run(
        go,
        n_tasks=n_tasks,
        unit=unit,
        collected=collected,
        stats_path=stats_path,
        concurrency=concurrency,
        detail=detail,
        s3_requests=[(0, chunks_module.chunk_count(shape, chunks) / n_tasks)],
    )
    click.echo(f"created {store} {array.shape} chunked {array.chunks} ({array.dtype})")
    click.echo(stats_module.summarize_coordinators(coordinator_stats))


@cli.command("validate-hashed-integers")
@click.argument("store")
@endpoint_url_option
@local_option
@coordinators_option
@concurrency_option
@retries_option
@stats_option
@stats_path_option
@chunks_per_invocation_option
def validate_hashed_integers_command(
    store: str,
    endpoint_url: str | None,
    local: bool,
    coordinators: int | None,
    concurrency: int,
    retries: int,
    stats: bool,
    stats_path: str | None,
    chunks_per_invocation: int,
) -> None:
    """
    Check every chunk of the array at STORE against the hash `hashed-integers`
    writes it with.
    """
    endpoint_url = resolve_endpoint_url(local, endpoint_url)
    store = store_url(store)
    collected = collect_stats(stats, stats_path)
    # opened here to size the bar, and so a missing store fails at the prompt
    array = open_array(store)
    n_tasks, unit, detail = batching(array.shape, array.chunks, chunks_per_invocation)
    coordinator_stats: list[dict[str, Any]] = []

    async def go(on_done: Callable[[int], None]) -> Any:
        async with Lambdas(
            endpoint_url=endpoint_url,
            coordinators=resolve_coordinators(coordinators),
            concurrency=concurrency,
            retries=retries,
            on_progress=on_done,
            coordinator_stats=coordinator_stats,
            invocation_stats=collected,
        ) as lambdas:
            await hashed_integers_module.validate_hashed_integers(
                lambdas, store, chunks_per_invocation=chunks_per_invocation
            )

    run(
        go,
        n_tasks=n_tasks,
        unit=unit,
        collected=collected,
        stats_path=stats_path,
        concurrency=concurrency,
        detail=detail,
        s3_requests=[(chunks_module.chunk_count(array.shape, array.chunks) / n_tasks, 0)],
    )
    click.echo(f"successfully validated {store}")
    click.echo(stats_module.summarize_coordinators(coordinator_stats))


@cli.command("rechunk")
@click.argument("source_store")
@click.argument("target_store")
@click.option("--target-chunks", type=SHAPE, required=True, help="Chunk shape of the new array.")
@click.option(
    "--memory",
    type=int,
    default=1024,
    show_default=True,
    help="Memory (MB) the rechunk function is deployed with.",
)
@click.option(
    "--min-mem-fraction",
    type=int,
    default=20,
    show_default=True,
    help="The min memory parameter as a fraction of memory.",
)
@click.option(
    "--shard-intermediates/--no-shard-intermediates",
    "shard_intermediates",
    default=False,
    show_default=True,
    help=(
        "Pack each intermediate array's chunks into shards of one copy chunk, so a task "
        "writes a single object instead of one per chunk. The target is never sharded. "
    ),
)
@click.option(
    "--keep-intermediates",
    is_flag=True,
    help=(
        "Leave intermediate arrays in place. They are otherwise deleted as soon as nothing "
        "reads them, which on a large rechunk is most of the bill."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace TARGET_STORE if it already holds an array, e.g. to re-run a failed rechunk.",
)
@endpoint_url_option
@local_option
@coordinators_option
@concurrency_option
@retries_option
@stats_option
@stats_path_option
def rechunk_command(
    source_store: str,
    target_store: str,
    target_chunks: tuple[int, ...],
    memory: int,
    min_mem_fraction: int,
    shard_intermediates: bool,
    keep_intermediates: bool,
    overwrite: bool,
    coordinators: int | None,
    endpoint_url: str | None,
    local: bool,
    retries: int,
    concurrency: int,
    stats: bool,
    stats_path: str | None,
) -> None:
    """
    Copy SOURCE_STORE to TARGET_STORE, rechunked to --target-chunks.

    Copy chunks are chosen to fit --memory unless --copy-chunks says
    otherwise. A rechunk that reshapes across axes needs several passes
    through intermediate arrays, written under the target and deleted as
    soon as nothing reads them (--keep-scratch keeps them);
    `rechunker-lambda plan` shows what will happen without running it.
    """
    endpoint_url = resolve_endpoint_url(local, endpoint_url)
    source_store = store_url(source_store)
    target_store = store_url(target_store, mkdir=True)
    collected = collect_stats(stats, stats_path)

    source = open_array(source_store)
    memory_bytes = memory * 1_000_000

    memory_bytes = memory * 1_000_000
    budget = plan_module.chunk_budget(memory_bytes, sharded=shard_intermediates)
    min_mem = budget // min_mem_fraction
    ops = plan_module.plan(
        source.shape,
        source.chunks,
        target_chunks,
        source.dtype.itemsize,
        budget,
        min_mem=min_mem,
        shard_intermediates=shard_intermediates,
    )
    coordinator_stats: list[dict[str, Any]] = []

    async def go(on_done: Callable[[int], None]) -> Any:
        async with Lambdas(
            endpoint_url=endpoint_url,
            coordinators=resolve_coordinators(coordinators),
            concurrency=concurrency,
            retries=retries,
            on_progress=on_done,
            coordinator_stats=coordinator_stats,
            invocation_stats=collected,
        ) as lambdas:
            return await rechunk_module.rechunk(
                lambdas,
                source_store,
                target_store,
                target_chunks=target_chunks,
                ops=ops,
                delete_intermediates=not keep_intermediates,
                overwrite=overwrite,
                on_warning=lambda message: click.echo(f"warning: {message}", err=True),
            )

    try:
        array = run(
            go,
            n_tasks=sum(op.n_tasks for op in ops),
            unit="copy chunks",
            collected=collected,
            stats_path=stats_path,
            concurrency=concurrency,
            s3_requests=[(op.fan_in, op.objects_written / op.n_tasks) for op in ops],
        )
    except ValueError as e:
        # e.g. copy chunks not a multiple of target chunks
        raise click.UsageError(str(e)) from e
    click.echo(
        f"rechunked {source_store} {source.chunks} -> {target_store} {array.chunks} "
        f"(shape {array.shape}, {array.dtype})"
    )
    click.echo(stats_module.summarize_coordinators(coordinator_stats))


@cli.command("plan")
@click.argument("store", required=False)
@click.option("--target-chunks", type=SHAPE, required=True, help="Chunk shape to rechunk to.")
@click.option("--shape", type=SHAPE, default=None, help="Array shape, if STORE is not given.")
@click.option(
    "--source-chunks", type=SHAPE, default=None, help="Current chunk shape, if STORE is not given."
)
@click.option(
    "--dtype", default="uint32", show_default=True, help="Array dtype, if STORE is not given."
)
@click.option(
    "--memory",
    type=int,
    default=1024,
    show_default=True,
    help="Memory (MB) the rechunk function is deployed with.",
)
@click.option(
    "--min-mem-fraction",
    type=int,
    default=20,
    show_default=True,
    help="The min memory parameter as a fraction of memory.",
)
@click.option(
    "--shard-intermediates/--no-shard-intermediates",
    "shard_intermediates",
    default=False,
    show_default=True,
    help=(
        "Pack each intermediate array's chunks into shards of one copy chunk, so a task "
        "writes a single object instead of one per chunk. The target is never sharded. "
    ),
)
def plan_command(
    store: str | None,
    target_chunks: tuple[int, ...],
    shape: tuple[int, ...] | None,
    source_chunks: tuple[int, ...] | None,
    dtype: str,
    memory: int,
    min_mem_fraction: int,
    shard_intermediates: bool = False,
) -> None:
    """
    Show how a rechunk would be carried out without running it.
    """
    if store is not None:
        array = open_array(store)
        shape, source_chunks, itemsize = array.shape, array.chunks, array.dtype.itemsize
    else:
        if shape is None or source_chunks is None:
            raise click.UsageError(
                "give STORE, or --shape and --source-chunks to plan without an array"
            )
        itemsize = np.dtype(dtype).itemsize

    memory_bytes = memory * 1_000_000
    budget = plan_module.chunk_budget(memory_bytes, sharded=shard_intermediates)
    min_mem = budget // min_mem_fraction
    ops = plan_module.plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize,
        budget,
        min_mem=min_mem,
        shard_intermediates=shard_intermediates,
    )
    click.echo(plan_module.summarize(ops, memory_bytes))


@cli.command("cloudwatch-report")
@click.argument("stats_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--function-name",
    required=True,
    help="Function the run invoked; its log group is /aws/lambda/<function-name>.",
)
@click.option(
    "--timeout",
    type=float,
    default=cloudwatch.DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for the Logs Insights query.",
)
@click.option("--slowest", type=int, default=5, show_default=True, help="Also list the N slowest.")
def cloudwatch_report_command(
    stats_file: str,
    function_name: str,
    timeout: float,
    slowest: int,
) -> None:
    """
    Split a run's per-invocation time into Lambda init, Lambda execution
    and everything else, by joining STATS_FILE (written by --stats-file)
    against the REPORT lines in CloudWatch Logs.

    Answers the question the client-side summary can't: whether a slow
    run was slow inside the function, in cold starts, or on the way
    there. Needs `logs:StartQuery`/`logs:GetQueryResults` and reads logs
    the functions already write.
    """
    log_group = cloudwatch.log_group_name(function_name)

    stats = stats_module.read_jsonl(stats_file)
    if not stats:
        raise click.ClickException(f"{stats_file} has no invocation records")

    windows = cloudwatch.query_windows(stats)
    span = windows[-1][1] - windows[0][0]
    queries = f" in {len(windows)} queries" if len(windows) > 1 else ""
    click.echo(f"querying {log_group} over {span}s for {len(stats)} request ids{queries}...")
    try:
        reports, truncated = cloudwatch.query_all_reports(
            boto3.client("logs"), log_group, windows, timeout=timeout
        )
    except (RuntimeError, BotoCoreError, ClientError) as e:
        raise click.ClickException(first_line(e)) from e

    if truncated:
        # every figure below, the cost especially, would silently be a fraction of the run
        click.echo(
            f"WARNING: {truncated} of {len(windows)} queries hit Logs Insights' "
            f"{cloudwatch.REPORT_LIMIT}-row limit, so some invocations are missing and "
            f"every total below understates the run. The window holds more log lines "
            f"than this run produced - narrow it by reporting on a shorter run.",
            err=True,
        )

    joined = cloudwatch.join(stats, reports)
    click.echo(cloudwatch.summarize(joined, cloudwatch.missing(stats, reports)))
    penalty = cloudwatch.cold_start_penalty(joined)
    if penalty is not None:
        click.echo(penalty)

    spend = cost.actual(joined)
    if spend is not None:
        click.echo(cost.summarize_actual(spend))

    if joined and slowest > 0:
        click.echo(cloudwatch.slowest(joined, slowest))


@cli.command("timeline")
@click.argument("stats_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Where to write the chart. [default: STATS_FILE with a .svg suffix]",
)
@click.option(
    "--function-name",
    default=None,
    help=(
        "Join CloudWatch REPORT lines from this function's log group, to draw when each "
        "invocation was actually running rather than only when it was dispatched."
    ),
)
@click.option(
    "--max-points",
    type=int,
    default=timeline_module.DEFAULT_MAX_POINTS,
    show_default=True,
    help="Thin to this many tasks by taking every nth. 0 plots every task.",
)
@click.option("--title", default=None, help="Title for the chart. [default: STATS_FILE]")
@click.option(
    "--timeout",
    type=float,
    default=cloudwatch.DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for each Logs Insights query.",
)
def timeline_command(
    stats_file: str,
    output_path: str | None,
    function_name: str | None,
    max_points: int,
    title: str | None,
    timeout: float,
) -> None:
    """
    Plot a run's invocations over time, from STATS_FILE (written by
    --stats-file).

    One row per invocation: when it was created, when it went on the
    wire and when it returned, plus - with --function-name - a band for
    when the function was actually running. The gap in front of that
    band is the queueing that `cloudwatch-report` can only summarise as
    `unaccounted`, and a multi-pass rechunk's copy operations separate
    into labelled blocks.

    Needs the diagnostics extra: `uv sync --extra diagnostics`.
    """
    stats = stats_module.read_jsonl(stats_file)
    if not stats:
        raise click.ClickException(f"{stats_file} has no invocation records")

    records: list[Any] = stats
    if function_name is not None:
        log_group = cloudwatch.log_group_name(function_name)
        windows = cloudwatch.query_windows(stats)
        queries = f" in {len(windows)} queries" if len(windows) > 1 else ""
        click.echo(f"querying {log_group} for {len(stats)} request ids{queries}...")
        try:
            reports, truncated = cloudwatch.query_all_reports(
                boto3.client("logs"), log_group, windows, timeout=timeout
            )
        except (RuntimeError, BotoCoreError, ClientError) as e:
            raise click.ClickException(first_line(e)) from e
        records = cloudwatch.join(stats, reports)
        missing = len(stats) - len(records)
        if truncated or missing:
            click.echo(
                f"note: {missing} of {len(stats)} invocations have no REPORT line and are "
                f"drawn without an execution band",
                err=True,
            )

    try:
        figure = timeline_module.generate_timeline(
            records, title=title or Path(stats_file).name, max_points=max_points
        )
    except ImportError as e:
        raise click.ClickException(
            f"the timeline command needs matplotlib: {first_line(e)} "
            "- install it with `uv sync --extra diagnostics`"
        ) from e

    output = output_path or str(Path(stats_file).with_suffix(".svg"))
    figure.savefig(output)
    click.echo(f"wrote {output} ({len(records)} invocations)")


if __name__ == "__main__":
    cli()

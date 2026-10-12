from dataclasses import dataclass
from math import ceil, prod

from rechunk_lambda_benchmarks.vendor.cubed.rechunk import multistage_regular_rechunking_plan

# peak = baseline + multiplier x chunk bytes, fitted to CloudWatch REPORT
# lines. The multiplier is rounded up so the prediction sits above every
# measured point: under-predicting costs the run. Fitted below ~400 MB,
# above which it is conservative.
PEAK_BASELINE_BYTES = 31_000_000
PEAK_MULTIPLIER = 2.0

# Added to the peak of a task that writes shards, in shards: zarrs'
# `ShardingCodec::encode_bounded` preallocates the whole encoded shard
# alongside the decoded data. Measured, and rounded up. Applied to
# `shard_bytes`, not the copy chunk: an overhanging shard is materialised whole.
SHARD_ENCODE_COPIES = 0.9

# How much larger than its copy chunk a shard can be (real plans: 1.11,
# 1.23), for `chunk_budget`, which runs before any shard shape exists. Not
# dead weight where the two are equal: without it a run can have no
# candidate plan inside the safety margin.
SHARD_OVERHANG = 1.25

# Fraction of configured memory to plan against: the peak model is
# extrapolated well past its measurements, and an OOM costs a whole run.
DEFAULT_SAFETY = 0.8

# cubed's rule for min_mem, the intermediate size at which rechunker stops adding stages.
MIN_MEM_FRACTION = 20

# The divisors of `max_mem` that `candidate_plans` tries: 1 to 512 in 12%
# steps, since a real rechunk's distinct plans can sit within 15% of each other.
MIN_MEM_DIVISORS = tuple(round(1.12**k, 3) for k in range(0, 56))


@dataclass(frozen=True)
class CopyOp:
    """
    One copy operation: read `copy_chunks` at a time from an array
    chunked `source_chunks`, writing an array chunked `target_chunks`.

    `sharded` packs the written array's chunks into `shards`. It changes
    no chunk shape or unit of work, only how many objects are written.
    """

    source_chunks: tuple[int, ...]
    copy_chunks: tuple[int, ...]
    target_chunks: tuple[int, ...]
    shape: tuple[int, ...]
    itemsize: int
    sharded: bool = False

    @property
    def n_tasks(self) -> int:
        """Invocations this operation dispatches."""
        return prod(ceil(n / c) for n, c in zip(self.shape, self.copy_chunks, strict=True))

    @property
    def fan_in(self) -> int:
        """Source chunks read per task."""
        return prod(ceil(c / s) for c, s in zip(self.copy_chunks, self.source_chunks, strict=True))

    @property
    def fan_out(self) -> int:
        """Target chunks written per task."""
        return prod(ceil(c / t) for c, t in zip(self.copy_chunks, self.target_chunks, strict=True))

    @property
    def copy_chunk_bytes(self) -> int:
        return self.itemsize * prod(self.copy_chunks)

    @property
    def peak_memory_bytes(self) -> int:
        """
        What a task is predicted to peak at. A sharded task also holds
        the whole encoded shard, and its decoded side is padded to the
        shard, so both terms are keyed on it. Reads low on sharded
        operations even so, which `DEFAULT_SAFETY` covers.
        """
        held = max(self.copy_chunk_bytes, self.shard_bytes)
        peak = PEAK_BASELINE_BYTES + PEAK_MULTIPLIER * held
        if self.sharded:
            peak += SHARD_ENCODE_COPIES * self.shard_bytes
        return int(peak)

    @property
    def shard_bytes(self) -> int:
        """
        One shard, at its full shape - which is what is allocated, even
        where the shard runs off the end of the array. Zero if the
        operation is not sharded.
        """
        shards = self.shards
        return 0 if shards is None else self.itemsize * prod(shards)

    @property
    def bytes_read(self) -> int:
        """
        Bytes fetched from the source array, which is not the same as
        the bytes needed: a Zarr chunk can only be read whole, so a copy
        chunk that overlaps a source chunk pays for all of it.

        Approximate at array edges, where the last chunk along an axis
        is short.
        """
        return self.n_tasks * self.fan_in * self.itemsize * prod(self.source_chunks)

    @property
    def chunks_read(self) -> int:
        """
        Source chunks fetched across the whole operation, counting each
        overlapped chunk once per task that reads it. Reads cost a flat
        ~8ms each, so this, not `bytes_read`, drives time and GET charges.
        """
        return self.n_tasks * self.fan_in

    @property
    def chunks_written(self) -> int:
        """Target chunks written, i.e. the chunks of the output array."""
        return prod(ceil(n / t) for n, t in zip(self.shape, self.target_chunks, strict=True))

    @property
    def shards(self) -> tuple[int, ...] | None:
        """
        The shard shape for the array this operation writes, or None if
        it is not sharded.

        The copy chunk wherever it can be, so a task covers whole shards
        and no two tasks write one object. On an axis the copy chunk
        spans without being a multiple of the target chunk (721 against
        240), a shard must be whole chunks instead; the single task
        along that axis covers any shard, so the whole axis is one.
        """
        if not self.sharded:
            return None
        axes = zip(self.shape, self.copy_chunks, self.target_chunks, strict=True)
        return tuple(
            copy if copy % target == 0 else target * ceil(n / target) for n, copy, target in axes
        )

    @property
    def objects_written(self) -> int:
        """Objects PUT to the store: `chunks_written`, unless sharded."""
        shards = self.shards
        if shards is None:
            return self.chunks_written
        return prod(ceil(n / s) for n, s in zip(self.shape, shards, strict=True))

    @property
    def target_aligned(self) -> bool:
        """
        Whether copy chunks align with target chunk boundaries.
        """
        return all(
            (c == n) or (c % t == 0)
            for n, c, t in zip(self.shape, self.copy_chunks, self.target_chunks, strict=True)
        )


def chunk_budget(
    lambda_memory_bytes: float, safety: float = DEFAULT_SAFETY, sharded: bool = False
) -> int:
    """
    The largest copy chunk that fits in a function of this memory size,
    inverting the peak-memory model: a 1769 MB function at the default
    safety gets ~660 MiB.

    `sharded` budgets for a task that writes shards, roughly a third of
    the unsharded budget.

    A bigger budget is not automatically better: fewer tasks than the
    coordinator's concurrency is both dearer and slower.
    """
    usable = lambda_memory_bytes * safety - PEAK_BASELINE_BYTES
    if usable <= 0:
        raise ValueError(
            f"a {lambda_memory_bytes / 1e6:.0f} MB function has no room for data after "
            f"the {PEAK_BASELINE_BYTES / 1e6:.0f} MB runtime baseline"
        )
    multiplier = PEAK_MULTIPLIER
    if sharded:
        multiplier = (PEAK_MULTIPLIER + SHARD_ENCODE_COPIES) * SHARD_OVERHANG
    return int(usable / multiplier)


def plan(
    shape: tuple[int, ...],
    source_chunks: tuple[int, ...],
    target_chunks: tuple[int, ...],
    itemsize: int,
    max_mem: int,
    min_mem: int | None = None,
    shard_intermediates: bool = False,
) -> list[CopyOp]:
    if min_mem is None:
        min_mem = min(max_mem // MIN_MEM_FRACTION, itemsize * prod(shape))

    stages = multistage_regular_rechunking_plan(
        shape=shape,
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        itemsize=itemsize,
        min_mem=min_mem,
        max_mem=max_mem,
    )

    ops = []
    chunks_in = tuple(source_chunks)
    # Adapted from the tail of cubed.core.ops._rechunk_plan: a stage
    # whose read and write chunks differ is two copy operations, a split
    # then a consolidate, with an intermediate array between them.
    for i, (read_chunks, int_chunks, write_chunks) in enumerate(stages):
        last_stage = i == len(stages) - 1
        chunks_out = tuple(target_chunks) if last_stage else write_chunks
        if read_chunks == write_chunks:
            pairs = [(read_chunks, chunks_out)]
        elif last_stage:
            pairs = [(read_chunks, int_chunks), (write_chunks, chunks_out)]
        else:
            pairs = [(read_chunks, int_chunks)]
        for j, (copy_chunks, op_target_chunks) in enumerate(pairs):
            last = last_stage and (j == len(pairs) - 1)
            ops.append(
                CopyOp(
                    source_chunks=chunks_in,
                    copy_chunks=tuple(copy_chunks),
                    target_chunks=tuple(op_target_chunks),
                    shape=tuple(shape),
                    itemsize=itemsize,
                    sharded=shard_intermediates and not last,
                )
            )
            chunks_in = tuple(op_target_chunks)

    unaligned = [k for k, op in enumerate(ops) if not op.target_aligned]
    if unaligned:
        raise AssertionError(
            f"copy operations {unaligned} are not aligned to their target chunks, which "
            f"the regular planner should never produce - this rechunk would need "
            f"irregular intermediate arrays, which are not yet supported"
        )
    return ops


def summarize(ops: list[CopyOp], memory_bytes: int | None = None) -> str:
    """A human-readable plan."""
    if not ops:
        return "nothing to do: the array is already chunked that way"

    def mb(n: float) -> str:
        return f"{n / 1e6:.0f} MB"

    array_bytes = ops[0].itemsize * prod(ops[0].shape)
    lines = [
        f"{len(ops)} copy operation(s), {sum(op.n_tasks for op in ops):,} tasks total",
        "",
        f"{'op':>3}  {'copy chunks':<24} {'-> chunks':<24} {'tasks':>7} "
        f"{'in':>5} {'out':>6} {'copy':>9} {'peak':>9}",
    ]
    for k, op in enumerate(ops):
        lines.append(
            f"{k:>3}  {op.copy_chunks!s:<24} {op.target_chunks!s:<24} "
            f"{op.n_tasks:>7,} {op.fan_in:>5} {op.fan_out:>6} "
            f"{mb(op.copy_chunk_bytes):>9} {mb(op.peak_memory_bytes):>9}"
        )

    if sharded := [(k, op) for k, op in enumerate(ops) if op.shards is not None]:
        lines += [
            "",
            f"{'op':>3}  {'shards':<24} {'objects':>10} {'chunks':>10}",
        ]
        for k, op in sharded:
            lines.append(
                f"{k:>3}  {op.shards!s:<24} {op.objects_written:>10,} {op.chunks_written:>10,}"
            )

    read = sum(op.bytes_read for op in ops)
    lines += [
        "",
        f"array:          {mb(array_bytes)}",
        f"read:           {mb(read)} across all operations ({read / array_bytes:.1f}x the array)",
    ]
    written = sum(op.objects_written for op in ops)
    if written != sum(op.chunks_written for op in ops):
        lines.append(
            f"written:        {written:,} objects, against "
            f"{sum(op.chunks_written for op in ops):,} chunks (sharded intermediates)"
        )
    if memory_bytes is not None:
        worst = max(op.peak_memory_bytes for op in ops)
        lines.append(
            f"peak memory:    {mb(worst)} predicted, of {mb(memory_bytes)} "
            f"configured ({100 * worst / memory_bytes:.0f}%)"
        )
    return "\n".join(lines)

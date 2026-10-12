from collections.abc import Callable
from pathlib import Path

import zarr
from map_lambda import Lambdas

from rechunk_lambda_benchmarks.chunks import ChunkCoord, chunk_coords
from rechunk_lambda_benchmarks.plan import CopyOp, chunk_budget, plan
from rechunk_lambda_benchmarks.spec import array_spec
from rechunk_lambda_benchmarks.store import path_to_store, store_url

COPY_FUNCTION = "copy"
DELETE_INTERMEDIATE_FUNCTION = "delete-intermediate"

INTERMEDIATE_PREFIX = "_intermediate"

MB = 1_000_000
DEFAULT_MEMORY_MB = 1024  # same as in rust/copy/Cargo.toml


async def rechunk(
    lambdas: Lambdas,
    source_store: str | Path,
    target_store: str | Path,
    *,
    target_chunks: ChunkCoord,
    ops: list[CopyOp] | None = None,
    memory_mb: int | None = None,
    shard_intermediates: bool = False,
    overwrite: bool = False,
    delete_intermediates: bool = True,
    on_warning: Callable[[str], None] | None = None,
) -> zarr.Array:
    """
    Rechunk the array at `source_store` into a new array at
    `target_store`, one task per copy chunk - a unit-of-work grid
    independent of both arrays' chunking.

    A rechunk across axes may take several copy operations, chained
    through intermediate arrays under `scratch_store`. Each intermediate
    is deleted once the operation reading it has finished, unless
    `delete_scratch` is False; a failed deletion goes to `on_warning`, or
    is reported by a caller's own `delete_store`.

    `overwrite` replaces an existing target, as a failed run leaves one.
    Intermediates are always overwritten.

    **Give `ops` if you have already planned**, not the inputs to plan
    them: two plans from the same inputs need not agree, and the failure
    prints one plan and runs another.

    Raises if the source fails `array_spec`'s validation (sharded, say,
    or "." separated): the lambdas cannot reproduce it faithfully.
    """
    source = store_url(source_store)
    target = store_url(target_store, mkdir=True)
    source_array = zarr.open_array(
        store=zarr.storage.ObjectStore(path_to_store(source)), zarr_format=3
    )
    if ops is None:
        max_mem = chunk_budget((memory_mb or DEFAULT_MEMORY_MB) * MB, sharded=shard_intermediates)
        ops = plan(
            source_array.shape,
            source_array.chunks,
            target_chunks,
            source_array.dtype.itemsize,
            max_mem=max_mem,
            shard_intermediates=shard_intermediates,
        )

    def create(
        store: str, chunks: ChunkCoord, overwrite: bool, shards: ChunkCoord | None = None
    ) -> zarr.Array:
        return zarr.create_array(
            store=zarr.storage.ObjectStore(path_to_store(store, mkdir=True)),
            shape=source_array.shape,
            chunks=chunks,
            # a whole number of copy chunks, so no two tasks write one shard
            shards=tuple(shards) if shards is not None else None,
            dtype=source_array.dtype,
            overwrite=overwrite,
        )

    # create arrays
    target_array = create(target, ops[-1].target_chunks, overwrite)
    intermediates = [
        store_url(f"{target}/{INTERMEDIATE_PREFIX}/stage{k}", mkdir=True)
        for k in range(len(ops) - 1)
    ]
    stores: list[str] = [source, *intermediates, target]
    arrays = [
        source_array,
        *(
            create(store, op.target_chunks, True, op.shards)
            for store, op in zip(intermediates, ops, strict=False)
        ),
        target_array,
    ]

    for k, op in enumerate(ops):
        await lambdas.for_each(
            COPY_FUNCTION,
            [list(coord) for coord in chunk_coords(arrays[k].shape, op.copy_chunks)],
            input_key="copy_chunk_coord",
            constant_args={
                "source_store": stores[k],
                "source_array_path": "/",
                "source_array_spec": array_spec(arrays[k]),
                "target_store": stores[k + 1],
                "target_array_path": "/",
                "target_array_spec": array_spec(arrays[k + 1]),
                "copy_chunk_shape": list(op.copy_chunks),
            },
        )

        # delete previous intermediate store
        if k >= 1 and delete_intermediates:
            try:
                await lambdas.invoke(DELETE_INTERMEDIATE_FUNCTION, store=intermediates[k - 1])
            except Exception as e:
                if on_warning is not None:
                    on_warning(f"could not request deletion of {intermediates[k - 1]}: {e}")
                raise

    return target_array

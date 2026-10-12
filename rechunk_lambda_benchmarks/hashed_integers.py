"""
The benchmark data: an array of deterministic, incompressible values,
and a check that it survived a rechunk.
"""

from pathlib import Path

import zarr
from map_lambda import Lambdas

from rechunk_lambda_benchmarks.chunks import chunk_batches
from rechunk_lambda_benchmarks.spec import array_spec
from rechunk_lambda_benchmarks.store import path_to_store, store_url

HASHED_INTEGERS_FUNCTION = "hashed-integers"
VALIDATE_FUNCTION = "validate-hashed-integers"

# Keep in sync with the dtype rust/hashed-integers writes.
HASHED_INTEGERS_DTYPE = "uint32"


async def hashed_integers(
    lambdas: Lambdas,
    store: str | Path,
    shape: tuple[int, ...],
    *,
    chunks: tuple[int, ...],
    chunks_per_invocation: int = 1,
) -> zarr.Array:
    """
    Create a uint32 array of hashed values at `store`, and return it.

    Each element's value depends only on its position in the array,
    never on how the array happens to be chunked - so the data is
    incompressible and survives a rechunk unchanged. Check it with
    `validate_hashed_integers`.

    `chunks_per_invocation` is how many chunks one invocation writes.
    """
    url = store_url(store, mkdir=True)
    array = zarr.create_array(
        store=zarr.storage.ObjectStore(path_to_store(url)),
        shape=shape,
        chunks=chunks,
        dtype=HASHED_INTEGERS_DTYPE,
    )
    batches = chunk_batches(shape, chunks, chunks_per_invocation)
    await lambdas.for_each(
        HASHED_INTEGERS_FUNCTION,
        [[list(coord) for coord in batch] for batch in batches],
        input_key="chunk_coords",
        # rust/hashed-integers Request, minus chunk_coords
        constant_args={
            "store": url,
            "array_path": "/",
            "array_spec": array_spec(array),
        },
    )
    return array


async def validate_hashed_integers(
    lambdas: Lambdas,
    store: str | Path,
    *,
    chunks_per_invocation: int = 1,
) -> None:
    """
    Check every chunk of the array at `store` against the hash
    `hashed-integers` writes it with.

    Raises `RuntimeError` if any chunk does not match.
    """
    url = store_url(store)
    array = zarr.open_array(store=zarr.storage.ObjectStore(path_to_store(url)), zarr_format=3)
    batches = chunk_batches(array.shape, array.chunks, chunks_per_invocation)
    await lambdas.for_each(
        VALIDATE_FUNCTION,
        [[list(coord) for coord in batch] for batch in batches],
        input_key="chunk_coords",
        # rust/validate-hashed-integers Request, minus chunk_coords
        constant_args={
            "store": url,
            "array_path": "/",
            "array_spec": array_spec(array),
        },
    )

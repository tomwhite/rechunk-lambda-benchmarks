import math
from collections.abc import Iterable
from itertools import batched, product

ChunkCoord = tuple[int, ...]


def chunk_coords(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterable[ChunkCoord]:
    """All chunk grid coordinates for `shape` chunked as `chunks`."""
    return product(*(range(-(-s // c)) for s, c in zip(shape, chunks, strict=True)))


def chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    """
    How many chunks `shape` has when chunked as `chunks`.
    """
    return math.prod(-(-s // c) for s, c in zip(shape, chunks, strict=True))


def chunk_batches(
    shape: tuple[int, ...], chunks: tuple[int, ...], per_invocation: int = 1
) -> Iterable[tuple[ChunkCoord, ...]]:
    """
    All chunk grid coordinates for `shape`, batched into the units of
    work one invocation handles.
    """
    return batched(chunk_coords(shape, chunks), per_invocation)

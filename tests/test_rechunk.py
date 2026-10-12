import numpy as np
import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL, Lambdas

from rechunk_lambda_benchmarks.rechunk import rechunk
from rechunk_lambda_benchmarks.store import path_to_store


def _store_uri(path):
    """
    A file:// URI for `path`, created first: `path_to_store` creates only a bare path's directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _make_source_array(tmp_path, shape, chunks, dtype="int32"):
    """A source array whose elements are their own row-major offsets."""
    store = _store_uri(tmp_path / "source")
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    zarr.create_array(
        store=zarr.storage.ObjectStore(path_to_store(store)),
        chunks=chunks,
        data=data,
    )
    return store, data


@pytest.mark.parametrize(
    ("shape", "source_chunks", "target_chunks"),
    [
        # 2-D
        ((4, 6), (2, 3), (2, 2)),
        # shape not a multiple of any of the three chunkings - edge chunks
        ((5, 5), (2, 2), (3, 3)),
        # 1-D
        ((10,), (3,), (2,)),
    ],
)
@pytest.mark.lambda_integration
@pytest.mark.asyncio
async def test_rechunk(tmp_path, shape, source_chunks, target_chunks):
    source_store, expected = _make_source_array(tmp_path, shape, source_chunks)
    target_store = _store_uri(tmp_path / "target")

    async with Lambdas(endpoint_url=LOCAL_ENDPOINT_URL) as lambdas:
        target_array = await rechunk(
            lambdas, source_store, target_store, target_chunks=target_chunks
        )

    assert target_array.shape == shape
    assert target_array.chunks == target_chunks
    assert target_array.dtype == "int32"
    np.testing.assert_array_equal(np.asarray(target_array[:]), expected)

    # re-opening from the store should see the same thing
    reopened = zarr.open_array(store=zarr.storage.ObjectStore(path_to_store(target_store)))
    np.testing.assert_array_equal(np.asarray(reopened[:]), expected)

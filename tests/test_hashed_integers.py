import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL, Lambdas

from rechunk_lambda_benchmarks.hashed_integers import hashed_integers, validate_hashed_integers


@pytest.mark.lambda_integration
@pytest.mark.asyncio
async def test_hashed_integers_and_validate(tmp_path):
    store = str(tmp_path / "ints")
    shape = (4, 6)
    chunks = (2, 3)

    async with Lambdas(endpoint_url=LOCAL_ENDPOINT_URL) as lambdas:
        await hashed_integers(lambdas, store, shape, chunks=chunks)
        await validate_hashed_integers(lambdas, store)


@pytest.mark.lambda_integration
@pytest.mark.asyncio
async def test_hashed_integers_and_validate_fails(tmp_path):
    store = str(tmp_path / "ints")
    shape = (4, 6)
    chunks = (2, 3)

    async with Lambdas(endpoint_url=LOCAL_ENDPOINT_URL) as lambdas:
        await hashed_integers(lambdas, store, shape, chunks=chunks)

        # overwrite a chunk
        zarr.open_array(store=store).blocks[0, 0] = 0

        with pytest.raises(RuntimeError, match="mismatch at local index 0"):
            await validate_hashed_integers(lambdas, store)

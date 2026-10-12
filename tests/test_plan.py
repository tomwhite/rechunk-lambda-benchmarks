from rechunk_lambda_benchmarks.plan import CopyOp, chunk_budget, plan

MB = 1_000_000

ERA5_TINY = ((2480, 721, 1440), (31, 721, 1440), (2480, 10, 10))
ERA5_MEDIUM = ((25854, 721, 1440), (31, 721, 1440), (25854, 10, 10))
ERA5_LARGE = ((258540, 721, 1440), (31, 721, 1440), (258540, 10, 10))
ERA5 = ((350640, 721, 1440), (31, 721, 1440), (350640, 10, 10))
WORKLOADS = [ERA5_TINY, ERA5_MEDIUM, ERA5_LARGE, ERA5]


def test_era5_large_plan():
    memory_mb = 1280
    min_mem_fraction = 50  # chosen arbitrarily
    shape, source_chunks, target_chunks = ERA5_LARGE
    max_mem = chunk_budget(memory_mb * MB)
    min_mem = max_mem // min_mem_fraction

    ops = plan(shape, source_chunks, target_chunks, itemsize=4, max_mem=max_mem, min_mem=min_mem)

    assert len(ops) == 4

    def copy_op(sc, cc, tc):
        return CopyOp(sc, cc, tc, shape=shape, itemsize=4)

    assert ops[0] == copy_op(source_chunks, (93, 721, 1440), (93, 160, 360))
    assert ops[1] == copy_op((93, 160, 360), (1302, 160, 360), (1302, 40, 120))
    assert ops[2] == copy_op((1302, 40, 120), (18228, 40, 120), (18228, 10, 40))
    assert ops[3] == copy_op((18228, 10, 40), (258540, 10, 40), target_chunks)


def test_era5_large_plan_sharded():
    memory_mb = 1280
    min_mem_fraction = 100  # chosen arbitrarily
    shape, source_chunks, target_chunks = ERA5_LARGE
    max_mem = chunk_budget(memory_mb * MB, sharded=True)
    min_mem = max_mem // min_mem_fraction

    ops = plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize=4,
        max_mem=max_mem,
        min_mem=min_mem,
        shard_intermediates=True,
    )

    assert len(ops) == 3

    def copy_op(sc, cc, tc, sharded=True):
        return CopyOp(sc, cc, tc, shape=shape, itemsize=4, sharded=sharded)

    assert ops[0] == copy_op(source_chunks, (62, 721, 1440), (62, 80, 160))
    assert ops[1] == copy_op((62, 80, 160), (3968, 80, 160), (3968, 10, 20))
    assert ops[2] == copy_op((3968, 10, 20), (258540, 10, 20), target_chunks, sharded=False)

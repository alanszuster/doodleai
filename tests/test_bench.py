"""Smoke test of the speed benchmark on a tiny model (it is meant to run on a GPU)."""

import pytest

pytest.importorskip("tensorflow")

import bench  # noqa: E402


def test_bench_times_the_variants_it_can_run_on_a_cpu():
    results = bench.run(
        ["baseline", "no-aug", "real-data"],  # real-data is skipped without a data dir
        size=16,
        classes=5,
        batch_size=8,
        steps=2,
        warmup=1,
        widths=[4, 8],
        blocks=[1, 1],
    )
    assert [r["variant"] for r in results] == ["baseline", "no-aug"]
    assert all(r["ms_per_step"] > 0 and r["img_per_s"] > 0 for r in results)

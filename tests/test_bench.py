"""Smoke test of the speed benchmark on a tiny model (it is meant to run on a GPU)."""

import pytest

pytest.importorskip("tensorflow")

import bench  # noqa: E402
from prepare import load_config  # noqa: E402


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


def test_bench_uses_the_configured_stem_stride():
    model = bench.build(
        16, 5, [4, 8], [1, 1], precision="fp32", xla=False, augment=False, stem_stride=2
    )
    assert model.get_layer(index=1).strides == (2, 2)


def test_bench_loads_the_requested_config_before_applying_cli_overrides(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(bench.DEFAULT_CONFIG.read_text().replace("batch_size = 512", "batch_size = 32"))
    monkeypatch.setattr("sys.argv", ["bench.py", "--config", str(path), "--size", "16"])
    received = {}
    monkeypatch.setattr(bench, "run", lambda variants, **kwargs: received.update(kwargs))
    bench.main()
    assert received["batch_size"] == 32
    assert received["size"] == 16
    assert received["stem_stride"] == load_config(path)["train"]["stem_stride"]

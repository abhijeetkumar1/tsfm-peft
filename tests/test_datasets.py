"""Registry wiring, config validation, and the shipped protocol configs.

The tests that actually download data are marked ``network`` and are excluded from CI.
"""

from pathlib import Path

import pytest

from tsfm_peft.config import DataConfig, load_yaml
from tsfm_peft.data.registry import DATASETS, available_datasets, get_spec, load_dataset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "data"


class TestRegistry:
    def test_v01_datasets_are_registered(self):
        assert available_datasets() == ("etth1", "nn5_daily", "synthetic")

    def test_specs_declare_a_license_and_a_source(self):
        # The README datasets table is generated from these; an empty field would ship a
        # dataset with no attribution.
        for spec in DATASETS.values():
            assert spec.license
            assert spec.description
            assert spec.source_url

    def test_downloaded_datasets_cite_an_https_source(self):
        for spec in DATASETS.values():
            if spec.is_generated:
                continue
            assert spec.source_url.startswith("https://")

    def test_only_the_smoke_fixture_is_generated(self):
        generated = {name for name, spec in DATASETS.items() if spec.is_generated}
        assert generated == {"synthetic"}

    def test_seasonality_matches_the_frequency(self):
        assert get_spec("etth1").seasonality == 24
        assert get_spec("nn5_daily").seasonality == 7

    def test_unknown_dataset_lists_the_available_ones(self):
        with pytest.raises(KeyError, match="etth1"):
            get_spec("etth2")

    def test_spec_to_dict_omits_the_loader(self):
        assert "loader" not in get_spec("etth1").to_dict()


class TestDataConfig:
    def test_rejects_an_unregistered_dataset(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            DataConfig(
                dataset="not_a_dataset",
                protocol={"horizon": 1, "context_length": 1, "n_test_windows": 1},
            )

    def test_rejects_an_unknown_scaler(self):
        with pytest.raises(ValueError, match="unknown scaler"):
            DataConfig(
                dataset="etth1",
                protocol={"horizon": 1, "context_length": 1, "n_test_windows": 1},
                scaler="minmax",
            )

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match=r"extra_inputs|Extra inputs"):
            DataConfig(
                dataset="etth1",
                protocol={"horizon": 1, "context_length": 1, "n_test_windows": 1},
                scalar="standard",
            )

    def test_load_yaml_rejects_a_non_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml(path, DataConfig)


class TestShippedConfigs:
    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
    def test_validates(self, path):
        config = load_yaml(path, DataConfig)
        assert config.dataset in available_datasets()

    @pytest.mark.parametrize(
        ("stem", "length"),
        [("etth1", 17420), ("nn5_daily", 791)],
    )
    def test_protocol_fits_the_real_series_length(self, stem, length):
        # Guards against a config edit that would only fail after a multi-megabyte
        # download; the lengths are fixed properties of the pinned files.
        config = load_yaml(CONFIG_DIR / f"{stem}.yaml", DataConfig)
        assert config.protocol.min_series_length <= length

    def test_etth1_windows_tile_the_tail_without_overlap(self):
        config = load_yaml(CONFIG_DIR / "etth1.yaml", DataConfig)
        assert config.protocol.effective_stride == config.protocol.horizon


@pytest.mark.network
class TestRealDatasets:
    """Downloads roughly 3 MB on first run, then reads from the cache."""

    @pytest.mark.parametrize(
        ("name", "n_series", "length"),
        [("etth1", 7, 17420), ("nn5_daily", 111, 791)],
    )
    def test_shape_and_provenance(self, name, n_series, length):
        dataset = load_dataset(name)
        spec = get_spec(name)
        assert len(dataset) == n_series
        assert set(dataset.lengths) == {length}
        assert dataset.license == spec.license
        assert dataset.freq == spec.freq

    def test_etth1_channels_are_named_after_the_csv_columns(self):
        assert load_dataset("etth1").series_ids == (
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT",
        )

    def test_nn5_series_are_non_negative_withdrawals(self):
        dataset = load_dataset("nn5_daily")
        assert all(series.values.min() >= 0 for series in dataset)

    @pytest.mark.parametrize("stem", ["etth1", "nn5_daily"])
    def test_shipped_config_builds_a_split(self, stem):
        config = load_yaml(CONFIG_DIR / f"{stem}.yaml", DataConfig)
        split = config.build_split()
        expected = config.protocol.n_test_windows * len(split.dataset)
        assert len(split.test_windows) == expected
        for series in split.dataset:
            plan = split.plans[series.series_id]
            assert plan.test_origins[-1] + config.protocol.horizon == len(series)

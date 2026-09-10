"""The generated smoke fixture."""

import numpy as np

from tsfm_peft.data.registry import get_spec, load_dataset
from tsfm_peft.data.synthetic import make_synthetic_dataset


class TestSyntheticDataset:
    def test_generation_is_deterministic(self):
        first = make_synthetic_dataset()
        second = make_synthetic_dataset()
        for a, b in zip(first.series, second.series, strict=True):
            np.testing.assert_array_equal(a.values, b.values)

    def test_seed_changes_the_series(self):
        default = make_synthetic_dataset()
        reseeded = make_synthetic_dataset(seed=1)
        assert not np.array_equal(default.series[0].values, reseeded.series[0].values)

    def test_shape_follows_the_arguments(self):
        dataset = make_synthetic_dataset(n_series=4, length=120, seasonality=12, freq="D")
        assert len(dataset) == 4
        assert dataset.lengths == (120,) * 4
        assert (dataset.freq, dataset.seasonality) == ("D", 12)

    def test_loads_through_the_registry_without_network(self):
        dataset = load_dataset("synthetic")
        assert dataset.name == "synthetic"
        assert (dataset.freq, dataset.seasonality) == ("h", 24)
        assert len(dataset) == 3

    def test_registry_marks_it_generated(self):
        assert get_spec("synthetic").is_generated
        assert not get_spec("etth1").is_generated

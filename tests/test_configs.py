"""The shipped experiment configs, checked as a set rather than one at a time.

Every published row points at one of these files, so the properties that make the rows
comparable are properties of the *set*: arms of a dataset must share a protocol, the DoRA row
must differ from the LoRA row in the method and nothing else, and the ablation rows must
differ in rank alone. None of that is visible when each file is only validated on its own,
and all of it is easy to break with a plausible-looking edit months from now.

These tests are torch-free and run in the CPU CI job.
"""

from pathlib import Path

import pytest
import yaml

from tsfm_peft.config import load_experiment
from tsfm_peft.models.registry import get_model_spec

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((REPO / "configs" / "experiments").glob("*.yaml"))

#: Fields a row is allowed to differ in without changing what it measures.
DESCRIPTIVE = {"name", "notes"}

ABLATION_RANKS = (4, 8, 32)


def experiment(stem: str):
    return load_experiment(REPO / "configs" / "experiments" / f"{stem}.yaml")


def differing_paths(left: dict, right: dict, prefix: str = "") -> set[str]:
    """Return the dotted paths at which two nested mappings disagree."""
    paths = set()
    for key in set(left) | set(right):
        path = f"{prefix}{key}"
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            paths |= differing_paths(a, b, f"{path}.")
        elif a != b:
            paths.add(path)
    return paths


def compared(stem: str) -> dict:
    """Return an experiment's fields with the purely descriptive ones removed."""
    dumped = load_experiment(REPO / "configs" / "experiments" / f"{stem}.yaml").model_dump()
    return {k: v for k, v in dumped.items() if k not in DESCRIPTIVE}


class TestEveryConfigLoads:
    def test_there_are_configs_to_check(self):
        # Guards the glob itself: a rename that emptied it would make every parametrised
        # test below vacuously pass.
        assert len(EXPERIMENTS) >= 10

    @pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
    def test_validates(self, path):
        assert load_experiment(path).name

    @pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
    def test_the_name_matches_the_filename(self, path):
        # The name becomes the artifact filename, so two configs sharing one would have the
        # second silently overwrite the first's results.
        assert load_experiment(path).name == path.stem

    def test_names_are_unique(self):
        names = [load_experiment(path).name for path in EXPERIMENTS]
        assert len(set(names)) == len(names)

    @pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
    def test_references_a_shared_data_config(self, path):
        # An inline protocol would let one arm's horizon drift from another's on the same
        # dataset. Every arm points at a file instead.
        payload = yaml.safe_load(path.read_text())
        assert isinstance(payload["data"], str)

    @pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
    def test_fixture_models_stay_on_the_fixture_dataset(self, path):
        # A random-init model paired with a real dataset would produce an artifact that
        # looks exactly like a benchmark row and means nothing.
        config = load_experiment(path)
        if get_model_spec(config.model.name).is_fixture:
            assert config.data.dataset == "synthetic"


class TestArmsAreComparable:
    @pytest.mark.parametrize("dataset", ["etth1", "nn5_daily"])
    def test_every_arm_shares_one_data_config(self, dataset):
        arms = [p for p in EXPERIMENTS if p.stem.startswith(f"{dataset}-")]
        assert len(arms) >= 3
        protocols = {load_experiment(path).data.protocol.model_dump_json() for path in arms}
        assert len(protocols) == 1

    @pytest.mark.parametrize("dataset", ["etth1", "nn5_daily"])
    def test_dora_differs_from_lora_only_in_the_method(self, dataset):
        difference = differing_paths(
            compared(f"{dataset}-timesfm-lora"), compared(f"{dataset}-timesfm-dora")
        )
        assert difference == {"model.options.peft.method"}

    @pytest.mark.parametrize("dataset", ["etth1", "nn5_daily"])
    def test_the_finetuned_arms_match_the_zero_shot_arm_where_it_overlaps(self, dataset):
        # The fine-tuned rows must be the same model, on the same device, at the same
        # precision, with the same clamp rule as the row they are compared against.
        zero_shot = compared(f"{dataset}-timesfm-zeroshot")["model"]["options"]
        lora = compared(f"{dataset}-timesfm-lora")["model"]["options"]
        shared = set(zero_shot) - {"peft"}
        assert {key: lora[key] for key in shared} == {key: zero_shot[key] for key in shared}


class TestRankAblation:
    @pytest.mark.parametrize("rank", ABLATION_RANKS)
    def test_differs_from_the_headline_row_only_in_rank_and_alpha(self, rank):
        difference = differing_paths(
            compared("etth1-timesfm-lora"), compared(f"etth1-timesfm-lora-r{rank}")
        )
        assert difference == {"model.options.peft.rank", "model.options.peft.alpha"}

    @pytest.mark.parametrize("rank", ABLATION_RANKS)
    def test_carries_the_rank_its_filename_claims(self, rank):
        assert experiment(f"etth1-timesfm-lora-r{rank}").model.options["peft"]["rank"] == rank

    def test_holds_the_scaling_constant_across_the_sweep(self):
        # alpha / rank is what multiplies the update. Fixed across rows, the sweep varies
        # capacity alone; if it drifted, a difference between rows could not be attributed
        # to rank.
        stems = ["etth1-timesfm-lora"] + [f"etth1-timesfm-lora-r{r}" for r in ABLATION_RANKS]
        scalings = set()
        for stem in stems:
            peft = experiment(stem).model.resolved_options().peft
            scalings.add(peft.alpha / peft.rank)
        assert scalings == {2.0}

    def test_the_headline_row_is_the_rank_16_point(self):
        # There is deliberately no -r16 file; the sweep's midpoint is the headline arm.
        assert experiment("etth1-timesfm-lora").model.options["peft"]["rank"] == 16
        assert not (REPO / "configs" / "experiments" / "etth1-timesfm-lora-r16.yaml").exists()

    def test_covers_a_range_of_at_least_eight_times(self):
        ranks = {*ABLATION_RANKS, 16}
        assert max(ranks) / min(ranks) >= 8

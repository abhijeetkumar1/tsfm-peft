"""Scaffolding-level tests: the package imports and the CLI runs."""

import pytest

import tsfm_peft
from tsfm_peft.cli import main


def test_version_is_exposed():
    assert isinstance(tsfm_peft.__version__, str)
    assert tsfm_peft.__version__


def test_cli_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tsfm-peft" in capsys.readouterr().out


def test_cli_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "Parameter-efficient fine-tuning" in capsys.readouterr().out

import importlib.util
from pathlib import Path

import pytest

from d_bridge import cli
from d_bridge.modules import hisg
from d_bridge.modules.hisg.runtime import run_experiment
from d_bridge.modules.hisg.storage import Neo4jStore
from d_bridge.modules.hisg.workflow import build_experiment_graph


def test_hisg_package_exposes_module_identity_and_business_apis():
    assert hisg.MODULE_ID == "hisg"
    assert hisg.MODULE_NAME == "HISG Builder"
    assert callable(run_experiment)
    assert isinstance(Neo4jStore, type)
    assert callable(build_experiment_graph)


@pytest.mark.parametrize("module_name", [
    "config",
    "data",
    "granularity",
    "prompts",
    "randomness",
    "reporting",
    "runtime",
    "sampling",
    "storage",
    "workflow",
])
def test_legacy_root_business_modules_do_not_exist(module_name):
    assert importlib.util.find_spec(f"d_bridge.{module_name}") is None


@pytest.mark.parametrize("argv", [
    ["hisg", "--env-file", "experiment.env", "--data-dir", "posts"],
    ["--env-file", "experiment.env", "--data-dir", "posts"],
])
def test_root_cli_runs_hisg_explicitly_and_by_default(monkeypatch, tmp_path, argv):
    settings = object()
    captured = {}

    def load(path):
        captured["env_file"] = path
        return settings

    monkeypatch.setattr(cli, "load_settings", load)

    def run(current_settings, data_dir):
        captured["settings"] = current_settings
        captured["data_dir"] = data_dir
        return tmp_path / "result" / "run" / "hisg"

    monkeypatch.setattr(cli, "run_experiment", run)

    assert cli.main(argv) == 0
    assert captured["env_file"] == Path("experiment.env")
    assert captured["data_dir"] == Path("posts")


def test_root_cli_rejects_unknown_module():
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["unknown-module"])
    assert exc_info.value.code == 2

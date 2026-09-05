"""Run research scripts outside the checkout and check notebook sources."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "code_construction/construct_codes.py",
    "syndrome_extraction/validate_schedules.py",
    "circuit_distance/estimate_distance.py",
    "circuit_simulation/simulate_memory.py",
    "bivariate_bicycle/simulate_memory.py",
    "surface_code/simulate_memory.py",
    "figures/overhead/plot_overhead.py",
    "figures/logical_error_rates/plot_error_rates.py",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_from_unrelated_directory(script, tmp_path):
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for folder in (
            "circuit_distance",
            "circuit_simulation",
            "bivariate_bicycle",
            "surface_code",
            "figures",
        )
        for path in (ROOT / folder).rglob("*.ipynb")
    ),
    ids=lambda path: path.stem,
)
def test_notebooks_are_unexecuted_and_syntax_valid(path):
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert source.strip()
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse(source)


def test_saved_schedule_paths_resolve_from_summary():
    summary = ROOT / "syndrome_extraction" / "schedule_validation_summary.json"
    for record in json.loads(summary.read_text()):
        path = Path(record["stim_path"])
        assert not path.is_absolute()
        assert (summary.parent / path).is_file()


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for folder in ("circuit_simulation", "bivariate_bicycle", "surface_code")
        for path in (ROOT / folder).glob("*.ipynb")
    ),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_notebook_commands_accept_both_bases_and_detector_modes(path, monkeypatch):
    import importlib

    monkeypatch.chdir(ROOT)
    notebook = json.loads(path.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if "completed = subprocess.run" in source:
            break
        exec(compile(source, str(path), "exec"), namespace)
        if "SCRIPT = " in source:
            namespace.update(BASIS="both", DECODING_MODE="both")
    cmd = namespace["cmd"]
    monkeypatch.setattr(sys, "argv", cmd[1:])
    runner = importlib.import_module(f"{path.parent.name}.simulate_memory")
    args = runner.parse_args()
    assert args.basis == "both"
    assert args.decoding_mode == "both"

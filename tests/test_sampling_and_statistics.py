"""Check sampling reuse, simulation parameters, and logical-error summaries."""

import ast
import csv
import json
from pathlib import Path

import pytest

from circuit_distance import estimate_distance as distance
from circuit_simulation.logical_error_statistics import upsert_summary_record

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cpu_count", [None, 1, 8])
def test_distance_defaults_work_on_small_machines(monkeypatch, cpu_count):
    monkeypatch.setattr(distance.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr("sys.argv", ["cornucopia-distance"])
    assert distance.parse_args().workers >= 1


def test_summary_keeps_distinct_memory_durations():
    records = []
    for cycles in (2, 3):
        upsert_summary_record(
            records,
            {
                "shots_decoded": 100,
                "decoder": "relaybp",
                "code_name": "surface_d3",
                "parameter_label": "[[9,1,3]]",
                "P": 3,
                "expected_d": 3,
                "basis": "Z",
                "decoding_mode": "xz",
                "p": 0.002,
                "cycles": cycles,
            },
        )
    assert len(records) == 2
    assert {record["cycles"] for record in records} == {2, 3}


def test_refit_csv_preserves_code_labels_containing_commas(tmp_path):
    notebook = json.loads(
        (ROOT / "figures/logical_error_rates/refit_error_rates.ipynb").read_text()
    )
    helper_names = {
        "failure_column_for_y",
        "yerr_from_failures",
        "write_xz_average_points",
    }
    helpers = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in helper_names:
                helpers.append(node)
    assert {node.name for node in helpers} == helper_names
    import math

    namespace = {
        "Path": Path,
        "csv": csv,
        "math": math,
        "Y_COLUMN": "LER_per_cycle_per_logical",
    }
    exec(
        compile(
            ast.Module(body=helpers, type_ignores=[]), "write_xz_average_points", "exec"
        ),
        namespace,
    )
    path = tmp_path / "rates.csv"
    namespace["write_xz_average_points"](
        [
            {
                "parameter_label": "[[252,130,6]]",
                "p": 0.002,
                "LER_per_cycle_per_logical": 1e-6,
                "failures": 4,
            }
        ],
        path,
    )
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert None not in row
    assert row["parameter_label"] == "[[252,130,6]]"
    assert float(row["p"]) == 0.002
    assert float(row["yerr"]) == pytest.approx(5e-7)


@pytest.fixture
def saved_samples(tmp_path):
    from circuit_simulation.memory_experiment import run_generate_condition
    from circuit_simulation.shot_data import artifact_paths
    from circuit_simulation.simulation_parameters import Condition

    condition = Condition("test_memory", "[[1,1,1]]", 1, 1, 1, 1, "Z", "xz", 0.2, 2)
    meta = {"num_detectors": 1, "num_observables": 1}
    paths = artifact_paths(tmp_path, condition)
    paths["dem"].parent.mkdir(parents=True)
    paths["dem"].write_text("error(0.2) D0 L0\n")

    def generate(*, shots=8, seed=7):
        run_generate_condition(
            condition,
            tmp_path,
            shots,
            4,
            2,
            seed,
            meta,
            force=False,
            progress_seconds=30,
        )

    generate()
    return condition, meta, paths, generate


def test_sample_reuse_and_extension_preserve_existing_chunks(tmp_path, saved_samples):
    from circuit_simulation.shot_data import load_shots_manifest

    condition, _, _, generate = saved_samples
    first = load_shots_manifest(tmp_path, condition)
    generate()
    assert load_shots_manifest(tmp_path, condition) == first
    generate(shots=12)
    extended = load_shots_manifest(tmp_path, condition)
    assert extended["chunks"][:2] == first["chunks"]
    assert sum(chunk["num_shots"] for chunk in extended["chunks"]) == 12


def test_new_seed_regenerates_stored_samples(tmp_path, saved_samples):
    from circuit_simulation.memory_experiment import expected_generate_tasks
    from circuit_simulation.shot_data import load_shots_manifest, shot_matches_task

    condition, meta, _, generate = saved_samples
    first = load_shots_manifest(tmp_path, condition)
    generate(seed=11)
    second = load_shots_manifest(tmp_path, condition)
    assert second["condition_seed"] == 11
    assert first["chunks"][0]["seed"] != second["chunks"][0]["seed"]
    tasks = expected_generate_tasks(tmp_path, condition, 8, 4, 11, meta)
    assert all(shot_matches_task(task) for task in tasks)


@pytest.mark.parametrize("changed", ["circuit", "samples", "record"])
def test_changed_sample_inputs_require_regeneration(tmp_path, saved_samples, changed):
    from circuit_simulation.shot_data import load_shots_manifest, shots_manifest_path

    condition, _, paths, generate = saved_samples
    first = load_shots_manifest(tmp_path, condition)
    if changed == "circuit":
        paths["dem"].write_text("error(0.3) D0 L0\n")
    elif changed == "samples":
        path = Path(first["chunks"][0]["path"])
        path.touch()
    else:
        shots_manifest_path(tmp_path, condition).write_text("{}")
    with pytest.raises(ValueError):
        load_shots_manifest(tmp_path, condition)
    generate()
    second = load_shots_manifest(tmp_path, condition)
    assert len(second["chunks"]) == 2
    assert second["manifest_hash"] != first["manifest_hash"]


def test_missing_chunks_cannot_be_reported_as_complete(tmp_path, saved_samples):
    from circuit_simulation.memory_experiment import expected_generate_tasks
    from circuit_simulation.shot_data import build_shots_manifest

    condition, meta, _, _ = saved_samples
    tasks = expected_generate_tasks(tmp_path, condition, 12, 4, 7, meta)
    with pytest.raises(ValueError, match="Missing or incompatible shot chunk"):
        build_shots_manifest(tmp_path, condition, tasks, meta, 12, 4, 7)


@pytest.mark.parametrize(
    "values",
    ["nan", "inf", "-0.001", "1.01", "0.002,0.002", "0.0020000001,0.0020000002"],
)
def test_noise_probabilities_reject_invalid_or_ambiguous_conditions(values):
    from circuit_simulation.memory_experiment import parse_p_list

    with pytest.raises(SystemExit):
        parse_p_list(values)


def test_noise_probabilities_include_the_noiseless_limit():
    from circuit_simulation.memory_experiment import parse_p_list

    assert parse_p_list("0, 0.002, 1") == [0.0, 0.002, 1.0]


@pytest.mark.parametrize(
    "module",
    [
        "circuit_simulation.memory_experiment",
        "bivariate_bicycle.simulate_memory",
        "surface_code.simulate_memory",
    ],
)
def test_cached_circuit_matches_exact_physical_condition(module, tmp_path):
    import importlib
    from dataclasses import replace

    from circuit_simulation.memory_experiment import load_artifact_meta
    from circuit_simulation.shot_data import artifact_paths
    from circuit_simulation.simulation_parameters import Condition

    runner = importlib.import_module(module)
    condition = Condition(
        "test_memory", "[[1,1,1]]", 1, 1, 1, 1, "Z", "xz", 0.0020000001, 2
    )
    meta = {
        "code_name": condition.code_name,
        "parameter_label": condition.parameter_label,
        "P": 1,
        "L": 1,
        "J": 1,
        "expected_d": 1,
        "basis": "Z",
        "decoding_mode": "xz",
        "p": condition.p_noise,
        "cycles": 2,
    }
    if hasattr(runner, "artifact_implementation_hash"):
        meta["artifact_implementation_hash"] = runner.artifact_implementation_hash()
    paths = artifact_paths(tmp_path, condition)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    paths["meta"].write_text(json.dumps(meta))
    assert load_artifact_meta(tmp_path, condition) == meta
    different = replace(condition, p_noise=0.0020000002)
    assert different.label == condition.label
    with pytest.raises(ValueError, match="physical condition"):
        load_artifact_meta(tmp_path, different)
    with pytest.raises(ValueError, match="physical condition"):
        runner.build_and_save_artifacts(None, different, tmp_path)

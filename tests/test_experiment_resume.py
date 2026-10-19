import json
import sys

import pytest
import torch
from filelock import FileLock

from mr_nwm import experiment, run_store
from mr_nwm.provenance import sha256_file
from mr_nwm.verification import compare_metrics


def invoke(monkeypatch, output, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--steps",
            "1",
            "--seeds",
            "11",
            "--variants",
            "mlp",
            "--batch-size",
            "4",
            "--eval-size",
            "4",
            "--latency-repeats",
            "2",
            "--checkpoints",
            "--output",
            str(output),
            *extra,
        ],
    )
    experiment.main()


def read_manifest(output):
    return json.loads((output / "metrics.json").read_text())


def test_interrupted_sweep_reuses_completed_rows_and_matches_uninterrupted_training(
    monkeypatch, tmp_path
):
    output = tmp_path / "interrupted"
    original = experiment.run_variant
    options = ("--steps", "2", "--variants", "full", "mlp", "--seeds", "11", "22")
    calls = []

    def interrupt_after_first(name, seed, args, device):
        calls.append((name, seed))
        if len(calls) == 2:
            raise RuntimeError("injected interruption")
        return original(name, seed, args, device)

    monkeypatch.setattr(experiment, "run_variant", interrupt_after_first)
    with pytest.raises(RuntimeError, match="injected interruption"):
        invoke(monkeypatch, output, *options)
    partial = read_manifest(output)
    assert len(partial["results"]) == 1
    first = partial["results"][0]
    checkpoint = output / "checkpoints/full_11.pt"
    original_bytes = checkpoint.read_bytes()
    calls.clear()

    def track(name, seed, args, device):
        calls.append((name, seed))
        return original(name, seed, args, device)

    monkeypatch.setattr(experiment, "run_variant", track)
    invoke(monkeypatch, output, *options, "--resume")
    resumed = read_manifest(output)
    assert calls == [("full", 22), ("mlp", 11), ("mlp", 22)]
    assert resumed["results"][0] == first
    assert checkpoint.read_bytes() == original_bytes
    assert len(resumed["resume_commands"]) == 1

    reference = tmp_path / "uninterrupted"
    monkeypatch.setattr(experiment, "run_variant", original)
    invoke(monkeypatch, reference, *options)
    uninterrupted = read_manifest(reference)
    for actual, expected in zip(resumed["results"], uninterrupted["results"]):
        assert actual["training_trace"] == expected["training_trace"]
        for split in ("iid", "ood", "composition"):
            compare_metrics(actual[split], expected[split], 0)
        name = f"{actual['variant']}_{actual['seed']}.pt"
        left = torch.load(output / "checkpoints" / name, weights_only=True)["state_dict"]
        right = torch.load(reference / "checkpoints" / name, weights_only=True)["state_dict"]
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_complete_resume_performs_no_training(monkeypatch, tmp_path):
    invoke(monkeypatch, tmp_path)
    original = read_manifest(tmp_path)["results"]

    def unexpected_training(*args):
        pytest.fail("completed sweep was retrained")

    monkeypatch.setattr(experiment, "run_variant", unexpected_training)
    invoke(monkeypatch, tmp_path, "--resume")
    assert read_manifest(tmp_path)["results"] == original


def test_resume_recomputes_metrics_instead_of_trusting_saved_values(monkeypatch, tmp_path):
    invoke(monkeypatch, tmp_path)
    manifest = read_manifest(tmp_path)
    manifest["results"][0]["iid"]["nmse_db"] += 1
    run_store.save_json(tmp_path / "metrics.json", manifest)
    original = (tmp_path / "metrics.json").read_bytes()
    with pytest.raises(RuntimeError, match="Metric mismatch"):
        invoke(monkeypatch, tmp_path, "--resume")
    assert (tmp_path / "metrics.json").read_bytes() == original


def test_existing_output_requires_an_explicit_restart_choice(monkeypatch, tmp_path):
    invoke(monkeypatch, tmp_path)
    original = (tmp_path / "metrics.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, tmp_path)
    assert error.value.code == 2
    assert (tmp_path / "metrics.json").read_bytes() == original
    invoke(monkeypatch, tmp_path, "--seeds", "22", "--overwrite")
    assert [row["seed"] for row in read_manifest(tmp_path)["results"]] == [22]


@pytest.mark.parametrize(
    "options", [("--steps", "2"), ("--batch-size", "8"), ("--horizon", "2"), ("--seeds", "22")]
)
def test_resume_rejects_changed_experiment_arguments(monkeypatch, tmp_path, options):
    invoke(monkeypatch, tmp_path)
    original = (tmp_path / "metrics.json").read_bytes()
    with pytest.raises(RuntimeError, match="Resume argument mismatch"):
        invoke(monkeypatch, tmp_path, *options, "--resume")
    assert (tmp_path / "metrics.json").read_bytes() == original


@pytest.mark.parametrize("field", ["source_sha256", "environment"])
def test_resume_rejects_changed_sources_or_runtime(monkeypatch, tmp_path, field):
    invoke(monkeypatch, tmp_path)
    manifest = read_manifest(tmp_path)
    if field == "source_sha256":
        manifest[field]["src/mr_nwm/experiment.py"] = "0" * 64
    else:
        manifest[field]["torch"] = "different-runtime"
    run_store.save_json(tmp_path / "metrics.json", manifest)
    original = (tmp_path / "metrics.json").read_bytes()
    with pytest.raises(RuntimeError, match="Resume (source|environment) mismatch"):
        invoke(monkeypatch, tmp_path, "--resume")
    assert (tmp_path / "metrics.json").read_bytes() == original


@pytest.mark.parametrize("corruption", ["weights", "metadata", "missing_digest"])
def test_resume_rejects_corrupt_or_unattributed_checkpoints(monkeypatch, tmp_path, corruption):
    invoke(monkeypatch, tmp_path)
    manifest = read_manifest(tmp_path)
    row = manifest["results"][0]
    path = tmp_path / "checkpoints/mlp_11.pt"
    saved = torch.load(path, weights_only=True)
    if corruption == "weights":
        saved["state_dict"]["network.0.weight"] += 1
        run_store.save_checkpoint(path, saved)
        message = "digest mismatch"
    elif corruption == "metadata":
        saved["seed"] = 22
        run_store.save_checkpoint(path, saved)
        row["checkpoint_sha256"] = sha256_file(path)
        message = "metadata mismatch"
    else:
        del row["checkpoint_sha256"]
        message = "Missing checkpoint digest"
    run_store.save_json(tmp_path / "metrics.json", manifest)
    original = (tmp_path / "metrics.json").read_bytes()
    with pytest.raises(RuntimeError, match=message):
        invoke(monkeypatch, tmp_path, "--resume")
    assert (tmp_path / "metrics.json").read_bytes() == original


def test_second_writer_cannot_change_an_active_run(monkeypatch, tmp_path):
    invoke(monkeypatch, tmp_path)
    original = (tmp_path / "metrics.json").read_bytes()
    with FileLock(tmp_path / ".run.lock", timeout=0):
        with pytest.raises(SystemExit) as error:
            invoke(monkeypatch, tmp_path, "--overwrite")
        assert error.value.code == 2
    assert (tmp_path / "metrics.json").read_bytes() == original


@pytest.mark.parametrize("kind", ["json", "checkpoint"])
def test_failed_atomic_replace_preserves_previous_artifact(monkeypatch, tmp_path, kind):
    save = run_store.save_json if kind == "json" else run_store.save_checkpoint
    path = tmp_path / "artifact"
    save(path, {"value": 1})
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(run_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        save(path, {"value": 2})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".pending-*"))


def test_nonfinite_gradients_abort_before_optimizer_or_checkpoint(monkeypatch, tmp_path):
    original = experiment.build_model

    def bad_gradient(name, horizon):
        model = original(name, horizon)
        model.network[-1].weight.register_hook(lambda gradient: gradient * float("nan"))
        return model

    monkeypatch.setattr(experiment, "build_model", bad_gradient)
    with pytest.raises(RuntimeError, match="non-finite"):
        invoke(monkeypatch, tmp_path)
    assert read_manifest(tmp_path)["results"] == []
    assert not (tmp_path / "checkpoints/mlp_11.pt").exists()


@pytest.mark.parametrize("options", [("--seeds", "11", "11"), ("--variants", "mlp", "mlp")])
def test_duplicate_run_keys_fail_before_creating_outputs(monkeypatch, tmp_path, options):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, output, *options)
    assert error.value.code == 2
    assert not output.exists()

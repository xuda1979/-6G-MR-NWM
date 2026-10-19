"""Atomic experiment artifacts and strict contracts for resuming completed runs."""

import json
import os
import tempfile
from pathlib import Path

import torch

from .verification import validate_result_rows

MANIFEST_VERSION = 2


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(path, value):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def save_checkpoint(path, value):
    _atomic_write(path, lambda stream: torch.save(value, stream))


def load_resume_manifest(path, current):
    """Only reuse rows from the same experiment, sources and runtime contract."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError("Resume requires a version-2 manifest created by this runner")
    if manifest["protocol"] != current["protocol"]:
        raise RuntimeError("Resume protocol mismatch")
    ignored = {"output", "resume", "overwrite"}
    previous_args = {
        key: value for key, value in manifest["arguments"].items() if key not in ignored
    }
    current_args = {key: value for key, value in current["arguments"].items() if key not in ignored}
    if previous_args != current_args:
        changed = sorted(
            key
            for key in previous_args.keys() | current_args.keys()
            if previous_args.get(key) != current_args.get(key)
        )
        raise RuntimeError(f"Resume argument mismatch: {', '.join(changed)}")
    if manifest["source_sha256"] != current["source_sha256"]:
        raise RuntimeError("Resume source mismatch; use a new output directory for changed code")
    for key in ("python", "torch", "numpy", "platform", "device", "threads"):
        if manifest["environment"][key] != current["environment"][key]:
            raise RuntimeError(f"Resume environment mismatch: {key}")
    validate_result_rows(manifest, complete=False)
    return manifest

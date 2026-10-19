"""Hash project-owned experiment sources without traversing installed environments."""

import hashlib
from pathlib import Path


def source_hashes(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    paths = list(root.glob("*.py"))
    for directory in ("src", "scripts", "tests"):
        paths.extend((root / directory).rglob("*.py"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if not path.is_symlink()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts
        )
    }

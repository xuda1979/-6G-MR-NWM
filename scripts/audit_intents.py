"""Measure the exact lexicon compiler on the controlled language splits."""

import hashlib
import json
from pathlib import Path

from mr_nwm.data import generate_batch
from mr_nwm.metrics import intent_metrics
from mr_nwm.rules import compile_objectives

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "results/metrics.json").read_text())
results = {}
for split in ("train", "test", "composition"):
    batch = generate_batch(manifest["arguments"]["eval_size"], 10, 900_000, intent_split=split)
    results[split] = intent_metrics(compile_objectives(batch.intents), batch.weights)
    if results[split]["intent_tv"] != 0:
        raise RuntimeError(f"Lexicon compiler failed the benchmark grammar: {split}")
paths = [Path(__file__), root / "src/mr_nwm/rules.py"]
output = {
    "baseline": "exact-objective-lexicon",
    "results": results,
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    },
    "note": "Only the benchmark grammar; no open-language capability implied.",
}
(root / "results/intent_baseline.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))

# Multimodal numerical precision review

Review date: 2026-10-21. Baseline: `c48db96`.

## Reproduced failures and corrections

The foundation input encoder previously narrowed raw ages and graph weights to
the model dtype before preprocessing. With FP16 parameters, an age of 86,400
seconds became infinity before `log1p`, although the logarithmic feature is about
11.37 and is representable. Small positive graph weights could instead become
zero before row normalization; multiplying a graph by a positive constant then
changed its messages. Wide-range FP64 metadata exposed the same ordering issue
for BF16 and FP32. New regression cases failed on the baseline implementation.

The encoder now computes logarithmic time features and graph normalization in at
least FP32, retaining FP64 metadata when supplied, before narrowing the resulting
features. Graph rows are scaled by their largest surviving edge before summing;
missing and expired nodes cannot send or receive messages, and isolated rows stay
zero. Valid numeric measurements that overflow the model dtype raise `ValueError`
before their adapter executes. Complex ages and adjacency are rejected explicitly.

The paper and [foundation guide](LANGUAGE_FOUNDATION.md) describe the calculation
order, explicit expiry policy, and advancing observation ages across macro windows.
The manuscript was rebuilt and visually reviewed at nine pages.

## Validation

Commands below ran from the repository root with the project environment active:

```bash
make lint test
make smoke
python scripts/verify_results.py --allow-source-changes --output runs/precision-reference.json
make paper
```

| Check | Result |
|---|---|
| Baseline suite before edits | 83 passed |
| Updated suite | 100 passed; no skipped tests |
| Focused modality/foundation suite | 37 passed |
| Ruff lint and formatting | Passed |
| Fresh two-update, eight-variant benchmark smoke | Passed; seven checkpoint digests checked |
| Resume of the completed smoke sweep | All eight rows re-evaluated and reused |
| Source-matched smoke verification | 520 deterministic scalar values matched at absolute tolerance `1e-5` |
| Historical checkpoint regression | 24 rows and 1,560 deterministic scalar values matched at absolute tolerance `1e-5` |
| Paper | Nine pages; final TeX log has no undefined references or overfull boxes |

An independent read-only verifier also recomputed graph messages using scalar
weighted averages and time features using Python `math.log1p`, without calling
the encoder's normalization logic. Nine dtype/edge-scale combinations covered
FP16, BF16, FP32 and scales `1e-100`, `1`, `1e100`, with FP64 graph metadata.
Missing/expired nodes, isolated rows, finite gradients, and downcast-overflow
rejection passed. Deliberately zeroing messages caused the independent comparison
to fail. Tiny offline Qwen models completed backward, an SGD optimizer step,
checkpoint save/load, exact prediction replay and preference-invariance checks
in each dtype. Parameter movement was confirmed in FP32; low-precision updates
can round away, so this is not a claim about successful half-precision training.

Runtime: Python 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.5.4, Transformers 4.57.6,
pytest 8.4.2 and Ruff 0.11.13, on Linux x86-64 CPU.

## Scope and evidence limits

Historical benchmark metrics and checkpoints were preserved. Their audit is an
explicit cross-revision deterministic regression, not a replay of training or
wall-clock latency. The 21 historical parameterized checkpoints predate saved
digest attribution; the fresh smoke checks digests for all seven parameterized
variants. The original three-seed synthetic results are not new performance gains.

The pretrained Qwen integration record remains evidence for its recorded source
revision. This review used offline random Qwen fixtures and did not download or
retest the full pretrained checkpoint. GPU kernels, synchronized radio data and
target-device performance remain unmeasured. Callers must preserve raw metadata
precision and calibrate measurement features; this patch cannot recover values
already rounded away or guarantee finite computation for arbitrary learned weights.

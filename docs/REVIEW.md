# Technical review and corrections

This is the historical draft/synthetic review. Current task-trained measured-trace
evidence is described in [PUBLIC_DATA.md](PUBLIC_DATA.md) and
[the measured-data results](../results/public_data/SUMMARY.md); the findings below
retain their original experimental scope.

The starting material consisted of two pasted manuscript/code drafts. This
release reconstructs executable source rather than running their embedded shell
or bundler scripts. The original pasted conversations are not included.

| Finding in the drafts | Correction in this release |
|---|---|
| Malformed Python dunder names, indentation, nested quotes and fences; `src/**init**.py`; invalid workflow YAML | Installable `src/mr_nwm` package, compilable sources, tests, and executable CI |
| A purported state-space expert with no temporal state in one draft | Explicit recurrent memory; sequence/step equivalence and memory-dependence tests |
| Random target words unrelated to input intent | Auditable phrase-derived objective targets; disjoint wording and composition tests |
| Unsupported PLM initialization, graph encoder, and sparse-MoE descriptions | Random semantic MLPs and dense routing stated explicitly; unimplemented extensions identified |
| Table values such as `-14.65 dB`, `0.18 ms`, and an unimplemented pure-LLM comparison | Removed; every published result derives from saved executable runs |
| Same initial state and actions run twice described as two macro windows | Predicted terminal state and summary advance into the next action window |
| Intent preferences directly change the simulator transition law | Preferences affect utility; exogenous context and actions determine physics |
| Unmasked multimodal attention and recurrence leak preferences into physical predictions | Physical queries cannot read intent; physics recurrence excludes intent tokens |
| Default dimensions produce an empty action slice in the original simulator | Explicit two-power/six-state simulator with checked dimensions |
| Detached baseline freezes the slow model and changes unrelated training objectives | Controlled gradient cut; comparison against the matching unshared architecture |
| All ablation claims rely on only training one of the allocated models | Every trainable variant independently trained on the same batches and budget |
| Only small/weak numeric baseline | Added a parameter-matched MLP and a standalone model using the same recurrent operator |
| Zero-initialized flow samples despite a Gaussian training base | Sampling requires explicit base noise and action conditioning |
| Action-independent macro prior can bypass the action-conditioned fast model | Deterministic predictor models state increments from state/action/context; optional flow remains a separate trajectory head |
| Variance-based error called NMSE without definition | Signal-energy NMSE defined in code and manuscript; feature and horizon breakdowns retained |
| Cosine mapped into a misleading percentage | Report raw cosine similarity and total variation distance |
| One cold host timer used as a deterministic DU latency claim | Warm-up, synchronization, repeated samples and p50/p95/p99; exclusions listed |
| Joint training or tensor sharing described as a proof of real-world generalization or deployment | Empirical hypothesis; no convergence, field-performance, or distributed-runtime claim |
| No references supporting major methods | Primary method papers and the current retrieved O-RAN architecture specification cited |

## Interpretation policy

The proposed nested architecture is a research hypothesis. Small-simulator gains
do not establish field value. In particular, overlap between three-seed variation
in sharing and bridge ablations must be reported rather than treated as a
universal advantage. The standalone recurrent control is important for deciding
whether the slow tier is justified by a specific application.

The main contribution of this release is a correct and measurable experiment,
including causal checks, strong controls, saved final weights, and a manuscript
whose scope matches the implementation. Open-domain semantics, stochastic
calibration, real-time deadlines, topology generalization, and practical network
control remain research tasks.

## Language-foundation integration

The project now additionally includes an actual pretrained language-transformer
path, continuous modality tokens with observation ages/periods, and optional
topology message passing. Qwen2.5-0.5B was loaded and its multimodal/gradient/causal
interfaces exercised, with a separate integration record. The reference numerical
benchmark remains the small randomly initialized model and must not be described
as a performance result for the language-foundation version. Raw CSI calibration,
image patch extraction, task training, and deployed timing remain unvalidated.

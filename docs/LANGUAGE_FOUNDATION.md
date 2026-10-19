# Language foundation, multimodal inputs, and multiple timescales

The intended MR-NWM backbone is a pretrained language transformer. The public
API is `mr_nwm.foundation.LanguageBasedWorldModel.from_pretrained(...)`. The
smaller model in `model.py` is kept intact as the reproducible experimental
reference; its scores must not be attributed to the language-foundation model.

## What is connected

| Input | How it enters the language backbone |
|---|---|
| Intent text | Actual pretrained-model tokenizer and token embedding |
| Exogenous textual evidence | A separate field, available to physical and semantic views |
| Current numerical state and measured context | Dedicated continuous projections |
| CSI and IQ | Continuous feature tokens; caller supplies explicit real/imaginary pairs |
| KPI/traffic series | Configurable continuous feature tokens |
| Topology | Node features with an optional normalized graph message-passing step |
| Maps/visual features | Configurable patch feature tokens; raw patch extraction is external |
| Previous fast summary | Numeric projection into the same language-transformer embedding width |

The generic front ends are trainable linear/MLP projections with modality
type and time embeddings. They retain absolute feature levels; any calibrated
unit conversion or fixed training-set normalization belongs in the data pipeline.
They are not pretrained radio or vision encoders.
Raw RF calibration, CSI feature construction, image patch extraction, and
clock synchronization remain the responsibility of the data pipeline.

## Information flow and shared parameters

The same pretrained language model processes two views:

1. **Physical view:** measured state/context, observed modalities, optional
   validated exogenous evidence, and a physical query token. It never reads
   operational preference text.
2. **Semantic view:** the same observed prefix followed by the preference text.
   A task-trained head compiles structured objective weights.

The projected physical context passes through a recurrent physics expert and
generates the FiLM/low-rank packet. When enabled, the fast model reuses that
same physics expert. This is a shared expert around the language trunk; the
implementation does not replace the pretrained transformer's internal layers
with wireless state-space operators.

Requests, applied actions, and observed external events have different causal
roles. A request to change the network should not be put in exogenous evidence
as though it had already occurred. The caller must validate evidence before
using that field. Actual actions enter the fast transition predictor directly.

## Three distinct clocks

| Clock | Representation and execution |
|---|---|
| Per-modality acquisition | `period_seconds`, `age_seconds`, and a validity mask per modality/token |
| Slow backbone refresh | Run two views of the language transformer once per macro window |
| Fast numerical prediction | Reuse a cached packet for K recurrent transitions; no transformer call |

These describe two learned execution rates receiving asynchronous measurements.
An even slower operator/knowledge/model-management layer can be added externally;
it is not a third implemented controller. `horizon=5` in the reference benchmark
is a discrete modeling choice, not a calibrated radio deadline.

Age values are relative to a caller-selected slow-update time. Future timestamps
are rejected. Missing tokens are masked. Optional `max_age_seconds` masks tokens
older than the configured limit, including their incoming and outgoing graph edges.
The acquisition period is not automatically used as an expiry limit.
Excessive input lengths raise an error before either language-model pass
so the caller must resample or pool streams explicitly. This module does not
schedule network messages or solve transport delay and stale-packet fallback.

For a fixed snapshot over multiple macro windows, supply the physical duration of
one macro window. The model advances measurement ages without mutating the snapshot:

```python
trajectory = model.rollout(
    inputs, initial, actions, context, macro_period_seconds=0.005,
)
```

`macro_period_seconds` is required when modalities are present and the rollout spans
multiple windows. No radio time is inferred from `horizon`. `FoundationInputs.advance(t)`
also supports manual advancement by a nonnegative number of seconds. For newly acquired
measurements or changing exogenous context, call the model per macro window with fresh
inputs and the preceding `sync_summary` and predicted terminal state. A fixed-snapshot
forecast does not generate future sensor measurements or refresh textual evidence.

## Example

```python
import torch
from mr_nwm.foundation import LanguageBasedWorldModel
from mr_nwm.modalities import ModalityObservation
from mr_nwm.data import generate_batch

model = LanguageBasedWorldModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    modality_dimensions={"csi": 4, "iq": 2, "topology": 3},
    freeze=True,
).eval()
batch = generate_batch(1, steps=5)
inputs = model.tokenize(
    ["Prioritize throughput and energy."],
    modalities={
        "csi": ModalityObservation(
            values=torch.randn(1, 3, 4),
            age_seconds=torch.zeros(1, 3),
            period_seconds=0.001,
        ),
    },
)
with torch.inference_mode():
    packet = model.forward_slow_tier(inputs, batch.initial, batch.context)
    state, memory = batch.initial, None
    for action in batch.actions.unbind(1):
        state, hidden, memory = model.forward_fast_tier_step(state, action, packet, memory)
```

Model weights are downloaded separately and retain their upstream license. The
wireless adapters and heads in this example are newly initialized; outputs are
not trained forecasts. Freeze mode disables gradients into language-model
weights while preserving autograd through the trunk into input adapters. Full
language-model fine-tuning is enabled with `freeze=False` and requires an
appropriate optimizer and memory budget. PEFT/LoRA fine-tuning of the language
backbone itself is not implemented by the dynamic low-rank fast-head mechanism.

## Validation and remaining work

The offline contract tests cover shared physics weights, causal separation,
changes in physical evidence/modalities, graph adjacency and missing tokens,
input ages, token budgets, and input-adapter gradients through a frozen trunk.
`scripts/pretrained_demo.py` additionally attempts a real pretrained checkpoint,
records its resolved revision and source hashes, and audits gradient and
preference invariance. A completed integration record, if available, is stored
separately from the small-network benchmark.

The architecture still needs synchronized radio datasets, numeric-adapter
alignment, task training, realistic out-of-distribution evaluation, and target
hardware measurements. Integration success cannot establish RAN accuracy,
general intent understanding, or a real-time guarantee.

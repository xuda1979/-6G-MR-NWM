# Reproduced synthetic benchmark

Mean ± sample standard deviation across initialization seeds. Lower NMSE is better.

| Model | Parameters | IID NMSE (dB) | OOD NMSE (dB) |
|---|---:|---:|---:|
| MR-NWM shared | 28,031 | -22.24 ± 0.20 | -9.58 ± 0.41 |
| MR-NWM unshared | 32,815 | -21.99 ± 1.13 | -9.34 ± 0.86 |
| No modulation | 28,031 | -21.64 ± 0.15 | -9.47 ± 0.88 |
| Detached bridge | 32,815 | -21.63 ± 0.96 | -9.56 ± 0.99 |
| Residual MLP | 5,382 | -14.36 ± 0.07 | -6.33 ± 0.23 |
| Matched-size MLP | 28,806 | -14.92 ± 0.35 | -7.15 ± 0.70 |
| Standalone recurrent | 5,398 | -21.19 ± 0.30 | -9.43 ± 0.37 |
| Persistence | 0 | -3.67 ± 0.00 | -6.87 ± 0.00 |

## Intent and decision audit

TV is total variation distance; regret is within the finite candidate set.

| Model | Wording TV | Composition TV | IID regret | Action-effect NMSE (dB) |
|---|---:|---:|---:|---:|
| MR-NWM shared | 0.118 ± 0.040 | 0.195 ± 0.046 | 0.0055 ± 0.0022 | -11.75 ± 0.27 |
| MR-NWM unshared | 0.114 ± 0.040 | 0.211 ± 0.061 | 0.0052 ± 0.0028 | -11.39 ± 0.97 |
| No modulation | 0.114 ± 0.040 | 0.187 ± 0.053 | 0.0060 ± 0.0022 | -11.00 ± 0.40 |
| Detached bridge | 0.111 ± 0.040 | 0.206 ± 0.056 | 0.0054 ± 0.0024 | -10.96 ± 0.86 |

## Host-observed batch-one latency

Mean of per-run quantiles in milliseconds; device synchronized. Timing varies by host.

| Model | Slow p50 | Fast p50 | Fast p95 | Fast p99 |
|---|---:|---:|---:|---:|
| MR-NWM shared | 0.591 ± 0.006 | 0.081 ± 0.001 | 0.167 ± 0.046 | 0.291 ± 0.097 |
| MR-NWM unshared | 0.590 ± 0.007 | 0.081 ± 0.001 | 0.164 ± 0.034 | 0.309 ± 0.114 |
| No modulation | 0.599 ± 0.004 | 0.081 ± 0.001 | 0.123 ± 0.013 | 0.169 ± 0.054 |
| Detached bridge | 0.602 ± 0.019 | 0.081 ± 0.000 | 0.119 ± 0.012 | 0.164 ± 0.069 |

## Interpretation

All scores come from `metrics.json`. This two-link normalized simulator does not establish performance on channel traces, a 3GPP simulator, a RIC, or a DU. There is no pretrained language model in the experiment. Hard power-budget compliance follows from candidate filtering, not from learned safety.

The detached bridge must be compared with the unshared model to isolate cross-scale gradient feedback; comparing it only with the shared model confounds gradient flow and parameter sharing.

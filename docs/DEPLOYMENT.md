# Computational API and deployment requirements

The reference implementation trains a single PyTorch module and exposes slow
and fast forward calls. It does not implement a network service, E2 service
model, asynchronous scheduler, or real-time DU executable.

The slow call returns a tensor packet. A fast step needs the current state,
candidate action, cached packet and current recurrent memory. The caller must
retain the returned memory for the rest of that macro window. After a window,
the full forward method resamples hidden states into a next-window summary.

| Concern | Reference behavior | Requirement for a distributed implementation |
|---|---|---|
| Parameter sharing | Identical Python module object when enabled | Consistent versioned weight copies on each process/device |
| Packet lifetime | Caller-managed in memory | Sequence number, expiry, version and normalization checks |
| Timing | Repeated host-observed inference timings | Hardware-specific end-to-end deadline and load/tail tests |
| Data | Synthetic arrays and separate measured-trace forecasts | Target-calibrated telemetry, actuator records and missing-data handling |
| Failure | Ordinary exceptions | Local fallback for stale/missing context and failed inference |
| Safety | Explicit finite power-budget candidate filter | Application-specific hard constraints and independent validation |
| Interfaces | Python tensor API | Selected O-RAN service model and supported actuators |
| Privacy/security | No credentials or network configuration needed | Deployment-specific authentication and data governance |

O-RAN Near-RT RIC loops operate around 10 ms to 1 s, with faster loops residing
inside E2 nodes where supported. The architecture specification does not turn
Python fast-step measurements into a sub-millisecond control guarantee, and an
arbitrary modulation packet is not automatically a standardized E2 message.

The [public measured-trace study](PUBLIC_DATA.md) supplies observational forecasts
from historical throughput/delay, radio indicators and speed. Those source
records do not supervise power/PRB interventions. The next practical step is
calibrated control simulation or recorded-action evaluation, followed by a small
testbed. The project should establish action-ranking value,
multi-step calibration, and sensitivity to stale context before integration with
live network control.

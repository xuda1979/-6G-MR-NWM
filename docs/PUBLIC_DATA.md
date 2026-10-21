# Public measured-data training

This study trains five-step observational forecasts on three free public wireless datasets.
It complements the [archived synthetic action-conditioned benchmark](../results/SUMMARY.md).
The two studies have different targets, units and evaluation contracts; their scores must
be reported separately. The new [measured-data results](../results/public_data/SUMMARY.md)
are generated from retained predictions and experiment manifests.

## Sources and permissions

| Dataset / identifier | Official source and immutable identity | Download used / terms |
|---|---|---|
| UCC4G: `ucc4g` | Raca et al., *Beyond Throughput: a 4G LTE Dataset with Channel and Context Metrics*, MMSys 2018, [DOI 10.1145/3204949.3208123](https://doi.org/10.1145/3204949.3208123); [Zenodo 1219679 v1](https://doi.org/10.5281/zenodo.1219679) | [LTE_Dataset.zip](https://zenodo.org/api/records/1219679/files/LTE_Dataset.zip/content), 2,252,245 bytes, 135 CSV traces. Zenodo metadata specifies **CC BY 4.0**. The neighboring `ns3_dataset.zip` is simulated and excluded. |
| UCC5G: `ucc5g` | Raca et al., *Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics*, MMSys 2020, [DOI 10.1145/3339825.3394938](https://doi.org/10.1145/3339825.3394938); [official repository](https://github.com/uccmisl/5Gdataset/tree/306d92eba81cb4f1195d8b8eba625666a02a10ec) | [5G-production-dataset.zip](https://raw.githubusercontent.com/uccmisl/5Gdataset/306d92eba81cb4f1195d8b8eba625666a02a10ec/5G-production-dataset.zip), 2,177,094 bytes, 83 CSV traces. The repository supplies **GPL-3.0**; it does not separately identify a data-specific license. Preserve that distinction and upstream attribution. No simulation framework is used. |
| CICV5G: `cicv5g` | Zhang et al., *5G communication delay dataset for cloud-based vehicle planning and control*, Scientific Data 2026, [DOI 10.1038/s41597-026-07239-7](https://doi.org/10.1038/s41597-026-07239-7); [archived dataset DOI 10.5281/zenodo.17475688](https://doi.org/10.5281/zenodo.17475688); [source pin](https://github.com/zxr805/CICV5G/tree/0ad0b8307b918b22754ff651f750e70c7960abd2) | Download 90 per-run TXT files, totaling 19,889,556 bytes, from the pinned tree. The dataset is **CC BY 4.0**, stated separately from the article's publication terms. Aggregated `all.txt`, spreadsheets and figures are excluded. |

Source licenses govern the upstream data; this repository's MIT code license does not
relicense them. Raw measurements are downloaded to ignored `runs/public-data-cache/`,
not vendored. The 5G collection includes the network modes present in its measurements;
its name does not imply that every recorded sample uses a pure 5G radio path.

The [source catalog](../src/mr_nwm/public_sources.py) pins archive SHA-256 identities
and CICV file paths, Git blob identities and byte counts. Downloads are authenticated
before parsing; cached files are checked again on use. Each experiment's source manifest
also retains individual file SHA-256 values, parsing counts and transformations.

## Causal inputs and targets

Each sample has **eight historical observations**, four feature channels and an explicit
four-channel validity mask. The channels are:

| Channel | UCC4G / UCC5G | CICV5G |
|---|---|---|
| Past target | Downlink bitrate, kbps divided by 1,000 to obtain Mbps | Previously observed round-trip delay, ms |
| Radio strength | RSRP | RSRP |
| Radio quality | SNR | SINR |
| Mobility | Speed, km/h divided by 3.6 to obtain m/s | Velocity, already m/s |

Targets are the next **five observed downlink rates** in Mbps or **five round-trip delays**
in ms. Missing history features retain a false mask and zero placeholder. No value is
backfilled from a future row. The final historical target must be observed and nonnegative;
all five target values must be finite and nonnegative. Future receive timestamps,
post-reception vehicle coordinates, future radio readings and future target values are
excluded from inputs. Each input period uses only its eight historical timestamps.
Token ages approximate uniformly spaced history using that causal median period;
the full-trace median is used only for offline gap segmentation, never as a model input.

These traces have no supervised power/PRB intervention labels or operator-objective
targets. The world-model interface uses zero action placeholders and a fixed forecast
task description. Results support **observational forecasting** on these datasets;
they do not establish action effects, optimized RAN control, learned intent parsing,
closed-loop policy return or real-time deployment guarantees.

## Splits, cleaning and coverage

The [loader](../src/mr_nwm/public_data.py) sorts timestamps and retains the last logger row
at each repeated timestamp. It removes identical normalized trace aliases before assigning
splits. Traces sharing an exact timestamp-and-feature event form one connected group, so
partial aliases stay together. The canonical group name determines its split:
`SHA256("mr-nwm-public-v1:" + group_name) mod 100`, with buckets 0–69 for training,
70–79 for validation and 80–99 for testing. This is a deterministic approximately
70/10/20 **group** split, not a promised window-count ratio or a future-date benchmark.

Within each unique trace, a gap larger than 1.5 times the median positive sampling period
starts a new segment. Windows of eight past plus five future observations, stride five,
cannot cross segment or trace boundaries. Missing targets invalidate a candidate window.
The reported horizon is five observations; its duration follows each trace's measured
period and is not a common sub-millisecond wireless deadline.

Executed preflight counts for the pinned sources and this window contract:

| Dataset | Raw rows / published run files | Unique traces / overlap groups | Gap boundaries | Accepted windows: train / validation / test |
|---|---:|---:|---:|---:|
| UCC4G | 174,523 / 135 | 135 / 133 | 18,347 | 1,358 / 610 / 229 |
| UCC5G | 188,711 / 83 | 83 / 83 | 19,366 | 4,110 / 752 / 998 |
| CICV5G | 155,330 / 90 | 89 / 89 | 3,633 | 18,053 / 4,155 / 6,618 |

Raw row totals count the downloaded files before cleaning. CICV contains one exact
duplicate run. Some retained traces have no sufficiently long gap-free segment, so not
every source group contributes a training/evaluation window. This strict gap policy
substantially reduces coverage and excludes disconnected intervals from forecast windows;
performance on those excluded intervals remains unmeasured.

## Training and comparison

The learned language version uses **HuggingFaceTB/SmolLM2-135M**, frozen at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Numeric adapters, context projections,
modulation and fast dynamics parameters are task-trained around this frozen backbone.
Continuous telemetry and its mask enter the multimodal interface; this is not training
the language backbone itself or proving a general language-understanding gain.

Comparators are the trained numeric MR-NWM, last-observation persistence and ridge
regression. The same held-out sample IDs and physical-unit scoring apply to every model.
The scaler fits **observed training histories only**, applies `log1p` to the target
channel, and standardizes channels with training statistics. Held-out records never fit
the transform. Missingness remains explicit. Predictions are converted back to physical
units before evaluation.

Each learned model/seed receives 300 AdamW updates with batch size eight, sampling all
eligible training windows uniformly with replacement. This is a bounded optimization
budget, not an assertion of a complete dataset pass or convergence. Matching updates
does not match inference FLOPs or wall-clock cost between the two architectures.

Validation uses at most 256 windows selected by the smallest SHA-256 values of stable
sample IDs. The best checkpoint is selected by validation MAE; its test forecasts are
scored after selection. Testing uses the same deterministic rule with a cap of 512:
UCC4G has only 229 accepted test windows, so all 229 are used. UCC5G and CICV5G each use
512. The primary metric is MAE over all selected windows and five horizons, in Mbps or
ms. Learned-model summaries show mean and sample standard deviation across seeds 11,
22 and 33. Persistence and ridge are deterministic controls. Seed variation is not a
confidence interval over deployment sites or independent trajectories.

## Reproduction and result locations

Install the optional language-model dependencies, then run one command per dataset:

```bash
python -m pip install -e '.[foundation,dev,paper]'
python scripts/train_public.py --dataset ucc4g --output runs/public-results/ucc4g --updates 300 --seeds 11 22 33 --threads 2
python scripts/train_public.py --dataset ucc5g --output runs/public-results/ucc5g --updates 300 --seeds 11 22 33 --threads 2
python scripts/train_public.py --dataset cicv5g --output runs/public-results/cicv5g --updates 300 --seeds 11 22 33 --threads 2
```

The runner records source hashes, resolved settings, data manifests, environment,
checkpoint identities and prediction files. Use a new output directory for a changed
measurement contract; `--resume` requires matching experiment identity and artifacts.
The complete language checkpoint is downloaded separately; saved adapters require the
named pinned backbone. Published experiment manifests are under
[`results/public_data/`](../results/public_data/), with each dataset's `metrics.json`
and the [combined result summary](../results/public_data/SUMMARY.md). Read those executed
records for scores; no prospective score is supplied by this guide.

The original [synthetic metrics](../results/metrics.json), Qwen integration-only record
and dated review/checkpoint audit files are historical evidence under their original
contracts. Their numeric values and dates are preserved. Current result descriptions
must identify the measured-data study explicitly rather than attributing its evidence
to the earlier synthetic or integration-only runs.

## 中文摘要

三套免费公开实测数据分别覆盖 4G／5G 吞吐与信道、车联网通信时延。输入仅使用八步历史
目标、RSRP、SNR／SINR、速度及缺失掩码，预测后五步；按完整运行组划分，并隔离重复、
重叠及大时间间隙。训练只拟合训练集归一化参数，比较冻结语言骨干周围的可训练模块、
数值模型及简单基线。公开结果报告实际单位误差与三种子波动；动作控制能力仍需带有
有效干预记录的数据及后续闭环实验验证。

# Source data and model notices

Code in this repository retains its existing license. Upstream measurements and
pretrained weights retain their separate terms and attribution.

- **UCC4G:** Raca, Quinlan, Zahran and Sreenan, *Beyond Throughput: a 4G LTE Dataset
  with Channel and Context Metrics*, MMSys 2018; dataset
  [Zenodo 1219679](https://doi.org/10.5281/zenodo.1219679), CC BY 4.0.
- **UCC5G:** Raca, Leahy, Sreenan and Quinlan, *Beyond Throughput, The Next
  Generation: A 5G Dataset with Channel and Context Metrics*, MMSys 2020;
  [official source](https://github.com/uccmisl/5Gdataset/tree/306d92eba81cb4f1195d8b8eba625666a02a10ec)
  supplies GPL-3.0 at repository level, without a separately identified data
  license. Preserve the source's terms for source-derived records.
- **CICV5G:** Zhang et al., *5G communication delay dataset for cloud-based vehicle
  planning and control*, Scientific Data 2026; dataset
  [Zenodo 17475688](https://doi.org/10.5281/zenodo.17475688), CC BY 4.0 for data.
- **SmolLM2-135M:** Hugging Face's
  [model card at the pinned revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2).
  The pretrained trunk is downloaded separately and is not included in adapter
  checkpoints. Reconstruct it with the exact recorded model revision.

`*_predictions.npz` retains transformed/source-derived test target values and
stable source IDs alongside predictions. It omits GPS coordinates, original
vehicle trajectories, identifiers and raw signal records. Dataset attribution,
terms, transformations and exact source file identities are recorded in each
`metrics.json` and [the protocol](../../docs/PUBLIC_DATA.md). Scores and plots
describe this forecast task, not results endorsed by the dataset creators.

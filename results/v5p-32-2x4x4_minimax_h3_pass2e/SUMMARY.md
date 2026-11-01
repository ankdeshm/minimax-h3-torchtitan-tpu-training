# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v5p-32` Results

- **Updated (UTC):** `2026-10-31T16:21:50.394553+00:00`
- **Cluster / Nodepool:** `tpu-v5p-32-pool` (`2x4x4`, `32` TPU `v5p` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | **24.188s** | **677.2** | `21,670` | `10.46` | **27.48%** | **27.48%** | `3.6320 -> 3.8916` | `OK` |

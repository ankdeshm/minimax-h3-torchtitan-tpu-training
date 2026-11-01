# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v5p-32` Results

- **Updated (UTC):** `2026-10-31T10:10:01.518281+00:00`
- **Cluster / Nodepool:** `tpu-v5p-32-pool` (`2x4x4`, `32` TPU `v5p` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | **24.888s** | **658.4** | `21,069` | `10.46` | **26.71%** | **26.71%** | `3.6320 -> 3.8829` | `OK` |
| 2 | `mh3_p2_02_fsdp32_splash_save_ac_s16k_pretrained` | `FSDP-32` | `SplashSaveAC (calibrated)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 3 | `mh3_p2_03_fsdp32_splash_save_sa1024_xla_lhs_s16k` | `FSDP-32` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`--- | Root Cause (first observed failure): | [0]: |   time  `) |
| 4 | `mh3_p2_04_fsdp32_splash_save_sa1024_xla_lhs_s16k_ga2` | `FSDP-32` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=-9` (` | W1031 10:00:58.456000 20969 site-packages/torch/distribut`) |

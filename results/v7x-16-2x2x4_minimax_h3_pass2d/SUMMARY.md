# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v7x-16` Results

- **Updated (UTC):** `2026-10-31T14:40:37.532435+00:00`
- **Cluster / Nodepool:** `tpu-v7x-16-pool` (`2x2x4`, `32` TPU `v7x` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 2 | `mh3_p2_02_fsdp32_splash_save_ac_s16k_pretrained` | `FSDP-32` | `SplashSaveAC (calibrated)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 3 | `mh3_p2_03_fsdp32_xla_lhs_bf16red_s16k` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`--------- | +-----------+ | | Anomalies | | +-----------+ | `) |
| 4 | `mh3_p2_04_fsdp32_xla_lhs_s16k_ga2` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`pu/_internal/utils/annotations.py", line 272, in wrapper |  `) |
| 5 | `mh3_p2_05_hsdp_2x16_xla_lhs_s16k` | `HSDP 2x16` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`+-----------+ | | Anomalies | | +-----------+ | --------- | `) |
| 6 | `mh3_p2_06_fsdp32_fp8_linear_quant_s16k` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `TpuFloat8Linear (rowwise FP8)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 7 | `mh3_p2_07_fsdp32_s32k_longvideo_5s_768p_16x9` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`1322 | num_cache_hits=1167 {88.2%} | num_compilations=12 | p`) |

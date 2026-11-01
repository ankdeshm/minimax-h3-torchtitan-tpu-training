# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v5p-32` Results

- **Updated (UTC):** `2026-10-31T11:05:39.417862+00:00`
- **Cluster / Nodepool:** `tpu-v5p-32-pool` (`2x4x4`, `32` TPU `v5p` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | **24.884s** | **658.4** | `21,069` | `10.46` | **26.72%** | **26.71%** | `3.6320 -> 3.2877` | `OK` |
| 2 | `mh3_p2_02_fsdp32_splash_save_ac_s16k_pretrained` | `FSDP-32` | `SplashSaveAC (calibrated)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 3 | `mh3_p2_03_fsdp32_splash_save_xla_lhs_bf16red_s16k` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`2020 | num_cache_hits=1834 {90.7%} | num_compilations=82 | p`) |
| 4 | `mh3_p2_04_fsdp32_splash_save_xla_lhs_s16k_ga2` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`s=3290 | num_cache_hits=3090 {93.9%} | num_compilations=0 | `) |
| 5 | `mh3_p2_05_hsdp_2x16_splash_save_xla_lhs_s16k` | `HSDP 2x16` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`2143 | num_cache_hits=1945 {90.7%} | num_compilations=57 | p`) |
| 6 | `mh3_p2_06_fsdp32_fp8_linear_quant_s16k` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `TpuFloat8Linear (rowwise FP8)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`                    ^^^^^^^^^^^^^^^^^^^^^ |     File "/usr/l`) |
| 7 | `mh3_p2_07_fsdp32_splash_save_s32k_longvideo_5s_768p_16x9` | `FSDP-32` | `SplashSaveAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`1324 | num_cache_hits=1167 {88.1%} | num_compilations=18 | p`) |

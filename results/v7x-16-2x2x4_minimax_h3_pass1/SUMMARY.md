# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v7x-16` Results

- **Updated (UTC):** `2026-10-31T06:21:56.568400+00:00`
- **Cluster / Nodepool:** `tpu-v7x-16-pool` (`2x2x4`, `32` TPU `v7x` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p1_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 2 | `mh3_p1_02_fsdp32_splash_save_ac_s16k_pretrained` | `FSDP-32` | `SplashSaveAC (32/50 blocks, 64% attn saved)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`pu/_internal/utils/annotations.py", line 272, in wrapper |  `) |
| 3 | `mh3_p1_03_fsdp32_splash_save_sa1024_xla_lhs_s16k` | `FSDP-32` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`lation cache evicted. | I0000 00:00:1793426906.020286   2959`) |
| 4 | `mh3_p1_04_fsdp32_splash_save_sa1024_xla_lhs_s16k_ga2` | `FSDP-32` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 5 | `mh3_p1_05_hsdp_2x16_splash_save_sa1024_xla_lhs_s16k` | `HSDP 2x16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`2195 | num_cache_hits=1983 {90.3%} | num_compilations=41 | p`) |
| 6 | `mh3_p1_06_fsdp32_fp8_linear_quant_s16k` | `FSDP-32` | `SplashSaveAC` | `1024` | `TpuFloat8Linear (tensorwise FP8)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`ax_h3_torchtitan.py", line 761, in parallelize_minimax_h3 | `) |
| 7 | `mh3_p1_07_fsdp32_splash_save_s32k_longvideo_5s_768p_16x9` | `FSDP-32` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | - | - | - | - | - | - | - | `rc=1` (`pu/_internal/utils/annotations.py", line 272, in wrapper |  `) |

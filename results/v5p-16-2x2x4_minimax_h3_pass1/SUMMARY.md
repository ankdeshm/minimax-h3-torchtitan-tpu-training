# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v5p-16` Results

- **Updated (UTC):** `2026-10-31T06:40:45.846713+00:00`
- **Cluster / Nodepool:** `tpu-v5p-16-pool` (`2x2x4`, `16` TPU `v5p` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p1_01_fsdp16_fullac_s16k_pretrained` | `FSDP-16` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`ailed with: e0101: RuntimeProgramAllocationFailure: |   Erro`) |
| 2 | `mh3_p1_02_fsdp16_splash_save_ac_s16k_pretrained` | `FSDP-16` | `SplashSaveAC (32/50 blocks, 64% attn saved)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`pu/_internal/utils/annotations.py", line 272, in wrapper |  `) |
| 3 | `mh3_p1_03_fsdp16_splash_save_sa1024_xla_lhs_s16k` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`072 | num_cache_hits=1872 {90.3%} | num_compilations=58 | pe`) |
| 4 | `mh3_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s16k_ga2` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=1` (`er failed with: e0101: RuntimeProgramAllocationFailure: |   `) |
| 5 | `mh3_p1_05_hsdp_2x8_splash_save_sa1024_xla_lhs_s16k` | `HSDP 2x8` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`ing_emitter.cc:5733 HLO: as_strided.12; HLO computation: mai`) |
| 6 | `mh3_p1_06_fsdp16_fp8_linear_quant_s16k` | `FSDP-16` | `SplashSaveAC` | `1024` | `TpuFloat8Linear (tensorwise FP8)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (` | W1031 06:22:28.048000 34508 site-packages/torch/distribut`) |
| 7 | `mh3_p1_07_fsdp16_splash_save_s32k_longvideo_5s_768p_16x9` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | - | - | - | - | - | - | - | `rc=-9` (` | W1031 06:31:42.698000 34575 site-packages/torch/distribut`) |

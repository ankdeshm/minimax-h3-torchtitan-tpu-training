# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v5p-16` Results

- **Updated (UTC):** `2026-10-31T10:28:54.496932+00:00`
- **Cluster / Nodepool:** `tpu-v5p-16-pool` (`2x2x4`, `16` TPU `v5p` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp16_fullac_s16k_pretrained` | `FSDP-16` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`ailed with: e0101: RuntimeProgramAllocationFailure: |   Erro`) |
| 2 | `mh3_p2_02_fsdp16_splash_save_ac_s16k_pretrained` | `FSDP-16` | `SplashSaveAC (calibrated)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`er failed with: e0101: RuntimeProgramAllocationFailure: |   `) |

# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v7x-16` Results

- **Updated (UTC):** `2026-10-31T16:30:19.736855+00:00`
- **Cluster / Nodepool:** `tpu-v7x-16-pool` (`2x2x4`, `32` TPU `v7x` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 2 | `mh3_p2_02_fsdp32_splash_save_ac_s16k_pretrained` | `FSDP-32` | `SplashSaveAC (calibrated)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=-9` (`lation cache evicted. | I0000 00:00:1793464215.946486   1693`) |

# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `v7x-16` Results

- **Updated (UTC):** `2026-10-31T17:12:41.988257+00:00`
- **Cluster / Nodepool:** `tpu-v7x-16-pool` (`2x2x4`, `32` TPU `v7x` ranks)
- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)
- **Checkpoint:** `/models/models/MiniMaxAI/MiniMax-H3/transformer` (`66.28 GB`, 638 tensors across 14 safetensors shards)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `mh3_p2_01_fsdp32_fullac_s16k_pretrained` | `FSDP-32` | `FullAC (recompute attn)` | `512` | `BF16 + FP32 heads` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 2 | `mh3_p2_02_fsdp32_fullac_sa1024_s16k_pretrained` | `FSDP-32` | `FullAC (sa_block=1024)` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 3 | `mh3_p2_03_fsdp32_xla_lhs_bf16red_s16k` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (` |       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^ |     File "/usr/lo`) |
| 4 | `mh3_p2_04_fsdp32_xla_lhs_s16k_ga2` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`ng signal 15 with default behavior | E1031 16:59:58.662307  `) |
| 5 | `mh3_p2_05_hsdp_2x16_xla_lhs_s16k` | `HSDP 2x16` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`er failed with: e0101: RuntimeProgramAllocationFailure: |   `) |
| 6 | `mh3_p2_06_fsdp32_fp8_linear_quant_s16k` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `TpuFloat8Linear (rowwise FP8)` | `16,384` | `1x` | - | - | - | - | - | - | - | `rc=1` (`bserved failure): | [0]: |   time      : 2026-10-31_17:11:46`) |
| 7 | `mh3_p2_07_fsdp32_s32k_longvideo_5s_768p_16x9` | `FSDP-32` | `FullAC + XLA LHS` | `512` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | - | - | - | - | - | - | - | `rc=-9` (`lib/python3.14/multiprocessing/resource_tracker.py:475: User`) |

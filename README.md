# `MiniMax-H3` (`33.14B` DiT) TorchTitan Training on TPU `v5p`

**PyTorch [`TorchTitan`](https://github.com/pytorch/torchtitan) + `TorchTPU 0.2.0` GA** architecture, distributed Rectified-Flow pre-training, Pallas `SplashAttention(is_causal=False)` + Activation Checkpointing, and `FSDP-32` vs `FSDP-16` scaling results for **[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)** (`33,139,340,704` parameters, `66.28 GB` across `14` safetensors shards, `638 / 638` keys) on **Cloud TPU `v5p-32` (`2x4x4`, `32` chips, `3,040 GiB` HBM)** and **`v5p-16` (`2x2x4`, `16` chips)**.

- **Directory Guide:** [`results/README.md`](./results/README.md) *(maps each pass folder to its configuration and outcome)*

---

## Key Results & Script Reference

| Run | Parallelism & Slice | Steady Step (`Tok/s`) | **MFU (%)** | Peak HBM | Script & Line Anchor | K8s Manifest | Result Files | Notes |
|---|---|---|---:|---:|---|---|---|---|
| **Optimized `FSDP-32` (`Pass 2E`)** | **`FSDP-32` on `v5p-32` (`2x4x4`, `S=16,384`)** | **`24.188s`** (`677.2`/rk · `21,670` tot) | **`27.48%`** | **`10.46 GiB`** | [`run_minimax_h3_v5p16_sweep.py#L209`](./scripts/run_minimax_h3_v5p16_sweep.py#L209-L227) · [`minimax_h3_torchtitan.py#L524`](./scripts/minimax_h3_torchtitan.py#L524-L565) | [`v5p-32-minimax-h3-pass2.yaml`](./k8s/v5p-32-minimax-h3-pass2.yaml) | [`SUMMARY.md`](./results/v5p-32-2x4x4_minimax_h3_pass2e/SUMMARY.md) · [`JSON`](./results/v5p-32-2x4x4_minimax_h3_pass2e/summary.json) · [`Log`](./results/v5p-32-2x4x4_minimax_h3_pass2e/mh3_p2_01_fsdp32_fullac_s16k_pretrained.log) | Sharding across `32` `v5p` chips (`FSDP-32`) + contiguous 3-modality AdaLN slice modulation ([`#L432`](./scripts/minimax_h3_torchtitan.py#L432-L460)) avoids XLA `RuntimeProgramAllocationFailure` seen on `FSDP-16` |
| **Multi-Step Loss Convergence (`Pass 2C`)** | **`FSDP-32` on `v5p-32` (`2x4x4`, `S=16,384`)** | **`24.884s`** (`658.4`/rk · `21,069` tot) | **`26.72%`** | **`10.46 GiB`** | [`run_minimax_h3_v5p16_sweep.py#L209`](./scripts/run_minimax_h3_v5p16_sweep.py#L209-L227) · [`parallelize_minimax_h3#L749`](./scripts/minimax_h3_torchtitan.py#L749-L810) | [`v5p-32-minimax-h3-pass2.yaml`](./k8s/v5p-32-minimax-h3-pass2.yaml) | [`SUMMARY.md`](./results/v5p-32-2x4x4_minimax_h3_pass2c/SUMMARY.md) · [`JSON`](./results/v5p-32-2x4x4_minimax_h3_pass2c/summary.json) · [`Log`](./results/v5p-32-2x4x4_minimax_h3_pass2c/mh3_p2_01_fsdp32_fullac_s16k_pretrained.log) | Verified Rectified-Flow loss decrease **`3.6320 -> 3.2877`** loading all `638 / 638` pretrained tensors ([`#L883`](./scripts/minimax_h3_torchtitan.py#L883-L950)) |
| **Initial `FSDP-32` Bring-Up (`Pass 2`)** | **`FSDP-32` on `v5p-32` (`2x4x4`, `S=16,384`)** | **`24.552s`** (`667.4`/rk · `21,357` tot) | **`27.08%`** | **`10.46 GiB`** | [`run_minimax_h3_v5p16_sweep.py#L209`](./scripts/run_minimax_h3_v5p16_sweep.py#L209-L227) | [`v5p-32-minimax-h3-pass2.yaml`](./k8s/v5p-32-minimax-h3-pass2.yaml) | [`SUMMARY.md`](./results/v5p-32-2x4x4_minimax_h3_pass2/SUMMARY.md) · [`JSON`](./results/v5p-32-2x4x4_minimax_h3_pass2/summary.json) · [`Log`](./results/v5p-32-2x4x4_minimax_h3_pass2/mh3_p2_01_fsdp32_fullac_s16k_pretrained.log) | Initial end-to-end `33.14B` omni-modal DiT training step on `v5p-32` (`524,288` omni tokens/step) |
| **`v5p-16` `FSDP-16` Memory Boundary (`Pass 2F`)** | `FSDP-16` on `v5p-16` (`2x2x4`, `S=16,384`) | XLA Compile HBM OOM | — | `>95 GiB` | [`run_minimax_h3_v5p16_sweep.py#L209`](./scripts/run_minimax_h3_v5p16_sweep.py#L209-L227) | [`v5p-16-minimax-h3-pass2.yaml`](./k8s/v5p-16-minimax-h3-pass2.yaml) | [`SUMMARY.md`](./results/v5p-16-2x2x4_minimax_h3_pass2f/SUMMARY.md) · [`JSON`](./results/v5p-16-2x2x4_minimax_h3_pass2f/summary.json) | `33.14B` (`19.27B` trunk + `13.05B` `96,768`-dim AdaLN) + XLA static buffers exceed `FSDP-16` on `95 GiB` `v5p`; resolved by `FSDP-32` (`10.46 GiB` live HBM) |

---

## 1. Architecture & TPU Alignment (`638 / 638` Safetensors Keys)

| Dimension | `MiniMaxAI/MiniMax-H3` Specification | TorchTitan + Cloud TPU Implementation (Code Anchor) |
|---|---|---|
| **Architecture (`MiniMaxH3Model`)** | Unified Single-Stream Omni-Modal DiT (`50` main blocks + `2` `TokenRefiner` blocks); **`19.27B` active per-token trunk** + **`13.05B` per-sample modality-specific AdaLN** (`adaln_proj`: `2688 -> 96768 = 3 × 6 × 5376`) | **`128×128` MXU tile alignment** in [`MiniMaxH3Model` (`#L524`)](./scripts/minimax_h3_torchtitan.py#L524-L565): `hidden_size = 5,376` (`42×128`), `56` heads × `head_dim = 128`, SwiGLU `ffn_dim = 14,336` (`112×128`, [`#L252`](./scripts/minimax_h3_torchtitan.py#L252-L295)). |
| **Attention Mechanism** | **Zero cross-attention** — unified 1D packed bidirectional full self-attention (`is_causal=False`) over `[text, audio, video]` + per-head `RMSNorm(128)` QK-Norm + 3D Axial RoPE `(64, 32, 32)` | Maps to TorchTitan's Pallas **`SplashAttention(is_causal=False)`** in [`MiniMaxH3Attention` (`#L298`)](./scripts/minimax_h3_torchtitan.py#L298-L352), with [`_full_ac_save_splash_policy` (`#L743`)](./scripts/minimax_h3_torchtitan.py#L743-L748) and `FullAC` per-block rematerialization ([`#L749`](./scripts/minimax_h3_torchtitan.py#L749-L810)). |
| **Scatter/Gather Elimination** | Reference `diffusers` forward uses `index_copy` + `300×` `[S, 5376]` `index_select` gathers per step for modality AdaLN | [`_modulate_packed_sequence` (`#L432`)](./scripts/minimax_h3_torchtitan.py#L432-L460) packs `[text, audio, video]` contiguously and applies broadcast slice modulation with zero XLA `DynamicSlice`/`Scatter`/`Gather` HLO ops. |
| **Sharded Checkpoint Streaming** | `14` safetensors shards (`66.28 GB` BF16 + FP32 input/output/time heads) | [`load_pretrained_minimax_h3_safetensors` (`#L883`)](./scripts/minimax_h3_torchtitan.py#L883-L950) streams all `638` tensors into `FSDP2` `DTensor` shards (`10.46 GiB` live HBM/chip on `v5p-32`). |

---

## 2. Scaling Summary (`v5p-16` OOM to `27.48%` MFU on `v5p-32`)

1. **`FSDP-16` XLA Static Allocation Limit (`Pass 1` / `Pass 2F` on `v5p-16`):**
   - Although sharded persistent state under `FSDP-16` is `~33.3 GiB/chip`, XLA's backward all-gather + `96,768`-dim AdaLN modulation gradient buffers trigger `e0101: RuntimeProgramAllocationFailure` on 16 `95 GiB` `v5p` chips ([`results/v5p-16-2x2x4_minimax_h3_pass2f/SUMMARY.md`](./results/v5p-16-2x2x4_minimax_h3_pass2f/SUMMARY.md)).
2. **Scaling to `FSDP-32` (`2x4x4` `v5p-32`) + `FullAC` + Pallas `SplashAttention(sa=512)` (`Pass 2` to `Pass 2E`):**
   - Increasing the shard degree to `FSDP-32` ([`k8s/v5p-32-minimax-h3-pass2.yaml`](./k8s/v5p-32-minimax-h3-pass2.yaml)) reduces live per-chip HBM to **`10.46 GiB`**, converges from pretrained weights (`loss: 3.6320 -> 3.2877` in [`Pass 2C`](./results/v5p-32-2x4x4_minimax_h3_pass2c/SUMMARY.md)), and reaches **`27.48%` MFU (`677.2` tok/s/rank = `21,670` cluster omni tok/s, `24.188s/step`)** in [`Pass 2E`](./results/v5p-32-2x4x4_minimax_h3_pass2e/SUMMARY.md).

---

## 3. Repository Structure & Quickstart

- [`docker/Dockerfile`](./docker/Dockerfile) — Python 3.14 + PyTorch 2.15 + TorchTPU `0.2.0` GA + TorchTitan image
- [`scripts/minimax_h3_torchtitan.py`](./scripts/minimax_h3_torchtitan.py) — Native TorchTitan `MiniMaxH3Model` (`638/638` HF keys) + `MiniMaxH3Trainer`
- [`scripts/run_minimax_h3_v5p16_sweep.py`](./scripts/run_minimax_h3_v5p16_sweep.py) — Parameterized multi-host `v5p-32` / `v5p-16` / `v7x-16` TorchTitan sweep runner
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS safetensors stager
- [`k8s/v5p-32-minimax-h3-pass2.yaml`](./k8s/v5p-32-minimax-h3-pass2.yaml) — 8-host (`32`-chip `v5p-32` `2x4x4`) Indexed Job manifest (`27.48%` MFU)
- [`k8s/v5p-16-minimax-h3-pass2.yaml`](./k8s/v5p-16-minimax-h3-pass2.yaml) & [`k8s/v7x-16-minimax-h3-pass2.yaml`](./k8s/v7x-16-minimax-h3-pass2.yaml) — `v5p-16` and `v7x-16` sweep manifests
- [`results/`](./results/README.md) — `SUMMARY.md`, `summary.json`, and per-experiment `.log` outputs

```bash
# 1. Stage MiniMaxAI/MiniMax-H3 (14 transformer shards + VAE + audio VAE, 77.32 GB) to GCS
kubectl apply -f k8s/stage-minimax-h3-job.yaml

# 2. Create ConfigMap & Launch 8-Host (32-Chip v5p-32 2x4x4) TorchTitan Sweep
kubectl create configmap minimax-h3-script   --from-file=minimax_h3_torchtitan.py=scripts/minimax_h3_torchtitan.py   --from-file=run_minimax_h3_v5p16_sweep.py=scripts/run_minimax_h3_v5p16_sweep.py
kubectl apply -f k8s/v5p-32-minimax-h3-pass2.yaml
```

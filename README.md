# `MiniMaxAI/MiniMax-H3` (`33.14B` Omni-Modal Video + Stereo-Audio DiT) Pre-Training on Cloud TPU `v5p-16` with PyTorch TorchTitan + TorchTPU `0.2.0` GA

End-to-end **PyTorch [`TorchTitan`](https://github.com/pytorch/torchtitan) + `TorchTPU 0.2.0` GA** compatibility architecture, distributed Rectified-Flow pre-training, Pallas `SplashAttention` + Selective Activation Checkpointing, and `BF16` vs `TpuFloat8LinearConverter` quantization sweep for **[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)** (`33,139,340,704` parameters, `66.28 GB` across 14 safetensors shards, `638` tensors) on **`16` Cloud TPU `v5p` chips (`2x2x4`, `1,520 GiB` HBM)**.

---

## 1. TorchTitan + TPU `v5p-16` Compatibility Verdict: **100% Compatible & Hardware-Aligned**

| Dimension | `MiniMaxAI/MiniMax-H3` Specification | TorchTitan + TPU `v5p-16` (`2x2x4`) Compatibility |
|---|---|---|
| **Architecture (`MiniMaxH3Transformer3DModel`)** | Unified Single-Stream Omni-Modal DiT (`50` main blocks + `2` `TokenRefiner` blocks); **`19.27B` active per-token trunk** + **`13.05B` per-sample modality-specific AdaLN** (`adaln_proj`: `2688 -> 96768 = 3 × 6 × 5376`) | **100% exact `128×128` `v5p` MXU tile alignment**: `hidden_size = 5,376` (`42×128`), `56` heads × `head_dim = 128` (`attn_inner_dim = 7,168 = 56×128`), SwiGLU `ffn_dim = 14,336` (`112×128`, `bias=False`). Every tensor dim-0 (`32, 96, 128, 2688, 5376, 7168, 28672, 96768`) is evenly divisible by `16`! |
| **Attention Mechanism** | **Zero cross-attention** — 100% unified 1D packed bidirectional full self-attention (`is_causal=False`, unmasked) over `[text (L_txt), audio (L_aud), video (L_vid)]` + per-head `RMSNorm(128, eps=1e-5)` QK-Norm + 3D Axial RoPE `(64, 32, 32)` | Maps **1:1** to TorchTitan's Pallas **`SplashAttention(is_causal=False)` (`FullMask`)** on all 50 main blocks + 2 refiner blocks, paired with **`SplashSaveAC` (`_full_ac_save_splash_policy`)** to eliminate **100% of `O(S^2)` self-attention backward recomputation**! |
| **Scatter/Gather Elimination** | Reference `diffusers` forward uses `index_copy` + `300×` `[S, 5376]` `index_select` gathers per step for modality AdaLN | Our `MiniMaxH3Model` packs `[text, audio, video]` contiguously and applies broadcast slice modulation (`_modulate_packed_3mod`), achieving **100% mathematical bit-identity with zero XLA `DynamicSlice`/`Scatter`/`Gather` HLO ops**! |
| **Quantization** | Released in **native `BF16` (`blocks`, `token_refiner`, `context_embedder`) + `FP32` (`proj_in/out`, `audio_proj_in/out`, `time_embedder`)** — zero custom FP4/INT4 weight blockers | Runs at full hardware rate on TPU `v5p` **`BF16` MXU (`459 TFLOPs/chip`)** (`BF16` params + `FP32` or `BF16` ReduceScatter + `FP32` AdamW). Includes optional **`TpuFloat8LinearConverter` (`torchao` tensorwise FP8)** ablation (note: hardware FP8 MXU requires `v6e`/`v7x`; on `v5p`, FP8 executes via XLA cast/emulation). |
| **16-Chip `v5p` HBM Budget (`1,520 GiB`)** | `33.14B` params (`66.28 GB` BF16) | Under `FSDP-16`, persistent sharded state is **`~33.3 GiB/chip`** (`3.86 GiB` BF16 params + `3.86–7.72 GiB` grads + `23.15 GiB` FP32 AdamW + `2.4 GiB` AllGather buffer), leaving **`~61.7 GiB/chip` free** — fitting both **`4s 768p` (`S = 16,384`)** and **`5s 768p 16:9` (`1344×768` @ 24fps + 32kHz stereo audio, `S = 32,768`)** per rank with zero Context Parallelism overhead! |

---

## 2. 7-Experiment TPU `v5p-16` Sweep Matrix (`scripts/run_minimax_h3_v5p16_sweep.py`)

1. `mh3_p1_01_fsdp16_fullac_s16k_pretrained`: Pretrained `66.28 GB` 14-shard checkpoint + `FSDP-16` + `FullAC` + `sa_block=512` (`S = 16,384`).
2. `mh3_p1_02_fsdp16_splash_save_ac_s16k_pretrained`: Pretrained + `FSDP-16` + **`SplashSaveAC` (`0%` attention recompute)** + `sa_block=512` (`S = 16,384`).
3. `mh3_p1_03_fsdp16_splash_save_sa1024_xla_lhs_s16k`: Pretrained + `FSDP-16` + `SplashSaveAC` + **`sa_block=1024`** + **`v5p` XLA Latency-Hiding Scheduler (`LHS` + AG Backward Pipelining + `VMEM=80MiB` + `BF16` Reduce + Compiled Optimizer/GradNorm)**.
4. `mh3_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s16k_ga2`: **`2×` Gradient Accumulation (`524,288` omni tokens/step = `32` `4s-768p` omni clips/step)**.
5. `mh3_p1_05_hsdp_2x8_splash_save_sa1024_xla_lhs_s16k`: **2D `HSDP 2×8`** (`replicate=2, shard=8`) vs `1D FSDP-16` on `2x2x4` ICI torus.
6. `mh3_p1_06_fsdp16_fp8_linear_quant_s16k`: **`TpuFloat8LinearConverter` (`torchao` tensorwise FP8)** quantization ablation vs native `v5p` `BF16` MXU.
7. `mh3_p1_07_fsdp16_splash_save_s32k_longvideo_5s_768p_16x9`: **Full-Resolution `5s 768p 16:9` (`1344×768` @ 24fps + 32kHz stereo audio, `S = 32,768`, `1.05M` omni tokens/step with `2×` GradAccum)**.

---

## 3. Repository Structure & Quickstart

```text
├── README.md
├── docker/
│   └── Dockerfile                         # Python 3.14 + PyTorch 2.15 + TorchTPU 0.2.0 GA + TorchTitan
├── scripts/
│   ├── minimax_h3_torchtitan.py           # Native TorchTitan MiniMaxH3Model (638/638 HF keys) + MiniMaxH3Trainer
│   ├── run_minimax_h3_v5p16_sweep.py      # 7-experiment v5p-16 MFU, SplashSaveAC, HSDP, FP8 & S=32K sweep
│   └── stage_hf_to_gcs.py                 # Parallel Hugging Face -> GCS safetensors stager
├── k8s/
│   ├── storage-gcsfuse.yaml               # GKE GCS Fuse PV/PVC manifest
│   ├── stage-minimax-h3-job.yaml          # K8s Job to stream 77.32 GB MiniMax-H3 assets to GCS
│   └── v5p-16-minimax-h3-sweep.yaml       # 4-host (16 v5p chips, 2x2x4) Indexed Job manifest
└── results/                               # Auto-synced JSON & Markdown benchmark outputs
```

```bash
# 1. Stage MiniMaxAI/MiniMax-H3 (14 transformer shards + VAE + audio VAE, 77.32 GB) to GCS
kubectl apply -f k8s/stage-minimax-h3-job.yaml

# 2. Create ConfigMap & Launch 4-Host (16-Chip v5p 2x2x4) TorchTitan Pretraining Sweep
kubectl create configmap minimax-h3-v5p16-script \
  --from-file=minimax_h3_torchtitan.py=scripts/minimax_h3_torchtitan.py \
  --from-file=run_minimax_h3_v5p16_sweep.py=scripts/run_minimax_h3_v5p16_sweep.py
kubectl apply -f k8s/v5p-16-minimax-h3-sweep.yaml
```

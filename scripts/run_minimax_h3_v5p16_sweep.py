#!/usr/bin/env python3
"""TPU v5p-16 (2x2x4, 16 chips) TorchTitan + TorchTPU 0.2.0 GA Pretraining Sweep for MiniMax-H3 (33.14B Omni DiT).

Runs on idle warm nodepool `tpu-v5p-16-pool` (only when zero Flint workloads are active on that pool).
Evaluates:
  1. Pretrained `MiniMaxAI/MiniMax-H3` (66.28 GB, 638 keys across 14 safetensors shards) streaming DTensor load
  2. `FullAC` vs `SplashSaveAC` (`_full_ac_save_splash_policy` eliminating 100% of O(S^2) self-attn recompute)
  3. Bidirectional Pallas `SplashAttention(is_causal=False)` block size (`512` vs `1024`) + `v5p` XLA LHS flags
  4. `2x` Gradient Accumulation (`524K` omni tokens/step) + `BF16` ReduceScatter + Compiled Optimizer/GradNorm
  5. `2D HSDP (2x8)` vs `1D FSDP-16` on `2x2x4` ICI torus
  6. `TpuFloat8LinearConverter` (`torchao` tensorwise FP8) quantization ablation vs native `v5p` `BF16` MXU
  7. Long-context `S=32,768` (`5s 768p 16:9` 24fps video + 32kHz stereo audio + text, zero CP overhead!)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

PEAK_BF16_TFLOPS_PER_CHIP = {"v5p": 459.0, "v6e": 918.0, "v7x": 1153.5}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?grad_norm:\s*(?P<gn>[0-9.]+)"
    r".*?memory:\s*(?P<mem>[0-9.]+)GiB.*?"
    r"tps:\s*(?P<tps>[0-9,]+).*?tflops:\s*(?P<tflops>[0-9.]+).*?mfu:\s*(?P<mfu>[0-9.]+)%",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"time_end_to_end:\s*(?P<sec>[0-9.]+)s", re.IGNORECASE)


def filter_supported_xla_flags(candidates: list[str]) -> str:
    """Validates candidate `--xla_*` flags against installed `libtpu.so` binary strings."""
    try:
        import libtpu

        so_dir = Path(libtpu.__file__).parent
        so_files = list(so_dir.glob("*.so"))
        blob = b"".join(p.read_bytes() for p in so_files) if so_files else b""
    except Exception:
        blob = b""
    if not blob:
        return " ".join(candidates)
    kept, dropped = [], []
    for flag in candidates:
        name = flag.lstrip("-").split("=", 1)[0].encode("utf-8")
        if name in blob:
            kept.append(flag)
        else:
            dropped.append(flag)
    print(f"[minimax-h3-sweep] Validated XLA flags: kept={len(kept)}, dropped={dropped}", flush=True)
    return " ".join(kept)


def probe_compile_opt_args(workdir: str) -> list[str]:
    code = (
        "from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig; "
        "c = TPUTrainerConfig(); out = []; "
        "hasattr(c, 'compile') and hasattr(c.compile, 'compile_grad_norm') and out.append('--compile.compile_grad_norm'); "
        "hasattr(c, 'compile') and hasattr(c.compile, 'optimizer_compile') and out.append('--compile.optimizer_compile=separate_graph'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_grad_norm') and out.append('--tpu_config.compile_grad_norm'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_optimizer') and out.append('--tpu_config.compile_optimizer'); "
        "print(' '.join(out))"
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = workdir
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=25)
        if res.returncode == 0 and res.stdout.strip():
            flags = res.stdout.strip().split()
            print(f"[minimax-h3-sweep] Probed compiled-optimizer flags: {flags}", flush=True)
            return flags
    except Exception as exc:
        print(f"[minimax-h3-sweep] probe_compile_opt_args warn: {exc}", flush=True)
    return []


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(2.0)
    return master_addr


def compute_exact_minimax_h3_flops_per_token(seq_len: int, l_txt: int = 512) -> float:
    """First-principles forward+backward (6N + 12*L*H*d_h*S) FLOPs/token for MiniMax-H3 (33.14B)."""
    hidden = 5376
    attn_inner = 7168  # 56 * 128
    ffn = 14336
    n_layers = 50
    n_refiner = 2
    # Active per-token parameters in 50 main blocks: QKV (3*5376*7168) + Out (7168*5376) + SwiGLU (3*5376*14336)
    main_block_params = n_layers * (4 * hidden * attn_inner + 3 * hidden * ffn)  # 19,267,584,000
    # Refiner active over l_txt / seq_len fraction + amortized AdaLN (13.05B / seq_len)
    refiner_params = n_refiner * (4 * hidden * attn_inner + 3 * hidden * ffn) * (l_txt / seq_len)
    adaln_amortized = (n_layers * 3 * (2688 * 96768)) / seq_len
    dense_flops_per_tok = 6.0 * (main_block_params + refiner_params + adaln_amortized)
    # Exact bidirectional (non-causal -> 2x causal!) attention FLOPs/token:
    # Forward: 4 * S * attn_inner per layer; Backward: 8 * S * attn_inner per layer => 12 * L * attn_inner * S
    attn_flops_per_tok = 12.0 * n_layers * attn_inner * seq_len + 12.0 * n_refiner * attn_inner * l_txt * (
        l_txt / seq_len
    )
    return dense_flops_per_tok + attn_flops_per_tok


def write_summary_files(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"# MiniMax-H3 (`33.14B` Omni-Modal Video+Audio DiT) — TorchTitan + TorchTPU `{summary['accel']}-{summary.get('phys_chips', summary['num_chips'])}` Results",
        "",
        f"- **Updated (UTC):** `{summary['updated_utc']}`",
        f"- **Cluster / Nodepool:** `{summary['nodepool']}` (`{summary['topology']}`, `{summary['num_chips']}` TPU `{summary['accel']}` ranks)",
        f"- **Model:** `MiniMaxAI/MiniMax-H3` (`33,139,340,704` total params; `19.27B` active per-token trunk + `13.05B` modality AdaLN)",
        f"- **Checkpoint:** `{summary['checkpoint_dir']}` (`66.28 GB`, 638 tensors across 14 safetensors shards)",
        "",
        "| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Tok/s/rank | Cluster Tok/s | Peak HBM (GiB) | TorchTitan MFU (%) | Exact Bidir-Attn MFU (%) | Loss (`step 1 -> N`) | Status |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, exp in enumerate(summary["experiments"], 1):
        if exp["status"] == "ok":
            lines.append(
                f"| {idx} | `{exp['name']}` | `{exp['parallelism']}` | `{exp['ac_mode']}` | `{exp['sa_block']}` | "
                f"`{exp['quant']}` | `{exp['seq_len']:,}` | `{exp['grad_accum']}x` | **{exp['steady_step_s']:.3f}s** | "
                f"**{exp['steady_tps_per_chip']:,.1f}** | `{exp['cluster_tps']:,.0f}` | `{exp['peak_mem_gib']:.2f}` | "
                f"**{exp['torchtitan_mfu_pct']:.2f}%** | **{exp['exact_bidir_mfu_pct']:.2f}%** | "
                f"`{exp['loss_first']:.4f} -> {exp['loss_last']:.4f}` | `OK` |"
            )
        else:
            lines.append(
                f"| {idx} | `{exp['name']}` | `{exp['parallelism']}` | `{exp['ac_mode']}` | `{exp['sa_block']}` | "
                f"`{exp['quant']}` | `{exp['seq_len']:,}` | `{exp['grad_accum']}x` | - | - | - | - | - | - | - | "
                f"`{exp['status']}` (`{exp.get('error', '')[:60]}`) |"
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel", default="v5p")
    parser.add_argument("--topology", default="2x2x4")
    parser.add_argument("--nnodes", type=int, default=4)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--nodepool", default="tpu-v5p-16-pool")
    parser.add_argument("--checkpoint-dir", default="/models/models/MiniMaxAI/MiniMax-H3/transformer")
    parser.add_argument("--results-root", default="/models/minimax-h3-results")
    parser.add_argument("--pass-name", default="pass2")
    args = parser.parse_args()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    base_port = int(os.environ.get("MASTER_PORT", "29500"))
    num_chips = args.nnodes * args.nproc_per_node
    phys_chips = num_chips // 2 if args.accel == "v7x" else num_chips
    peak_tflops = PEAK_BF16_TFLOPS_PER_CHIP[args.accel]

    # Stage minimax_h3_torchtitan.py into /torchtitan-native so `python -m minimax_h3_torchtitan` resolves cleanly
    workdir = Path("/torchtitan-native")
    src_mod = Path("/scripts/minimax_h3_torchtitan.py")
    dst_mod = workdir / "minimax_h3_torchtitan.py"
    shutil.copy2(src_mod, dst_mod)

    # Syntax preflight without initializing multi-host PJRT outside torchrun
    import py_compile

    py_compile.compile(str(dst_mod), doraise=True)
    print(f"[rank={node_rank}] Preflight syntax OK: {dst_mod}", flush=True)

    resolved_ip = wait_for_master_dns(master_addr)
    print(f"[rank={node_rank}] Resolved MASTER_ADDR={master_addr} -> {resolved_ip}", flush=True)

    base_xla_flags = ""
    lhs_candidates = [
        "--xla_enable_async_all_gather=true",
        "--xla_enable_async_all_reduce=true",
        "--xla_tpu_enable_ag_backward_pipelining=true",
        "--xla_tpu_enable_ici_ag_pipelining=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_max_concurrent_async_all_gathers=2",
        "--xla_latency_hiding_scheduler_rerun=2",
    ]
    if args.accel == "v7x":
        lhs_candidates.extend(
            [
                "--xla_tpu_enable_sparse_core_collective_aggregator=true",
                "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
                "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
                "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
                "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
                "--xla_tpu_dvfs_p_state=7",
            ]
        )
    lhs_xla_flags = filter_supported_xla_flags(lhs_candidates)
    compiled_opt_flags = probe_compile_opt_args(str(workdir))

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / f"{args.accel}-{phys_chips}-{args.topology}_minimax_h3_{args.pass_name}"
    hsdp_shard = num_chips // 2

    experiments = [
        {
            "name": f"mh3_p2_01_fsdp{num_chips}_fullac_s16k_pretrained",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "FullAC (recompute attn)",
            "save_splash": False,
            "sa_block": 512,
            "quant": "BF16 + FP32 heads",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 1,
            "reduce_dtype": "float32",
            "steps": 8,
            "load_pretrained": True,
            "xla_flags": base_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_02_fsdp{num_chips}_splash_save_ac_s16k_pretrained",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC (calibrated)",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 + FP32 heads",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 1,
            "reduce_dtype": "float32",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": base_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_03_fsdp{num_chips}_splash_save_xla_lhs_bf16red_s16k",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC + XLA LHS",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_04_fsdp{num_chips}_splash_save_xla_lhs_s16k_ga2",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC + XLA LHS",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 2,
            "reduce_dtype": "bfloat16",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_05_hsdp_2x{hsdp_shard}_splash_save_xla_lhs_s16k",
            "parallelism": f"HSDP 2x{hsdp_shard}",
            "dp_rep": 2,
            "dp_shard": hsdp_shard,
            "ac_mode": "SplashSaveAC + XLA LHS",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 8,
            "load_pretrained": False,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_06_fsdp{num_chips}_fp8_linear_quant_s16k",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC + XLA LHS",
            "save_splash": True,
            "sa_block": 512,
            "quant": "TpuFloat8Linear (rowwise FP8)",
            "use_fp8": True,
            "seq_len": 16384,
            "mb_tokens": 16384,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 8,
            "load_pretrained": False,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": f"mh3_p2_07_fsdp{num_chips}_splash_save_s32k_longvideo_5s_768p_16x9",
            "parallelism": f"FSDP-{num_chips}",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC + XLA LHS",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 2,
            "reduce_dtype": "bfloat16",
            "steps": 8,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
    ]

    summary = {
        "run_timestamp": ts,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accel": args.accel,
        "topology": args.topology,
        "nodepool": args.nodepool,
        "num_chips": num_chips,
        "phys_chips": phys_chips,
        "peak_bf16_tflops_per_chip": peak_tflops,
        "checkpoint_dir": args.checkpoint_dir,
        "experiments": [],
    }

    for exp_idx, exp in enumerate(experiments):
        port = base_port + exp_idx * 7
        global_batch_tokens = exp["mb_tokens"] * exp["grad_accum"] * num_chips
        cmd = [
            "torchrun",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            "--rdzv_backend=static",
            f"--rdzv_endpoint={master_addr}:{port}",
            f"--node_rank={node_rank}",
            "-m",
            "minimax_h3_torchtitan",
            "--module=minimax_h3_torchtitan",
            "--config=minimax_h3_33b_tpu",
            f"--hf_checkpoint_dir={args.checkpoint_dir}",
            f"--training.max_context_length={exp['seq_len']}",
            f"--training.num_tokens_per_microbatch_per_dp_rank={exp['mb_tokens']}",
            f"--training.num_tokens_per_train_step={global_batch_tokens}",
            f"--training.steps={exp['steps']}",
            f"--training.mixed_precision_reduce={exp['reduce_dtype']}",
            f"--parallelism.data_parallel_replicate_degree={exp['dp_rep']}",
            f"--parallelism.data_parallel_shard_degree={exp['dp_shard']}",
            f"--sa_block={exp['sa_block']}",
            "--debug.seed=42",
        ]
        if exp["save_splash"]:
            cmd.append("--save_splash_in_ac")
        else:
            cmd.append("--no-save_splash_in_ac")
        if exp["use_fp8"]:
            cmd.append("--use_fp8_linear")
        else:
            cmd.append("--no-use_fp8_linear")
        if exp["load_pretrained"]:
            cmd.append("--load_pretrained_weights")
        else:
            cmd.append("--no-load_pretrained_weights")
        if exp["use_compiled_opt"]:
            cmd.extend(["--use_compiled_opt", "--use_compiled_grad_norm", *compiled_opt_flags])
        else:
            cmd.extend(["--no-use_compiled_opt", "--no-use_compiled_grad_norm"])

        env = os.environ.copy()
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env["PYTHONPATH"] = str(workdir)
        env["LIBTPU_INIT_ARGS"] = exp["xla_flags"]
        env["PJRT_DEVICE"] = "TPU"

        print(f"\n{'=' * 88}", flush=True)
        print(f"[rank={node_rank}] >>> EXP {exp_idx + 1}/{len(experiments)}: {exp['name']} <<<", flush=True)
        print(f"[rank={node_rank}] CMD: {' '.join(cmd)}", flush=True)
        print(f"{'=' * 88}", flush=True)

        log_lines: list[str] = []
        step_records: list[dict] = []
        t_start = time.time()
        last_activity = [time.time()]
        fatal_seen_at: list[float] = []
        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        import signal
        import threading

        stop_wd = threading.Event()

        def _watchdog() -> None:
            while not stop_wd.wait(3.0):
                if proc.poll() is not None:
                    return
                now = time.time()
                idle_s = now - last_activity[0]
                fatal_wait_s = (now - fatal_seen_at[0]) if fatal_seen_at else 0.0
                if fatal_wait_s > 10.0 or idle_s > 540.0:
                    reason = f"fatal_cpp_err({fatal_wait_s:.1f}s)" if fatal_seen_at else f"idle_timeout({idle_s:.1f}s)"
                    print(f"[rank={node_rank}] [watchdog] Killing stuck torchrun group ({reason})", flush=True)
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    return

        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        wd_thread.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            last_activity[0] = time.time()
            sys.stdout.write(line)
            sys.stdout.flush()
            log_lines.append(line)
            if not fatal_seen_at and (
                "ExecutionTask failed: RESOURCE_EXHAUSTED" in line
                or "RuntimeProgramAllocationFailure" in line
                or "START_SESSION failed" in line
                or "RuntimeUnexpectedCoreHalt" in line
                or "SLICE_FAILURE" in line
            ):
                fatal_seen_at.append(time.time())
            m = STEP_RE.search(line)
            if m and ("[titan]" in line or "root - INFO" in line):
                step_num = int(m.group("step"))
                if not step_records or step_records[-1]["step"] != step_num:
                    tm = TIME_RE.search(line)
                    step_records.append(
                        {
                            "step": step_num,
                            "loss": float(m.group("loss")),
                            "grad_norm": float(m.group("gn")),
                            "mem_gib": float(m.group("mem")),
                            "tps": float(m.group("tps").replace(",", "")),
                            "tflops": float(m.group("tflops")),
                            "mfu_pct": float(m.group("mfu")),
                            "step_s": float(tm.group("sec")) if tm else 0.0,
                        }
                    )
        stop_wd.set()
        rc = proc.wait()
        wall_s = time.time() - t_start

        if node_rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{exp['name']}.log").write_text("".join(log_lines))
            steady = step_records[3:] if len(step_records) >= 4 else step_records[-3:]
            if rc == 0 and steady:
                avg_tps = sum(r["tps"] for r in steady) / len(steady)
                avg_step_s = sum(r["step_s"] for r in steady if r["step_s"] > 0) / max(
                    1, len([r for r in steady if r["step_s"] > 0])
                )
                avg_tt_mfu = sum(r["mfu_pct"] for r in steady) / len(steady)
                peak_mem = max(r["mem_gib"] for r in step_records)
                exact_flops_tok = compute_exact_minimax_h3_flops_per_token(exp["seq_len"])
                exact_tflops = (avg_tps * exact_flops_tok) / 1e12
                exact_mfu = 100.0 * exact_tflops / peak_tflops
                rec = {
                    **exp,
                    "status": "ok",
                    "wall_s": round(wall_s, 2),
                    "steady_step_s": round(avg_step_s, 4),
                    "steady_tps_per_chip": round(avg_tps, 2),
                    "cluster_tps": round(avg_tps * num_chips, 1),
                    "peak_mem_gib": round(peak_mem, 2),
                    "torchtitan_mfu_pct": round(avg_tt_mfu, 2),
                    "exact_bidir_tflops_per_chip": round(exact_tflops, 2),
                    "exact_bidir_mfu_pct": round(exact_mfu, 2),
                    "loss_first": step_records[0]["loss"],
                    "loss_last": step_records[-1]["loss"],
                    "steps_recorded": step_records,
                }
            else:
                tail_err = "".join(log_lines[-25:]).strip().replace("\n", " | ")
                rec = {
                    **exp,
                    "status": f"rc={rc}",
                    "wall_s": round(wall_s, 2),
                    "error": tail_err[-400:],
                    "steps_recorded": step_records,
                }
            summary["experiments"].append(rec)
            summary["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_summary_files(out_dir, summary)
            print(
                f"[rank=0] Saved {exp['name']} -> {out_dir / 'SUMMARY.md'} (status={rec['status']})",
                flush=True,
            )
        time.sleep(8.0)

    print(f"[rank={node_rank}] ALL {len(experiments)} MINIMAX-H3 EXPERIMENTS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()

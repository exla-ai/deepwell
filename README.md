# Deepwell
Train efficiently on Blackwell via low-precision kernels, planning, and MoE support.



## Usage

```
import deepwell as dw

# 1) Probe hardware and model graph
hw = dw.probe()
ir = dw.capture(model)

# 2) Autoplan parallelism and kernels
plan = dw.autoplan(
    ir,
    hw=hw,
    seq_len=32_768,
    global_batch=4096,
    arch="blackwell-sm100",      # set to 'blackwell-sm100' or 'blackwell-sm120' based on probe()
    moe={"experts": 8, "capacity_factor": 1.25},
)

# 3) Compile: bind Blackwell kernels, precision policy + fallbacks
engine = dw.compile(
    ir,
    plan=plan,
    precision="mxfp8",              # FP4 via NVFP4 later, guarded
    fallback="safe",                # BF16 fallback for fragile layers
)

# 4) Sanity check
dw.dryrun(engine)

# 5) Train
trainer = dw.Trainer.from_engine(
    engine,
    optimizer="adamw_dw",
    dataloader=train_loader,
    elastic=True,                   # assumes external launcher and re-shardable checkpoints
)
trainer.fit()

# 6) (Optional) Export engine artifact for inference
artifact = dw.export(engine, path="llama70b_b200.engine")

```

## Why are we building this? 
- Blackwell 5th‑gen Tensor Cores introduce MXFP8 and NVFP4/FP4 with microscaling. Vanilla PyTorch lacks the graph‑level planner or automatic precision fallback. Deepwell bridges that.
- CUTLASS exposes Blackwell kernels (`tcgen05.mma`, grouped GEMM) but leaves integration and orchestration to users. Deepwell provides a full training scaffold.
- MXFP8 on Blackwell requires both non‑transposed and transposed MXFP8 tensors; transposing MXFP8 implies requantization. This is why the precision policy must manage transpose points (NVIDIA Docs).


### What exists (training-relevant)
- Transformer Engine (TE): Provides Blackwell‑supported MXFP8 and NVFP4/FP4 tensor ops; handles transpose quantization; does not offer graph‑level planning.
- CUTLASS (Blackwell): Contains `tcgen05.mma` on SM100, block‑scaled MXFP8/NVFP4 kernels, and grouped‑GEMM examples.
- cuBLAS/cuBLASLt: Grouped‑GEMM exists and serves as a baseline; we prefer CUTLASS Blackwell block‑scaled grouped‑GEMM when shapes allow.
- Megablocks: Implements MoE grouped GEMM on Hopper; treat as Hopper‑tuned prior art with migration notes.
- ThunderKittens: CUDA DSL for custom kernel writing; Blackwell posts exist—keep as an option, not a dependency.

### Gaps Deepwell should fill
- **Autoplan**: PP/TP/EP/SP + tiling + activation and optimizer placement from NVLink fabric and HBM. Missing upstream.
- **Precision policy**: per‑layer MXFP8/NVFP4 (FP4) with BF16 fallbacks; checkpoint policies; guardrail tests; transpose‑point management because MXFP8 transpose implies requantization (NVIDIA Docs).
- **MoE block**: grouped‑GEMM fprop/dgrad/wgrad with TE epilogues, router logic, capacity factors; shipped as a stable drop‑in.
- **TMEM and `tcgen05.mma` utilization**: plan‑driven kernel binding must manage accumulator residency in TMEM on SM100.

### Deepwell (training-first) scope
- `probe()` → B200/B100, HBM size, NVLink graph.
- `capture()` → FX graph IR.
- `autoplan()` → PP/TP/EP/SP, tiles, grouped‑GEMM configs, activation and optimizer placement; `arch` set to `blackwell-sm100|sm120` once exact SM is detected via `probe()` (CUTLASS docs distinguish both).
- `compile(...)` → Bind TE and CUTLASS kernels, apply fallback policy, emit runnable engine.
- `dryrun()` → memory + comm/compute overlap checks.
- `Trainer.from_engine()` → enforce plan; overlap comm/compute; activation remat; ZeRO/FSDP wiring; elasticity assumes re‑shardable checkpoints and an external launcher.
- `save()` → weights + optimizer + Plan + scales + precision policy.

(Blackwell TE and NVLink context justify these choices. FP4 and NVFP4 are supported by Blackwell 5th‑gen Tensor Cores; FP4 is experimental for training here with guarded fallbacks.)

### Minimal viable spine

- Target: run dense LLaMA‑style blocks with MXFP8 on Blackwell and show BF16 parity ±ε (NVIDIA Developer, arXiv).

#### Repo layout
```
deepwell/
  __init__.py
  probe.py               # NVML + CUDA + NCCL topology
  capture.py             # FX graph capture + op tagging
  ir.py                  # nodes, tensors, parallelizable regions
  plan/
    __init__.py
    planner.py           # PP/TP/EP/SP, tiles, placement
    costs.py             # mem, comm, compute models
  precision/
    policy.py            # per-layer dtype, scale, guardrails
    scales.py            # block/tensor scales, amax tracking
  kernels/
    te_linear.py         # TE MXFP8 linears/attn wrappers
    cutlass_grouped.py   # SM100 grouped GEMM bindings
  compile.py             # bind kernels, materialize executors
  engine.py              # executable plan + runtime hooks
  dryrun.py              # memory + overlap simulator
  trainer.py             # overlap, remat, DP→FSDP/PP/TP
  moe/
    router.py            # top-k, capacity factor, losses
    block.py             # grouped-GEMM fprop/dgrad/wgrad
  checkpoints/
    format_v1.py         # weights, opt, scales, plan, policy
  tests/
    test_mxfp8_parity.py
    test_grouped_gemm.py
    test_planner.py
    test_dryrun.py
  docs/
    ...
```

#### Step order and acceptance

- probe()
  - Collect: device count, SM version, HBM per device, NVLink adjacency, PCIe hops.
  - Use NVML and NCCL to build a fabric graph.
  - Accept: JSON with devices, links, hbw_gbps estimates.

- capture() + IR
  - FX‑trace a PyTorch model. Annotate blocks: {embed, attn, mlp, norm, router, experts}.
  - Accept: IR prints that align with layer counts and parameter shapes.

- Precision policy
  - Default: MXFP8 everywhere except embeddings, first/last linear, logits, norms in BF16. MXFP8 transpose points handled with re‑quantized tensors and dual‑buffered scales. Guard: auto‑fallback to BF16 on instabilities (overflow, loss spike). (NVIDIA Docs)
  - Accept: unit test that toggles layer policies and preserves numerical range.

- TE bindings (MXFP8)
  - Wrap TE linear and attention with required transposed/non‑transposed MXFP8 tensor paths and amax/scale management.
  - Accept: microbench shows MXFP8 path runs and produces close outputs vs BF16 on random input.

- CUTLASS SM100 grouped GEMM
  - Build a minimal PyTorch extension for grouped GEMM with block‑scaled FP8 and hooks for TMEM‑friendly epilogues. Start with CUTLASS example shapes. (GitHub)
  - Accept: forward + backward against PyTorch reference within tolerance; throughput beats cuBLASLt grouped‑GEMM baseline for those shapes. (NVIDIA Developer)

- autoplan() v0
  - Inputs: IR, hw, seq_len, global_batch, MoE config.
  - Outputs: PP stage map, TP degree per block, EP degree for experts, SP on long‑seq, tiles, grouped‑GEMM configs, placement of activations and optimizer states across NVLink.
  - Use simple heuristics first: satisfy HBM, then maximize intra‑node NVLink utilization, then TP.
  - Accept: dryrun() shows <5% bubble estimate and no OOM for a known config.

- compile() + engine
  - Bind TE ops or CUTLASS kernels per plan. Emit runnable engine with streams and events for overlap.
  - Accept: end‑to‑end forward‑backward step executes.

- dryrun()
  - Simulator for memory, comm‑compute overlap, and pipeline bubbles using plan metadata and rough kernel cost models.
  - Accept: predicts memory within 10% of runtime measured peak on a small run.

- Trainer.from_engine() v0
  - Start with DP only. Add activation remat. Wire ZeRO‑1 fused AdamW later.
  - Accept: Llama‑7B batch‑N step runs, loss decreases, checkpoint writes scales + policy.

- Parity harness (M1 exit)
  - Run a 1–2B toy on MXFP8 vs BF16 for N steps. Compare loss curves and eval ppl within tight bounds. Cite MXFP8 parity expectation. (NVIDIA Developer, arXiv)

## Milestones

### M1: MXFP8 paths + parity harness vs BF16
- TE linear/attention wrappers with transpose‑aware scales and required non‑transposed/transposed MXFP8 tensors; CUTLASS SM100 GEMM integration; unit tests; parity harness vs BF16 ±ε.

### M2: MoE grouped GEMM + router logic
- Grouped‑GEMM fprop/dgrad/wgrad wired; L2‑reuse "grouping"; router logic; capacity factors; drop‑in DeepwellMoE; baseline against Megablocks behavior.

### M3: Autoplan + dryrun
- Fabric‑aware planner for PP×TP×EP×SP and tiling based on NVLink + HBM; cost model accounts for HBM fit, NVLink hops, grouped‑GEMM fusion level, TMEM residency constraints, and attention FLOPs vs seq_len; simulator for memory + bubbles.

### M4: Runtime training support with elasticity
- Trainer with overlap (comm/compute) via explicit streams per parallelism axis, activation remat, ZeRO/FSDP; elastic restart and checkpoint schema; note that elasticity assumes an external launcher and re‑shardable checkpoints.

### M5: FP4 experimental with guarded fallbacks
- FP4 and NVFP4 supported by Blackwell 5th‑gen Tensor Cores; FP4 is experimental for training here. Guarded FP4 kernels with SNR checks and per‑layer fallbacks; short BF16 repair fine‑tunes on divergence; accuracy guardrails. (NVIDIA Developer)

### M6: Docs and performance recipes
- Llama-70B dense and MoE reference runs; reproducible configs; perf dashboards.

### M7: Extras
Extras to add (next docs sections):

- **Precision Policy Guide**: defaults and when to override per layer; rationale for transpose‑point management.
- **MoE Cookbook**: grouped‑GEMM configs, router losses, capacity factors; Hopper→Blackwell migration notes.
- **Kernel Notes**: `tcgen05.mma` (SM100), TMEM residency, epilogue fusion pointers to CUTLASS examples; arch strings `blackwell-sm100|sm120`.

## Immediate TODOs (one week of focused work)

- Boot `probe()` with NVML and NCCL topo dump.
- FX `capture()` plus IR nodes and tags.
- Implement `precision.policy` with scale buffers and BF16 fallbacks at named layers.
- TE MXFP8 wrappers for Linear and Attention with transpose‑aware scales. (NVIDIA Docs)
- CUTLASS grouped GEMM extension for a fixed expert MLP shape; add autograd. (GitHub)
- `compile()` to stitch TE ops and grouped GEMM based on a hard‑coded plan.
- `dryrun()` memory model v0.
- Parity harness script vs BF16.
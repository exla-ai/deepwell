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
    arch="blackwell",
    moe={"experts": 8, "capacity_factor": 1.25},
)

# 3) Compile: bind Blackwell kernels, precision policy + fallbacks
engine = dw.compile(
    ir,
    plan=plan,
    precision="mxfp8",              # FP4 later, guarded
    fallback="safe",                # BF16 fallback for fragile layers
)

# 4) Sanity check
dw.dryrun(engine)

# 5) Train
trainer = dw.Trainer.from_engine(
    engine,
    optimizer="adamw_dw",
    dataloader=train_loader,
    elastic=True,
)
trainer.fit()

# 6) (Optional) Export engine artifact for inference
artifact = dw.export(engine, path="llama70b_b200.engine")

```

## Why are we building this? 
- Blackwell TE introduces MXFP8/FP4 and microscaling. Vanilla PyTorch lacks the graph-level planner or automatic precision fallback. Deepwell bridges that.
- CUTLASS exposes Blackwell kernels (`tcgen05.mma`, grouped GEMM) but leaves integration and orchestration to users. Deepwell provides a full training scaffold. 


### What exists (training-relevant)
- Transformer Engine (TE): Provides Blackwell-supported FP8/MXFP8 and NVFP4 tensor ops; handles transpose quantization; does not offer graph-level planning.
- CUTLASS (Blackwell): Contains `tcgen05.mma`, block-scaled FP8/FP4 kernels, and grouped GEMM examples.
- cuBLAS/cuBLASLt: General grouped-GEMM support; not optimized for Blackwell micro-tensor paths.
- Megablocks: Implements MoE grouped GEMM on Hopper; not tuned for Blackwell.
- ThunderKittens: CUDA DSL for custom kernel writing; no Blackwell deployments yet. 

### Gaps Deepwell should fill
- **Autoplan**: PP/TP/EP/SP + tiling + KV/optimizer placement from NVLink fabric and HBM. Missing upstream.
- **Precision policy**: per-layer MXFP8/FP4 with BF16 fallbacks; checkpoint policies; guardrail tests. TE doesn't prescribe this.
- **MoE block**: grouped-GEMM fprop/dgrad/wgrad with TE epilogues, router logic, capacity factors; shipped as a stable drop-in.
- **TMEM/tcgen05 utilization**: plan-driven kernel binding for SM100 fused micro-tensor memory usage.

### Deepwell (training-first) scope
- `probe()` → B200/B100, HBM size, NVLink graph.
- `capture()` → FX graph IR.
- `autoplan()` → PP/TP/EP/SP, tiles, grouped-GEMM configs, KV/opt placement.
- `compile(...)` → Bind TE and CUTLASS kernels, apply fallback policy, emit runnable engine.
- `dryrun()` → memory + comm/compute overlap checks.
- `Trainer.from_engine()` → enforce plan; overlap comm/compute; activation remat; ZeRO/FSDP wiring.
- `save()` → weights + optimizer + Plan + scales + precision policy.

(Blackwell TE and NVLink context justify these choices.)

## Milestones

### M1: MXFP8 paths + parity harness vs BF16
- TE linear/attention with required MXFP8 transposed/non-transposed tensors; CUTLASS SM100 GEMM integration; unit tests.

### M2: MoE grouped GEMM + router logic
- Grouped-GEMM fprop/dgrad/wgrad wired; L2-reuse "grouping"; router logic; capacity factors; drop-in DeepwellMoE.

### M3: Autoplan + dryrun
- Fabric-aware planner for PP/TP/EP/SP and tiling based on NVLink + HBM; simulator for memory + bubbles.

### M4: Runtime training support with elasticity
- Trainer with overlap (comm/compute), activation remat, ZeRO/FSDP; elastic restart and checkpoint schema.

### M5: FP4 experimental with guarded fallbacks
- Guarded FP4 kernels + per-layer fallbacks; short repair fine-tune hooks; accuracy guardrails.

### M6: Docs and performance recipes
- Llama-70B dense and MoE reference runs; reproducible configs; perf dashboards.

### M7: Extras
Extras to add (next docs sections):

- **Precision Policy Guide**: defaults and when to override per layer.
- **MoE Cookbook**: grouped-GEMM configs, router losses, capacity factors; Hopper→Blackwell migration notes.
- **Kernel Notes**: tcgen05.mma, TMEM residency, epilogue fusion pointers to CUTLASS examples.
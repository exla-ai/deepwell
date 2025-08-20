# Deepwell
Blackwell GPU-native training and inference scaffold that makes models run in FP4/FP8 with optimal kernels, memory use, and parallelism - without users hand-tuning 


## Usage

```
import deepwell as dw

# 1) Probe hardware and model graph
hw = dw.probe()                     # B200/B100, HBM, NVLink fabric
graph = dw.capture(model)           # FX graph from PyTorch

# 2) Autoplan parallelism + kernels for Blackwell
plan = dw.autoplan(
    graph,
    hw=hw,
    seq_len=32_768,
    global_batch=4096,
    arch="blackwell",
    moe={"experts": 8, "capacity_factor": 1.25},  # optional
)

# 3) Compile: swap in Blackwell kernels (tcgen05.mma), MXFP8 policy
engine = dw.compile(
    graph,
    plan=plan,
    precision="mxfp8",              # later: "fp4" guarded
    policy="safe",                  # per-layer fallback table
)

# 4) Calibrate MXFP8/FP4 scales (short warmup)
dw.calibrate(engine, calib_iter, steps=500)

# 5a) Train
trainer = dw.Trainer(
    engine,
    optimizer="adamw_dw",
    dataloader=train_loader,
    elastic=True,                   # fault tolerant
)
trainer.fit()

# 5b) Or export for inference
artifact = ,w.export(engine, path="llama70b_b200.engine")

```
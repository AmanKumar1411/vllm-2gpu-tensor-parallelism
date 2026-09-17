# vLLM 2-GPU Tensor Parallelism on 2× NVIDIA A100-SXM4-40GB
## Complete Debugging, Experiment Log, Commands, Failures, Fixes, and Final State

**Date of investigation:** 17 September 2026  
**Platform:** Vast.ai rented GPU instance  
**Primary goal:** Run Qwen/Qwen3.5-9B with vLLM using **2-GPU tensor parallelism (`tensor_parallel_size=2`)** across two NVIDIA A100 GPUs.

---

# 1. Executive Summary

This document records the complete step-by-step investigation of a vLLM 2-GPU tensor-parallel setup on a rented Vast.ai instance.

The instance exposed:

- **2 × NVIDIA A100-SXM4-40GB**
- **Driver:** 595.84
- **CUDA reported by `nvidia-smi`:** 13.2
- **MIG:** Disabled
- **vLLM:** 0.29.0
- **NCCL:** 2.30.7
- **Model:** `Qwen/Qwen3.5-9B`
- **Tensor parallelism:** 2
- **Initial vLLM template configuration:** `--max-num-seqs 8`, `--max-model-len 32000`

The investigation uncovered multiple separate issues.

### Issue 1 — vLLM TP=2 workers were spinning at 100% GPU utilization before model loading

Initial `nvidia-smi` showed:

```text
GPU 0: 100% GPU utilization, only 711 MiB used
GPU 1: 100% GPU utilization, only 711 MiB used
```

Both GPUs had `VLLM::Worker` processes.

This was not normal inference. The workers were stuck during distributed startup/communication initialization.

### Issue 2 — The two A100s had no active NVLink path

`nvidia-smi topo -m` showed:

```text
GPU0 <-> GPU1 = NODE
```

rather than:

```text
GPU0 <-> GPU1 = NV#
```

and:

```text
nvidia-smi nvlink --status
```

reported all NVLink links inactive.

However, CUDA P2P capability still reported `OK`, and PyTorch reported:

```text
0 -> 1: True
1 -> 0: True
```

So the GPUs were visible and peer-access capability was exposed, even though the topology was PCIe/host-bridge based rather than NVLink.

### Issue 3 — NCCL/P2P path caused the TP=2 startup hang

The decisive experiment was:

```bash
NCCL_P2P_DISABLE=1
```

With P2P enabled, NCCL initialized and connected rings but vLLM then hung before model loading.

With:

```bash
NCCL_P2P_DISABLE=1
```

vLLM progressed into actual model loading.

This was the main successful workaround discovered during the investigation.

### Issue 4 — Qwen3.5/vLLM startup still did not become API-ready

After fixing the P2P-related startup hang, the model workers successfully allocated approximately 9.8–10.3 GB per GPU, proving that TP=2 model allocation was working.

However:

```bash
ss -lntp | grep 18001
```

never showed the API server listening, and:

```bash
curl http://localhost:18001/v1/models
```

continued to fail with:

```text
curl: (7) Failed to connect to localhost port 18001
```

Several additional attempts were made:

- vLLM V1 runner
- `--enforce-eager`
- `--language-model-only`
- shorter `--max-model-len 8192`
- chunked prefill
- disabling custom all-reduce
- disabling P2P
- disabling symmetric-memory-related paths

These did not produce a final end-to-end serving result during the recorded session.

### Final conclusion

The investigation **successfully solved the TP=2 NCCL/P2P startup hang** and demonstrated that:

```text
TP=2
+
NCCL_P2P_DISABLE=1
+
2 × A100
```

can progress to model allocation/loading.

However, the complete goal of getting `Qwen/Qwen3.5-9B` to a working OpenAI-compatible HTTP endpoint on `localhost:18001` was **not fully completed in the recorded session**.

The remaining issue was narrowed to a later stage of vLLM/Qwen3.5 initialization after model memory allocation.

---

# 2. Hardware and Software Environment

## 2.1 GPUs

The system reported:

```text
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB
```

Each GPU has:

```text
40,960 MiB VRAM
```

The relevant initial `nvidia-smi` information was:

```text
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB
```

MIG was disabled on both:

```text
MIG M. = Disabled
```

This was appropriate for a normal two-GPU tensor-parallel experiment.

---

## 2.2 NVIDIA driver and CUDA

Initial output:

```text
NVIDIA-SMI 595.84
Driver Version: 595.84
CUDA Version: 13.2
```

---

## 2.3 vLLM

The Vast.ai template launched:

```text
vllm serve Qwen/Qwen3.5-9B
```

and reported:

```text
version 0.29.0
```

---

## 2.4 NCCL

vLLM reported:

```text
vLLM is using nccl==2.30.7
```

NCCL itself reported:

```text
NCCL version 2.30.7+cuda13.3
```

This is worth noting: `nvidia-smi` reported CUDA 13.2 through the driver, while the bundled NCCL build reported `+cuda13.3`.

---

## 2.5 Model

The model used throughout the investigation was:

```text
Qwen/Qwen3.5-9B
```

vLLM resolved it as:

```text
Qwen3_5ForConditionalGeneration
```

---

# 3. Goal of the Experiment

The desired architecture was:

```text
                 vLLM
                  |
             Tensor Parallel
                  |
        +---------+---------+
        |                   |
     A100 GPU 0          A100 GPU 1
      40 GB                 40 GB
```

The intended vLLM setting was:

```bash
--tensor-parallel-size 2
```

With tensor parallelism, each GPU participates in the same model execution group.

The experiment was primarily intended to test **2-GPU tensor parallelism on rented A100s**, rather than because Qwen3.5-9B required two A100s for VRAM capacity.

---

# 4. First Mistake: Incorrect `nvidia-smi` Command

The first commands were:

```bash
nividia smi
```

and:

```bash
nvidia smi
```

Both failed because the command is:

```bash
nvidia-smi
```

Correct command:

```bash
nvidia-smi
```

---

# 5. Initial `nvidia-smi` Diagnosis

The first correct output showed:

```text
GPU 0:
A100-SXM4-40GB
Memory: 711 MiB / 40960 MiB
GPU Utilization: 100%
Power: ~56 W

GPU 1:
A100-SXM4-40GB
Memory: 711 MiB / 40960 MiB
GPU Utilization: 100%
Power: ~60 W
```

Processes:

```text
GPU 0 -> VLLM::Worker -> ~702 MiB
GPU 1 -> VLLM::Worker -> ~702 MiB
```

This immediately raised the question:

> Why are both GPUs at 100% utilization when no inference request has been sent?

The answer was **not normal inference**.

The amount of GPU memory was far too small for the Qwen3.5-9B model to have been loaded.

---

# 6. Continuous Monitoring with `nvidia-smi dmon`

Command used:

```bash
nvidia-smi dmon -s pucm -d 1
```

Important results repeated for many samples:

```text
GPU 0:
power ~55–56 W
SM    100%
FB    711 MB

GPU 1:
power ~59–60 W
SM    100%
FB    711 MB
```

The memory-engine utilization was effectively zero while SM utilization stayed at 100%.

This indicated that the processes were executing/spinning but were not actually performing normal model inference.

---

# 7. Per-Process Monitoring with `nvidia-smi pmon`

Command:

```bash
nvidia-smi pmon -i 0,1 -s um -c 10
```

Repeated output:

```text
GPU 0  PID 4704  VLLM::Worker  SM 99%
GPU 1  PID 4737  VLLM::Worker  SM 99%
```

GPU memory per process:

```text
~702 MB
```

This confirmed that the high GPU utilization originated specifically from the vLLM worker processes.

---

# 8. Discovering the Vast.ai vLLM Template

Command:

```bash
ps aux | grep -i vllm
```

Important processes:

```text
/opt/supervisor-scripts/vllm.sh

vllm serve Qwen/Qwen3.5-9B
```

The vLLM command was running with a configuration equivalent to:

```text
model = Qwen/Qwen3.5-9B
tensor_parallel_size = 2
max_num_seqs = 8
max_model_len = 32000
```

The vLLM engine also showed:

```text
enable_prefix_caching = True
enable_chunked_prefill = True
```

and initially:

```text
disable_custom_all_reduce = False
```

---

# 9. Checking the HTTP Server

Command:

```bash
curl http://localhost:8000/v1/models
```

This did not return a usable model response.

Then:

```bash
ss -lntp
```

showed several Vast.ai/portal services, but the actual vLLM engine had not yet become a normal serving endpoint.

The Vast.ai environment also exposed various Caddy/portal services. This is important because the template was not simply a bare `vllm serve` installation.

---

# 10. First Major Topology Investigation

Command:

```bash
nvidia-smi topo -m
```

Result:

```text
        GPU0    GPU1
GPU0     X      NODE
GPU1    NODE     X
```

NVIDIA defines:

```text
NODE = PCIe/host-bridge connection within the same NUMA node
NV#  = bonded NVLink connection
```

Therefore the environment was not exposing:

```text
GPU0 <-> NVLink <-> GPU1
```

Instead it showed:

```text
GPU0 <-> NODE/PCIe topology <-> GPU1
```

---

# 11. Checking NVLink

Command:

```bash
nvidia-smi nvlink --status
```

Result:

```text
NVML: Unable to retrieve Nvlink information as all links are inActive
```

for both A100s.

Therefore:

```text
Active NVLink = NO
```

This is an important limitation for high-performance tensor parallelism.

---

# 12. Checking CUDA P2P Read/Write Support

Commands:

```bash
nvidia-smi topo -p2p r
```

and:

```bash
nvidia-smi topo -p2p w
```

Both returned:

```text
GPU0 GPU1
GPU0 X  OK
GPU1 OK X
```

Therefore NVIDIA's topology utility reported CUDA P2P support as available in both directions.

This initially suggested that direct GPU peer access should work.

---

# 13. Checking P2P from PyTorch

Command:

```bash
python - <<'PY'
import torch

print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print("0 -> 1:", torch.cuda.can_device_access_peer(0, 1))
print("1 -> 0:", torch.cuda.can_device_access_peer(1, 0))
PY
```

Result:

```text
GPU count: 2
0 NVIDIA A100-SXM4-40GB
1 NVIDIA A100-SXM4-40GB
0 -> 1: True
1 -> 0: True
```

So all of the following were true:

```text
GPU visible to CUDA                 YES
GPU visible to PyTorch              YES
Peer-access capability              YES
P2P read reported OK                YES
P2P write reported OK               YES
NVLink active                       NO
```

This became an important part of the later diagnosis.

---

# 14. Initial TP=2 vLLM Log

The vLLM log showed:

```text
world_size=2 rank=0 local_rank=0 backend=nccl
world_size=2 rank=1 local_rank=1 backend=nccl
vLLM is using nccl==2.30.7
```

After that, the process appeared to stall.

At the same time:

```text
GPU 0 = ~100%
GPU 1 = ~100%
VRAM ~= 711 MiB each
```

This meant the hang occurred during distributed initialization before model weights were loaded.

---

# 15. Understanding `VLLM::Worker`

The process list showed:

```text
VLLM::Worker
```

for each GPU.

This is normal for tensor parallelism.

For `tensor_parallel_size=2`:

```text
Rank 0 -> GPU 0
Rank 1 -> GPU 1
```

The key issue was not that vLLM could not see the second GPU. It clearly could.

The problem was that the workers were stuck while establishing the distributed execution environment.

---

# 16. Important Cleanup Discovery: `supervisorctl stop vllm`

Command:

```bash
supervisorctl status
```

showed:

```text
vllm RUNNING
```

Then:

```bash
supervisorctl stop vllm
```

returned:

```text
vllm: stopped
```

However, `nvidia-smi` still showed the old `VLLM::Worker` processes.

Therefore:

> Stopping the supervisor service did not necessarily immediately remove the already spawned vLLM worker processes.

This was why explicit process cleanup became necessary.

---

# 17. Explicit vLLM Process Cleanup

Command:

```bash
pkill -9 -f 'VLLM::Worker|vllm serve|VLLM::EngineCore' 2>/dev/null
```

This was used before subsequent manual experiments.

---

# 18. First Manual TP=1 Test

The first manual command was:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --max-num-seqs 8
```

This failed with:

```text
OSError: [Errno 98] Address already in use
```

This did NOT indicate a model or GPU failure.

It simply meant another server/process was already using the default vLLM port.

---

# 19. Corrected TP=1 Test with a Separate Port

The command was changed to:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --port 18001
```

This time vLLM successfully progressed through:

```text
Initializing a V1 LLM engine
world_size=1
Using V2 Model Runner
Loading model from scratch...
```

and:

```text
model.safetensors.index.json: 100%
```

This proved that:

- CUDA itself worked.
- One A100 could run vLLM.
- The model was accessible.
- Single-GPU model initialization could progress.
- The previous `Address already in use` issue was only a port conflict.

The TP=1 run was interrupted manually during model loading.

---

# 20. Testing TP=2 with Custom All-Reduce Disabled

The next experiment was:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=INFO \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

The purpose was to isolate vLLM's custom all-reduce implementation.

NCCL produced useful information:

```text
NCCL version 2.30.7+cuda13.3
```

and eventually:

```text
ncclCommInitRank ... Init COMPLETE
```

for both ranks.

It also showed:

```text
P2P Chunksize set to 131072
```

and:

```text
Channel ... via P2P/CUMEM
```

followed by:

```text
Connected all rings
```

So this experiment proved:

> NCCL communicator initialization itself can complete successfully.

However, the workers still failed to progress into model loading and the GPUs continued spinning.

---

# 21. Critical Discovery: NCCL P2P Disable

The decisive experiment was:

```bash
NCCL_P2P_DISABLE=1 \
VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
VLLM_USE_NCCL_SYMM_MEM=0 \
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=INFO \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

NCCL explicitly confirmed:

```text
NCCL_P2P_DISABLE set by environment to 1
```

and changed its channels from direct P2P to:

```text
via SHM/direct
```

Most importantly, after NCCL initialization, vLLM progressed to:

```text
Loading model from scratch...
```

This was the first clear evidence that:

```text
NCCL P2P disabled
```

was the key workaround for the TP=2 startup hang.

---

# 22. Comparing P2P Enabled vs Disabled

## P2P enabled

Observed:

```text
NCCL Init COMPLETE
Connected all rings
GPU workers remain busy
Model does not load
```

## P2P disabled

Observed:

```text
NCCL Init COMPLETE
Connected all rings
Loading model from scratch...
Model memory begins allocating
```

This was the strongest causal debugging result in the investigation.

---

# 23. Why the Topology Was Suspicious

The GPU pair had:

```text
nvidia-smi topo -m
GPU0 <-> GPU1 = NODE
```

and:

```text
NVLink = inactive
```

while P2P capability APIs reported that P2P was supported.

This created the following environment:

```text
             CPU / PCIe topology
              /          \
         GPU 0            GPU 1
       A100 40GB        A100 40GB
```

instead of:

```text
GPU 0 ===== NVLink ===== GPU 1
```

For tensor parallelism, this matters because tensor-parallel ranks exchange intermediate data frequently.

---

# 24. Symmetric-Memory Experiments

Another attempted workaround was to disable symmetric-memory paths:

```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0
VLLM_USE_NCCL_SYMM_MEM=0
```

However, simply disabling these settings **did not fix the problem when NCCL P2P remained enabled**.

Therefore the results were:

```text
Symmetric memory disabled only
        ↓
Still hangs
```

versus:

```text
NCCL_P2P_DISABLE=1
        ↓
Progresses to model loading
```

This helped narrow the root cause.

---

# 25. Model Memory Allocation After the P2P Fix

With P2P disabled, the vLLM workers eventually allocated approximately:

```text
GPU 0: ~10.3 GB
GPU 1: ~10.3 GB
```

For example:

```text
GPU 0: 10303 MiB / 40960 MiB
GPU 1: 10303 MiB / 40960 MiB
```

with:

```text
VLLM::Worker_TP0
VLLM::Worker_TP1
```

This was a major milestone.

It proved that:

```text
Tensor parallel rank 0 -> GPU 0
Tensor parallel rank 1 -> GPU 1
```

were both participating in model allocation.

---

# 26. Why 0% GPU Utilization Was Not Automatically a Failure

At several points after model memory allocation, `nvidia-smi` showed:

```text
GPU utilization = 0%
```

while the vLLM workers remained alive.

This does not automatically mean the server is broken.

`GPU-Util` measures recent GPU kernel activity. During startup/initialization, processes can spend time doing CPU-side work, synchronization, allocation, compilation, or other initialization tasks.

Therefore:

```text
0% GPU utilization
```

and:

```text
~10 GB VRAM allocated
```

can coexist during initialization.

The real indicator of server readiness was:

```bash
ss -lntp | grep 18001
```

and eventually:

```bash
curl http://localhost:18001/v1/models
```

---

# 27. API Readiness Problem

Even after the workers were alive and model memory was allocated:

```bash
ss -lntp | grep 18001
```

returned nothing.

Therefore no process was listening on:

```text
localhost:18001
```

Consequently:

```bash
curl http://localhost:18001/v1/models
```

returned:

```text
curl: (7) Failed to connect to localhost port 18001 after 0 ms: Couldn't connect to server
```

At this point, the debugging problem had changed from:

```text
NCCL/P2P startup hang
```

to:

```text
later vLLM/Qwen3.5 initialization or serving startup issue
```

---

# 28. Process Tree During the Later Initialization

The following processes remained alive:

```text
vllm serve Qwen/Qwen3.5-9B

VLLM::EngineCore

VLLM::Worker_TP0

VLLM::Worker_TP1
```

For example:

```text
PID 19726   vllm serve
PID 20306   VLLM::EngineCore
PID 20580   VLLM::Worker_TP0
PID 20627   VLLM::Worker_TP1
```

The processes were still consuming CPU.

Therefore this was not always an immediate crash; some configurations appeared to continue performing initialization without opening the HTTP port.

---

# 29. V2 Model Runner Investigation

The initial configuration used:

```text
Using V2 Model Runner
```

because vLLM 0.29.0 was using the newer model-runner path by default in the observed configuration.

Because Qwen3.5 is a hybrid architecture involving full attention plus GDN/linear-attention components, we tested forcing the older runner.

Environment variable:

```bash
VLLM_USE_V2_MODEL_RUNNER=0
```

---

# 30. V1 Model Runner Test

Command:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

This produced:

```text
Initializing a V1 LLM engine
```

Then:

```text
world_size=2
rank 0
rank 1
```

The workers successfully loaded approximately:

```text
~10.3 GB per GPU
```

Again, however:

```bash
ss -lntp | grep 18001
```

showed nothing.

So:

```text
Switching from V2 to V1
```

did not by itself fully solve the HTTP serving problem.

---

# 31. `--enforce-eager` Experiment

Another attempt was:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --port 18001
```

vLLM explicitly reported:

```text
Enforce eager set, disabling torch.compile and CUDAGraphs.
```

So this experiment removed the normal compilation/CUDA-graph initialization path.

However, the process later exited and GPU memory returned to approximately:

```text
4 MiB
```

with no serving endpoint.

Therefore:

```text
--enforce-eager
```

did not provide a complete solution.

---

# 32. Qwen3.5 Text-Only / Language-Model-Only Experiment

Because the objective was text generation rather than multimodal inference, another experiment used:

```bash
--language-model-only
```

Command:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --language-model-only \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --disable-custom-all-reduce \
  --port 18001
```

vLLM confirmed:

```text
All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
```

This was a reasonable diagnostic reduction because it removed multimodal processing from the test.

---

# 33. Shorter Context-Length Experiment

The same test reduced:

```text
max_model_len = 32000
```

to:

```text
max_model_len = 8192
```

The purpose was to reduce KV-cache requirements and eliminate 32K context as a possible cause.

The model still allocated approximately:

```text
~9.9 GB per GPU
```

but the HTTP server still did not become available.

Therefore:

```text
32K context alone was not the complete explanation.
```

---

# 34. Current Known-Good Partial Configuration

The strongest working configuration discovered was:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

This configuration achieved:

```text
2-GPU TP initialization
+
NCCL workaround
+
model memory allocation on both GPUs
```

However, the API endpoint did not become ready during the recorded session.

---

# 35. The Later `language-model-only` Configuration

A more constrained configuration was:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --language-model-only \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --disable-custom-all-reduce \
  --port 18001
```

This got as far as:

```text
V1 LLM engine
world_size=2
```

and allocated approximately:

```text
9877 MiB / GPU
```

But:

```bash
ss -lntp | grep 18001
```

still returned nothing.

Thus this also remained a partial success.

---

# 36. Commands Used During the Investigation

## NVIDIA GPU information

```bash
nvidia-smi
```

```bash
nvidia-smi dmon -s pucm -d 1
```

```bash
nvidia-smi pmon -i 0,1 -s um -c 10
```

```bash
nvidia-smi topo -m
```

```bash
nvidia-smi topo -p2p r
```

```bash
nvidia-smi topo -p2p w
```

```bash
nvidia-smi nvlink --status
```

---

## CUDA/PyTorch P2P

```bash
python - <<'PY'
import torch

print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print("0 -> 1:", torch.cuda.can_device_access_peer(0, 1))
print("1 -> 0:", torch.cuda.can_device_access_peer(1, 0))
PY
```

---

## Process inspection

```bash
ps aux | grep -i vllm
```

```bash
ps -fp <PID>
```

```bash
ps -ef | grep -E 'vllm|VLLM::'
```

```bash
ps -o pid,ppid,%cpu,%mem,etime,stat,cmd -p <PIDs>
```

---

## Supervisor

```bash
supervisorctl status
```

```bash
supervisorctl stop vllm
```

Important lesson:

> Stopping the supervisor service did not always eliminate the worker processes, so explicit process cleanup was needed.

---

## Process cleanup

```bash
pkill -9 -f 'VLLM::Worker|vllm serve|VLLM::EngineCore' 2>/dev/null
```

---

## Network/socket checks

```bash
ss -lntp
```

```bash
ss -lntp | grep 18001
```

---

## API testing

```bash
curl http://localhost:18001/v1/models
```

Chat completion test:

```bash
curl http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "Explain tensor parallelism in one sentence."}
    ],
    "max_tokens": 50
  }'
```

---

## Monitoring

```bash
watch -n 1 nvidia-smi
```

```bash
watch -n 1 'nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader'
```

---

## Files/log inspection

```bash
tail -n 200 /var/log/portal/vllm.log
```

```bash
grep -iE "error|warn|nccl|p2p|allreduce|cuda|hang|waiting" \
  /var/log/portal/vllm.log | tail -n 100
```

```bash
find /tmp -maxdepth 2 -type f -iname '*vllm*' 2>/dev/null
```

---

# 37. vLLM Commands Tested

## Original Vast.ai-style TP=2

Conceptually:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000
```

Observed result:

```text
TP=2 workers spin
GPU ~=100%
VRAM ~=711 MiB
```

---

## TP=1

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --port 18001
```

Reached model loading.

---

## TP=2 without custom all-reduce

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=INFO \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

NCCL initialized, but direct P2P path still resulted in a startup hang.

---

## TP=2 with P2P disabled

```bash
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

This progressed into model loading.

---

## TP=2 + V1

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

Loaded model state on both GPUs, but HTTP server remained unavailable.

---

## TP=2 + V1 + eager mode

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --port 18001
```

This did not produce a final serving state.

---

## TP=2 + language-model-only + 8K

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --language-model-only \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --disable-custom-all-reduce \
  --port 18001
```

Reached approximately:

```text
GPU0 ~9.9 GB
GPU1 ~9.9 GB
```

but still no HTTP listener.

---

# 38. What Each Experiment Proved

| Experiment | Result | What it proved |
|---|---|---|
| `nvidia-smi` | Both A100s visible | GPU access works |
| MIG status | Disabled | GPUs are not partitioned into MIG instances |
| `topo -m` | NODE | GPU pair uses PCIe/host topology in this environment |
| `nvlink --status` | Inactive | No active NVLink path exposed |
| P2P read/write | OK | Driver reports P2P capability |
| PyTorch peer access | True | CUDA exposes peer access |
| Original TP=2 | 100% SM, ~711 MiB | Distributed startup was stuck |
| TP=1 | Model loading begins | Single GPU/vLLM/model path works |
| TP=2 + disable custom all-reduce | NCCL Init COMPLETE | NCCL communicator itself can initialize |
| TP=2 + P2P disabled | Model loading starts | P2P path was the key startup blocker |
| Symmetric-memory disabled only | Still hangs | Symmetric-memory flags alone were not the fix |
| V1 runner | Still no HTTP endpoint | V2 was not the only remaining issue |
| `--enforce-eager` | Process eventually exited | Disabling compile/CUDAGraphs did not solve it |
| `--language-model-only` + 8K | ~9.9 GB/GPU, no API | Text-only + reduced context still didn't complete serving |

---

# 39. Bugs That Were Actually Solved

## Bug A — Misinterpreting 100% GPU utilization

### Symptom

```text
GPU = 100%
VRAM = ~711 MiB
```

### Root cause

vLLM TP workers were stuck during distributed initialization rather than performing inference.

### Resolution

Debugged using:

```bash
nvidia-smi dmon
nvidia-smi pmon
ps
NCCL_DEBUG=INFO
```

---

## Bug B — TP=2 NCCL/P2P startup hang

### Symptom

```text
NCCL Init COMPLETE
Connected all rings
```

followed by no model loading.

### Root cause isolated to

```text
Direct GPU P2P path
```

in this Vast.ai topology.

### Resolution

```bash
NCCL_P2P_DISABLE=1
```

This was the most important successful fix.

---

## Bug C — Port conflict during TP=1 test

### Symptom

```text
OSError: [Errno 98] Address already in use
```

### Resolution

Use a separate port:

```bash
--port 18001
```

and clean existing processes.

---

## Bug D — Supervisor stopped but workers remained

### Symptom

```bash
supervisorctl stop vllm
```

returned success, but `nvidia-smi` still showed:

```text
VLLM::Worker
```

### Resolution

Explicit process cleanup:

```bash
pkill -9 -f 'VLLM::Worker|vllm serve|VLLM::EngineCore' 2>/dev/null
```

---

# 40. Issues Not Fully Solved

## Remaining Issue A — vLLM API server never became ready

Even after:

```text
TP=2
P2P disabled
V1 runner
model memory allocated
```

the server never opened:

```text
localhost:18001
```

in the recorded session.

Therefore:

```bash
curl http://localhost:18001/v1/models
```

continued to fail.

---

## Remaining Issue B — Exact post-model-load blocker

At the end of the session, the exact code path responsible for the later initialization stall had not yet been identified.

The next diagnostic should have been worker stack inspection rather than continuing to change flags blindly.

Recommended diagnostics:

```bash
ps -o pid,ppid,%cpu,%mem,etime,stat,wchan:32,cmd \
-p <engine_pid>,<worker0_pid>,<worker1_pid>
```

If available:

```bash
py-spy dump --pid <worker0_pid>
py-spy dump --pid <worker1_pid>
```

Also:

```bash
pidstat -d -p <worker0_pid>,<worker1_pid> 1 5
```

These were proposed but were **not completed in the recorded session**.

---

# 41. Important Lessons from the Investigation

## Lesson 1 — GPU utilization alone is misleading

A GPU at:

```text
100% utilization
```

does not necessarily mean a model is serving traffic.

The initial 100% utilization was actually part of a stalled distributed initialization path.

---

## Lesson 2 — VRAM usage is a better startup indicator

The transition:

```text
~711 MiB
```

to:

```text
~9.8–10.3 GB
```

was a much stronger indication that vLLM had progressed to actual model loading/allocation.

---

## Lesson 3 — NCCL logs are extremely valuable

Using:

```bash
NCCL_DEBUG=INFO
```

revealed:

```text
Init COMPLETE
```

and:

```text
via P2P/CUMEM
```

which allowed the investigation to distinguish NCCL initialization from the later model-loading problem.

---

## Lesson 4 — P2P capability and functional P2P are not exactly the same debugging question

The system reported:

```text
P2P read = OK
P2P write = OK
torch.cuda.can_device_access_peer() = True
```

yet:

```text
NCCL P2P enabled → TP=2 startup hang
NCCL P2P disabled → model loading progresses
```

This is why the actual runtime experiment was more informative than relying only on capability checks.

---

## Lesson 5 — Topology matters for tensor parallelism

The topology showed:

```text
NODE
```

and NVLink was inactive.

For high-performance tensor parallelism, an A100 pair with an active NVLink topology would generally be preferable to a PCIe-only/host-bridge path.

---

## Lesson 6 — Do not change too many variables simultaneously unless necessary

The best diagnostic sequence came from isolating one major variable:

```text
P2P enabled
    ↓
hang

P2P disabled
    ↓
progress
```

That gave a strong causal signal.

Later experiments became more complex and therefore less clean diagnostically.

---

# 42. Recommended Clean Starting Configuration for This Host

Based on the evidence from this session, the cleanest baseline for this particular host is:

```bash
pkill -9 -f 'vllm serve|VLLM::Worker|VLLM::EngineCore' 2>/dev/null
```

then:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

This should be considered:

> **A known-good partial initialization configuration, not a proven fully serving production configuration.**

The investigation demonstrated that it can reach substantial model allocation on both A100s.

---

# 43. Recommended Benchmark Environment for Real TP Performance Testing

If the real objective is not just "make TP=2 run" but:

> Measure / learn high-performance 2-GPU tensor parallelism on A100

then a better rented machine would be one where:

```bash
nvidia-smi topo -m
```

shows:

```text
        GPU0    GPU1
GPU0     X      NV#
GPU1    NV#      X
```

and:

```bash
nvidia-smi nvlink --status
```

shows active links.

The current machine instead showed:

```text
GPU0 <-> GPU1 = NODE
```

and all NVLink links inactive.

That makes it a less suitable host for studying the performance characteristics of NVLink-based A100 tensor parallelism.

---

# 44. Minimal Debugging Decision Tree

The entire investigation can be reduced to the following:

```text
Start vLLM TP=2
       |
       v
GPU 100%, VRAM ~700 MiB?
       |
      YES
       |
       v
Check NCCL_DEBUG=INFO
       |
       v
NCCL Init / P2P path hangs?
       |
      YES
       |
       v
Try NCCL_P2P_DISABLE=1
       |
       +----------+
       |          |
     FAIL       PROGRESS
       |          |
       |          v
       |     Model starts loading
       |          |
       |          v
       |     VRAM rises to ~10 GB/GPU
       |          |
       |          v
       |     Is API listening?
       |       /        \
       |     YES         NO
       |      |           |
       |      v           v
       |   Test API   Inspect worker
       |               stack / logs
```

---

# 45. Submission-Friendly Final Findings

### Environment

```text
GPU:              2 × NVIDIA A100-SXM4-40GB
VRAM:             40 GB each
Driver:           595.84
CUDA:             13.2 (driver-reported)
vLLM:             0.29.0
NCCL:             2.30.7
Model:            Qwen/Qwen3.5-9B
Tensor Parallel: 2
```

### GPU topology

```text
GPU0 <-> GPU1: NODE
NVLink:        inactive
CUDA P2P:      reported available
```

### Main startup failure

```text
TP=2 + direct P2P
        ↓
workers spin
        ↓
~100% GPU utilization
        ↓
~711 MiB VRAM
        ↓
model never loads
```

### Main workaround discovered

```bash
NCCL_P2P_DISABLE=1
```

### Result of workaround

```text
NCCL initializes
        ↓
TP ranks connect
        ↓
model starts loading
        ↓
~9.8–10.3 GB allocated per GPU
```

### Remaining problem

```text
Model memory allocated
        ↓
vLLM worker processes alive
        ↓
HTTP server still not listening on :18001
```

Therefore the investigation successfully isolated and worked around the **TP=2 P2P/NCCL startup hang**, but did not complete a full successful OpenAI-compatible serving test for Qwen3.5-9B within the recorded session.

---

# 46. Exact Commands Collected in One Place

## Environment inspection

```bash
nvidia-smi
nvidia-smi dmon -s pucm -d 1
nvidia-smi pmon -i 0,1 -s um -c 10
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi nvlink --status
```

## CUDA P2P test

```bash
python - <<'PY'
import torch
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("0 -> 1:", torch.cuda.can_device_access_peer(0, 1))
print("1 -> 0:", torch.cuda.can_device_access_peer(1, 0))
PY
```

## Process management

```bash
supervisorctl status
supervisorctl stop vllm
pkill -9 -f 'vllm serve|VLLM::Worker|VLLM::EngineCore' 2>/dev/null
ps aux | grep -i vllm
ps -ef | grep -E 'vllm|VLLM::'
```

## Socket/API debugging

```bash
ss -lntp
ss -lntp | grep 18001
curl http://localhost:18001/v1/models
```

## Recommended TP=2 workaround configuration

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --max-model-len 32000 \
  --disable-custom-all-reduce \
  --port 18001
```

---

# 47. Final Technical Summary

The experiment demonstrated that the two A100 GPUs were correctly exposed to CUDA and vLLM, but their topology was PCIe/host-bridge (`NODE`) rather than active NVLink.

The initial TP=2 startup repeatedly produced the characteristic pattern:

```text
2 × VLLM::Worker
GPU utilization ~100%
VRAM ~711 MiB
```

NCCL logs showed successful communicator setup but the application failed to progress to model loading when direct GPU P2P was enabled.

Disabling NCCL P2P with:

```bash
NCCL_P2P_DISABLE=1
```

was the key workaround. After doing so, vLLM reached model loading and allocated roughly 9.8–10.3 GB on each A100.

Subsequent experiments with V1 model runner, eager mode, text-only mode, chunked prefill, shorter context length, and disabled custom/symmetric-memory all-reduce were attempts to solve a separate later-stage initialization problem. Those experiments did not produce a final working HTTP endpoint during the recorded session.

Therefore the most important result is:

> **2-GPU tensor parallelism itself was successfully initialized and model memory was successfully distributed across both A100s. The remaining unresolved problem was getting the Qwen3.5-9B vLLM server to finish startup and expose the OpenAI-compatible API on port 18001.**

---

# 48. External References Used During the Investigation

These are the main types of references consulted while interpreting the behavior:

- NVIDIA `nvidia-smi` and topology documentation
- NVIDIA NCCL environment and troubleshooting documentation
- vLLM CLI and troubleshooting documentation
- vLLM GitHub issues/RFC discussions related to:
  - TP startup hangs
  - NCCL/P2P behavior
  - PCIe/VM topology
  - Qwen3.5 hybrid attention
  - vLLM model-runner behavior

Representative links:

- https://docs.nvidia.com/deploy/nvidia-smi/
- https://docs.nvidia.com/deeplearning/nccl/
- https://docs.vllm.ai/en/latest/cli/serve/
- https://docs.vllm.ai/en/latest/usage/troubleshooting/
- https://github.com/vllm-project/vllm/issues/51513

---

# 49. Final Status

**TP=2 NCCL/P2P startup problem:** ✅ Isolated and worked around

**2-GPU model allocation:** ✅ Demonstrated

**Both A100s actively used by vLLM:** ✅ Demonstrated

**Direct P2P path on this host:** ⚠️ Problematic for this workload

**Active NVLink between the GPUs:** ❌ Not exposed/active

**Qwen3.5-9B model fully serving over `localhost:18001`:** ❌ Not completed in the recorded session

**Next recommended debugging step:** inspect the Python worker stack traces after model allocation to identify exactly what the workers are waiting on before the API server opens port 18001.

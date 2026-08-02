# Paper 9 — Results (validation source)

**Status: ALL RQs COMPLETE 2026-07-19 (RQ2: 41 rows = 36-image battery +
2.4.3 makeup + 4-image NVIDIA_DISABLE_REQUIRE override leg).**
All numbers computed by `analyze.py` from `data/*.csv`; miners:
`mine_dockerhub.py`, `mine_ngc.py`, `mine_pypi.py`, `mine_issues.py`,
`smoke_suite.py`. Collection date: 2026-07-18. USER MUST VALIDATE every
number below against `analyze.py` output before any draft cites it.

## Testbed (RQ2)
Windows 11 + Docker Desktop (WSL2 backend), NVIDIA RTX 3050 Ti Laptop 4GB
(sm_86, Ampere), driver 555.97 (CUDA 12.5 usermode ceiling),
nvidia-container-toolkit legacy mode. Deliberately mid-life driver: the
median practitioner machine, not a fresh install.

## RQ1 — Matrix reality

- nvidia/cuda Docker Hub: **1,784 tags** (1,760 parse to the structured
  scheme `<cuda>-<cudnn?>-<flavor>-<os>`).
- **Combinatorial growth**: tags per CUDA minor rose 35 (11.0) → 226 (13.0);
  distinct OS bases per minor **7 → 14** across the same span.
- **Publication window per CUDA minor collapsed mid-12.x**: 11.0–11.4 tags
  span 531–571 days of pushes; 12.2 spans 11d, 12.4 18d, 12.5 42d — then
  12.8 (458d) and 12.9 (377d) become the new long-lived lines. (Long 11.x
  spans partly reflect batch re-pushes ending 2023-11; state as observed
  push-activity windows, not support commitments.)
- pytorch/pytorch Hub: 168 structured tags; **24/45 framework versions ship
  ≥2 CUDA builds**; 16 distinct CUDA versions across the image history
  (10.0 → 13.2).
- NGC monthlies (release-notes mining, 96 pytorch + 58 tensorflow releases
  2017–2026): **CUDA minor bumps every 4 months (median)** in both; pytorch
  line has **25 distinct minimum-driver requirements** in 96 releases —
  a new driver floor roughly every 4 months.
- PyPI: torch median release gap 49d vs vllm 10d, tensorflow 12d.
  **25/49 torch releases carry nvidia-\* wheel pins** (exact cudnn pin per
  release; 7 distinct cudnn versions) — the "CUDA via pip" channel
  (nvidia-cudnn-cu1x etc.) covers only the recent half of torch history.

## RQ3 — Staleness (Docker Hub, age of last tag push at 2026-07-18)

| repo | tags | median age (d) | >1y old |
|---|---|---|---|
| tensorflow/tensorflow | 1,201 | 2,531 | 97.7% |
| pytorch/pytorch | 180 | 772 | 73.3% |
| nvidia/cuda | 1,784 | 591 | 60.8% |
| ollama/ollama | 1,077 | 317 | 44.9% |
| vllm/vllm-openai | 485 | 80 | 10.3% |

TensorFlow's Docker Hub GPU line is by far the stalest measured (median tag
~7y, 97.7% >1y). **Do NOT call it abandoned** — checked 2026-07-26 against
https://www.tensorflow.org/install/docker: there is NO deprecation notice, Docker
is still presented as the recommended way to run TF on a GPU, and
`tensorflow/tensorflow:latest-gpu` is still the image named. So the honest claim
is publishing staleness, not abandonment, and not "distribution moved elsewhere".
The sharper finding: the officially recommended tag is simultaneously the stalest
in the sample AND (per RQ2) silently falls back to CPU on driver 555.97.
Docker Hub push activity and a vendor's stated support policy are different
things; only the former is measured here.

## RQ4 — Practitioner pain (GitHub issues, 6 repos, 138,591 total issues)

- Narrow error-signature share of ALL issues ("CUDA error" phrase):
  pytorch 6.3%, tensorflow 4.7%, **vllm 9.3%**, llama.cpp 6.7%,
  nvidia-container-toolkit 13.4%, ollama 4.9%.
- Caveat (stated in paper): broad terms ("CUDA version", "cudnn") hit
  issue-template environment dumps; only error-signature queries
  (cuda_error / no_kernel_image / forward_compat / mismatch) are reported
  as pain signal.
- Time-to-close (fetched compat-issue samples, n=8,335):
  frameworks median 10–27d — but **NVIDIA/nvidia-container-toolkit median
  516d, p90 1,367d** (n=239). The thinnest layer in the stack is the
  slowest to fix.

## RQ2 — Measured breakage (COMPLETE)

**Headline matrix** (full per-test rows in `data/rq2_smoke.csv`, raw logs
`logs/rq2/`): 41 tested rows; every framework image that starts, works —
100% of pytorch/pytorch (10/10, 2021→2025), NGC pytorch (3/3 incl. CUDA
12.9), TF 2.4.3→2.16.1 (5/5), vllm, ollama pass GPU compute on driver
555.97. The only failures: (a) all four nvidia/cuda ≥12.6 base images
(toolkit admission gate), (b) the same four with the gate disabled
(segfault at runtime init), (c) tensorflow:latest-gpu (silent CPU
fallback), (d) CUDA 11.0 nvcc not knowing sm_86, (e) one never-published
tag (2.4.4-gpu).

**Failure taxonomy (measured):**
| class | trigger | observed |
|---|---|---|
| toolkit-requirement-gate | NVIDIA_REQUIRE_CUDA label > driver | 4 images, clean refusal, actionable msg |
| segfault-on-init | gate bypassed, cudart ≥12.6 on 12.5 driver | 4 images, rc=139, ZERO diagnostics |
| silent-cpu-fallback | cuDNN init fails inside TF | exit 0, gpus=[], invisible in CI |
| arch-unknown-to-compiler | nvcc older than GPU arch | CUDA 11.0 `-arch=sm_86` |
| tag-gone | never-published version tag | tensorflow:2.4.4-gpu |

Details:

- **nvidia/cuda ladder 11.0→12.5 (11 images): everything works.** t1
  nvidia-smi, default nvcc compile+run, PTX-JIT (compute_50 → sm_86 JIT)
  all pass. Single exception: CUDA 11.0's nvcc predates sm_86 →
  `nvcc fatal: Value 'sm_86' is not defined` (arch-unknown-to-compiler;
  default + PTX paths still pass on 11.0).
- **Hard admission cliff at 12.6**: nvidia/cuda 12.6.3 / 12.8.1 / 12.9.1 /
  13.0.1 all fail EVERY test including bare nvidia-smi — container never
  starts: `nvidia-container-cli: requirement error: unsatisfied condition:
  cuda>=12.6` (toolkit enforces the image's NVIDIA_REQUIRE_CUDA label
  against driver 555.97).
- **Gate bypass leg (NVIDIA_DISABLE_REQUIRE=true, 4 gated images)**:
  containers start, nvidia-smi works, nvcc compiles — but the compiled
  vecadd **segfaults (rc=139) at CUDA runtime init** on all four
  (12.6.3/12.8.1/12.9.1/13.0.1), zero diagnostic output. Same binary+test
  passes on ≤12.5 images on the same stack. The gate converts a bare
  segfault into an actionable error for the raw-toolkit path — justified
  there, NOT mere policy.
- **Frameworks cross the same boundary successfully**:
  pytorch/pytorch:2.6.0-cuda12.6 and 2.7.1-cuda12.8 (pip nvidia-* wheel
  stack) and NGC pytorch:25.06 (CUDA 12.9, bundled compat libs) all pass
  fully on the same 555.97 driver. The compat boundary is real for raw
  CUDA apps but framework distribution channels carry their own
  forward-compat machinery — the matrix's difficulty depends on which
  channel ships your CUDA. pytorch 1.8.1-cuda11.1 (2021) also still fully
  works (backward compat solid).
- **Gate-conservatism extends to 12.8**: pytorch/pytorch:2.7.1-cuda12.8
  (torch 2.7.1+cu128, cuDNN 9.7) passes fully on the same 555.97 driver.
- **Silent CPU fallback (worst class)**: tensorflow/tensorflow:latest-gpu
  (TF 2.21.0) detects **zero GPUs** and exits 0 — cuDNN plugin fails with
  `CUDNN_STATUS_INTERNAL_ERROR`, TF warns and "Skipping registering GPU
  devices"; the canonical floating tag silently computes on CPU while
  2.9.3/2.12.0/2.15.0/2.16.1-gpu all pass GPU matmul. No nonzero exit, no
  exception — invisible in CI.
- tensorflow:2.4.4-gpu never existed on Hub (kept as tag-gone row;
  2.4.3-gpu tested instead — passes, so TF GPU images work 2.4.3→2.16.1
  on this driver; only the floating latest-gpu is broken).
- ollama/ollama:latest starts, sees the GPU, reports
  `total_vram="4.0 GiB"` (t6-style inference test deferred to Paper 8).
- vllm/vllm-openai:v0.6.3 torch stack fully functional (model-load OOM
  scope excluded on 4GB — Paper 8 territory).

## Threats / integrity notes
- ~~Single GPU, single driver version~~ → RESOLVED 2026-07-24: the native-Linux
  leg added a SECOND driver point (RTX 3070, driver 580.95), which showed the
  admission cliff moves with the driver. Both points are still consumer
  Ampere-class, so datacenter cards (where CUDA forward compat IS supported)
  remain untested — that is the honest residual scope limit.
- ~~WSL2 backend: the ≥12.6 segfault should be replicated on native Linux
  before claiming universality~~ → RESOLVED 2026-07-24 by the replication below.
  Native CUDA returns a clean named error ("forward compatibility was attempted
  on non supported HW", rc=1) for the same impossible situation that segfaults
  silently (rc=139) under WSL2. So the SILENT-SEGFAULT PRESENTATION is
  WSL2-specific; the underlying incompatibility is not. Do NOT state the
  segfault as a universal property — paper.tex now scopes it in the abstract,
  C2, RQ2 body, threats and conclusion. The ≤12.5 pass / ≥12.6 fail contrast on
  the SAME stack stands regardless.
- Docker Hub search-count style metrics avoided entirely; only tag metadata.
- GitHub search API total_counts are approximate (GitHub-documented); shares
  reported at the precision GitHub returns.
- NGC release-notes parsing: framework_version field extracted for only
  41/96 pytorch pages (format drift) — cuda + driver fields near-complete
  (96/96, 88/96); cudnn column noisy, re-extract before citing.
- tensorflow/tensorflow "abandonment" claim: RESOLVED 2026-07-26. Official TF
  docs show NO deprecation of the Docker GPU path — it is still recommended and
  still names latest-gpu. The abandonment wording was WRONG and has been removed
  from paper.tex and this file; we now claim measured publishing staleness only.

## NATIVE-LINUX REPLICATION (added 2026-07-24) — resolves WSL2 + single-driver caveats

Native Linux (vast.ai RTX 3070, driver 580.95.05, confirmed KVM via
systemd-detect-virt), docker + nvidia-container-toolkit, nvidia/cuda images
via nested docker. data/native-linux/native_cuda_smoke.csv:

| CUDA image (devel-ubuntu24.04) | normal | bypass |
|---|---|---|
| 12.8.1 | STARTED, rc=0 (works) | rc=0 |
| 13.0.1 | STARTED, rc=0 (works) | rc=0 |
| 13.1.1 | STARTED, **rc=1 "forward compatibility was attempted on non supported HW"** | rc=1 (same) |
| 13.2.1 | STARTED, rc=0 (works) | rc=0 |
| 13.3.0 | STARTED, rc=0 (works) | rc=0 |
| 12.6.3 | PULLFAIL (tag n/a on hub) | — |

**FINDING 1 — cliff moves with driver (multi-driver point):** on driver 580.95
(CUDA-13 era) every image that was GATED on the local 555.97 (CUDA 12.5) driver
now ADMITS/starts. The toolkit gate is driver-relative — cliff LOCATION set by
driver, existence by image-vs-driver. Resolves the "single measured point"
threat (now 2 points).

**FINDING 2 — segfault is WSL2-specific (the important one):** where WSL2 gave
rc=139 SEGFAULT with ZERO diagnostics on a forward-compat-impossible image,
native Linux gives a CLEAN NAMED error: "forward compatibility was attempted on
non supported HW" (rc=1). Same impossible situation (cudart > driver on consumer
Ampere, forward-compat unsupported on consumer GPUs), different presentation:
WSL2 silent segfault vs native actionable error. Scopes P9's segfault-on-init
class as WSL2-path-specific. Resolves the flagged WSL2 caveat.

Anomaly (reported, not over-interpreted): 13.2.1/13.3.0 run clean where 13.1.1
fails forward-compat on the SAME driver — image-specific forward-compat lib
packaging difference. Bypass (NVIDIA_DISABLE_REQUIRE) identical to normal for
all (no toolkit gate triggered on this newer driver).

Integrated into paper.tex RQ2 + threats + future-work. Cost ~$0.10 vast
(3070 ~30min), VM destroyed, credit $2.84. First smoke run (run1) had a
docker-not-ready race → re-ran clean (smoke2) with pre-pull; run1 log kept.

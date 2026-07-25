# Paper 9 — "The CUDA Compatibility Tax: Mining Driver/Toolkit/Framework Breakage Across the AI Container Ecosystem"

**Genre:** mining study (Paper-4 DNA aimed at the AI stack). $0.
**Venue path:** MSR → UCC → IEEE Access.
**Why intense/viral:** "CUDA version hell" is the single most-complained-about
AI-infra problem; no one has measured it. All public data: NGC catalog,
Docker Hub (pytorch/*, tensorflow/*, nvidia/cuda), PyPI wheels metadata,
GitHub issues.

## RQs
- RQ1: Matrix reality — for the AI base images (nvidia/cuda tags, NGC
  PyTorch/TF/Triton monthly releases, official pytorch images): which CUDA
  toolkit / driver / cuDNN / framework combinations does the ecosystem
  actually ship, and how fast do supported windows move? (Tag mining +
  manifest/label extraction.)
- RQ2: Breakage surface — empirical compat testing on the local GPU (driver
  555.97): pull top-N images, run a standard CUDA + framework smoke suite in
  each; failure taxonomy (driver-too-old, PTX forward-compat, cuDNN mismatch,
  arch not compiled ...). Real measured breakage, not documentation claims.
- RQ3: Staleness + CVE exposure — Paper-4 metrics on AI images: tag recency,
  base-OS staleness, scanner findings on the most-pulled AI images (pinned
  scanner version).
- RQ4: Practitioner pain mined — GitHub issue mining (pytorch, vllm,
  llama.cpp, nvidia-container-toolkit): share of issues that are
  version-compat, time-to-resolution, recurring patterns.

## Method notes
Registry APIs + `docker manifest inspect` for metadata at scale; local pulls
bounded (top ~40 images) for RQ2 smoke runs; disk cleaned per batch.

## Cost
$0. Disk + bandwidth only.

## Status
- [ ] Tag/metadata miners (NGC, Docker Hub, PyPI)
- [ ] Smoke suite + local RQ2 runs
- [ ] Issue miner
- [ ] results.md → user validates → 10pp draft

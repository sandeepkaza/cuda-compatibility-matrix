"""RQ2 follow-up: rerun toolkit-gated images with NVIDIA_DISABLE_REQUIRE=true.

Quantifies how much the NVIDIA_REQUIRE_CUDA admission gate over-blocks:
for each image refused by `unsatisfied condition: cuda>=X`, disable the gate
and run the same t1/t3/t4/t5 battery. Rows appended to rq2_smoke.csv with
`+norequire` suffix on the image name.
"""

import csv
import re
import subprocess
import time
from pathlib import Path

from smoke_suite import (DATA, FIELDS, LOGS, VECADD, classify, sh)

GATED = [
    "nvidia/cuda:12.6.3-devel-ubuntu24.04",
    "nvidia/cuda:12.8.1-devel-ubuntu24.04",
    "nvidia/cuda:12.9.1-devel-ubuntu24.04",
    "nvidia/cuda:13.0.1-devel-ubuntu24.04",
]

ENV = ["-e", "NVIDIA_DISABLE_REQUIRE=true"]


def run(image, log):
    row = {"image": image + "+norequire", "kind": "cuda-devel"}
    rc, out, dt = sh(["docker", "pull", image], 3600, log)
    row["pull_s"] = f"{dt:.0f}"
    if rc != 0:
        row["t1_smi"] = classify(rc, out)
        return row
    rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all", *ENV,
                     "--entrypoint", "nvidia-smi", image, "-L"], 180, log)
    row["t1_smi"] = classify(rc, out)
    rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all", *ENV,
                     "--entrypoint", "/bin/sh", image, "-c",
                     "nvcc --version 2>/dev/null | grep release"], 180, log)
    m = re.search(r"release (\d+\.\d+)", out)
    row["toolkit_cuda"] = m.group(1) if m else ""
    row["t2_libs"] = "ok" if rc == 0 else classify(rc, out)
    heredoc = f"cat > /tmp/v.cu <<'EOF'\n{VECADD}\nEOF\n"
    for tname, flags in [("t3_nvcc_default", ""),
                         ("t4_nvcc_sm86", "-arch=sm_86"),
                         ("t5_ptx_jit", "-gencode arch=compute_50,code=compute_50")]:
        script = heredoc + f"nvcc {flags} /tmp/v.cu -o /tmp/v 2>&1 && /tmp/v"
        rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all", *ENV,
                         "--entrypoint", "/bin/bash", image, "-c", script],
                        600, log)
        ok = rc == 0 and "VECADD_OK" in out
        row[tname] = "ok" if ok else classify(rc if rc else 1, out)
    subprocess.run(["docker", "rmi", image], capture_output=True)
    return row


def main():
    out = DATA / "rq2_smoke.csv"
    with open(out, encoding="utf-8") as f:
        done = {r["image"] for r in csv.DictReader(f)}
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        for image in GATED:
            if image + "+norequire" in done:
                print(f"skip {image}", flush=True)
                continue
            print(f"=== {image} +norequire", flush=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", image)
            with open(LOGS / f"{safe}+norequire.log", "w", encoding="utf-8") as log:
                row = run(image, log)
            w.writerow(row)
            f.flush()
            print(f"  {row}", flush=True)


if __name__ == "__main__":
    main()

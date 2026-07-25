"""RQ1 miner: PyPI wheel metadata for the CUDA wheel ecosystem.

Sources (all PyPI JSON API, $0):
 1. torch / tensorflow / jax releases: per-release wheel filenames give the
    python-version x platform support surface; upload dates give cadence.
 2. nvidia-* CUDA runtime wheels (nvidia-cudnn-cu11/cu12, nvidia-cublas-cu12,
    ...): the "CUDA via pip" distribution channel, split by cuXX suffix.
 3. torch dependency pins on nvidia-* wheels per release (requires_dist),
    showing exactly which cuDNN/NCCL each torch release hard-pins.

Outputs: pypi_releases.csv (one row per release), pypi_torch_nvidia_pins.csv.
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

UA = {"User-Agent": "cuda-compat-study/1.0 (academic mining; contact: sandeep.kaza@gmail.com)"}

FRAMEWORKS = ["torch", "tensorflow", "jax", "jax-cuda12-plugin", "onnxruntime-gpu",
              "vllm", "triton"]
NVIDIA_WHEELS = [
    "nvidia-cudnn-cu11", "nvidia-cudnn-cu12", "nvidia-cudnn-cu13",
    "nvidia-cublas-cu11", "nvidia-cublas-cu12", "nvidia-cublas-cu13",
    "nvidia-nccl-cu11", "nvidia-nccl-cu12", "nvidia-nccl-cu13",
    "nvidia-cuda-runtime-cu11", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13",
    "nvidia-cuda-nvcc-cu11", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-nvcc-cu13",
    "nvidia-tensorrt", "tensorrt",
]


def get_json(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))


def mine_package(pkg):
    d = get_json(f"https://pypi.org/pypi/{pkg}/json")
    if d is None:
        print(f"  {pkg}: 404", flush=True)
        return [], {}
    rows = []
    pins = {}
    for ver, files in d.get("releases", {}).items():
        if not files:
            continue
        wheels = [f for f in files if f["filename"].endswith(".whl")]
        pythons = sorted({f["python_version"] for f in wheels})
        plats = sorted({f["filename"].rsplit("-", 1)[-1].removesuffix(".whl")
                        for f in wheels})
        first_upload = min(f["upload_time_iso_8601"] for f in files)
        total_size = sum(f["size"] for f in files)
        rows.append({
            "package": pkg,
            "version": ver,
            "first_upload": first_upload,
            "n_wheels": len(wheels),
            "python_tags": ";".join(pythons),
            "platform_tags": ";".join(plats),
            "total_size_bytes": total_size,
            "yanked": any(f.get("yanked") for f in files),
        })
    # requires_dist of the LATEST version only comes from top-level; per-version
    # pins need per-version JSON (fetched for torch below).
    return rows, d.get("info", {})


def torch_nvidia_pins():
    """Per-torch-release nvidia-* dependency pins (the hard compat contract)."""
    d = get_json("https://pypi.org/pypi/torch/json")
    versions = [v for v, files in d["releases"].items() if files]
    rows = []
    for v in sorted(versions):
        dv = get_json(f"https://pypi.org/pypi/torch/{v}/json")
        if not dv:
            continue
        reqs = dv["info"].get("requires_dist") or []
        nv = [r for r in reqs if r.lower().startswith(("nvidia-", "triton"))]
        for r in nv:
            rows.append({"torch_version": v,
                         "upload": min(f["upload_time_iso_8601"] for f in d["releases"][v]),
                         "pin": r})
        if not nv:
            rows.append({"torch_version": v,
                         "upload": min(f["upload_time_iso_8601"] for f in d["releases"][v]),
                         "pin": ""})
        time.sleep(0.2)
    out = DATA / "pypi_torch_nvidia_pins.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["torch_version", "upload", "pin"])
        w.writeheader()
        w.writerows(rows)
    print(f"torch nvidia pins: {len(rows)} rows -> {out.name}", flush=True)


def main():
    all_rows = []
    for pkg in FRAMEWORKS + NVIDIA_WHEELS:
        rows, _ = mine_package(pkg)
        all_rows.extend(rows)
        print(f"{pkg}: {len(rows)} releases", flush=True)
        time.sleep(0.3)
    out = DATA / "pypi_releases.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"total {len(all_rows)} package releases -> {out.name}", flush=True)
    torch_nvidia_pins()


if __name__ == "__main__":
    main()

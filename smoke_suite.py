"""RQ2: empirical compat smoke suite on the local GPU.

Host: Windows 11, Docker Desktop (WSL2), RTX 3050 Ti Laptop 4GB (sm_86),
driver 555.97 (CUDA 12.5 usermode ceiling). Images newer than CUDA 12.5
exercise the forward-compat / PTX-JIT path — measured, not documented.

Per image: pull (timed) -> run test battery -> docker rmi. Sequential, so
peak disk = one image. Results: data/rq2_smoke.csv + logs/rq2/<image>.log.

Tests:
  T1 nvidia-smi          container-toolkit driver injection
  T2 cuda-libs           report /usr/local/cuda version + runtime libs
  T3 nvcc-default        (devel only) compile vecadd w/ nvcc defaults, run
  T4 nvcc-sm86           (devel only) explicit -arch=sm_86, run
  T5 ptx-jit             (devel only) PTX-only compute_50 -> JIT on sm_86
  T6 torch               (pytorch imgs) is_available, matmul, cudnn ver
  T7 tf                  (tf imgs) list GPUs, matmul on /GPU:0
"""

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
LOGS = HERE / "logs" / "rq2"
DATA.mkdir(exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

VECADD = r"""
#include <cstdio>
__global__ void add(float*a,float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(){int n=1<<20;size_t s=n*sizeof(float);float*a,*b,*c;cudaMallocManaged(&a,s);cudaMallocManaged(&b,s);cudaMallocManaged(&c,s);
for(int i=0;i<n;i++){a[i]=1.f;b[i]=2.f;}add<<<(n+255)/256,256>>>(a,b,c,n);cudaError_t e=cudaDeviceSynchronize();
if(e!=cudaSuccess){printf("CUDA_FAIL %s\n",cudaGetErrorString(e));return 1;}
printf("VECADD_OK %.1f\n",c[0]);return 0;}
"""

TORCH_TEST = (
    "import torch,json;"
    "r={'torch':torch.__version__,'cuda':torch.version.cuda,"
    "'avail':torch.cuda.is_available()};"
    "r['cudnn']=torch.backends.cudnn.version() if r['avail'] else None;"
    "r['dev']=torch.cuda.get_device_name(0) if r['avail'] else None;"
    "r['cap']=('%d.%d'%torch.cuda.get_device_capability(0)) if r['avail'] else None;"
    "x=torch.rand(512,512,device='cuda')@torch.rand(512,512,device='cuda') if r['avail'] else None;"
    "r['matmul']=float(x.sum()) if x is not None else None;"
    "print('TORCH_JSON',json.dumps(r))"
)

TF_TEST = (
    "import tensorflow as tf,json;"
    "g=[d.name for d in tf.config.list_physical_devices('GPU')];"
    "r={'tf':tf.__version__,'gpus':g};\n"
    "if g:\n"
    "  with tf.device('/GPU:0'):\n"
    "    x=tf.linalg.matmul(tf.random.uniform((512,512)),tf.random.uniform((512,512)))\n"
    "  r['matmul']=float(tf.reduce_sum(x))\n"
    "print('TF_JSON',json.dumps(r))"
)

# (image, kind) kind: cuda-devel | cuda-runtime | torch | tf | ngc | vllm | ollama
IMAGES = [
    # nvidia/cuda ladder across toolkit majors (devel = has nvcc)
    ("nvidia/cuda:11.0.3-devel-ubuntu20.04", "cuda-devel"),
    ("nvidia/cuda:11.2.2-devel-ubuntu20.04", "cuda-devel"),
    ("nvidia/cuda:11.4.3-devel-ubuntu20.04", "cuda-devel"),
    ("nvidia/cuda:11.6.2-devel-ubuntu20.04", "cuda-devel"),
    ("nvidia/cuda:11.8.0-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.0.1-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.1.1-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.2.2-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.3.2-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.4.1-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.5.1-devel-ubuntu22.04", "cuda-devel"),
    ("nvidia/cuda:12.6.3-devel-ubuntu24.04", "cuda-devel"),
    ("nvidia/cuda:12.8.1-devel-ubuntu24.04", "cuda-devel"),
    ("nvidia/cuda:12.9.1-devel-ubuntu24.04", "cuda-devel"),
    ("nvidia/cuda:13.0.1-devel-ubuntu24.04", "cuda-devel"),
    # official pytorch images across CUDA generations (runtime flavor)
    ("pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", "torch"),
    ("pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime", "torch"),
    ("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", "torch"),
    ("pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime", "torch"),
    ("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime", "torch"),
    # tensorflow gpu ladder
    ("tensorflow/tensorflow:2.4.3-gpu", "tf"),  # 2.4.4-gpu never published (kept in CSV as tag-gone)
    ("tensorflow/tensorflow:2.9.3-gpu", "tf"),
    ("tensorflow/tensorflow:2.12.0-gpu", "tf"),
    ("tensorflow/tensorflow:2.15.0-gpu", "tf"),
    ("tensorflow/tensorflow:2.16.1-gpu", "tf"),
    ("tensorflow/tensorflow:latest-gpu", "tf"),
    # NGC monthlies (large; keep few) — torch inside
    ("nvcr.io/nvidia/pytorch:22.12-py3", "torch"),
    ("nvcr.io/nvidia/pytorch:24.05-py3", "torch"),
    ("nvcr.io/nvidia/pytorch:25.06-py3", "torch"),
    # inference servers
    ("vllm/vllm-openai:v0.6.3", "vllm"),
    ("ollama/ollama:latest", "ollama"),
]


def sh(args, timeout, log):
    t0 = time.time()
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        log.write(f"\n$ {' '.join(args)}\n[rc={p.returncode} {time.time()-t0:.1f}s]\n{out}\n")
        return p.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired:
        log.write(f"\n$ {' '.join(args)}\nTIMEOUT after {timeout}s\n")
        return -9, "TIMEOUT", time.time() - t0


def classify(rc, out):
    """Failure taxonomy from error text."""
    if rc == 0:
        return "ok"
    o = out.lower()
    if "timeout" in o and rc == -9:
        return "timeout"
    if "forward compatibility" in o or "cuda_error_system_driver_mismatch" in o:
        return "driver-too-old"
    if "no kernel image" in o or "cuda_error_no_binary_for_gpu" in o:
        return "arch-not-compiled"
    if "ptx" in o and ("unsupported" in o or "invalid" in o):
        return "ptx-version"
    if "cudnn" in o:
        return "cudnn-mismatch"
    if "libcuda" in o or "driver version is insufficient" in o or "error 803" in o:
        return "driver-too-old"
    if "out of memory" in o:
        return "oom"
    if "modulenotfounderror" in o or "importerror" in o:
        return "import-error"
    if "manifest" in o and "not found" in o:
        return "tag-gone"
    return "other"


def test_image(image, kind, writer, f):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", image)
    with open(LOGS / f"{safe}.log", "w", encoding="utf-8") as log:
        row = {"image": image, "kind": kind, "pull_s": "", "size_mb": ""}
        rc, out, dt = sh(["docker", "pull", image], 3600, log)
        row["pull_s"] = f"{dt:.0f}"
        if rc != 0:
            row["t1_smi"] = classify(rc, out)
            writer.writerow(row)
            f.flush()
            return
        rc2, out2, _ = sh(["docker", "image", "inspect", image,
                           "--format", "{{.Size}}"], 60, log)
        if rc2 == 0:
            row["size_mb"] = f"{int(out2.strip())/1e6:.0f}"

        rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                         "--entrypoint", "nvidia-smi", image, "-L"], 180, log)
        row["t1_smi"] = classify(rc, out)

        ep = "/bin/sh" if kind != "ollama" else "/bin/sh"
        rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all", "--entrypoint",
                         ep, image, "-c",
                         "cat /usr/local/cuda/version* 2>/dev/null; "
                         "nvcc --version 2>/dev/null | grep release; "
                         "ldconfig -p 2>/dev/null | grep -E 'libcudart|libcudnn' | head -5"],
                        180, log)
        m = re.search(r"release (\d+\.\d+)", out)
        row["toolkit_cuda"] = m.group(1) if m else ""
        row["t2_libs"] = "ok" if rc == 0 else classify(rc, out)

        if kind == "cuda-devel":
            heredoc = f"cat > /tmp/v.cu <<'EOF'\n{VECADD}\nEOF\n"
            for tname, nvccflags in [
                ("t3_nvcc_default", ""),
                ("t4_nvcc_sm86", "-arch=sm_86"),
                ("t5_ptx_jit", "-gencode arch=compute_50,code=compute_50"),
            ]:
                script = (heredoc +
                          f"nvcc {nvccflags} /tmp/v.cu -o /tmp/v 2>&1 && /tmp/v")
                rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                                 "--entrypoint", "/bin/bash", image, "-c", script],
                                600, log)
                ok = rc == 0 and "VECADD_OK" in out
                row[tname] = "ok" if ok else classify(rc if rc else 1, out)
        elif kind == "torch":
            rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                             "--entrypoint", "python", image, "-c", TORCH_TEST],
                            600, log)
            m = re.search(r'TORCH_JSON ({.*})', out)
            ok = False
            if rc == 0 and m:
                import json as _json
                ok = _json.loads(m.group(1)).get("matmul") is not None
            row["t6_torch"] = "ok" if ok else classify(rc if rc else 1, out)
            row["detail"] = m.group(1)[:300] if m else (out.strip().splitlines()[-1][:300] if out.strip() else "")
        elif kind == "tf":
            rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                             "--entrypoint", "python", image, "-c", TF_TEST],
                            600, log)
            ok = rc == 0 and '"matmul"' in out
            row["t7_tf"] = "ok" if ok else ("no-gpu-visible" if rc == 0 else classify(1, out))
            m = re.search(r"TF_JSON (.+)", out)
            row["detail"] = m.group(1)[:300] if m else out.strip().splitlines()[-1][:300] if out.strip() else ""
        elif kind == "vllm":
            # 4GB VRAM: attempt tiny model load is out of scope; check torch stack inside
            rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                             "--entrypoint", "python3", image, "-c", TORCH_TEST],
                            600, log)
            ok = rc == 0 and "TORCH_JSON" in out
            row["t6_torch"] = "ok" if ok else classify(rc if rc else 1, out)
            m = re.search(r"TORCH_JSON (.+)", out)
            row["detail"] = m.group(1)[:300] if m else ""
        elif kind == "ollama":
            # serve then query version + GPU detection line from logs
            rc, out, _ = sh(["docker", "run", "--rm", "--gpus", "all",
                             "--entrypoint", "/bin/sh", image, "-c",
                             "ollama serve & sleep 8; ollama --version 2>&1"],
                            240, log)
            row["detail"] = out.strip().splitlines()[-1][:300] if out.strip() else ""
            row["t2_libs"] = "ok" if "version" in out.lower() else classify(rc, out)

        writer.writerow(row)
        f.flush()
    # keep base 12.4.1 (session test image); drop everything else
    if image != "nvidia/cuda:12.4.1-base-ubuntu22.04":
        subprocess.run(["docker", "rmi", image], capture_output=True)


FIELDS = ["image", "kind", "pull_s", "size_mb", "toolkit_cuda", "t1_smi",
          "t2_libs", "t3_nvcc_default", "t4_nvcc_sm86", "t5_ptx_jit",
          "t6_torch", "t7_tf", "detail"]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out = DATA / "rq2_smoke.csv"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f)}
    mode = "a" if done else "w"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not done:
            w.writeheader()
        for image, kind in IMAGES:
            if image in done:
                print(f"skip (done): {image}", flush=True)
                continue
            if only and only not in image:
                continue
            print(f"=== {image} [{kind}]", flush=True)
            t0 = time.time()
            try:
                test_image(image, kind, w, f)
            except Exception as e:
                print(f"  EXC {e}", flush=True)
                w.writerow({"image": image, "kind": kind, "detail": f"EXC {e}"[:300]})
                f.flush()
            print(f"  done in {time.time()-t0:.0f}s", flush=True)
    print("RQ2 smoke complete", flush=True)


if __name__ == "__main__":
    main()

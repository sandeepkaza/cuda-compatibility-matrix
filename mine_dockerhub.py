"""RQ1/RQ3 miner: Docker Hub tag metadata for AI base images.

Pulls the full tag list (name, last_pushed, full_size, architectures) for
each target repo via the public Hub v2 API and writes one CSV per repo plus
a combined parsed CSV with CUDA/cuDNN/flavor/OS fields extracted from tag
names where the repo uses a structured scheme.

$0 / unauthenticated; Hub allows anonymous tag listing (rate limit ~ enough
with page_size=100 and polite sleep).
"""

import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

REPOS = [
    "nvidia/cuda",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "ollama/ollama",
    "vllm/vllm-openai",
]

UA = {"User-Agent": "cuda-compat-study/1.0 (academic mining; contact: sandeep.kaza@gmail.com)"}


def get_json(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  retry {i+1} after {e} ({wait}s)", flush=True)
            time.sleep(wait)


def mine_repo(repo):
    out = DATA / f"hub_tags_{repo.replace('/', '_')}.csv"
    rows = []
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"
    page = 0
    while url:
        d = get_json(url)
        page += 1
        for t in d.get("results", []):
            archs = sorted({f"{im.get('os')}/{im.get('architecture')}"
                            for im in t.get("images", []) if im.get("architecture")})
            digests = {im.get("digest") for im in t.get("images", []) if im.get("digest")}
            rows.append({
                "repo": repo,
                "tag": t["name"],
                "last_pushed": t.get("tag_last_pushed") or "",
                "full_size": t.get("full_size") or "",
                "architectures": ";".join(archs),
                "digest": sorted(digests)[0] if digests else "",
            })
        url = d.get("next")
        if page % 10 == 0:
            print(f"  {repo}: page {page}, {len(rows)} tags", flush=True)
        time.sleep(0.3)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{repo}: {len(rows)} tags -> {out.name}", flush=True)
    return rows


# nvidia/cuda tag scheme: 12.4.1-cudnn-runtime-ubuntu22.04, 11.8.0-base-ubi8, ...
NVCUDA_RE = re.compile(
    r"^(?P<cuda>\d+\.\d+(?:\.\d+)?)"
    r"(?:-(?P<cudnn>cudnn\d*))?"
    r"-(?P<flavor>base|runtime|devel)"
    r"-(?P<os>.+)$"
)
# pytorch/pytorch: 2.3.1-cuda12.1-cudnn8-runtime
PYTORCH_RE = re.compile(
    r"^(?P<fw>\d+\.\d+(?:\.\d+)?)"
    r"-cuda(?P<cuda>\d+\.\d+)"
    r"-cudnn(?P<cudnn>\d+)"
    r"-(?P<flavor>runtime|devel)$"
)
# tensorflow/tensorflow: 2.16.1-gpu, 2.16.1-gpu-jupyter, latest-gpu
TF_RE = re.compile(r"^(?P<fw>\d+\.\d+\.\d+)-gpu(?:-jupyter)?$")


def parse_all(all_rows):
    out = DATA / "hub_tags_parsed.csv"
    parsed = []
    for r in all_rows:
        repo, tag = r["repo"], r["tag"]
        m = None
        d = {"repo": repo, "tag": tag, "last_pushed": r["last_pushed"],
             "framework_version": "", "cuda": "", "cudnn": "", "flavor": "", "os": ""}
        if repo == "nvidia/cuda":
            m = NVCUDA_RE.match(tag)
            if m:
                d.update(cuda=m["cuda"], cudnn=m["cudnn"] or "", flavor=m["flavor"], os=m["os"])
        elif repo == "pytorch/pytorch":
            m = PYTORCH_RE.match(tag)
            if m:
                d.update(framework_version=m["fw"], cuda=m["cuda"],
                         cudnn=m["cudnn"], flavor=m["flavor"])
        elif repo == "tensorflow/tensorflow":
            m = TF_RE.match(tag)
            if m:
                d.update(framework_version=m["fw"], flavor="gpu")
        if m:
            parsed.append(d)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(parsed[0].keys()))
        w.writeheader()
        w.writerows(parsed)
    print(f"parsed {len(parsed)} structured tags -> {out.name}", flush=True)


def main():
    all_rows = []
    for repo in REPOS:
        try:
            all_rows.extend(mine_repo(repo))
        except Exception as e:
            print(f"FAILED {repo}: {e}", flush=True)
    parse_all(all_rows)


if __name__ == "__main__":
    main()

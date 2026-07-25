"""RQ1 miner: NGC catalog tags + per-release CUDA/driver/framework versions.

Two sources:
 1. NGC v2 API (public, guest-pull repos): tag list + per-tag push dates,
    sizes, scan status -> ngc_images_<repo>.csv
 2. NVIDIA framework container release notes (docs.nvidia.com): per monthly
    release (e.g. 24.05) the shipped CUDA toolkit, cuDNN, NCCL, TensorRT,
    framework version, and REQUIRED DRIVER -> ngc_release_matrix.csv
    This is the ground truth for "how fast do supported windows move".
"""

import csv
import json
import re
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

UA = {"User-Agent": "cuda-compat-study/1.0 (academic mining; contact: sandeep.kaza@gmail.com)"}

NGC_REPOS = ["pytorch", "tensorflow", "tritonserver", "tensorrt", "nemo"]

# release-notes doc slugs per repo (frameworks docs site)
RELNOTES = {
    "pytorch": "pytorch-release-notes",
    "tensorflow": "tensorflow-release-notes",
}


def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise  # permanent; caller probes slug variants
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))


def mine_ngc_images(repo):
    # NGC ignores page-size/page-number and returns the full image list
    url = f"https://api.ngc.nvidia.com/v2/repos/nvidia/{repo}/images"
    d = json.loads(get(url))
    rows = []
    for im in d.get("images", []):
        archs = sorted({v.get("architecture", "") for v in im.get("architectureVariants", [])})
        rows.append({
            "repo": f"nvidia/{repo}",
            "tag": im.get("tag", ""),
            "pushed": im.get("pushedDate", ""),
            "size": im.get("size", ""),
            "architectures": ";".join(a for a in archs if a),
            "scan_date": im.get("scanDate", ""),
        })
    out = DATA / f"ngc_images_{repo}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"nvidia/{repo}: {len(rows)} tags -> {out.name}", flush=True)
    return rows


# ---- release notes scraping ----------------------------------------------
# Each monthly release page: https://docs.nvidia.com/deeplearning/frameworks/
#   pytorch-release-notes/rel-24-05.html (older: rel_21-05.html variants)

VER_PATTERNS = {
    "cuda": re.compile(r"(?:NVIDIA )?CUDA(?:[® ]|&#174;|&reg;)*\s*(\d+\.\d+(?:\.\d+)*)", re.I),
    "cudnn": re.compile(r"cuDNN[^0-9]{0,20}(\d+\.\d+(?:\.\d+)*)", re.I),
    "nccl": re.compile(r"NCCL[^0-9]{0,20}(\d+\.\d+(?:\.\d+)*)", re.I),
    "tensorrt": re.compile(r"TensorRT[^0-9]{0,20}(\d+\.\d+(?:\.\d+)*)", re.I),
    "driver_min": re.compile(r"(?:[Dd]river (?:release )?)(\d{3}(?:\.\d+)*)"),
}
FW_PATTERNS = {
    "pytorch": re.compile(r"PyTorch[^0-9]{0,30}?(\d+\.\d+\.\d+[a-z0-9.+]*)", re.I),
    "tensorflow": re.compile(r"TensorFlow[^0-9]{0,30}?(\d+\.\d+\.\d+)", re.I),
}


def strip_tags(html):
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html)


def mine_relnotes(repo, slug, releases):
    rows = []
    for rel in releases:
        # doc slug uses rel-XX-YY (dots -> dashes); py3 suffix not part of slug
        # >=21.x pages use rel-XX-YY.html; older use rel_XX-YY.html
        text = None
        for prefix in ("rel-", "rel_"):
            url = (f"https://docs.nvidia.com/deeplearning/frameworks/{slug}/"
                   f"{prefix}{rel.replace('.', '-')}.html")
            try:
                text = strip_tags(get(url))
                break
            except Exception:
                continue
        if text is None:
            print(f"  {repo} {rel}: fetch failed (both slug variants)", flush=True)
            continue
        row = {"repo": repo, "release": rel, "url": url}
        for k, pat in VER_PATTERNS.items():
            m = pat.search(text)
            row[k] = m.group(1) if m else ""
        m = FW_PATTERNS[repo].search(text)
        row["framework_version"] = m.group(1) if m else ""
        rows.append(row)
        time.sleep(0.3)
    out = DATA / f"ngc_relnotes_{repo}.csv"
    fields = ["repo", "release", "framework_version", "cuda", "cudnn", "nccl",
              "tensorrt", "driver_min", "url"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    got = sum(1 for r in rows if r["cuda"])
    print(f"{repo} release notes: {len(rows)} pages, {got} with CUDA ver -> {out.name}", flush=True)


def main():
    all_tags = {}
    for repo in NGC_REPOS:
        try:
            all_tags[repo] = mine_ngc_images(repo)
        except Exception as e:
            print(f"FAILED nvidia/{repo}: {e}", flush=True)
    for repo, slug in RELNOTES.items():
        # monthly releases YY.MM from tag list (e.g. 24.05 from 24.05-py3)
        rels = sorted({m.group(1) for r in all_tags.get(repo, [])
                       if (m := re.match(r"^(\d{2}\.\d{2})", r["tag"]))})
        mine_relnotes(repo, slug, rels)


if __name__ == "__main__":
    main()

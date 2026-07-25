"""P9 analysis: reads data/*.csv, prints stats blocks for results.md.

Sections:
  A. RQ1 matrix reality — CUDA ladder shipped on Docker Hub; NGC monthly
     CUDA/driver window movement; PyPI wheel cadence + torch nvidia pins.
  B. RQ3 staleness — tag recency distributions per repo.
  C. RQ4 practitioner pain — compat-issue shares, time-to-resolution.
  D. RQ2 smoke results (when rq2_smoke.csv complete).

Run: python analyze.py > analysis.txt  (numbers only; results.md hand-written
from this output so every claim traces to a computed line)
"""

import csv
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).parent / "data"
NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


def load(name):
    p = DATA / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dt(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "n/a"


print("=" * 70)
print("A. RQ1 — MATRIX REALITY")
print("=" * 70)

parsed = load("hub_tags_parsed.csv")
nv = [r for r in parsed if r["repo"] == "nvidia/cuda"]
print(f"\nnvidia/cuda structured tags: {len(nv)}")
cuda_minor = Counter(".".join(r["cuda"].split(".")[:2]) for r in nv)
print("tags per CUDA minor:", dict(sorted(cuda_minor.items(),
      key=lambda kv: [int(x) for x in kv[0].split(".")])))

# distinct OS bases per CUDA minor — combinatorial surface
os_per = defaultdict(set)
for r in nv:
    os_per[".".join(r["cuda"].split(".")[:2])].add(r["os"])
sizes = {k: len(v) for k, v in sorted(os_per.items(),
         key=lambda kv: [int(x) for x in kv[0].split(".")])}
print("distinct OS bases per CUDA minor:", sizes)

# lifespan of each CUDA minor on hub (first->last push of its tags)
print("\nCUDA minor lifespan on Docker Hub (first push -> last push, days):")
for minor in sorted(os_per, key=lambda s: [int(x) for x in s.split(".")]):
    pushes = [dt(r["last_pushed"]) for r in nv
              if r["cuda"].startswith(minor)]
    pushes = [p for p in pushes if p]
    if len(pushes) >= 2:
        span = (max(pushes) - min(pushes)).days
        print(f"  {minor}: {span}d, last activity {max(pushes).date()}")

pt = [r for r in parsed if r["repo"] == "pytorch/pytorch"]
print(f"\npytorch/pytorch structured tags: {len(pt)}")
fw_cuda = sorted({(r['framework_version'], r['cuda']) for r in pt})
cudas_per_fw = defaultdict(set)
for f, c in fw_cuda:
    cudas_per_fw[f].add(c)
multi = {f: sorted(cs) for f, cs in cudas_per_fw.items() if len(cs) > 1}
print(f"framework versions with >1 CUDA build: {len(multi)}/{len(cudas_per_fw)}")
cuda_span = Counter(c for _, c in fw_cuda)
print("distinct CUDA versions across pytorch images:", sorted(cuda_span))

# NGC window movement
for repo in ("pytorch", "tensorflow"):
    rn = load(f"ngc_relnotes_{repo}.csv")
    rn = [r for r in rn if r["cuda"]]
    print(f"\nNGC {repo} release notes: {len(rn)} releases with CUDA version")
    # CUDA minor change cadence
    seq = []
    for r in sorted(rn, key=lambda r: r["release"]):
        minor = ".".join(r["cuda"].split(".")[:2])
        seq.append((r["release"], minor, r["driver_min"]))
    changes = [(a, b) for a, b in zip(seq, seq[1:]) if a[1] != b[1]]
    print(f"  CUDA minor bumps across {len(seq)} monthlies: {len(changes)}")
    if changes:
        # months between bumps
        def mnum(rel):
            y, m = rel.split(".")[:2]
            return int(y) * 12 + int(m)
        bump_rels = [b[0] for _, b in changes]
        gaps = [mnum(b) - mnum(a) for a, b in zip(bump_rels, bump_rels[1:])]
        if gaps:
            print(f"  months between CUDA minor bumps: med {statistics.median(gaps)}, "
                  f"mean {statistics.mean(gaps):.1f}")
    drivers = [r["driver_min"] for r in rn if r["driver_min"]]
    print(f"  distinct min-driver requirements: {len(set(drivers))}")

pins = load("pypi_torch_nvidia_pins.csv")
by_ver = defaultdict(list)
for r in pins:
    if r["pin"]:
        by_ver[r["torch_version"]].append(r["pin"])
print(f"\ntorch releases with nvidia-* wheel pins: {len(by_ver)} "
      f"(of {len({r['torch_version'] for r in pins})})")
cudnn_pins = {}
for v, ps in sorted(by_ver.items()):
    for p in ps:
        m = re.match(r"nvidia-cudnn-cu(\d+)[^0-9]*==?\s*([\d.]+)", p)
        if m:
            cudnn_pins[v] = f"cu{m.group(1)}=={m.group(2)}"
print(f"exact cudnn pins: {len(cudnn_pins)} torch versions, "
      f"{len(set(cudnn_pins.values()))} distinct cudnn versions")

rel = load("pypi_releases.csv")
for pkg in ("torch", "tensorflow", "vllm"):
    rows = [r for r in rel if r["package"] == pkg and not r["version"].endswith("rc")]
    dates = sorted(d for r in rows if (d := dt(r["first_upload"])))
    if len(dates) > 3:
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        print(f"{pkg}: {len(dates)} releases, med gap {statistics.median(gaps)}d")

print()
print("=" * 70)
print("B. RQ3 — STALENESS")
print("=" * 70)
for name in ("nvidia_cuda", "pytorch_pytorch", "tensorflow_tensorflow",
             "ollama_ollama", "vllm_vllm-openai"):
    rows = load(f"hub_tags_{name}.csv")
    ages = [(NOW - d).days for r in rows if (d := dt(r["last_pushed"]))]
    if ages:
        ages.sort()
        stale = sum(1 for a in ages if a > 365)
        print(f"{name}: {len(ages)} tags, median age {ages[len(ages)//2]}d, "
              f">1y old: {pct(stale, len(ages))}")

print()
print("=" * 70)
print("C. RQ4 — PRACTITIONER PAIN")
print("=" * 70)
counts = load("gh_issue_counts.csv")
by_repo = defaultdict(dict)
for r in counts:
    by_repo[r["repo"]][r["query"]] = int(r["count"])
print(f"\n{'repo':<32}{'total':>8}{'cuda_err':>9}{'no_kimg':>8}"
      f"{'fwd_cmp':>8}{'mismatch':>9}{'err_share':>10}")
for repo, q in by_repo.items():
    tot = q.get("TOTAL_ISSUES", 0)
    ce = q.get("cuda_error", 0)
    print(f"{repo:<32}{tot:>8}{ce:>9}{q.get('no_kernel_image',0):>8}"
          f"{q.get('forward_compat',0):>8}{q.get('mismatch',0):>9}"
          f"{pct(ce, tot):>10}")
print("\nNOTE: 'cuda_version'/'driver_version'/'cudnn' counts are inflated by "
      "issue-template env dumps (collect_env) — report as mention-rate only, "
      "use cuda_error/no_kernel_image/forward_compat as the pain signal.")

det = load("gh_issues_detail.csv")
print(f"\nissue details: {len(det)}")
for repo in by_repo:
    rows = [r for r in det if r["repo"] == repo]
    closed = [r for r in rows if r["closed_at"]]
    ttrs = []
    for r in closed:
        c, o = dt(r["closed_at"]), dt(r["created_at"])
        if c and o:
            ttrs.append((c - o).days)
    if ttrs:
        openpct = pct(len(rows) - len(closed), len(rows))
        print(f"  {repo}: {len(rows)} fetched, open {openpct}, "
              f"time-to-close med {statistics.median(ttrs):.0f}d "
              f"(p90 {sorted(ttrs)[int(len(ttrs)*0.9)]}d)")

print()
print("=" * 70)
print("D. RQ2 — SMOKE (partial until battery completes)")
print("=" * 70)
smoke = load("rq2_smoke.csv")
print(f"images tested: {len(smoke)}")
for r in smoke:
    tests = {k: v for k, v in r.items()
             if k.startswith("t") and k[1].isdigit() and v}
    print(f"  {r['image']}: {tests}")

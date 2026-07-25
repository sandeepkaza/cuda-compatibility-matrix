"""RQ4 miner: practitioner pain — version-compat issues on GitHub.

For each target repo:
  * total issue count (baseline denominator)
  * count of issues matching each compat query (search total_count)
  * up to 1000 matched issues per repo (search cap) with created/closed/state
    -> time-to-resolution and recurrence analysis

Auth: token from `gh auth token` (5000 core / 30 search req/min).
Search queries hit title+body; counts are per-query and a deduped union is
computed from the fetched issue numbers.
"""

import csv
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True,
                       text=True, check=True).stdout.strip()
HDRS = {"Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "cuda-compat-study/1.0"}

REPOS = [
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "vllm-project/vllm",
    "ggml-org/llama.cpp",
    "NVIDIA/nvidia-container-toolkit",
    "ollama/ollama",
]

# compat query set; quoted phrases target version-compat pain specifically
QUERIES = {
    "cuda_version": '"CUDA version"',
    "driver_version": '"driver version"',
    "cudnn": "cudnn",
    "cuda_error": '"CUDA error"',
    "incompatible": "cuda incompatible",
    "no_kernel_image": '"no kernel image"',
    "ptx": "ptx",
    "forward_compat": '"forward compatibility"',
    "mismatch": "cuda mismatch",
}


def get_json(url, retries=6):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=60) as r:
                remaining = int(r.headers.get("X-RateLimit-Remaining", "99"))
                reset = int(r.headers.get("X-RateLimit-Reset", "0"))
                d = json.load(r)
                if remaining <= 1:
                    wait = max(reset - time.time(), 0) + 2
                    print(f"  rate limit, sleeping {wait:.0f}s", flush=True)
                    time.sleep(wait)
                return d
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset", "0"))
                wait = max(reset - time.time(), 0) + 5 if reset else 65
                print(f"  {e.code}, sleeping {wait:.0f}s", flush=True)
                time.sleep(min(wait, 300))
            elif i == retries - 1:
                raise
            else:
                time.sleep(2 ** (i + 1))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"exhausted retries: {url}")


def search_count(q):
    url = "https://api.github.com/search/issues?per_page=1&q=" + urllib.parse.quote(q)
    d = get_json(url)
    time.sleep(2.2)  # 30 search req/min
    return d.get("total_count", 0)


def search_issues(q, cap=1000):
    """Fetch up to `cap` (API max 1000) results, newest first."""
    items = []
    for page in range(1, cap // 100 + 1):
        url = ("https://api.github.com/search/issues?per_page=100"
               f"&page={page}&sort=created&order=desc&q=" + urllib.parse.quote(q))
        d = get_json(url)
        batch = d.get("items", [])
        items.extend(batch)
        time.sleep(2.2)
        if len(batch) < 100:
            break
    return items


def main():
    counts_rows = []
    issue_rows = []
    seen = set()  # (repo, number)
    for repo in REPOS:
        base = f"repo:{repo} is:issue"
        total = search_count(base)
        counts_rows.append({"repo": repo, "query": "TOTAL_ISSUES",
                            "count": total})
        print(f"{repo}: {total} total issues", flush=True)
        for name, q in QUERIES.items():
            full = f"{base} {q}"
            n = search_count(full)
            counts_rows.append({"repo": repo, "query": name, "count": n})
            print(f"  {name}: {n}", flush=True)
        # fetch details for the two highest-signal queries per repo
        for name in ("cuda_version", "cuda_error"):
            full = f"{base} {QUERIES[name]}"
            for it in search_issues(full):
                key = (repo, it["number"])
                if key in seen:
                    continue
                seen.add(key)
                issue_rows.append({
                    "repo": repo,
                    "number": it["number"],
                    "matched_query": name,
                    "title": (it.get("title") or "")[:200],
                    "state": it.get("state", ""),
                    "created_at": it.get("created_at", ""),
                    "closed_at": it.get("closed_at") or "",
                    "comments": it.get("comments", 0),
                    "labels": ";".join(l["name"] for l in it.get("labels", [])),
                })
        print(f"  fetched {sum(1 for r in issue_rows if r['repo'] == repo)} issue details", flush=True)

    with open(DATA / "gh_issue_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "query", "count"])
        w.writeheader()
        w.writerows(counts_rows)
    with open(DATA / "gh_issues_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(issue_rows[0].keys()))
        w.writeheader()
        w.writerows(issue_rows)
    print(f"counts: {len(counts_rows)} rows; details: {len(issue_rows)} issues", flush=True)


if __name__ == "__main__":
    main()

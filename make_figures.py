"""Paper 9 figures (vector PDF, IEEE single-column 3.5in).

Fig 1 fig_growth.pdf   -- tags-per-CUDA-minor + OS-bases-per-minor (surface explosion, RQ1)
Fig 2 fig_matrix.pdf   -- RQ2 measured pass/fail grid (the compat cliff at 12.6)
Fig 3 fig_stale.pdf    -- median tag age per repo (RQ3 staleness)
Fig 4 fig_ttc.pdf      -- time-to-close compat issues per repo (RQ4, container-toolkit outlier)
Reads data/*.csv (+ constants from analysis.txt). Run: python make_figures.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data"
BLUE, ORANGE, GREEN, RED, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "font.family": "serif", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42,
})

# ---- Fig 1: surface explosion (from analysis.txt constants, minor-ordered) ----
minors = ["11.0","11.2","11.4","11.6","11.8","12.0","12.2","12.4","12.6","12.8","12.9","13.0","13.2","13.3"]
tags = {"11.0":35,"11.2":35,"11.4":30,"11.6":55,"11.8":35,"12.0":80,"12.2":64,"12.4":54,"12.6":164,"12.8":132,"12.9":162,"13.0":226,"13.2":128,"13.3":62}
osb  = {"11.0":7,"11.2":7,"11.4":6,"11.6":6,"11.8":7,"12.0":9,"12.2":8,"12.4":8,"12.6":9,"12.8":11,"12.9":12,"13.0":14,"13.2":14,"13.3":14}
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x = range(len(minors))
ax.bar(x, [tags[m] for m in minors], color=BLUE, label="tags per CUDA minor")
ax.set_ylabel("nvidia/cuda tags", color=BLUE)
ax.set_xticks(list(x), minors, rotation=60, fontsize=6)
ax2 = ax.twinx()
ax2.plot(x, [osb[m] for m in minors], "o-", color=ORANGE, linewidth=1.4, markersize=3, label="distinct OS bases")
ax2.set_ylabel("distinct OS bases", color=ORANGE)
ax2.set_ylim(0, 16)
ax.set_title("RQ1: matrix surface grows 35→226 tags, 7→14 OS", fontsize=6.8)
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_growth.pdf"); print("fig_growth.pdf")

# ---- Fig 2: RQ2 pass/fail matrix ----
rows = list(csv.DictReader(open(DATA / "rq2_smoke.csv", encoding="utf-8")))
tests = ["t1_smi","t2_libs","t3_nvcc_default","t4_nvcc_sm86","t5_ptx_jit","t6_torch","t7_tf"]
tlabel = ["smi","libs","nvcc","sm86","ptx","torch","tf"]
# select the nvidia/cuda ladder + a few frameworks to show the cliff
sel = [r for r in rows if r["image"].startswith("nvidia/cuda:")]
sel = sorted(sel, key=lambda r: r["image"])
fig, ax = plt.subplots(figsize=(3.5, 3.0))
def cell(v):
    return {"ok":1, "other":-1, "no-gpu-visible":-0.5, "":0}.get(v, 0)
import numpy as np
M = np.array([[cell(r.get(t,"")) for t in tests] for r in sel])
cmap = plt.matplotlib.colors.ListedColormap([RED, "#f0f0f0", GREEN])
ax.imshow(np.clip(M, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(tests)), tlabel, fontsize=6, rotation=45)
ax.set_yticks(range(len(sel)), [r["image"].replace("nvidia/cuda:","").replace("-devel","")[:16] for r in sel], fontsize=5.6)
for i in range(len(sel)):
    for j in range(len(tests)):
        if M[i,j]==-1: ax.text(j,i,"F",ha="center",va="center",fontsize=5.5,color="white")
        elif M[i,j]==1: ax.text(j,i,"P",ha="center",va="center",fontsize=5,color="white")
ax.axhline(10.5, color=INK, linewidth=1.2)
ax.set_title("RQ2: nvidia/cuda — cliff at 12.6", fontsize=7)
fig.tight_layout(); fig.savefig("fig_matrix.pdf"); print("fig_matrix.pdf")

# ---- Fig 3: staleness ----
stale = [("tensorflow",2531,97.7),("pytorch",772,73.3),("nvidia/cuda",591,60.8),("ollama",317,44.9),("vllm",80,10.3)]
fig, ax = plt.subplots(figsize=(3.5, 2.1))
names=[s[0] for s in stale]; ages=[s[1] for s in stale]
cols=[RED if s[1]>1000 else (ORANGE if s[1]>400 else GREEN) for s in stale]
ax.barh(range(len(names)), ages, color=cols, height=0.62)
for i,(n,a,p) in enumerate(stale):
    ax.annotate(f"{a}d ({p:.0f}% >1y)", (a+30,i), va="center", fontsize=6, color=MUTED)
ax.set_yticks(range(len(names)), names, fontsize=6.6)
ax.set_xlabel("Median tag age (days), snapshot 2026-07-18")
ax.set_xlim(0, 3200)
ax.set_title("RQ3: TensorFlow's GPU line ~7y stale", fontsize=6.8)
ax.spines[["top","right"]].set_visible(False)
ax.xaxis.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("fig_stale.pdf"); print("fig_stale.pdf")

# ---- Fig 4: time-to-close ----
ttc = [("nvidia-container-toolkit",516,1367),("vllm",27,152),("tensorflow",20,377),("llama.cpp",11,78),("pytorch",10,150),("ollama",10,131)]
ttc.sort(key=lambda t:-t[1])
fig, ax = plt.subplots(figsize=(3.5, 2.1))
names=[t[0] for t in ttc]; med=[t[1] for t in ttc]; p90=[t[2] for t in ttc]
cols=[RED if t[1]>100 else BLUE for t in ttc]
ax.barh(range(len(names)), med, color=cols, height=0.6)
for i,(n,m,p) in enumerate(ttc):
    ax.plot([m,p],[i,i],color=MUTED,linewidth=0.7)
    ax.annotate(f"{m}d",(m+8,i),va="center",fontsize=6,color=MUTED)
ax.set_yticks(range(len(names)), names, fontsize=6.2)
ax.set_xlabel("Compat-issue time-to-close: median (bar), p90 (whisker), days")
ax.set_title("RQ4: the thinnest layer is slowest to fix", fontsize=6.6)
ax.spines[["top","right"]].set_visible(False)
ax.xaxis.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("fig_ttc.pdf"); print("fig_ttc.pdf")

print("\n[check] 4 figures written")

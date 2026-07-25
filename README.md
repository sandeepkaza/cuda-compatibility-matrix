# cuda-compat-matrix

Companion artifact for the paper **"The CUDA Compatibility Tax: A Measurement of the Container, Framework, and Driver Version Matrix"**.

A four-axis measurement of the CUDA container/framework/driver compatibility matrix: surface growth, measured breakage (41-image smoke battery), staleness, and issue-resolution pain.

## What's here

`data/*.csv` — Docker Hub tags, NGC release notes, PyPI pins, GitHub issue counts/details, and the 41-image smoke matrix (`rq2_smoke.csv`). Miners: `mine_dockerhub.py`, `mine_ngc.py`, `mine_pypi.py`, `mine_issues.py`, `smoke_suite.py`.

## Reproduce

Requires Python 3 (plus, per paper, Docker/kind/a GPU host — see the paper's
Method section). Every number in the paper is regenerated from the committed
data by:

```bash
python3 analyze.py   # RQ1-RQ4 from data/*.csv
```

Each stage writes its intermediate data, so the path from raw measurement to
every reported figure is inspectable without re-running the measurement.

## Citation

See `CITATION.cff`. A DOI will be added after Zenodo archival. Venue path:
MSR -> UCC -> IEEE Access.

## License

MIT (`LICENSE`). Author: Venkata Sandeep Chowdary Kaza <sandeep.kaza@gmail.com>, ORCID 0009-0001-9454-8995.

## AI-assistance disclosure

AI-assisted scripting and editing (Anthropic Claude) was employed under the
author's direction; the author defined the research questions, designed the
method, ran the measurements, and validated every reported number.

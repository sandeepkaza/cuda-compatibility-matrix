"""fig_pipeline.png — Paper 9 compat-matrix mining + smoke-test pipeline."""
from diagrams import Cluster, Diagram, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.onprem.container import Docker
from diagrams.onprem.compute import Server
graph_attr={"dpi":"300","fontsize":"13","pad":"0.2","splines":"spline","nodesep":"0.35","ranksep":"0.6"}
node_attr={"fontsize":"11"}
with Diagram("",filename="fig_pipeline",outformat="png",show=False,graph_attr=graph_attr,node_attr=node_attr,direction="LR"):
    with Cluster("mining (snapshot 2026-07-18)"):
        hub=Python("mine_dockerhub.py\nnvidia/cuda 1784 tags\n+ pytorch/tf/ollama/vllm")
        ngc=Python("mine_ngc.py\n96 pytorch + 58 tf\nmonthly relnotes")
        pypi=Python("mine_pypi.py\ntorch/tf/vllm\nnvidia-* wheel pins")
        iss=Python("mine_issues.py\n138,591 issues, 6 repos\ncompat error signatures")
    raw=Storage("data/*.csv\nhub_tags / ngc_relnotes /\npypi_pins / gh_issues")
    with Cluster("measured breakage (RTX 3050 4GB, driver 555.97)"):
        smoke=Docker("smoke_suite.py\n41 images x t1..t7\nsmi/nvcc/ptx/torch/tf")
    an=Python("analyze.py\nRQ1 growth RQ2 cliff\nRQ3 staleness RQ4 pain")
    out=Storage("results.md + figures")
    hub>>raw; ngc>>raw; pypi>>raw; iss>>raw
    raw>>an; smoke>>Edge(label="pass/fail matrix")>>an>>out

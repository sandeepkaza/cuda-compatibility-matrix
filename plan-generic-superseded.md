# Paper 9 — "What Cross-Cloud Traffic Actually Costs: A Measured Latency, Throughput, and Egress-Price Matrix"

**Genre:** network + FinOps measurement. **Venue path:** IEEE CLOUD → UCC → IEEE Access.
**Gap (web-verified 2026-07-18):** egress pricing is documented (calculators
abound) but there is no published empirical matrix joining measured
inter-cloud network performance WITH metered cost — search results
explicitly note the absence of empirical latency measurements.

## RQs
- RQ1: Performance matrix — pairwise latency/jitter/throughput (iperf3 +
  ping/TCP RTT) between small VMs/clusters on Azure, AWS, OCI, GCP: same-region
  pairs where geography allows + one cross-continent pair; internet path
  (baseline) per pair; ≥24h sampling windows for time-of-day effects.
- RQ2: Metered egress truth — transfer a known corpus (10–50 GB) each
  direction on each pair; reconcile bytes sent vs billed egress line items
  (published price vs actual bill; NAT/LB hidden-cost surcharges the
  calculators warn about, measured).
- RQ3: Application consequence — replay Paper 3's replication channel and
  Paper 4-style artifact pulls over the matrix: what the numbers mean for DR
  RPO/RTO (ties to P3's placement lever) and for multi-cloud CI.
- RQ4 (FinOps): $/GB effective vs advertised, per pair per direction — the
  paper's headline table.

## Method notes
- Cheapest viable VMs; iperf3 servers ephemeral; every transfer logged
  client-side AND reconciled against the four bills (needs a few days'
  billing lag — schedule the bill pull as a follow-up step).
- Egress spend is the experiment: cap at ~$25 of deliberate transfer.

## Cost
VMs trivial; egress budget ~$25; total <$40.

## Status
- [ ] Probe/transfer harness + time-of-day scheduler
- [ ] Runs: pair matrix × direction × window
- [ ] Bill reconciliation (user pulls invoices) → results.md → 10pp draft

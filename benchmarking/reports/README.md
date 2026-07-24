# Campaign write-ups

Dated result folders from both benchmarking tracks (see the
[benchmarking overview](../README.md) for how the tracks relate). Each
folder holds a `report.md` plus the distilled per-campaign CSVs; raw
per-rank traces are not kept in git (multi-GB — stored in S3, available
on request).

| Campaign | Track | Status | Headline |
|---|---|---|---|
| [`2026-06-11_pr153_ddp_mlp_g4dn/`](2026-06-11_pr153_ddp_mlp_g4dn/report.md) | Wall-clock | current | +1.02% single-GPU throughput, ≈0% 2-node DDP (v0.3.1) |
| [`2026-07-19_v035_ddp_mlp_g4dn/`](2026-07-19_v035_ddp_mlp_g4dn/report.md) | Wall-clock | current | +0.95% single-GPU, +0.41% 4-GPU DDP (v0.3.5) |
| [`2026-07-21_ondemand_1_2_4node_g4dn/`](2026-07-21_ondemand_1_2_4node_g4dn/report.md) | Attribution | **superseded** | GIL-stress diagnostic only — configs unknowingly ran the GIL stress probe in every cell; not normal-overhead evidence |
| [`2026-07-22_clean_1_2_4node_g4dn/`](2026-07-22_clean_1_2_4node_g4dn/report.md) | Attribution | current | `trace_auto` adds < 1 ms per rank per step (0.17–0.58 ms) across 1/2/4 nodes × bs 256/512/1024 |

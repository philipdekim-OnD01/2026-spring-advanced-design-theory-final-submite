# Results and interpretation

## Latency

GPU end-to-end latency was lower at every model width. At w0.25, GPU and DLA
latencies were `0.142 ms` and `0.207 ms`. At w16 they were `0.629 ms` and
`2.753 ms`, making DLA `4.37x` slower.

The experiment therefore does not support using DLA to minimize batch-1
single-request latency. DLA scheduling overhead and its lower compute ceiling
become more visible as model width grows.

## Compute power

Idle-adjusted compute power was lower on DLA at every width. GPU dynamic power
rose from approximately `5.0 W` to `25.8 W`, while DLA remained between
approximately `3.6 W` and `5.4 W`. At w16, DLA dynamic power was `80.6%` lower.

This supports using DLA as a low-power offload path. A practical heterogeneous
policy is to keep latency-critical inference on GPU and route power-sensitive
or background inference to DLA, leaving GPU capacity available for other work.

## Scope of the claim

Lower instantaneous power does not automatically imply lower energy per
inference because DLA execution takes longer. The power graph supports a claim
about compute-power budget and thermal load, not an unconditional claim that
DLA always minimizes energy.

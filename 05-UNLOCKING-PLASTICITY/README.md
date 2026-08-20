# Unlocking Plasticity Rules: replication package

This repository supports the behavioral results reported in REI Labs' article
[*Unlocking Plasticity Rules*](https://reilabs.org/blog/unlocking-plasticity-rules).
It covers Temporal Context Projection (TCP) and Counterfactual Utility
Plasticity (CUP) on POPGym RepeatPrevious.

The package contains reproduction code, exact Domain declarations, seed-level
results, episode-level learning curves, frozen-state checks, and final CUP
structure summaries. It contains no decision traces.

## Evidence included

### RepeatPreviousEasy

Five matched runs compare:

1. present-only input (`memoryless`);
2. TCP without CUP (`tcp`);
3. TCP with CUP (`tcp_cup`).

Each run begins from empty task state, receives 510 online interactions, and is
then frozen for 16 unseen episodes. TCP reached `1.0000` mean return and action
accuracy in every run. The present-only control reached `-0.5109` mean return
and `0.2445` action accuracy.

### RepeatPreviousMedium

Five paired ten-episode runs compare TCP with TCP+CUP. TCP+CUP changed mean
learning-curve return from `0.2839` to `0.3167`, an `11.5%` relative increase,
and won four of five seeds. The paired 95% interval crosses zero, so this is not
a conclusive superiority estimate. Final CUP structures are retained for every
seed.

### Wisconsin control ladder

The article's four-condition aggregate is recorded under `results/wisconsin/`.
The supplied research artifacts did not include its exact runner, twelve-feature
manifest, or per-order results. This repository therefore marks Wisconsin as
reported aggregate only and does not claim to reproduce it.

## Start here

- `RESULTS.md` explains the reported numbers and attribution.
- `REPRODUCTION.md` gives the pinned environment and commands.
- `domains/resolved/` contains the five public Domain variants.
- `analysis/verify_results.py` checks the package without Core or POPGym.

Run the static verification:

```bash
python analysis/verify_results.py
```

## Scope

RepeatPreviousHard, preliminary probes, the superseded August 16 Easy result,
and unrelated Adapt-1 benchmarks are intentionally absent. TCP receives causal
credit for the Easy memory result. CUP receives no return credit on Easy.

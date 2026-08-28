# Adapt-1 user-ready replication experiments

This bundle contains six isolated experiments:

1. `01-ALFWORLD-COMPOSITIONAL`: compositional policy learning and frozen ID/OOD evaluation on ALFWorld.
2. `02-CAUSALAB-INTERVENTION`: interactive causal discovery across the public 3-node through 7-node CausaLab suites.
3. `03-TRADING-SIMULATION`: a small-scale, one-pass online market adaptation simulation using a sanitized numeric stream.
4. `04-SYMBOLIC-ALCHEMY`: sequential action learning on the official fixed Symbolic Alchemy task bank.
5. `05-UNLOCKING-PLASTICITY`: matched POPGym RepeatPrevious studies of Temporal Context Projection (TCP) and Counterfactual Utility Plasticity (CUP), including frozen evaluation and control results.
6. `06-AUTONOMOUS-DOMAINS`: Domain declarations for two verified autonomous structure-discovery experiments.

Each directory contains setup and Domain guidance together with the retained
artifacts needed to check its reference results. Artifact scope varies by
experiment; the repository contains no credentials or raw per-decision traces.

Run each experiment in a separate virtual environment. Keep the listed seeds, task counts, step limits, benchmark revisions, and Adapt-1 version fixed when checking the reference scores.

Symbolic Alchemy uses stochastic action selection and online fitting. Compare a clean run with the documented reference range. An exact return match is not expected.

API credentials must be supplied through environment variables. No credential is stored in this bundle.

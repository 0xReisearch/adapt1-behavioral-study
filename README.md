# Adapt-1 user-ready replication experiments

This bundle contains four isolated experiments:

1. `01-ALFWORLD-COMPOSITIONAL`: compositional policy learning and frozen ID/OOD evaluation on ALFWorld.
2. `02-CAUSALAB-INTERVENTION`: interactive causal discovery across the public 3-node through 7-node CausaLab suites.
3. `03-TRADING-SIMULATION`: a small-scale, one-pass online market adaptation simulation using a sanitized numeric stream.
4. `04-SYMBOLIC-ALCHEMY`: sequential action learning on the official fixed Symbolic Alchemy task bank.

Each directory contains setup instructions, a harness build guide, a Domain design guide, and the recorded reference scores. The package contains no prior execution logs, evaluation histories, credentials, or task-level model responses.

Run each experiment in a separate virtual environment. Keep the listed seeds, task counts, step limits, benchmark revisions, and Adapt-1 version fixed when checking the reference scores.

Symbolic Alchemy uses stochastic action selection and online fitting. Compare a clean run with the documented reference range. An exact return match is not expected.

API credentials must be supplied through environment variables. No credential is stored in this bundle.

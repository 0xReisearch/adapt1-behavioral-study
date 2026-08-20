# Domain declarations

`resolved/` contains the five Domain variants used by the public experiments:

- `easy_memoryless.json`
- `easy_tcp.json`
- `easy_tcp_cup.json`
- `medium_tcp.json`
- `medium_tcp_cup.json`

The Easy declarations differ only in temporal-context and CUP settings. The
Medium pair differs only in CUP settings. Easy uses a generic maximum lag of 16;
Medium uses a generic maximum lag of 64. Neither declaration identifies the
rewarding lag.

The files use stable public Domain IDs. The harness appends the learner seed to
the runtime ID so each run receives isolated state.

No Wisconsin Domain is included. The article's reported Wisconsin experiment
used twelve component predictors and 78 singleton/pair coalitions. The available
30-feature API-level run specification describes a different experiment.


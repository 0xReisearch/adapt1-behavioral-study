# Detailed results

## RepeatPreviousEasy: causal TCP ablation

All values below are means across five independent runs. Dispersion for frozen
return and accuracy is the sample standard deviation across the five seed-level
means.

| Variant | Frozen return | Frozen action accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| Present-only | -0.5109 +/- 0.0325 | 0.2445 +/- 0.0162 | 1.4185 | 0.7652 |
| TCP | **1.0000 +/- 0.0000** | **1.0000 +/- 0.0000** | **0.5451** | **0.2795** |
| TCP + CUP | **1.0000 +/- 0.0000** | **1.0000 +/- 0.0000** | 0.7227 | 0.3758 |

TCP improved frozen return by `+1.5109`. The five paired gains ranged from
`+1.4896` to `+1.5677`. Their paired 95% t interval was
`[+1.4706, +1.5513]`.

TCP alone reached the return ceiling and produced the lowest probability loss.
CUP therefore receives no causal credit for solving RepeatPreviousEasy.

Detailed files:

- `results/repeat_previous_easy/per_seed.csv`
- `results/repeat_previous_easy/episode_returns.csv`
- `results/repeat_previous_easy/aggregate.json`
- `results/repeat_previous_easy/runs/*.json`

## RepeatPreviousMedium: CUP acquisition and topology

| Variant | Mean learning-curve return | Last-five return | Final episode |
|---|---:|---:|---:|
| TCP | 0.2839 +/- 0.1139 | 0.7578 | 0.9444 |
| TCP + CUP | **0.3167 +/- 0.0661** | **0.8067** | **0.9722** |

TCP+CUP improved the mean learning curve by `+0.0328`, or `11.5%`, and won four
of five paired seeds. The paired 95% t interval was
`[-0.0565, +0.1221]`. This interval includes zero.

Seed 0 reached `1.0000` over sixteen frozen episodes in both conditions. The
Medium result concerns acquisition and learned structure, not a higher frozen
ceiling.

The final topology summaries preserve the article's two examples:

- Seed 0 consolidated `learned_model` and inhibited
  `contextual_memory x learned_model`.
- Seed 3 inhibited `contextual_memory` alone and consolidated
  `contextual_memory x learned_model`.

Detailed files:

- `results/repeat_previous_medium/per_seed.csv`
- `results/repeat_previous_medium/episode_returns.csv`
- `results/repeat_previous_medium/final_cup_structures.json`
- `results/repeat_previous_medium/aggregate.json`
- `results/repeat_previous_medium/runs/*.json`

## Wisconsin: reported control-ladder aggregate

| Composition rule | Mean log loss | Mean balanced accuracy |
|---|---:|---:|
| Disabled | 0.6635 | 0.5000 |
| Fixed uniform | 0.3288 | 0.8858 |
| CUP | 0.3087 | 0.8726 |
| Best past-loss coalition | 0.2493 | 0.8761 |

CUP reduced mean log loss by `0.0201` relative to fixed uniform fusion in all
thirty reported orders. Fixed uniform retained a reported `0.0131`
balanced-accuracy advantage. Best-coalition selection had lower log loss than
CUP in all thirty reported orders.

These values are included because the article reports them. They are not marked
as reproduced here. See `results/wisconsin/STATUS.md`.


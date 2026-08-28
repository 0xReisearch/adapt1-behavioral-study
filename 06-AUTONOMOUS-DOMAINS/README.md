# Autonomous Domain experiment declarations

## 1. Experiment scope

This directory contains the two Domain declarations used to verify autonomous
structure discovery from observed events, actions, and outcomes.

- `domain-a.json` covers sequential, action-conditioned transition learning.
- `domain-b.json` covers direct intervention and causal-binding discovery.

Both declarations begin with empty transition `input_paths`, an unresolved
intervention binding, and no authored causal-variable list. Adapt-1 must infer
eligible predictive paths and causal bindings from the event stream.

## 2. Domain A

Domain A enables bound-transition learning, transition prediction, autonomous
projection, and causal-graph induction. It supplies the public observation
surface, legal action surface, and measurable outcome channel without supplying
an effect map, hidden topology, reversibility prior, or bijective assignment.

The declaration is intended to verify whether an action-conditioned learner can
construct useful temporal structure when the predictive inputs are initially
unspecified.

## 3. Domain B

Domain B enables transition prediction, autonomous projection, and causal-graph
induction over observation and intervention-result events. Its predictor paths,
intervention binding, and causal variables are initially unresolved.

The declaration is intended to verify whether Adapt-1 can identify usable
predictors and intervention relationships from structured event streams.

## 4. Required API behavior

Use an Adapt-1 API version that supports autonomous projection and causal-binding
discovery. Domain creation must preserve empty `input_paths`, an empty
`intervention_path`, and an empty causal-variable declaration.

Each experiment must use a fresh Domain and fresh session identity. Submit only
observations available at decision time, the executed action or intervention,
and the resulting outcome. Do not inject discovered paths or bindings from an
earlier run.

## 5. Validate the declarations

From this directory, verify the JSON files and their checksums:

```bash
python -m json.tool domain-a.json >/dev/null
python -m json.tool domain-b.json >/dev/null
sha256sum -c SHA256SUMS.txt
```

Successful validation confirms that both declarations match the published
experiment files.

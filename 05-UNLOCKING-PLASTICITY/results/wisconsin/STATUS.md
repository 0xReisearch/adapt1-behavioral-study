# Wisconsin artifact status

Status: **reported aggregate only**.

The article reports a thirty-order matched ladder with disabled composition,
fixed uniform fusion, CUP, and best past-loss coalition selection. The supplied
Wisconsin JSON instead compares CUP with a global prior, online logistic
regression, and Gaussian naive Bayes. It cannot reproduce the article's control
ladder.

The supplied `WISCONSIN_CUP_RUN` specification also defines a different future
study. It uses all thirty public features and compares no CUP, order-1 CUP, and
order-2 CUP among registered posterior sources. It explicitly does not
reproduce the older twelve-component, 78-coalition result.

Before Wisconsin can be marked reproducible here, recover and add:

1. the exact twelve-feature manifest and preprocessing;
2. the component-predictor and coalition construction code;
3. the disabled, fixed-uniform, CUP, and best-coalition control implementation;
4. all thirty per-order metrics;
5. the analyzer that regenerates the article table;
6. hashes for those inputs and outputs.

No synthetic per-order values or substitute Domain are included.


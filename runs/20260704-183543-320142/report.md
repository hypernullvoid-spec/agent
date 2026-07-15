# Neo run report — 20260704-183543-320142
*Generated 2026-07-04 18:35:46*

## Task
Predict target from f1..f3. Metric: RMSE (lower better).

## Outcome
**Best validation metric: `0.3018`** (lower is better), found at step 2 via `draft`.

Solution saved to `best_solution.py`.

## Search statistics

- Steps executed: 4 (4 good, 0 buggy)
- Wall time: 4s
- Models: code=`mock:smoke-code`, feedback=`mock:smoke-fb`
- LLM usage: 4 calls · 0 in / 0 out tokens

## Solution tree

```
├─ [0] draft (good) metric=3.926
├─ [1] draft (good) metric=3.926
├─ [2] draft (good) metric=0.3018 ★
└─ [3] draft (good) metric=0.3018
```

## Metric history (good nodes)

| step | stage | metric | exec time |
|-----:|-------|-------:|----------:|
| 0 | draft | 3.926 | 0.9s |
| 1 | draft | 3.926 | 0.9s |
| 2 | draft | 0.3018 | 0.9s |
| 3 | draft | 0.3018 | 0.9s |

## Best solution plan

plan: add squared features to capture curvature

## Best solution analysis

reviewed

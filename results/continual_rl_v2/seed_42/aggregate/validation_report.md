# Continual RL V2 — Validation Report

Seed: `42`

Independent test evaluated: **NO**

Validation is used only for retrospective model assessment.

## no_replay

| Stage | GSM8K | Algebra | Competition | MBPP | Avg seen |
|---:|---:|---:|---:|---:|---:|
| 0 | 46.00% | — | — | — | 46.00% |
| 1 | 47.33% | 18.00% | — | — | 32.67% |
| 2 | 46.67% | 18.00% | 12.00% | — | 25.56% |
| 3 | 47.33% | 15.00% | 8.00% | 24.44% | 23.69% |

- Final average accuracy: **23.69%**
- Final average forgetting: **2.33 points**
- Average backward transfer: **-1.89 points**

## fixed_replay

| Stage | GSM8K | Algebra | Competition | MBPP | Avg seen |
|---:|---:|---:|---:|---:|---:|
| 0 | 46.00% | — | — | — | 46.00% |
| 1 | 45.33% | 17.00% | — | — | 31.17% |
| 2 | 45.33% | 17.00% | 13.00% | — | 25.11% |
| 3 | 48.00% | 17.00% | 17.00% | 22.22% | 26.06% |

- Final average accuracy: **26.06%**
- Final average forgetting: **0.00 points**
- Average backward transfer: **+2.00 points**

## adaptive_replay

| Stage | GSM8K | Algebra | Competition | MBPP | Avg seen |
|---:|---:|---:|---:|---:|---:|
| 0 | 46.00% | — | — | — | 46.00% |
| 1 | 46.67% | 17.00% | — | — | 31.83% |
| 2 | 45.33% | 18.00% | 14.00% | — | 25.78% |
| 3 | 46.67% | 17.00% | 12.00% | 0.00% | 18.92% |

- Final average accuracy: **18.92%**
- Final average forgetting: **1.00 points**
- Average backward transfer: **-0.44 points**


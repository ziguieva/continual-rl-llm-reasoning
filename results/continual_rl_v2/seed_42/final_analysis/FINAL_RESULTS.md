# Continual & Data-Efficient RL for LLM Reasoning

## Final validation results — Seed 42

| Strategy | GSM8K | Algebra | Competition | MBPP | Final avg. | Forgetting | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Replay | 47.33% | 15.00% | 8.00% | 24.44% | 23.69% | 2.33 pts | -1.89 pts |
| Fixed Replay | 48.00% | 17.00% | 17.00% | 22.22% | 26.06% | 0.00 pts | +2.00 pts |
| Adaptive Replay | 46.67% | 17.00% | 12.00% | 24.44% | 25.03% | 1.00 pts | -0.44 pts |

## Main observations

- Fixed Replay obtains the highest final average accuracy: **26.06%**.
- Fixed Replay improves final average accuracy by **+2.36 points** relative to No Replay.
- Adaptive Replay improves final average accuracy by **+1.33 points** relative to No Replay.
- Average forgetting falls from **2.33** points without replay to **0.00** with Fixed Replay.
- Fixed Replay produces positive average backward transfer (**+2.00 points**).
- Adaptive Replay reaches **25.03%** final average accuracy, between Fixed Replay and No Replay.

## Interpretation

For seed 42, replay improves retention across the sequential reasoning curriculum. Fixed Replay provides the strongest overall trade-off between final performance and retention.

The current Adaptive Replay controller does not yet demonstrate a clear advantage over Fixed Replay. During this experiment its replay allocations were effectively identical to the fixed allocation, so differences between both strategies cannot be attributed confidently to adaptive allocation.

Therefore the main supported conclusion is that **replay mitigates forgetting in this continual reasoning setting**, while the benefit of adaptive replay remains to be demonstrated.

## Limitations

- Results reported here correspond to one seed (seed 42).
- The Qwen2.5-0.5B model shows limited absolute performance on advanced mathematics.
- The adaptive replay controller did not create meaningfully different replay allocations.
- Validation was used only retrospectively; the independent 714-example test set remains closed.

## Independent test status

**Not evaluated.** The 714-example independent test set remains untouched.

# Continual & Data-Efficient Reinforcement Learning for LLM Reasoning

A local research project investigating **continual learning**, **reinforcement learning**, and **experience replay** for sequential reasoning adaptation in small language models.

The project studies whether a language model can learn new mathematical and programming tasks while preserving previously acquired reasoning capabilities.

The experiments are conducted with **Qwen2.5-0.5B-Instruct**, LoRA adapters, supervised fine-tuning, and GRPO-style optimization with verifiable rewards.

---

## Research Question

The main question investigated in this project is:

> **Can experience replay reduce catastrophic forgetting when a small language model is sequentially trained on mathematical reasoning and programming tasks?**

A secondary question investigates whether an adaptive replay mechanism can provide a better trade-off between learning new tasks and retaining previous capabilities.

---

## Continual Learning Sequence

The model is trained sequentially across four reasoning domains:

| Stage | Domain | Dataset |
|---|---|---|
| Stage 0 | Applied mathematics | GSM8K |
| Stage 1 | Advanced algebra | MATH |
| Stage 2 | Competition mathematics | MATH |
| Stage 3 | Python programming | MBPP |

Stage 0 is shared by all continual-learning strategies.

The following three strategies are then compared:

- **No Replay** — training only on the current task.
- **Fixed Replay** — 20% of the training groups are sampled from previous tasks.
- **Adaptive Replay** — replay allocation is determined using estimates of previous-task performance.

---

## Model

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct

Training uses parameter-efficient fine-tuning with LoRA.
The complete workflow was executed locally on Apple Silicon using PyTorch MPS.
Training Method
The project combines:
- Supervised Fine-Tuning (SFT)
- Group Relative Policy Optimization (GRPO)
- Verifiable rewards
- Experience replay
- Continual-learning evaluation
- Retrospective forgetting analysis
For the final continual-learning experiments, the reinforcement-learning reward is correctness-only:
reward = 1 if the answer is correct
reward = 0 otherwise

Mathematical answers are evaluated using deterministic answer verification.
Programming solutions are evaluated using executable MBPP unit tests inside an isolated Docker sandbox.
Experimental Budget
Each continual-learning stage uses:
40 training groups
×
4 generated responses per group
=
160 generations per stage

For replay strategies:
32 current-task groups
+
8 replay groups
=
40 total groups

This corresponds to a replay ratio of:
20%

Validation Benchmark
The retrospective validation benchmark contains 440 problems:
Domain	Validation examples
GSM8K	150
Advanced Algebra	100
Competition Mathematics	100
MBPP	90
Total	440


Validation examples are not used for training.
A separate 714-example independent test set remains closed and is not used in the reported model-selection experiments.
Final Results
Final retrospective validation results for seed 42:
Strategy	GSM8K	Algebra	Competition	MBPP	Average	Forgetting	BWT
No Replay	47.33%	15.00%	8.00%	24.44%	23.69%	2.33	-1.89
Fixed Replay	48.00%	17.00%	17.00%	22.22%	26.06%	0.00	+2.00
Adaptive Replay	46.67%	17.00%	12.00%	24.44%	25.03%	1.00	-0.44


Main Result
The strongest configuration observed in the seed-42 experiment is:
Fixed Replay

with a final average validation accuracy of:
26.06%

compared with:
23.69% — No Replay
25.03% — Adaptive Replay

Fixed Replay also reduces average measured forgetting from:
2.33 percentage points

to:
0.00 percentage points

and produces positive backward transfer:
+2.00 percentage points

The results therefore support experience replay as an effective mechanism for reducing forgetting in this continual reasoning setting.
The current Adaptive Replay controller does not yet demonstrate a clear advantage over Fixed Replay.
Figures
Final Average Accuracy

Fixed Replay obtains the highest overall final accuracy.
Final Accuracy by Domain

Fixed Replay performs particularly well on previously learned mathematical domains, while No Replay and Adaptive Replay obtain slightly stronger MBPP performance.
Average Forgetting

Fixed Replay reduces measured average forgetting to zero in the seed-42 experiment.
Backward Transfer

Fixed Replay is the only evaluated strategy with positive average backward transfer.
Project Structure
continual-rl-llm-reasoning/
│
├── data/
│   └── benchmark_v2_v1/
│
├── experiments/
│   ├── benchmark construction
│   ├── baseline evaluation
│   ├── supervised fine-tuning
│   ├── GRPO experiments
│   ├── continual-learning experiments
│   └── final analysis
│
├── src/
│
├── results/
│   └── continual_rl_v2/
│       └── seed_42/
│           └── final_analysis/
│
├── paper/
│   └── continual_rl_reasoning_report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md

Model checkpoints and local virtual environments are intentionally excluded from the repository.
Final Research Report
The complete report describing the methodology, continual-learning setup, results, limitations, and conclusions is available here:
[**Read the full research report (PDF)**](paper/continual_rl_reasoning_report.pdf)
Installation
Clone the repository:
git clone https://github.com/ziguieva/continual-rl-llm-reasoning.git
cd continual-rl-llm-reasoning

Create a Python virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Final Analysis
The final continual-learning analysis can be reproduced from previously generated evaluation results with:
python experiments/55_analyze_continual.py

This script generates:
final_results.csv
continual_accuracy_matrix.csv
final_metrics.json
accuracy_matrices.json
01_final_average_accuracy.png
02_final_domain_accuracy.png
03_average_forgetting.png
04_backward_transfer.png
trajectory_no_replay.png
trajectory_fixed_replay.png
trajectory_adaptive_replay.png

No model training is required for this analysis step.
Metrics
Forgetting
For domain \(d\):
F_d = max historical accuracy on d - final accuracy on d

Lower values indicate better retention.
Backward Transfer
For domain \(d\):
BWT_d = final accuracy - accuracy immediately after learning domain d

Positive values indicate that subsequent training improved performance on an earlier task.
Limitations
The current results should be interpreted with several limitations:
1. The final continual-learning comparison currently reports one random seed (seed 42).
2. Qwen2.5-0.5B has limited absolute performance on advanced mathematical reasoning.
3. The first adaptive replay controller used small training-only probes.
4. The adaptive controller produced replay allocations that were effectively identical to Fixed Replay in the seed-42 experiment.
5. Replay methods use the same total training-group budget as No Replay but fewer current-task examples.
6. The independent 714-example test set remains intentionally closed.
Therefore, the results demonstrate an experimental trend rather than statistical superiority across random seeds.
Future Work
Potential extensions include:
- Multi-seed evaluation
- Improved adaptive replay allocation
- Larger language models
- More robust forgetting estimators
- Uncertainty-aware replay
- Better plasticity-retention control
- Evaluation on BBH
- Evaluation on HumanEval+
- Final evaluation on the untouched independent test benchmark
A possible adaptive replay priority function is:
priority =
    α × forgetting
  + β × current weakness
  + γ × uncertainty

This could allow replay to focus more effectively on capabilities that are both weak and at risk of being forgotten.
Conclusion
This project studies continual reasoning adaptation under a constrained computational budget.
The main experimental finding is:
Experience replay mitigates forgetting during sequential reasoning adaptation in the studied setting.

For seed 42, Fixed Replay provides the strongest observed balance between new-task learning and retention:
Final Average Accuracy : 26.06%
Average Forgetting     : 0.00 pts
Backward Transfer      : +2.00 pts

Adaptive Replay also improves over No Replay, but its advantage over a simple fixed replay policy remains to be demonstrated.
Author
Evan Jean-Christ Zigui
Engineering research project — 2026

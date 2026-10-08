# Continual & Data-Efficient RL for LLM Reasoning

Research project on **continual learning for language model reasoning**, with a focus on catastrophic forgetting and experience replay.

The project investigates whether a small language model can sequentially learn mathematical and programming tasks while preserving previously acquired capabilities.

## Overview

The experiments use:

- **Qwen2.5-0.5B-Instruct**
- LoRA parameter-efficient fine-tuning
- Supervised Fine-Tuning (SFT)
- GRPO-style reinforcement learning
- Verifiable correctness rewards
- Experience replay
- Retrospective continual-learning evaluation

All experiments were conducted locally on Apple Silicon with PyTorch MPS.

## Continual Learning Sequence

| Stage | Domain | Dataset |
|---|---|---|
| 0 | Applied Mathematics | GSM8K |
| 1 | Advanced Algebra | MATH |
| 2 | Competition Mathematics | MATH |
| 3 | Python Programming | MBPP |

Three continual-learning strategies are compared:

- **No Replay**
- **Fixed Replay**
- **Adaptive Replay**

Fixed Replay uses a 20% replay ratio.

## Validation Benchmark

The retrospective validation benchmark contains **440 problems**:

| Domain | Examples |
|---|---:|
| GSM8K | 150 |
| Advanced Algebra | 100 |
| Competition Mathematics | 100 |
| MBPP | 90 |
| **Total** | **440** |

Validation examples are not used for training.

The independent test benchmark remains untouched during model selection.

## Main Results

Final retrospective validation results for **seed 42**:

| Strategy | GSM8K | Algebra | Competition | MBPP | Average | Forgetting | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Replay | 47.33% | 15.00% | 8.00% | 24.44% | 23.69% | 2.33 | -1.89 |
| **Fixed Replay** | **48.00%** | **17.00%** | **17.00% | 22.22% | **26.06%** | **0.00** | **+2.00** |
| Adaptive Replay | 46.67% | **17.00%** | 12.00% | **24.44%** | 25.03% | 1.00 | -0.44 |

### Key Finding

**Fixed Replay provides the strongest overall result in the seed-42 experiment.**

It achieves:

- **26.06%** final average accuracy
- **0.00** average measured forgetting
- **+2.00** percentage points of backward transfer

Compared with No Replay, Fixed Replay improves final average accuracy by **2.37 percentage points** while providing substantially better retention of previous reasoning tasks.

The current Adaptive Replay strategy also improves over No Replay, but does not outperform Fixed Replay.

## Training Budget

Each continual-learning stage uses:

```text
40 training groups
× 4 generations
= 160 generations
```

Replay strategies use:

```text
32 current-task groups
+ 8 replay groups
= 40 groups
```

The final reinforcement-learning reward is correctness-only:

```text
reward = 1 if correct
reward = 0 otherwise
```

Mathematical answers are evaluated using deterministic verification.

MBPP solutions are evaluated with executable unit tests inside an isolated Docker environment.

## Repository Structure

```text
continual-rl-llm-reasoning/
├── data/
├── experiments/
├── results/
├── src/
├── paper/
├── requirements.txt
├── .gitignore
└── README.md
```

Model checkpoints, virtual environments and private benchmark grading keys are excluded from the repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/ziguieva/continual-rl-llm-reasoning.git
cd continual-rl-llm-reasoning
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Final Analysis

The final analysis can be regenerated from the saved evaluation results with:

```bash
python experiments/55_analyze_continual.py
```

This step does not require model training.

## Limitations

- Final continual-learning results currently use one random seed: **42**.
- The base model is relatively small: **Qwen2.5-0.5B-Instruct**.
- The first Adaptive Replay controller produced replay allocations very close to Fixed Replay.
- Replay methods use the same total training-group budget as No Replay but fewer current-task examples.
- The independent test set is intentionally kept closed during model selection.

## Conclusion

The experiments support the use of **experience replay to mitigate forgetting during sequential reasoning adaptation**.

For seed 42, Fixed Replay provides the best observed balance between learning new tasks and retaining previous capabilities.

## Author

**Evan Jean-Christ Zigui**

Engineering research project — 2026
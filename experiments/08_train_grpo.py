
import json
import random
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = ROOT / "data/processed/train.jsonl"

OUTPUT_DIR = ROOT / "models/grpo_maths_smoke"

SEED = 42
MAX_STEPS = 2
NUM_GENERATIONS = 2

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "GPU MPS indisponible. "
        "Vérifie ton environnement PyTorch."
    )

print("=" * 60)
print("CONTINUAL RL - GRPO SMOKE TEST")
print("=" * 60)

print("Device : MPS")
print(f"Modèle : {MODEL_NAME}")

# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as file:

    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

maths = [
    row
    for row in records
    if row["domain"] == "maths"
]

combined = [
    row
    for row in maths
    if "(" in row["question"]
]

other = [
    row
    for row in maths
    if "(" not in row["question"]
]

rng = random.Random(SEED)

rng.shuffle(combined)
rng.shuffle(other)

if len(combined) < 8 or len(other) < 12:
    raise ValueError(
        "Pas assez d'exercices pour le test."
    )

selected = combined[:8] + other[:12]

rng.shuffle(selected)

# ============================================================
# 3. PRÉPARATION DU DATASET GRPO
# ============================================================

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en "
    "mathématiques, algèbre et Python. "
    "Résous précisément le problème. "
    "Termine par une ligne commençant "
    "par REPONSE: suivie directement "
    "du nombre entier trouvé. "
    "N'ajoute aucun texte après cette ligne."
)

train_records = []

for row in selected:

    train_records.append({
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": row["question"]
            }
        ],
        "answer": int(row["answer"])
    })

train_dataset = Dataset.from_list(
    train_records
)

print(f"Exercices : {len(train_dataset)}")
print("Opérations combinées : 8")
print("Autres opérations : 12")

# ============================================================
# 4. EXTRACTION DE LA RÉPONSE
# ============================================================

def extract_answer(text):

    pattern = (
        r"^\s*"
        r"(?:REPONSE|RÉPONSE|RESPONSE|"
        r"RESULTAT|RÉSULTAT)"
        r"\s*:\s*"
        r"(-?\d+)(?:\.0+)?"
        r"\s*[.!]?\s*$"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if matches:
        return int(matches[-1])

    match = re.fullmatch(
        r"\s*(-?\d+)(?:\.0+)?\s*[.!]?\s*",
        text
    )

    if match:
        return int(match.group(1))

    return None

# ============================================================
# 5. FONCTION DE RÉCOMPENSE
# ============================================================

def reward_correctness(
    completions,
    answer,
    **kwargs
):

    rewards = []

    for completion, expected in zip(
        completions,
        answer
    ):

        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion

        predicted = extract_answer(text)

        reward = (
            1.0
            if predicted == int(expected)
            else 0.0
        )

        rewards.append(reward)

    return rewards

# ============================================================
# 6. TEST DE LA FONCTION DE RÉCOMPENSE
# ============================================================

assert reward_correctness(
    ["REPONSE: 48", "REPONSE: 40"],
    [48, 48]
) == [1.0, 0.0]

print("Fonction de récompense : OK")

# ============================================================
# 7. CHARGEMENT DU MODÈLE
# ============================================================

print("\nChargement du modèle...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
)

model.config.use_cache = False

# ============================================================
# 8. CONFIGURATION LORA
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

# ============================================================
# 9. CONFIGURATION GRPO
# ============================================================

training_args = GRPOConfig(
    output_dir=str(OUTPUT_DIR),

    max_steps=MAX_STEPS,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,

    num_generations=NUM_GENERATIONS,
    max_completion_length=64,

    learning_rate=1e-5,

    beta=0.0,

    loss_type="grpo",

    optim="adamw_torch",

    fp16=False,
    bf16=False,

    gradient_checkpointing=True,

    dataloader_pin_memory=False,

    logging_steps=1,

    save_strategy="no",

    report_to="none",

    seed=SEED,

    use_vllm=False,

    remove_unused_columns=False
)

print(f"Appareil du Trainer : {training_args.device}")

# ============================================================
# 10. INITIALISATION DU TRAINER
# ============================================================

trainer = GRPOTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    reward_funcs=reward_correctness,

    peft_config=lora_config,

    processing_class=tokenizer
)

# ============================================================
# 11. ENTRAÎNEMENT
# ============================================================

print("\nDémarrage du GRPO...")

trainer.train()

# ============================================================
# 12. SAUVEGARDE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

with open(
    OUTPUT_DIR / "training_log.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        trainer.state.log_history,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("GRPO SMOKE TEST TERMINÉ")
print("=" * 60)

print(f"Modèle sauvegardé : {OUTPUT_DIR}")

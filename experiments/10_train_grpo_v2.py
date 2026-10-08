
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
OUTPUT_DIR = ROOT / "models/grpo_maths_v2"

SEED = 42
MAX_STEPS = 30
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 96
TEMPERATURE = 1.1
TOP_P = 0.95

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError("GPU MPS indisponible.")

DEVICE = "mps"

print("=" * 60)
print("CONTINUAL RL - GRPO V2")
print("=" * 60)

print(f"Modèle : {MODEL_NAME}")
print(f"GPU : {DEVICE}")
print(f"Étapes : {MAX_STEPS}")
print(f"Générations par prompt : {NUM_GENERATIONS}")

# ============================================================
# 2. CHARGEMENT DU DATASET
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

if len(maths) < 120:
    raise ValueError("Dataset maths incomplet.")

rng = random.Random(SEED)
rng.shuffle(maths)

SYSTEM_PROMPT = (
    "Tu es un assistant de mathématiques. "
    "Effectue les calculs avec précision. "
    "Tu peux raisonner brièvement. "
    "Termine par une ligne au format "
    "REPONSE: nombre. "
    "N'ajoute rien après cette ligne."
)

train_records = []

for row in maths:

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

train_dataset = Dataset.from_list(train_records)

print(f"\nExercices : {len(train_dataset)}")

# ============================================================
# 3. EXTRACTION DES RÉPONSES
# ============================================================

def extract_answer(text):

    pattern = (
        r"^\s*\**"
        r"(?:R[ÉE]PONSE|RESPONSE|"
        r"R[ÉE]SULTAT|RESULTAT)"
        r"\**\s*:\s*"
        r"(?:x\s*=\s*)?"
        r"(-?\d+)(?:\.0+)?"
        r"\s*[.!]?\s*$"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if matches:
        return int(matches[-1]), True

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None, False

    last_line = lines[-1]

    # Réponse numérique seule
    match = re.fullmatch(
        r"-?\d+(?:\.0+)?[.!]?",
        last_line
    )

    if match:
        value = re.search(r"-?\d+", last_line)
        return int(value.group()), False

    # Résultat final sous la forme x = 5
    match = re.fullmatch(
        r"x\s*=\s*(-?\d+)(?:\.0+)?[.!]?",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1)), False

    # Dernière phrase : le résultat est 15
    match = re.search(
        r"(?:est|is|vaut|affiche)\s+"
        r"[`*]*(-?\d+)(?:\.0+)?[`*]*"
        r"[.!]?\s*$",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1)), False

    return None, False

# ============================================================
# 4. FONCTION DE RÉCOMPENSE
# ============================================================

def calculate_reward(text, expected):

    predicted, explicit = extract_answer(text)

    if predicted is None:
        return 0.0

    # Bonus limité pour respecter le format
    format_bonus = 0.05 if explicit else 0.0

    # Réponse exacte : récompense maximale
    if predicted == expected:
        return 1.0 + format_bonus

    # Récompense partielle selon l'erreur relative
    error = abs(predicted - expected)

    scale = max(1, abs(expected))

    closeness = 1.0 / (1.0 + error / scale)

    partial_reward = 0.4 * closeness

    return partial_reward + format_bonus


def reward_reasoning(
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

        reward = calculate_reward(
            text,
            int(expected)
        )

        rewards.append(float(reward))

    return rewards

# ============================================================
# 5. TESTS UNITAIRES DE LA RÉCOMPENSE
# ============================================================

assert calculate_reward(
    "REPONSE: 48", 48
) == 1.05

assert calculate_reward(
    "REPONSE: 48", 40
) < 1.0

assert calculate_reward(
    "Réponse non reconnue", 48
) == 0.0

print("Fonction de récompense : OK")

# ============================================================
# 6. CHARGEMENT DU MODÈLE
# ============================================================

print("\nChargement de Qwen...")

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
).to(DEVICE)

model.eval()

# ============================================================
# 7. CALIBRATION DES RÉCOMPENSES
# ============================================================

print("\nCalibration des récompenses...")

combined = [
    row
    for row in maths
    if "(" in row["question"]
]

simple = [
    row
    for row in maths
    if "(" not in row["question"]
]

calibration_samples = (
    combined[:6] + simple[:2]
)

variable_groups = 0

for index, row in enumerate(
    calibration_samples,
    start=1
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": row["question"]
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=NUM_GENERATIONS,
            max_new_tokens=MAX_COMPLETION_LENGTH,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    rewards = []

    for output in outputs:

        generated = output[input_length:]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        reward = calculate_reward(
            response,
            int(row["answer"])
        )

        rewards.append(round(reward, 4))

    # Le groupe fournit-il un signal relatif ?
    if max(rewards) - min(rewards) > 1e-6:
        variable_groups += 1

    print(
        f"[{index}/{len(calibration_samples)}] "
        f"Réponse attendue : {row['answer']} | "
        f"Récompenses : {rewards}"
    )

print("-" * 60)

print(
    f"Groupes avec récompenses variables : "
    f"{variable_groups}/{len(calibration_samples)}"
)

if variable_groups == 0:

    raise RuntimeError(
        "Calibration : aucune variation de récompense. "
        "Entraînement interrompu pour éviter "
        "un nouveau GRPO sans signal."
    )

print("Calibration : OK")

# ============================================================
# 8. CONFIGURATION DU MODÈLE POUR L'ENTRAÎNEMENT
# ============================================================

model.config.use_cache = False

model.train()

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

    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,

    num_generations=NUM_GENERATIONS,

    max_completion_length=MAX_COMPLETION_LENGTH,

    temperature=TEMPERATURE,
    top_p=TOP_P,

    learning_rate=3e-5,
    lr_scheduler_type="constant",

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

    remove_unused_columns=False,

    use_vllm=False,

    seed=SEED
)

print(
    f"\nDevice Trainer : "
    f"{training_args.device}"
)

# ============================================================
# 10. INITIALISATION DU TRAINER
# ============================================================

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=reward_reasoning,
    peft_config=lora_config,
    processing_class=tokenizer
)

# ============================================================
# 11. ENTRAÎNEMENT
# ============================================================

print("\nDémarrage de GRPO V2...")

trainer.train()

# ============================================================
# 12. ANALYSE DES MÉTRIQUES
# ============================================================

logs = trainer.state.log_history

training_logs = [
    log
    for log in logs
    if "grad_norm" in log
]

nonzero_gradients = sum(
    float(log["grad_norm"]) > 1e-10
    for log in training_logs
)

variable_reward_steps = sum(
    float(
        log.get("frac_reward_zero_std", 1.0)
    ) < 1.0
    for log in training_logs
)

print("\n" + "=" * 60)
print("DIAGNOSTIC GRPO V2")
print("=" * 60)

print(
    f"Étapes avec gradient non nul : "
    f"{nonzero_gradients}/{len(training_logs)}"
)

print(
    f"Étapes avec variation de récompense : "
    f"{variable_reward_steps}/{len(training_logs)}"
)

if nonzero_gradients == 0:

    print(
        "Attention : aucun gradient non nul détecté."
    )

else:

    print(
        "Un signal d'optimisation a été détecté."
    )

# ============================================================
# 13. SAUVEGARDE DU MODÈLE
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

# ============================================================
# 14. SAUVEGARDE DES MÉTRIQUES
# ============================================================

with open(
    OUTPUT_DIR / "training_log.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        logs,
        file,
        indent=4,
        ensure_ascii=False
    )

diagnostics = {
    "model": MODEL_NAME,
    "steps": MAX_STEPS,
    "num_generations": NUM_GENERATIONS,
    "calibration_variable_groups": variable_groups,
    "nonzero_gradient_steps": nonzero_gradients,
    "variable_reward_steps": variable_reward_steps
}

with open(
    OUTPUT_DIR / "diagnostics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        diagnostics,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("GRPO V2 TERMINÉ")
print("=" * 60)

print(f"Modèle : {OUTPUT_DIR}")
print("Métriques sauvegardées.")

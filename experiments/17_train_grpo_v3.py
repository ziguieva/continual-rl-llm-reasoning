
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

OUTPUT_DIR = ROOT / "models/grpo_maths_v3"

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
print("CONTINUAL RL - GRPO V3")
print("=" * 60)

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

print(f"Questions : {len(train_dataset)}")

# ============================================================
# 3. EXTRACTION STRICTE
# ============================================================

NUMBER = r"(-?\d+)(?:\.0+)?"

LABEL = (
    r"(?:R[ÉE]PONSE|RESPONSE|"
    r"R[ÉE]SULTAT|RESULTAT)"
)

FINAL_PATTERN = re.compile(
    rf"^\s*{LABEL}\s*:\s*"
    rf"(?:x\s*=\s*)?"
    rf"{NUMBER}\s*[.!]?\s*$",
    re.IGNORECASE
)

# Reconnaît aussi les anciens marqueurs mal formés
# pour détecter les contradictions.
EARLIER_PATTERN = re.compile(
    r"^\s*(?:R[ÉE]PONSE|RESPONSE|"
    r"R[ÉE]SULTAT|RESULTAT|"
    r"REPMESSE(?:MENT)?)"
    r"\s*:\s*(-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE
)


def extract_final_answer(text):

    lines = [
        line.strip().strip(" *`")
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None, False

    # La dernière ligne doit contenir
    # une réponse explicite.
    final_match = FINAL_PATTERN.fullmatch(
        lines[-1]
    )

    if not final_match:
        return None, False

    final_answer = int(
        final_match.group(1)
    )

    # Vérification des réponses annoncées
    # précédemment dans le texte.
    for line in lines[:-1]:

        match = EARLIER_PATTERN.fullmatch(
            line
        )

        if match:

            earlier = float(
                match.group(1)
            )

            if earlier != final_answer:
                return final_answer, False

    return final_answer, True

# ============================================================
# 4. RÉCOMPENSE V3
# ============================================================

def calculate_reward(text, expected):

    predicted, consistent = extract_final_answer(
        text
    )

    # Pas de réponse finale exploitable.
    if predicted is None:
        return 0.0

    # Réponse contradictoire.
    if not consistent:
        return 0.0

    # Réponse exacte.
    if predicted == expected:
        return 1.0

    # Signal de proximité très faible.
    # Une erreur ne reçoit jamais
    # une récompense comparable à 1.
    error = abs(predicted - expected)

    return 0.05 / (1.0 + error)


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
# 5. TESTS DE LA RÉCOMPENSE
# ============================================================

assert calculate_reward(
    "REPONSE: 48", 48
) == 1.0

assert calculate_reward(
    "REPONSE: 40", 48
) < 0.05

assert calculate_reward(
    "REPONSE: 40\nREPONSE: 48", 48
) == 0.0

assert calculate_reward(
    "Le résultat est 48", 48
) == 0.0

assert calculate_reward(
    "REPONSE: nombre", 48
) == 0.0

print("Tests de récompense : OK")

# ============================================================
# 6. CHARGEMENT DU MODÈLE ORIGINAL
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
# 7. CALIBRATION
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

        rewards.append(
            calculate_reward(
                response,
                int(row["answer"])
            )
        )

    if max(rewards) - min(rewards) > 1e-6:
        variable_groups += 1

    print(
        f"[{index}/{len(calibration_samples)}] "
        f"Récompenses : "
        f"{[round(r, 4) for r in rewards]}"
    )

print(
    f"\nGroupes variables : "
    f"{variable_groups}/{len(calibration_samples)}"
)

if variable_groups == 0:
    raise RuntimeError(
        "Aucune variation de récompense. "
        "Entraînement interrompu."
    )

# ============================================================
# 8. CONFIGURATION LORA
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

print("\nDémarrage GRPO V3...")

trainer.train()

# ============================================================
# 12. DIAGNOSTIC
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
print("DIAGNOSTIC GRPO V3")
print("=" * 60)

print(
    f"Étapes avec gradient non nul : "
    f"{nonzero_gradients}/{len(training_logs)}"
)

print(
    f"Étapes avec variation de récompense : "
    f"{variable_reward_steps}/{len(training_logs)}"
)

# ============================================================
# 13. SAUVEGARDE
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
print("GRPO V3 TERMINÉ")
print("=" * 60)

print(f"Modèle : {OUTPUT_DIR}")

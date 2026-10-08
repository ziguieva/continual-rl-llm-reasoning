
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_ADAPTER = ROOT / "models/grpo_maths_v2"

TRAIN_FILE = ROOT / "data/processed/train.jsonl"

OUTPUT_DIR = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

SEED = 42
MAX_LENGTH = 256
MAX_STEPS = 60
LEARNING_RATE = 1e-4

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError("GPU MPS indisponible.")

DEVICE = "mps"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("CONTINUAL LEARNING - ALGEBRA SFT")
print("=" * 60)

# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

with TRAIN_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

algebra = [
    row
    for row in records
    if row["domain"] == "algebre"
]

if len(algebra) != 120:
    raise ValueError(
        f"120 exercices attendus, {len(algebra)} trouvés."
    )

print(f"Questions d'algèbre : {len(algebra)}")

# ============================================================
# 3. GÉNÉRATION DES DÉMONSTRATIONS
# ============================================================

def build_solution(question, expected):

    equation = question.split(":", 1)[-1].strip()

    number = r"(-?\d+)"

    # Cas 1 : ax + b = c
    match = re.fullmatch(
        rf"{number}x \+ {number} = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if a * expected + b != c:
            raise ValueError(question)

        return (
            f"On soustrait {b} des deux côtés.\n"
            f"{a}x = {c} - {b} = {c - b}\n"
            f"On divise par {a}.\n"
            f"x = {expected}\n"
            f"REPONSE: {expected}"
        )

    # Cas 2 : ax - b = c
    match = re.fullmatch(
        rf"{number}x - {number} = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if a * expected - b != c:
            raise ValueError(question)

        return (
            f"On ajoute {b} des deux côtés.\n"
            f"{a}x = {c} + {b} = {c + b}\n"
            f"On divise par {a}.\n"
            f"x = {expected}\n"
            f"REPONSE: {expected}"
        )

    # Cas 3 : a(x + b) = c
    match = re.fullmatch(
        rf"{number}\(x \+ {number}\) = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if a * (expected + b) != c:
            raise ValueError(question)

        return (
            f"On divise les deux côtés par {a}.\n"
            f"x + {b} = {c // a}\n"
            f"On soustrait {b}.\n"
            f"x = {c // a} - {b} = {expected}\n"
            f"REPONSE: {expected}"
        )

    # Cas 4 : x/a + b = c
    match = re.fullmatch(
        rf"x/{number} \+ {number} = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if a == 0 or expected != a * (c - b):
            raise ValueError(question)

        return (
            f"On soustrait {b} des deux côtés.\n"
            f"x/{a} = {c - b}\n"
            f"On multiplie par {a}.\n"
            f"x = {c - b} * {a} = {expected}\n"
            f"REPONSE: {expected}"
        )

    raise ValueError(
        f"Équation non reconnue : {equation}"
    )

# ============================================================
# 4. CONSTRUCTION DES EXEMPLES
# ============================================================

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

examples = []

for row in algebra:

    solution = build_solution(
        row["question"],
        int(row["answer"])
    )

    examples.append({
        "question": row["question"],
        "solution": solution
    })

print("Démonstrations générées et vérifiées.")

# ============================================================
# 5. CHARGEMENT DU TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ============================================================
# 6. PRÉPARATION DES TOKENS
# ============================================================

class AlgebraDataset(Dataset):

    def __init__(self, examples):

        self.samples = []

        for example in examples:

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": example["question"]
                }
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            target = (
                example["solution"]
                + tokenizer.eos_token
            )

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False
            )["input_ids"]

            target_ids = tokenizer(
                target,
                add_special_tokens=False
            )["input_ids"]

            input_ids = prompt_ids + target_ids

            if len(input_ids) > MAX_LENGTH:
                raise ValueError(
                    "Exemple trop long. "
                    "Augmenter MAX_LENGTH."
                )

            # Le modèle apprend uniquement à
            # générer la réponse de l'assistant.
            labels = (
                [-100] * len(prompt_ids)
                + target_ids
            )

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


train_dataset = AlgebraDataset(examples)

# ============================================================
# 7. COLLATOR
# ============================================================

def data_collator(features):

    batch_size = len(features)

    max_length = max(
        len(item["input_ids"])
        for item in features
    )

    input_ids = torch.full(
        (batch_size, max_length),
        tokenizer.pad_token_id,
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long
    )

    for index, item in enumerate(features):

        length = len(item["input_ids"])

        input_ids[index, :length] = torch.tensor(
            item["input_ids"]
        )

        attention_mask[index, :length] = 1

        labels[index, :length] = torch.tensor(
            item["labels"]
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ============================================================
# 8. CHARGEMENT DE QWEN ET DE L'ADAPTATEUR V2
# ============================================================

print("\nChargement du modèle...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager"
).to(DEVICE)

model = PeftModel.from_pretrained(
    base_model,
    str(INPUT_ADAPTER),
    is_trainable=True
)

model.config.use_cache = False

model.train()

model.print_trainable_parameters()

# ============================================================
# 9. CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    max_steps=MAX_STEPS,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant",

    optim="adamw_torch",

    max_grad_norm=1.0,

    fp16=False,
    bf16=False,

    gradient_checkpointing=False,

    dataloader_pin_memory=False,

    remove_unused_columns=False,

    logging_steps=5,

    save_strategy="no",

    report_to="none",

    seed=SEED,
    data_seed=SEED
)

# ============================================================
# 10. INITIALISATION DU TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer
)

# ============================================================
# 11. ENTRAÎNEMENT SÉQUENTIEL
# ============================================================

print("\nDébut du SFT sur l'algèbre...")

training_result = trainer.train()

# ============================================================
# 12. SAUVEGARDE
# ============================================================

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

training_info = {
    "base_model": MODEL_NAME,
    "previous_adapter": str(INPUT_ADAPTER),
    "new_adapter": str(OUTPUT_DIR),
    "task": "algebre",
    "method": "SFT",
    "train_examples": len(train_dataset),
    "steps": MAX_STEPS,
    "learning_rate": LEARNING_RATE,
    "training_loss": training_result.training_loss,
    "history": trainer.state.log_history
}

with (
    OUTPUT_DIR / "training_log.json"
).open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_info,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 13. FIN
# ============================================================

print("\n" + "=" * 60)
print("ALGEBRA SFT TERMINÉ")
print("=" * 60)

print(
    f"Training loss : "
    f"{training_result.training_loss:.4f}"
)

print(
    f"Modèle sauvegardé : {OUTPUT_DIR}"
)

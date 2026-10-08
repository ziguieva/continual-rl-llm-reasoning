
import ast
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

INPUT_ADAPTER = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

TRAIN_FILE = ROOT / "data/processed/train.jsonl"

OUTPUT_DIR = (
    ROOT
    / "models/grpo_maths_v2_then_algebra_sft_then_code_sft"
)

SEED = 42
MAX_LENGTH = 256
MAX_STEPS = 60
LEARNING_RATE = 1e-4

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "GPU MPS indisponible."
    )

DEVICE = "mps"

if not INPUT_ADAPTER.joinpath(
    "adapter_config.json"
).exists():
    raise FileNotFoundError(
        f"Adaptateur M2 introuvable : {INPUT_ADAPTER}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("CONTINUAL LEARNING - CODE SFT")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Modèle initial : {INPUT_ADAPTER}")
print(f"Étapes : {MAX_STEPS}")

# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(TRAIN_FILE)

with TRAIN_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

code_records = [
    row
    for row in records
    if row["domain"] == "code"
]

if len(code_records) != 120:
    raise ValueError(
        f"120 exercices attendus, "
        f"{len(code_records)} trouvés."
    )

print(
    f"\nExercices Python : {len(code_records)}"
)

# ============================================================
# 3. GÉNÉRATION DES SOLUTIONS
# ============================================================

def build_solution(question, expected):

    expression = question.strip().splitlines()[-1]
    expression = expression.strip()

    # --------------------------------------------------------
    # CAS 1 : PUISSANCE
    # print(a**b + c)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*\*\*\s*"
        r"(\d+)\s*\+\s*(\d+)\s*\)",
        expression
    )

    if match:

        a, b, c = map(
            int,
            match.groups()
        )

        power = a ** b
        result = power + c

        if result != expected:
            raise ValueError(
                f"Réponse incorrecte : {question}"
            )

        return (
            f"On calcule d'abord la puissance.\n"
            f"{a}**{b} = {power}\n"
            f"On ajoute {c}.\n"
            f"{power} + {c} = {result}\n"
            f"REPONSE: {result}"
        )

    # --------------------------------------------------------
    # CAS 2 : DIVISION ENTIÈRE
    # print(a // b)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*//\s*(\d+)\s*\)",
        expression
    )

    if match:

        a, b = map(
            int,
            match.groups()
        )

        if b == 0:
            raise ValueError(
                "Division par zéro."
            )

        result = a // b

        if result != expected:
            raise ValueError(
                f"Réponse incorrecte : {question}"
            )

        return (
            "L'opérateur // réalise "
            "une division entière.\n"
            f"{a} // {b} = {result}\n"
            f"REPONSE: {result}"
        )

    # --------------------------------------------------------
    # CAS 3 : SOMME D'UNE LISTE
    # print(sum([...]))
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*sum\(\s*"
        r"(\[[\d,\s-]+\])"
        r"\s*\)\s*\)",
        expression
    )

    if match:

        values = ast.literal_eval(
            match.group(1)
        )

        if (
            not isinstance(values, list)
            or not values
            or not all(
                type(value) is int
                for value in values
            )
        ):
            raise ValueError(
                "Liste Python invalide."
            )

        result = sum(values)

        if result != expected:
            raise ValueError(
                f"Réponse incorrecte : {question}"
            )

        calculation = " + ".join(
            str(value)
            for value in values
        )

        return (
            "La fonction sum() additionne "
            "les éléments de la liste.\n"
            f"{calculation} = {result}\n"
            f"REPONSE: {result}"
        )

    # --------------------------------------------------------
    # CAS 4 : LONGUEUR D'UNE LISTE
    # print(len([...]))
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*len\(\s*"
        r"(\[[\d,\s-]+\])"
        r"\s*\)\s*\)",
        expression
    )

    if match:

        values = ast.literal_eval(
            match.group(1)
        )

        if (
            not isinstance(values, list)
            or not all(
                type(value) is int
                for value in values
            )
        ):
            raise ValueError(
                "Liste Python invalide."
            )

        result = len(values)

        if result != expected:
            raise ValueError(
                f"Réponse incorrecte : {question}"
            )

        return (
            "La fonction len() retourne "
            "le nombre d'éléments.\n"
            f"La liste contient {result} éléments.\n"
            f"REPONSE: {result}"
        )

    # --------------------------------------------------------
    # CAS 5 : MODULO
    # print(a % b + c)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*%\s*"
        r"(\d+)\s*\+\s*(\d+)\s*\)",
        expression
    )

    if match:

        a, b, c = map(
            int,
            match.groups()
        )

        if b == 0:
            raise ValueError(
                "Modulo par zéro."
            )

        remainder = a % b
        result = remainder + c

        if result != expected:
            raise ValueError(
                f"Réponse incorrecte : {question}"
            )

        return (
            "L'opérateur % calcule "
            "le reste de la division entière.\n"
            f"{a} % {b} = {remainder}\n"
            f"{remainder} + {c} = {result}\n"
            f"REPONSE: {result}"
        )

    raise ValueError(
        f"Expression non reconnue : {expression}"
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

for row in code_records:

    solution = build_solution(
        row["question"],
        int(row["answer"])
    )

    examples.append({
        "question": row["question"],
        "solution": solution
    })

print(
    "Solutions Python générées et vérifiées."
)

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
# 6. PRÉPARATION DES DONNÉES SFT
# ============================================================

class CodeDataset(Dataset):

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

            input_ids = (
                prompt_ids + target_ids
            )

            if len(input_ids) > MAX_LENGTH:
                raise ValueError(
                    "Exemple trop long. "
                    "Augmenter MAX_LENGTH."
                )

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


train_dataset = CodeDataset(
    examples
)

print(
    f"Exemples préparés : "
    f"{len(train_dataset)}"
)

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

        length = len(
            item["input_ids"]
        )

        input_ids[index, :length] = torch.tensor(
            item["input_ids"],
            dtype=torch.long
        )

        attention_mask[index, :length] = 1

        labels[index, :length] = torch.tensor(
            item["labels"],
            dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ============================================================
# 8. CHARGEMENT DU MODÈLE M2
# ============================================================

print("\nChargement du modèle M2...")

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
# 9. CONFIGURATION SFT
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
# 11. ENTRAÎNEMENT SUR LA PROGRAMMATION
# ============================================================

print("\nDébut du SFT sur Python...")

training_result = trainer.train()

# ============================================================
# 12. SAUVEGARDE DU MODÈLE M3
# ============================================================

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

training_info = {
    "base_model": MODEL_NAME,

    "previous_adapter": str(
        INPUT_ADAPTER
    ),

    "new_adapter": str(
        OUTPUT_DIR
    ),

    "task": "code",

    "method": "SFT",

    "train_examples": len(
        train_dataset
    ),

    "steps": MAX_STEPS,

    "learning_rate": LEARNING_RATE,

    "training_loss": (
        training_result.training_loss
    ),

    "history": (
        trainer.state.log_history
    )
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
print("CODE SFT TERMINÉ")
print("=" * 60)

print(
    f"Training loss : "
    f"{training_result.training_loss:.4f}"
)

print(
    f"Modèle M3 sauvegardé : "
    f"{OUTPUT_DIR}"
)

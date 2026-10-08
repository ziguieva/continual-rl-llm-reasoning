
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
    ROOT / "models"
    / "grpo_maths_v2_then_algebra_sft"
    "_then_code_replay20"
)

SEED = 42

MAX_LENGTH = 256
MAX_STEPS = 60
LEARNING_RATE = 1e-4

N_CODE = 96
N_MATHS = 12
N_ALGEBRE = 12

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "GPU Apple MPS indisponible."
    )

DEVICE = "mps"

if not (
    INPUT_ADAPTER / "adapter_config.json"
).exists():
    raise FileNotFoundError(
        f"Adaptateur M2 introuvable : "
        f"{INPUT_ADAPTER}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("CONTINUAL LEARNING - EXPERIENCE REPLAY")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Modèle initial : {INPUT_ADAPTER}")
print(f"Étapes : {MAX_STEPS}")

# ============================================================
# 2. CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT
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

maths_records = [
    row for row in records
    if row["domain"] == "maths"
]

algebra_records = [
    row for row in records
    if row["domain"] == "algebre"
]

code_records = [
    row for row in records
    if row["domain"] == "code"
]

if (
    len(maths_records) != 120
    or len(algebra_records) != 120
    or len(code_records) != 120
):
    raise ValueError(
        "Le jeu d'entraînement doit contenir "
        "120 exemples par domaine."
    )

# Échantillonnage reproductible.
rng = random.Random(SEED)

rng.shuffle(maths_records)
rng.shuffle(algebra_records)
rng.shuffle(code_records)

selected_maths = maths_records[:N_MATHS]
selected_algebra = algebra_records[:N_ALGEBRE]
selected_code = code_records[:N_CODE]

selected_records = (
    selected_code
    + selected_maths
    + selected_algebra
)

rng.shuffle(selected_records)

assert len(selected_records) == 120

print("\nComposition du replay :")
print(f"Code : {N_CODE}")
print(f"Maths : {N_MATHS}")
print(f"Algèbre : {N_ALGEBRE}")

# Sauvegarde des identifiants pour
# reproduire exactement l'expérience.
manifest = {
    "seed": SEED,
    "source": str(TRAIN_FILE),
    "counts": {
        "code": N_CODE,
        "maths": N_MATHS,
        "algebre": N_ALGEBRE
    },
    "selected_ids": [
        row["id"]
        for row in selected_records
    ]
}

with (
    OUTPUT_DIR / "replay_manifest.json"
).open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 3. SOLUTIONS MATHÉMATIQUES
# ============================================================

def build_maths_solution(question, expected):

    expression = (
        question.split(":", 1)[-1].strip()
    )

    return (
        f"On effectue le calcul : "
        f"{expression} = {expected}.\n"
        f"REPONSE: {expected}"
    )

# ============================================================
# 4. SOLUTIONS ALGÉBRIQUES
# ============================================================

def build_algebra_solution(question, expected):

    equation = (
        question.split(":", 1)[-1].strip()
    )

    number = r"(-?\d+)"

    # ax + b = c
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

    # ax - b = c
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

    # a(x + b) = c
    match = re.fullmatch(
        rf"{number}\(x \+ {number}\) = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if a == 0 or a * (expected + b) != c:
            raise ValueError(question)

        divided = c // a

        return (
            f"On divise les deux côtés par {a}.\n"
            f"x + {b} = {divided}\n"
            f"On soustrait {b}.\n"
            f"x = {divided} - {b} = {expected}\n"
            f"REPONSE: {expected}"
        )

    # x/a + b = c
    match = re.fullmatch(
        rf"x/{number} \+ {number} = {number}",
        equation
    )

    if match:

        a, b, c = map(int, match.groups())

        if (
            a == 0
            or expected != a * (c - b)
        ):
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
# 5. SOLUTIONS PYTHON
# ============================================================

def build_code_solution(question, expected):

    expression = (
        question.strip().splitlines()[-1].strip()
    )

    # Cas 1 : puissance
    # print(a**b + c)
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
            raise ValueError(question)

        return (
            f"On calcule d'abord la puissance.\n"
            f"{a}**{b} = {power}\n"
            f"On ajoute {c}.\n"
            f"{power} + {c} = {result}\n"
            f"REPONSE: {result}"
        )

    # Cas 2 : division entière
    # print(a // b)
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
            raise ValueError(question)

        return (
            "L'opérateur // réalise "
            "une division entière.\n"
            f"{a} // {b} = {result}\n"
            f"REPONSE: {result}"
        )

    # Cas 3 : somme d'une liste
    # print(sum([...]))
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
            raise ValueError(question)

        result = sum(values)

        if result != expected:
            raise ValueError(question)

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

    # Cas 4 : longueur d'une liste
    # print(len([...]))
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
            raise ValueError(question)

        result = len(values)

        if result != expected:
            raise ValueError(question)

        return (
            "La fonction len() retourne "
            "le nombre d'éléments.\n"
            f"La liste contient "
            f"{result} éléments.\n"
            f"REPONSE: {result}"
        )

    # Cas 5 : modulo
    # print(a % b + c)
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
            raise ValueError(question)

        return (
            "L'opérateur % calcule le reste "
            "de la division entière.\n"
            f"{a} % {b} = {remainder}\n"
            f"{remainder} + {c} = {result}\n"
            f"REPONSE: {result}"
        )

    raise ValueError(
        f"Expression non reconnue : "
        f"{expression}"
    )

# ============================================================
# 6. CONSTRUCTION DES EXEMPLES
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

for row in selected_records:

    expected = int(
        row["answer"]
    )

    domain = row["domain"]

    if domain == "maths":

        solution = build_maths_solution(
            row["question"],
            expected
        )

    elif domain == "algebre":

        solution = build_algebra_solution(
            row["question"],
            expected
        )

    elif domain == "code":

        solution = build_code_solution(
            row["question"],
            expected
        )

    else:

        raise ValueError(
            f"Domaine inconnu : {domain}"
        )

    examples.append({
        "question": row["question"],
        "solution": solution,
        "domain": domain
    })

print(
    "\nDémonstrations générées et vérifiées."
)

# ============================================================
# 7. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

tokenizer.padding_side = "right"

# ============================================================
# 8. DATASET SFT
# ============================================================

class ReplayDataset(Dataset):

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

        return len(
            self.samples
        )

    def __getitem__(self, index):

        return self.samples[
            index
        ]


train_dataset = ReplayDataset(
    examples
)

print(
    f"Exemples préparés : "
    f"{len(train_dataset)}"
)

# ============================================================
# 9. COLLATOR
# ============================================================

def data_collator(features):

    batch_size = len(
        features
    )

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

    for index, item in enumerate(
        features
    ):

        length = len(
            item["input_ids"]
        )

        input_ids[
            index, :length
        ] = torch.tensor(
            item["input_ids"],
            dtype=torch.long
        )

        attention_mask[
            index, :length
        ] = 1

        labels[
            index, :length
        ] = torch.tensor(
            item["labels"],
            dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ============================================================
# 10. CHARGEMENT DE M2
# ============================================================

print("\nChargement de M2...")

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
# 11. CONFIGURATION DE L'ENTRAÎNEMENT
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
# 12. TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    data_collator=data_collator,

    processing_class=tokenizer
)

# ============================================================
# 13. ENTRAÎNEMENT AVEC REPLAY
# ============================================================

print(
    "\nDébut de l'entraînement "
    "avec Experience Replay..."
)

training_result = trainer.train()

# ============================================================
# 14. SAUVEGARDE
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

    "method": "SFT_with_replay",

    "steps": MAX_STEPS,

    "learning_rate": LEARNING_RATE,

    "train_examples": len(
        train_dataset
    ),

    "replay_ratio": (
        N_MATHS + N_ALGEBRE
    ) / len(train_dataset),

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
# 15. BILAN FINAL
# ============================================================

print("\n" + "=" * 60)
print("EXPERIENCE REPLAY TERMINÉ")
print("=" * 60)

print(
    f"Training loss : "
    f"{training_result.training_loss:.4f}"
)

print(
    f"Ratio de replay : "
    f"{training_info['replay_ratio']:.0%}"
)

print(
    f"Modèle sauvegardé : "
    f"{OUTPUT_DIR}"
)

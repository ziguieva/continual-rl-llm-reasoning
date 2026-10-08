
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
    TrainingArguments,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_ADAPTER = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

TRAIN_FILE = (
    ROOT / "data/processed/train.jsonl"
)

REPLAY_MANIFEST = (
    ROOT
    / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_replay20"
    / "replay_manifest.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_control96"
)

SEED = 42
MAX_LENGTH = 256
MAX_STEPS = 60
LEARNING_RATE = 1e-4
N_REPEATS = 24

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    raise RuntimeError("GPU Apple MPS indisponible.")

DEVICE = "mps"

for path in [
    TRAIN_FILE,
    REPLAY_MANIFEST,
    INPUT_ADAPTER / "adapter_config.json",
]:
    if not path.is_file():
        raise FileNotFoundError(path)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 60)
print("CONTINUAL LEARNING - CONTROL 96 CODE")
print("=" * 60)

# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

with TRAIN_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

records_by_id = {
    row["id"]: row
    for row in records
}

if len(records_by_id) != len(records):
    raise ValueError(
        "Identifiants dupliqués dans le dataset."
    )

with REPLAY_MANIFEST.open(
    "r",
    encoding="utf-8",
) as file:

    replay_manifest = json.load(file)

replay_ids = replay_manifest["selected_ids"]

if len(replay_ids) != 120:
    raise ValueError(
        "Le manifeste doit contenir 120 exemples."
    )

if any(
    example_id not in records_by_id
    for example_id in replay_ids
):
    raise ValueError(
        "Identifiants absents du dataset."
    )

# Les 96 exercices de code sont exactement
# ceux utilisés par M3-R.

code_ids = [
    example_id
    for example_id in replay_ids
    if records_by_id[example_id]["domain"] == "code"
]

old_ids = [
    example_id
    for example_id in replay_ids
    if records_by_id[example_id]["domain"] != "code"
]

if len(code_ids) != 96:
    raise ValueError(
        f"96 exercices de code attendus, "
        f"{len(code_ids)} trouvés."
    )

if len(set(code_ids)) != 96:
    raise ValueError(
        "Les exercices de code doivent être distincts."
    )

if len(old_ids) != N_REPEATS:
    raise ValueError(
        "24 anciens exercices attendus."
    )

# ============================================================
# 3. CONSTRUCTION DU GROUPE TÉMOIN
# ============================================================

rng = random.Random(SEED)

repeated_code_ids = rng.sample(
    code_ids,
    N_REPEATS,
)

repeat_iterator = iter(
    repeated_code_ids
)

# Nous conservons les positions des exercices
# de code présents dans le manifeste du replay.
# Les 24 positions des anciennes tâches sont
# remplacées par des répétitions de code.

control_ids = []

for example_id in replay_ids:

    domain = records_by_id[
        example_id
    ]["domain"]

    if domain == "code":

        control_ids.append(
            example_id
        )

    else:

        control_ids.append(
            next(repeat_iterator)
        )

if len(control_ids) != 120:
    raise ValueError(
        "Le groupe témoin doit contenir "
        "120 présentations."
    )

if len(set(control_ids)) != 96:
    raise ValueError(
        "Le groupe témoin doit contenir "
        "96 exercices distincts."
    )

control_records = [
    records_by_id[example_id]
    for example_id in control_ids
]

if any(
    row["domain"] != "code"
    for row in control_records
):
    raise ValueError(
        "Le groupe témoin ne doit "
        "contenir que du code."
    )

print("\nComposition du groupe témoin :")
print("Exercices de code distincts : 96")
print("Répétitions de code : 24")
print("Total des présentations : 120")

# ============================================================
# 4. SAUVEGARDE DU MANIFESTE
# ============================================================

control_manifest = {
    "seed": SEED,
    "source": str(TRAIN_FILE),
    "reference_manifest": str(
        REPLAY_MANIFEST
    ),
    "unique_code_examples": 96,
    "repeated_code_examples": N_REPEATS,
    "total_presentations": len(
        control_records
    ),
    "selected_ids": control_ids,
    "repeated_ids": repeated_code_ids,
}

with (
    OUTPUT_DIR / "control_manifest.json"
).open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        control_manifest,
        file,
        indent=4,
        ensure_ascii=False,
    )

# ============================================================
# 5. SOLUTIONS PYTHON
# ============================================================

# Cette fonction reprend exactement les
# démonstrations utilisées pour les exercices
# de code de l'étape 23.

def build_code_solution(question, expected):

    expression = (
        question.strip()
        .splitlines()[-1]
        .strip()
    )

    # --------------------------------------------------------
    # CAS 1 : PUISSANCE
    # print(a**b + c)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*\*\*\s*"
        r"(\d+)\s*\+\s*(\d+)\s*\)",
        expression,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
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

    # --------------------------------------------------------
    # CAS 2 : DIVISION ENTIÈRE
    # print(a // b)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*//\s*(\d+)\s*\)",
        expression,
    )

    if match:

        a, b = map(
            int,
            match.groups(),
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

    # --------------------------------------------------------
    # CAS 3 : SOMME D'UNE LISTE
    # print(sum([...]))
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*sum\(\s*"
        r"(\[[\d,\s-]+\])"
        r"\s*\)\s*\)",
        expression,
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

    # --------------------------------------------------------
    # CAS 4 : LONGUEUR D'UNE LISTE
    # print(len([...]))
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*len\(\s*"
        r"(\[[\d,\s-]+\])"
        r"\s*\)\s*\)",
        expression,
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
            raise ValueError(question)

        return (
            "La fonction len() retourne "
            "le nombre d'éléments.\n"
            f"La liste contient "
            f"{result} éléments.\n"
            f"REPONSE: {result}"
        )

    # --------------------------------------------------------
    # CAS 5 : MODULO
    # print(a % b + c)
    # --------------------------------------------------------

    match = re.fullmatch(
        r"print\(\s*(\d+)\s*%\s*"
        r"(\d+)\s*\+\s*(\d+)\s*\)",
        expression,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
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
# 6. PRÉPARATION DES EXEMPLES
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

for row in control_records:

    expected = int(
        row["answer"]
    )

    solution = build_code_solution(
        row["question"],
        expected,
    )

    examples.append({
        "question": row["question"],
        "solution": solution,
    })

print(
    "\nSolutions générées et vérifiées."
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

class ControlDataset(Dataset):

    def __init__(self, examples):

        self.samples = []

        for example in examples:

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": example["question"],
                },
            ]

            prompt = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            target = (
                example["solution"]
                + tokenizer.eos_token
            )

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = (
                prompt_ids + target_ids
            )

            if len(input_ids) > MAX_LENGTH:
                raise ValueError(
                    "Exemple trop long."
                )

            labels = (
                [-100] * len(prompt_ids)
                + target_ids
            )

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        return self.samples[
            index
        ]


train_dataset = ControlDataset(
    examples
)

# ============================================================
# 9. COLLATOR
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
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
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
            dtype=torch.long,
        )

        attention_mask[
            index, :length
        ] = 1

        labels[
            index, :length
        ] = torch.tensor(
            item["labels"],
            dtype=torch.long,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ============================================================
# 10. CHARGEMENT DE M2
# ============================================================

print("\nChargement de M2...")

base_model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE)
)

model = PeftModel.from_pretrained(
    base_model,
    str(INPUT_ADAPTER),
    is_trainable=True,
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
    data_seed=SEED,
)

# ============================================================
# 12. INITIALISATION DU TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ============================================================
# 13. ENTRAÎNEMENT DU GROUPE TÉMOIN
# ============================================================

print(
    "\nEntraînement de M3-C "
    "sur 96 exercices de code "
    "avec 24 répétitions..."
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

    "method": "SFT_code_control96",

    "seed": SEED,

    "unique_code_examples": 96,

    "repeated_code_examples": N_REPEATS,

    "total_presentations": 120,

    "steps": MAX_STEPS,

    "learning_rate": LEARNING_RATE,

    "training_loss": (
        training_result.training_loss
    ),

    "history": (
        trainer.state.log_history
    ),
}

with (
    OUTPUT_DIR / "training_log.json"
).open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        training_info,
        file,
        indent=4,
        ensure_ascii=False,
    )

# ============================================================
# 15. FIN
# ============================================================

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TÉMOIN TERMINÉ")
print("=" * 60)

print(
    f"Training loss : "
    f"{training_result.training_loss:.4f}"
)

print(
    f"Modèle sauvegardé : "
    f"{OUTPUT_DIR}"
)


import argparse
import ast
import gc
import json
import random
import re

from fractions import Fraction
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

REFERENCE_MANIFEST = (
    ROOT
    / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_replay20"
    / "replay_manifest.json"
)

STUDY_DIR = (
    ROOT / "models/replay_multiseed_v1"
)

RESULTS_DIR = ROOT / "results"

MAX_LENGTH = 256

MAX_STEPS = 60

LEARNING_RATE = 1e-4

ALLOWED_SEEDS = (42, 123, 2026)

RATIOS = (0, 10, 20)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

# ============================================================
# 2. ARGUMENTS DU TERMINAL
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Entraîner les modèles avec 0 %, "
        "10 % et 20 % de replay."
    )
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    choices=ALLOWED_SEEDS,
    help="Graine expérimentale.",
)

args = parser.parse_args()

SEED = args.seed

STUDY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 65)
print("CONTINUAL LEARNING - MULTI-SEED REPLAY")
print("=" * 65)

print(f"Seed : {SEED}")
print(f"Device : {DEVICE}")
print(f"Ratios : {RATIOS}")
print(f"Étapes par entraînement : {MAX_STEPS}")

if DEVICE != "mps":
    raise RuntimeError(
        "GPU Apple MPS indisponible."
    )

# ============================================================
# 3. VÉRIFICATION DES FICHIERS
# ============================================================

required_files = [
    TRAIN_FILE,
    REFERENCE_MANIFEST,
    INPUT_ADAPTER / "adapter_config.json",
]

for path in required_files:

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

# ============================================================
# 4. CHARGEMENT DES DONNÉES
# ============================================================

with TRAIN_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    train_records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

train_by_id = {
    row["id"]: row
    for row in train_records
}

if len(train_by_id) != len(train_records):

    raise ValueError(
        "Identifiants dupliqués."
    )

with REFERENCE_MANIFEST.open(
    "r",
    encoding="utf-8",
) as file:

    reference = json.load(file)

reference_ids = reference["selected_ids"]

if len(reference_ids) != 120:

    raise ValueError(
        "Le manifeste de référence "
        "doit contenir 120 exemples."
    )

if any(
    example_id not in train_by_id
    for example_id in reference_ids
):

    raise ValueError(
        "Identifiants absents du train."
    )

# ============================================================
# 5. MÊMES 96 EXERCICES DE CODE POUR TOUS
# ============================================================

code_ids = [
    example_id
    for example_id in reference_ids
    if train_by_id[example_id]["domain"] == "code"
]

maths_ids = [
    example_id
    for example_id in reference_ids
    if train_by_id[example_id]["domain"] == "maths"
]

algebra_ids = [
    example_id
    for example_id in reference_ids
    if train_by_id[example_id]["domain"] == "algebre"
]

if (
    len(code_ids) != 96
    or len(set(code_ids)) != 96
):

    raise ValueError(
        "96 exercices de code distincts attendus."
    )

if (
    len(maths_ids) != 12
    or len(algebra_ids) != 12
):

    raise ValueError(
        "12 exercices de maths et "
        "12 d'algèbre attendus."
    )

if len(set(reference_ids)) != 120:

    raise ValueError(
        "Le manifeste contient des doublons."
    )

print("\nDonnées communes vérifiées.")

print("Code distinct : 96")
print("Maths disponibles : 12")
print("Algèbre disponible : 12")

# ============================================================
# 6. ÉCHANTILLONNAGE REPRODUCTIBLE
# ============================================================

# Les répétitions de code sont communes
# et emboîtées entre les configurations.

code_rng = random.Random(SEED)

repeated_code_ids = code_rng.sample(
    code_ids,
    24,
)

# Les 6 anciens exercices utilisés pour
# le replay 10 % sont inclus dans ceux
# du replay 20 %.

old_rng = random.Random(
    SEED + 1
)

shuffled_maths = maths_ids.copy()

shuffled_algebra = algebra_ids.copy()

old_rng.shuffle(
    shuffled_maths
)

old_rng.shuffle(
    shuffled_algebra
)

# ============================================================
# 7. CONSTRUCTION DES TROIS CONFIGURATIONS
# ============================================================

def build_configuration(ratio):

    if ratio == 0:

        extra_code = (
            repeated_code_ids[:24]
        )

        extra_maths = []

        extra_algebra = []

    elif ratio == 10:

        extra_code = (
            repeated_code_ids[:12]
        )

        extra_maths = (
            shuffled_maths[:6]
        )

        extra_algebra = (
            shuffled_algebra[:6]
        )

    elif ratio == 20:

        extra_code = []

        extra_maths = (
            shuffled_maths[:12]
        )

        extra_algebra = (
            shuffled_algebra[:12]
        )

    else:

        raise ValueError(
            f"Ratio invalide : {ratio}"
        )

    selected_ids = (
        code_ids
        + extra_code
        + extra_maths
        + extra_algebra
    )

    if len(selected_ids) != 120:

        raise ValueError(
            "120 présentations attendues."
        )

    if len(set(
        example_id
        for example_id in selected_ids
        if train_by_id[example_id]["domain"] == "code"
    )) != 96:

        raise ValueError(
            "Les 96 exercices distincts "
            "doivent être conservés."
        )

    order_rng = random.Random(
        SEED + 10000
    )

    order_rng.shuffle(
        selected_ids
    )

    selected_records = [
        train_by_id[example_id]
        for example_id in selected_ids
    ]

    counts = {
        "code_distinct": 96,
        "code_repetitions": len(extra_code),
        "maths": len(extra_maths),
        "algebre": len(extra_algebra),
        "total": len(selected_ids),
    }

    manifest = {
        "seed": SEED,
        "ratio": ratio,
        "source": str(TRAIN_FILE),
        "initial_adapter": str(INPUT_ADAPTER),
        "counts": counts,
        "selected_ids": selected_ids,
        "repeated_code_ids": extra_code,
        "maths_ids": extra_maths,
        "algebra_ids": extra_algebra,
    }

    return (
        selected_records,
        manifest,
    )

# ============================================================
# 8. VALIDATION DES CALCULS MATHÉMATIQUES
# ============================================================

def calculate_maths(expression):

    tree = ast.parse(
        expression,
        mode="eval",
    )

    def visit(node):

        if isinstance(
            node,
            ast.Expression,
        ):

            return visit(
                node.body
            )

        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
        ):

            return Fraction(
                node.value
            )

        if isinstance(
            node,
            ast.UnaryOp,
        ):

            value = visit(
                node.operand
            )

            if isinstance(
                node.op,
                ast.USub,
            ):

                return -value

            if isinstance(
                node.op,
                ast.UAdd,
            ):

                return value

        if isinstance(
            node,
            ast.BinOp,
        ):

            left = visit(
                node.left
            )

            right = visit(
                node.right
            )

            if isinstance(
                node.op,
                ast.Add,
            ):

                return left + right

            if isinstance(
                node.op,
                ast.Sub,
            ):

                return left - right

            if isinstance(
                node.op,
                ast.Mult,
            ):

                return left * right

            if isinstance(
                node.op,
                ast.Div,
            ):

                if right == 0:
                    raise ValueError(
                        "Division par zéro."
                    )

                return left / right

        raise ValueError(
            "Expression mathématique non autorisée."
        )

    return visit(
        tree
    )

# ============================================================
# 9. SOLUTIONS MATHÉMATIQUES
# ============================================================

def build_maths_solution(
    question,
    expected,
):

    expression = (
        question.split(":", 1)[-1].strip()
    )

    calculated = calculate_maths(
        expression
    )

    if calculated != Fraction(
        expected
    ):

        raise ValueError(
            f"Réponse incorrecte : {question}"
        )

    return (
        f"On effectue le calcul : "
        f"{expression} = {expected}.\n"
        f"REPONSE: {expected}"
    )

# ============================================================
# 10. SOLUTIONS ALGÉBRIQUES
# ============================================================

def build_algebra_solution(
    question,
    expected,
):

    equation = (
        question.split(":", 1)[-1].strip()
    )

    number = r"(-?\d+)"

    # Cas 1 : ax + b = c

    match = re.fullmatch(
        rf"{number}x \+ {number} = {number}",
        equation,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
        )

        if a == 0 or a * expected + b != c:

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
        equation,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
        )

        if a == 0 or a * expected - b != c:

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
        equation,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
        )

        if (
            a == 0
            or a * (expected + b) != c
        ):

            raise ValueError(question)

        divided = (
            c // a
        )

        return (
            f"On divise les deux côtés par {a}.\n"
            f"x + {b} = {divided}\n"
            f"On soustrait {b}.\n"
            f"x = {divided} - {b} = {expected}\n"
            f"REPONSE: {expected}"
        )

    # Cas 4 : x/a + b = c

    match = re.fullmatch(
        rf"x/{number} \+ {number} = {number}",
        equation,
    )

    if match:

        a, b, c = map(
            int,
            match.groups(),
        )

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
        f"Équation inconnue : {equation}"
    )

# ============================================================
# 11. SOLUTIONS PYTHON
# ============================================================

def build_code_solution(
    question,
    expected,
):

    expression = (
        question.strip()
        .splitlines()[-1]
        .strip()
    )

    # Cas 1 : puissance
    # print(a**b + c)

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
            "On calcule d'abord la puissance.\n"
            f"{a}**{b} = {power}\n"
            f"On ajoute {c}.\n"
            f"{power} + {c} = {result}\n"
            f"REPONSE: {result}"
        )

    # Cas 2 : division entière
    # print(a // b)

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

    # Cas 3 : somme d'une liste
    # print(sum([...]))
    
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
                "Liste invalide."
            )

        result = sum(
            values
        )

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
                "Liste invalide."
            )

        result = len(
            values
        )

        if result != expected:

            raise ValueError(question)

        return (
            "La fonction len() retourne "
            "le nombre d'éléments.\n"
            f"La liste contient {result} éléments.\n"
            f"REPONSE: {result}"
        )

    # Cas 5 : modulo
    # print(a % b + c)

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
        f"Expression inconnue : {expression}"
    )

# ============================================================
# 12. CONSTRUCTION DES DÉMONSTRATIONS
# ============================================================

def build_examples(records):

    examples = []

    for row in records:

        expected = int(
            row["answer"]
        )

        domain = row["domain"]

        if domain == "maths":

            solution = build_maths_solution(
                row["question"],
                expected,
            )

        elif domain == "algebre":

            solution = build_algebra_solution(
                row["question"],
                expected,
            )

        elif domain == "code":

            solution = build_code_solution(
                row["question"],
                expected,
            )

        else:

            raise ValueError(
                f"Domaine inconnu : {domain}"
            )

        examples.append({
            "id": row["id"],
            "domain": domain,
            "question": row["question"],
            "solution": solution,
        })

    return examples

# ============================================================
# 13. TOKENIZER COMMUN
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
# 14. DATASET SFT
# ============================================================

class ContinualDataset(Dataset):

    def __init__(self, examples):

        self.samples = []

        self.target_tokens = 0

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

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
                prompt_ids
                + target_ids
            )

            if len(input_ids) > MAX_LENGTH:

                raise ValueError(
                    f"Exemple trop long : "
                    f"{example['id']}"
                )

            labels = (
                [-100] * len(prompt_ids)
                + target_ids
            )

            self.target_tokens += len(
                target_ids
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

        return self.samples[index]

# ============================================================
# 15. COLLATOR
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

    for index, item in enumerate(features):

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
# 16. ENTRAÎNEMENT D'UNE CONFIGURATION
# ============================================================

def train_configuration(ratio):

    output_dir = (
        STUDY_DIR
        / f"seed_{SEED}"
        / f"replay_{ratio}"
    )

    adapter_file = (
        output_dir / "adapter_config.json"
    )

    if adapter_file.exists():

        print(
            f"\nReplay {ratio}% : "
            "modèle déjà sauvegardé."
        )

        return None

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, manifest = build_configuration(
        ratio
    )

    examples = build_examples(
        records
    )

    dataset = ContinualDataset(
        examples
    )

    manifest["target_tokens"] = (
        dataset.target_tokens
    )

    with (
        output_dir / "training_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)

    print(
        f"SEED {SEED} - REPLAY {ratio}%"
    )

    print("=" * 65)

    print(
        f"Composition : {manifest['counts']}"
    )

    print(
        f"Tokens supervisés : "
        f"{dataset.target_tokens}"
    )

    # Chaque configuration repart
    # exactement du même adaptateur M2.

    random.seed(SEED)

    torch.manual_seed(SEED)

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

    # ========================================================
    # CONFIGURATION SFT
    # ========================================================

    training_args = TrainingArguments(
        output_dir=str(output_dir),

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ========================================================
    # ENTRAÎNEMENT
    # ========================================================

    training_result = trainer.train()

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    training_info = {
        "seed": SEED,

        "ratio": ratio,

        "method": "SFT_with_replay",

        "initial_adapter": str(
            INPUT_ADAPTER
        ),

        "output_adapter": str(
            output_dir
        ),

        "counts": manifest["counts"],

        "train_examples": len(
            dataset
        ),

        "target_tokens": (
            dataset.target_tokens
        ),

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
        output_dir / "training_log.json"
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

    print(
        f"\nTraining loss : "
        f"{training_result.training_loss:.4f}"
    )

    print(
        f"Modèle sauvegardé : {output_dir}"
    )

    # Libération de la mémoire
    # avant la configuration suivante.

    result = {
        "seed": SEED,
        "ratio": ratio,
        "training_loss": (
            training_result.training_loss
        ),
        "target_tokens": (
            dataset.target_tokens
        ),
        "adapter": str(
            output_dir
        ),
    }

    del trainer
    del model
    del base_model
    del dataset

    gc.collect()

    if DEVICE == "mps":

        torch.mps.empty_cache()

    return result

# ============================================================
# 17. EXPÉRIENCE COMPLÈTE POUR UNE GRAINE
# ============================================================

def main():

    results = []

    for ratio in RATIOS:

        result = train_configuration(
            ratio
        )

        if result is not None:

            results.append(
                result
            )

    summary_file = (
        RESULTS_DIR
        / f"replay_multiseed_seed_{SEED}_training.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "seed": SEED,
                "results": results,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)

    print(
        f"EXPÉRIENCE SEED {SEED} TERMINÉE"
    )

    print("=" * 65)

    for ratio in RATIOS:

        output_dir = (
            STUDY_DIR
            / f"seed_{SEED}"
            / f"replay_{ratio}"
        )

        print(
            f"Replay {ratio:2d}% : "
            f"{output_dir}"
        )

    print(
        f"\nRésumé : {summary_file}"
    )

# ============================================================
# 18. EXÉCUTION
# ============================================================

if __name__ == "__main__":

    main()

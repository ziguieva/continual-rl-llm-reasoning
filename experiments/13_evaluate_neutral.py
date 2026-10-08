
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = ROOT / "models/grpo_maths_v2"

DATA_FILE = ROOT / "data/processed/validation.jsonl"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DTYPE = torch.float32

MAX_NEW_TOKENS = 256

DOMAINS = ["maths", "algebre", "code"]

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=None
)

args = parser.parse_args()

if args.limit is not None and args.limit <= 0:
    parser.error("--limit doit être positif.")

# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"Adaptateur introuvable : {ADAPTER_DIR}"
    )

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if args.limit is not None:
    dataset = dataset[:args.limit]

if not dataset:
    raise ValueError("Dataset vide.")

print("=" * 60)
print("CONTINUAL RL - NEUTRAL EVALUATION")
print("=" * 60)

print(f"Modèle : {MODEL_NAME}")
print(f"Adaptateur : {ADAPTER_DIR}")
print(f"Device : {DEVICE}")
print(f"Questions : {len(dataset)}")

# ============================================================
# 4. CHARGEMENT DU MODÈLE ORIGINAL
# ============================================================

print("\nChargement de Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE
).to(DEVICE)

model.eval()

# ============================================================
# 5. EXTRACTION DE LA RÉPONSE
# ============================================================

def extract_answer(response, domain):

    lines = [
        line.strip().strip(" *`").strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    # Priorité au résultat final.
    last_line = lines[-1]

    # REPONSE: 42
    # RÉSULTAT: 42
    # RESPONSE: 42
    pattern = (
        r"(?:R[ÉE]PONSE|RESPONSE|"
        r"R[ÉE]SULTAT|RESULTAT)"
        r"\s*:\s*"
        r"(?:x\s*=\s*)?"
        r"(-?\d+)(?:\.0+)?[.!]?"
    )

    match = re.fullmatch(
        pattern,
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # Nombre entier seul
    match = re.fullmatch(
        r"(-?\d+)(?:\.0+)?[.!]?",
        last_line
    )

    if match:
        return int(match.group(1))

    # Équation finale : x = 42
    match = re.fullmatch(
        r"x\s*=\s*(-?\d+)(?:\.0+)?[.!]?",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # Calcul terminé : 22 * 19 = 418
    if domain == "maths":

        match = re.fullmatch(
            r"[0-9\s()+*/.-]+"
            r"=\s*(-?\d+)(?:\.0+)?",
            last_line
        )

        if match:
            return int(match.group(1))

    # Phrase finale :
    # "Le résultat est 42."
    match = re.search(
        r"(?:est|is|vaut|affiche)\s+"
        r"[`*]*(-?\d+)(?:\.0+)?[`*]*"
        r"[.!]?\s*$",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None

# ============================================================
# 6. GÉNÉRATION
# ============================================================

def generate_response(current_model, question, domain):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
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

        outputs = current_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        tokens,
        skip_special_tokens=True
    ).strip()

    predicted = extract_answer(response, domain)

    eos_reached = (
        tokens[-1].item() == tokenizer.eos_token_id
        if len(tokens) > 0
        else False
    )

    truncated = (
        len(tokens) >= MAX_NEW_TOKENS
        and not eos_reached
    )

    return {
        "predicted": predicted,
        "response": response,
        "tokens": len(tokens),
        "truncated": truncated
    }

# ============================================================
# 7. ÉVALUATION
# ============================================================

def evaluate(current_model, label):

    print(f"\nÉvaluation : {label}")

    current_model.eval()

    results = {}

    for index, example in enumerate(
        dataset,
        start=1
    ):

        result = generate_response(
            current_model,
            example["question"],
            example["domain"]
        )

        expected = int(example["answer"])

        result["correct"] = (
            result["predicted"] == expected
        )

        results[example["id"]] = result

        if index % 10 == 0 or index == len(dataset):
            print(
                f"Progression : "
                f"{index}/{len(dataset)}"
            )

    return results

# ============================================================
# 8. ÉVALUATION DU MODÈLE ORIGINAL
# ============================================================

baseline = evaluate(
    model,
    "Qwen original"
)

# ============================================================
# 9. CHARGEMENT DE L'ADAPTATEUR
# ============================================================

print("\nChargement de GRPO V2...")

trained_model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR)
)

trained_model.to(DEVICE)
trained_model.eval()

# ============================================================
# 10. ÉVALUATION APRÈS GRPO
# ============================================================

trained = evaluate(
    trained_model,
    "Qwen + GRPO V2"
)

# ============================================================
# 11. COMPARAISON
# ============================================================

comparison = []

for example in dataset:

    key = example["id"]

    before = baseline[key]
    after = trained[key]

    comparison.append({
        "id": key,
        "domain": example["domain"],
        "question": example["question"],
        "expected": example["answer"],

        "base_prediction": before["predicted"],
        "grpo_prediction": after["predicted"],

        "base_correct": before["correct"],
        "grpo_correct": after["correct"],

        "base_tokens": before["tokens"],
        "grpo_tokens": after["tokens"],

        "base_truncated": before["truncated"],
        "grpo_truncated": after["truncated"],

        "base_response": before["response"],
        "grpo_response": after["response"]
    })

# ============================================================
# 12. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows):

    total = len(rows)

    if not total:
        return None

    base_correct = sum(
        row["base_correct"]
        for row in rows
    )

    grpo_correct = sum(
        row["grpo_correct"]
        for row in rows
    )

    improved = sum(
        not row["base_correct"]
        and row["grpo_correct"]
        for row in rows
    )

    regressed = sum(
        row["base_correct"]
        and not row["grpo_correct"]
        for row in rows
    )

    return {
        "total": total,
        "base_correct": base_correct,
        "grpo_correct": grpo_correct,

        "base_accuracy": round(
            100 * base_correct / total, 2
        ),

        "grpo_accuracy": round(
            100 * grpo_correct / total, 2
        ),

        "improved": improved,
        "regressed": regressed,

        "base_missing": sum(
            row["base_prediction"] is None
            for row in rows
        ),

        "grpo_missing": sum(
            row["grpo_prediction"] is None
            for row in rows
        ),

        "base_tokens": sum(
            row["base_tokens"]
            for row in rows
        ),

        "grpo_tokens": sum(
            row["grpo_tokens"]
            for row in rows
        ),

        "base_truncated": sum(
            row["base_truncated"]
            for row in rows
        ),

        "grpo_truncated": sum(
            row["grpo_truncated"]
            for row in rows
        )
    }

# ============================================================
# 13. RÉSULTATS PAR DOMAINE
# ============================================================

summary = {
    "model": MODEL_NAME,
    "adapter": str(ADAPTER_DIR),
    "dataset": "validation",
    "prompt": SYSTEM_PROMPT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "domains": {},
    "global": calculate_metrics(comparison)
}

print("\n" + "=" * 60)
print("COMPARAISON PAR DOMAINE")
print("=" * 60)

for domain in DOMAINS:

    subset = [
        row
        for row in comparison
        if row["domain"] == domain
    ]

    metrics = calculate_metrics(subset)

    if metrics is None:
        continue

    summary["domains"][domain] = metrics

    print(f"\n{domain.upper()}")

    print(
        f"Avant : {metrics['base_accuracy']:.2f}%"
    )

    print(
        f"Après : {metrics['grpo_accuracy']:.2f}%"
    )

    print(
        f"Améliorées : {metrics['improved']}"
    )

    print(
        f"Dégradées : {metrics['regressed']}"
    )

    print(
        f"Prédictions manquantes : "
        f"{metrics['base_missing']} -> "
        f"{metrics['grpo_missing']}"
    )

# ============================================================
# 14. BILAN GLOBAL
# ============================================================

global_metrics = summary["global"]

delta = (
    global_metrics["grpo_accuracy"]
    - global_metrics["base_accuracy"]
)

print("\n" + "=" * 60)
print("BILAN GLOBAL")
print("=" * 60)

print(
    f"Avant GRPO : "
    f"{global_metrics['base_accuracy']:.2f}%"
)

print(
    f"Après GRPO : "
    f"{global_metrics['grpo_accuracy']:.2f}%"
)

print(f"Différence : {delta:+.2f} points")

print(
    f"Tokens : {global_metrics['base_tokens']}"
    f" -> {global_metrics['grpo_tokens']}"
)

# ============================================================
# 15. SAUVEGARDE
# ============================================================

suffix = (
    "_smoke"
    if args.limit is not None
    else ""
)

csv_path = (
    RESULTS_DIR
    / f"grpo_v2_neutral_validation{suffix}.csv"
)

json_path = (
    RESULTS_DIR
    / f"grpo_v2_neutral_validation{suffix}.json"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=comparison[0].keys()
    )

    writer.writeheader()
    writer.writerows(comparison)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nÉvaluation terminée.")
print(f"CSV : {csv_path}")
print(f"JSON : {json_path}")


import argparse
import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = ROOT / "models/grpo_maths_v2"

DATA_FILE = ROOT / "data/processed/validation.jsonl"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DTYPE = torch.float32

MAX_NEW_TOKENS = 256

DOMAINS = ["maths", "algebre", "code"]

SYSTEM_PROMPT = (
    "Tu es un assistant de mathématiques. "
    "Effectue les calculs avec précision. "
    "Tu peux raisonner brièvement. "
    "Termine par une ligne au format "
    "REPONSE: nombre. "
    "N'ajoute rien après cette ligne."
)

# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--domain",
    choices=["maths", "algebre", "code", "all"],
    default="maths"
)

args = parser.parse_args()

# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

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

if args.domain != "all":

    dataset = [
        row
        for row in dataset
        if row["domain"] == args.domain
    ]

if not dataset:
    raise ValueError("Dataset vide.")

if not (
    ADAPTER_DIR / "adapter_config.json"
).exists():

    raise FileNotFoundError(
        f"Adaptateur introuvable : {ADAPTER_DIR}"
    )

print("=" * 60)
print("CONTINUAL RL - GRPO V2 EVALUATION")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Modèle : {MODEL_NAME}")
print(f"Adaptateur : {ADAPTER_DIR}")
print(f"Domaine : {args.domain}")
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
# 5. EXTRACTION DES RÉPONSES
# ============================================================

def extract_answer(response, domain):

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
        response,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if matches:
        return int(matches[-1])

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    last_line = lines[-1]

    # Réponse numérique seule
    match = re.fullmatch(
        r"(-?\d+)(?:\.0+)?[.!]?",
        last_line
    )

    if match:
        return int(match.group(1))

    # Réponse algébrique finale
    match = re.fullmatch(
        r"x\s*=\s*(-?\d+)(?:\.0+)?[.!]?",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # Dernière phrase contenant le résultat
    match = re.search(
        r"(?:est|is|vaut|affiche)\s+"
        r"[`*]*(-?\d+)(?:\.0+)?[`*]*"
        r"[.!]?\s*$",
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # Calcul arithmétique terminé par = nombre
    if domain == "maths":

        match = re.fullmatch(
            r"[0-9\s()+*/.-]+"
            r"=\s*(-?\d+)(?:\.0+)?",
            last_line
        )

        if match:
            return int(match.group(1))

    return None

# ============================================================
# 6. GÉNÉRATION
# ============================================================

def generate_response(
    current_model,
    question,
    domain
):

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

    generated = outputs[0][input_length:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    predicted = extract_answer(
        response,
        domain
    )

    return {
        "prediction": predicted,
        "response": response,
        "tokens": len(generated)
    }

# ============================================================
# 7. ÉVALUATION D'UN MODÈLE
# ============================================================

def evaluate(current_model, label):

    current_model.eval()

    print(f"\nÉvaluation : {label}")

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
            result["prediction"] == expected
        )

        results[example["id"]] = result

        status = (
            "OK"
            if result["correct"]
            else "ERREUR"
        )

        print(
            f"[{index}/{len(dataset)}] "
            f"{example['domain']:8s} | "
            f"{status:6s} | "
            f"Attendu : {expected} | "
            f"Prédit : {result['prediction']}"
        )

    return results

# ============================================================
# 8. ÉVALUATION AVANT GRPO
# ============================================================

baseline = evaluate(
    model,
    "Qwen original"
)

# ============================================================
# 9. CHARGEMENT DE L'ADAPTATEUR GRPO V2
# ============================================================

print("\nChargement de l'adaptateur GRPO V2...")

grpo_model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR)
)

grpo_model.to(DEVICE)
grpo_model.eval()

# ============================================================
# 10. ÉVALUATION APRÈS GRPO
# ============================================================

trained = evaluate(
    grpo_model,
    "Qwen + GRPO V2"
)

# ============================================================
# 11. COMPARAISON QUESTION PAR QUESTION
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

        "base_prediction": before["prediction"],
        "grpo_prediction": after["prediction"],

        "base_correct": before["correct"],
        "grpo_correct": after["correct"],

        "base_tokens": before["tokens"],
        "grpo_tokens": after["tokens"],

        "base_response": before["response"],
        "grpo_response": after["response"]
    })

# ============================================================
# 12. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows):

    total = len(rows)

    if total == 0:
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

    changed = sum(
        row["base_response"]
        != row["grpo_response"]
        for row in rows
    )

    base_accuracy = (
        100 * base_correct / total
    )

    grpo_accuracy = (
        100 * grpo_correct / total
    )

    return {
        "questions": total,
        "base_accuracy": round(base_accuracy, 2),
        "grpo_accuracy": round(grpo_accuracy, 2),
        "accuracy_delta": round(
            grpo_accuracy - base_accuracy,
            2
        ),
        "improved": improved,
        "regressed": regressed,
        "changed_responses": changed,
        "base_tokens": sum(
            row["base_tokens"]
            for row in rows
        ),
        "grpo_tokens": sum(
            row["grpo_tokens"]
            for row in rows
        )
    }

# ============================================================
# 13. RÉSULTATS PAR DOMAINE
# ============================================================

summary = {
    "model": MODEL_NAME,
    "adapter": str(ADAPTER_DIR),
    "domain": args.domain,
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
        f"Différence : "
        f"{metrics['accuracy_delta']:+.2f} points"
    )

    print(
        f"Questions améliorées : {metrics['improved']}"
    )

    print(
        f"Questions dégradées : {metrics['regressed']}"
    )

# ============================================================
# 14. RÉSULTATS GLOBAUX
# ============================================================

global_metrics = summary["global"]

print("\n" + "=" * 60)
print("COMPARAISON FINALE GRPO V2")
print("=" * 60)

print(
    f"Avant GRPO : "
    f"{global_metrics['base_accuracy']:.2f}%"
)

print(
    f"Après GRPO : "
    f"{global_metrics['grpo_accuracy']:.2f}%"
)

print(
    f"Différence : "
    f"{global_metrics['accuracy_delta']:+.2f} points"
)

print(
    f"Questions améliorées : "
    f"{global_metrics['improved']}"
)

print(
    f"Questions dégradées : "
    f"{global_metrics['regressed']}"
)

print(
    f"Réponses modifiées : "
    f"{global_metrics['changed_responses']}"
)

print(
    f"Tokens avant : "
    f"{global_metrics['base_tokens']}"
)

print(
    f"Tokens après : "
    f"{global_metrics['grpo_tokens']}"
)

# ============================================================
# 15. SAUVEGARDE
# ============================================================

suffix = args.domain

csv_path = (
    RESULTS_DIR
    / f"grpo_v2_compare_{suffix}.csv"
)

json_path = (
    RESULTS_DIR
    / f"grpo_v2_compare_{suffix}.json"
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

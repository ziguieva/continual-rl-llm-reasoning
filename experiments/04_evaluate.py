
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "mps"
    else torch.float32
)

MAX_NEW_TOKENS = 256

DOMAINS = ["maths", "algebre", "code"]

# ============================================================
# 2. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    choices=["validation", "test"],
    default="validation"
)

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

data_path = DATA_DIR / f"{args.split}.jsonl"

with open(
    data_path,
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
    raise ValueError("Le dataset est vide.")

# Vérification des données
ids = set()
questions = set()

for example in dataset:

    assert example["domain"] in DOMAINS
    assert isinstance(example["answer"], int)

    assert example["id"] not in ids
    assert example["question"] not in questions

    ids.add(example["id"])
    questions.add(example["question"])

print("=" * 60)
print("CONTINUAL RL - DATASET BASELINE")
print("=" * 60)

print(f"Split : {args.split}")
print(f"Questions : {len(dataset)}")
print(f"Modèle : {MODEL_NAME}")
print(f"GPU : {DEVICE}")

# ============================================================
# 4. CHARGEMENT DU MODÈLE
# ============================================================

print("\nChargement du modèle...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE
).to(DEVICE)

model.eval()

print("Modèle chargé.")

# ============================================================
# 5. GÉNÉRATION
# ============================================================

def generate_response(question):

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant spécialisé en "
                "mathématiques, algèbre et Python. "
                "Résous précisément le problème. "
                "Termine par une ligne commençant "
                "par REPONSE: suivie directement "
                "du nombre entier trouvé. "
                "N'ajoute aucun texte après cette ligne."
            )
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

    start = time.perf_counter()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    input_length = inputs["input_ids"].shape[1]

    tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        tokens,
        skip_special_tokens=True
    ).strip()

    token_count = len(tokens)

    eos_reached = (
        tokens[-1].item() == tokenizer.eos_token_id
        if token_count > 0
        else False
    )

    truncated = (
        token_count >= MAX_NEW_TOKENS
        and not eos_reached
    )

    return response, token_count, truncated, elapsed

# ============================================================
# 6. EXTRACTION DE LA RÉPONSE
# ============================================================

def extract_answer(response):

    # Format demandé : REPONSE: 15
    # Variantes : RÉPONSE: 15, RESPONSE: 15

    pattern = (
        r"^\s*\**"
        r"(?:R[ÉE]PONSE|RESPONSE)"
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
        return int(matches[-1]), "explicit"

    # Réponse entièrement numérique
    match = re.fullmatch(
        r"\s*(?:x\s*=\s*)?"
        r"(-?\d+)(?:\.0+)?\s*[.!]?\s*",
        response
    )

    if match:
        return int(match.group(1)), "numeric"

    # Dernière phrase indiquant un résultat
    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if lines:

        last_line = lines[-1]

        match = re.search(
            r"(?:est|is|vaut|affiche)\s+"
            r"[`*]*(-?\d+)(?:\.0+)?[`*]*"
            r"[.!]?\s*$",
            last_line,
            flags=re.IGNORECASE
        )

        if match:
            return int(match.group(1)), "fallback"

    return None, "not_found"

# ============================================================
# 7. FICHIERS DE SORTIE
# ============================================================

suffix = (
    "_smoke"
    if args.limit is not None
    else ""
)

output_name = f"baseline_{args.split}{suffix}"

csv_path = RESULTS_DIR / f"{output_name}.csv"

json_path = RESULTS_DIR / f"{output_name}.json"

fields = [
    "id",
    "domain",
    "question",
    "expected",
    "predicted",
    "correct",
    "extraction_method",
    "format_error",
    "truncated",
    "token_count",
    "elapsed_seconds",
    "full_response"
]

# ============================================================
# 8. ÉVALUATION
# ============================================================

results = []

print("\nDébut de l'évaluation...\n")

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()

    for index, example in enumerate(
        dataset,
        start=1
    ):

        response, tokens, truncated, elapsed = (
            generate_response(
                example["question"]
            )
        )

        predicted, method = extract_answer(
            response
        )

        expected = example["answer"]

        correct = (
            predicted is not None
            and predicted == expected
        )

        record = {
            "id": example["id"],
            "domain": example["domain"],
            "question": example["question"],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "extraction_method": method,
            "format_error": predicted is None,
            "truncated": truncated,
            "token_count": tokens,
            "elapsed_seconds": round(elapsed, 3),
            "full_response": response
        }

        results.append(record)

        writer.writerow(record)
        file.flush()

        status = "OK" if correct else "ERREUR"

        print(
            f"[{index}/{len(dataset)}] "
            f"{example['domain']:8s} | "
            f"{status:6s} | "
            f"Attendu: {expected} | "
            f"Prédit: {predicted}"
        )

# ============================================================
# 9. CALCUL DES MÉTRIQUES
# ============================================================

summary = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "split": args.split,
    "max_new_tokens": MAX_NEW_TOKENS,
    "domains": {},
    "global": {}
}

print("\n" + "=" * 60)
print("RÉSULTATS")
print("=" * 60)

for domain in DOMAINS:

    subset = [
        result
        for result in results
        if result["domain"] == domain
    ]

    total = len(subset)

    correct = sum(
        result["correct"]
        for result in subset
    )

    accuracy = (
        100 * correct / total
        if total
        else 0.0
    )

    format_errors = sum(
        result["format_error"]
        for result in subset
    )

    truncated = sum(
        result["truncated"]
        for result in subset
    )

    summary["domains"][domain] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "format_errors": format_errors,
        "truncated": truncated
    }

    print(
        f"{domain.upper():10s} : "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

# ============================================================
# 10. SCORE GLOBAL
# ============================================================

total = len(results)

correct = sum(
    result["correct"]
    for result in results
)

accuracy = 100 * correct / total

format_errors = sum(
    result["format_error"]
    for result in results
)

truncated = sum(
    result["truncated"]
    for result in results
)

total_tokens = sum(
    result["token_count"]
    for result in results
)

summary["global"] = {
    "correct": correct,
    "total": total,
    "accuracy": accuracy,
    "format_errors": format_errors,
    "truncated": truncated,
    "generated_tokens": total_tokens
}

print("-" * 60)

print(f"Score global : {accuracy:.2f}%")
print(f"Erreurs de format : {format_errors}")
print(f"Générations tronquées : {truncated}")
print(f"Tokens générés : {total_tokens}")

# ============================================================
# 11. SAUVEGARDE JSON
# ============================================================

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

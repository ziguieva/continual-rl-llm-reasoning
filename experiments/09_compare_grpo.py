
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

ADAPTER_DIR = ROOT / "models/grpo_maths_smoke"

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

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en "
    "mathématiques, algèbre et Python. "
    "Résous précisément le problème. "
    "Termine par une ligne commençant "
    "par REPONSE: suivie directement "
    "du nombre entier trouvé. "
    "N'ajoute aucun texte après cette ligne."
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

parser.add_argument(
    "--limit",
    type=int,
    default=None
)

args = parser.parse_args()

if args.limit is not None and args.limit <= 0:
    parser.error("--limit doit être positif.")

# ============================================================
# 3. VÉRIFICATION DES FICHIERS
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

if not (
    ADAPTER_DIR / "adapter_config.json"
).exists():
    raise FileNotFoundError(
        "Adaptateur LoRA introuvable : "
        f"{ADAPTER_DIR}"
    )

print("=" * 60)
print("CONTINUAL RL - BASE VS GRPO")
print("=" * 60)

print(f"GPU : {DEVICE}")
print(f"Domaine : {args.domain}")

# ============================================================
# 4. CHARGEMENT DU DATASET
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

if args.limit is not None:
    dataset = dataset[:args.limit]

if not dataset:
    raise ValueError("Aucune question à évaluer.")

print(f"Questions : {len(dataset)}")

# ============================================================
# 5. CHARGEMENT DU MODÈLE ORIGINAL
# ============================================================

print("\nChargement de Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE
).to(DEVICE)

model.eval()

# ============================================================
# 6. EXTRACTION DE LA RÉPONSE
# ============================================================

def extract_answer(response):

    pattern = (
        r"^\s*"
        r"(?:REPONSE|RÉPONSE|RESPONSE|"
        r"RESULTAT|RÉSULTAT)"
        r"\s*:\s*"
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

    match = re.fullmatch(
        r"\s*(-?\d+)(?:\.0+)?\s*[.!]?\s*",
        response
    )

    if match:
        return int(match.group(1))

    return None

# ============================================================
# 7. GÉNÉRATION
# ============================================================

def generate_response(current_model, question):

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
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated = outputs[0][input_length:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    predicted = extract_answer(response)

    return {
        "response": response,
        "predicted": predicted,
        "tokens": len(generated)
    }

# ============================================================
# 8. ÉVALUATION D'UN MODÈLE
# ============================================================

def evaluate(current_model, name):

    print(f"\nÉvaluation : {name}")

    results = {}

    for index, example in enumerate(
        dataset,
        start=1
    ):

        result = generate_response(
            current_model,
            example["question"]
        )

        expected = int(example["answer"])

        result["correct"] = (
            result["predicted"] == expected
        )

        results[example["id"]] = result

        status = (
            "OK"
            if result["correct"]
            else "ERREUR"
        )

        print(
            f"[{index}/{len(dataset)}] "
            f"{status:6s} | "
            f"Attendu: {expected} | "
            f"Prédit: {result['predicted']}"
        )

    return results

# ============================================================
# 9. ÉVALUATION DU MODÈLE ORIGINAL
# ============================================================

baseline = evaluate(
    model,
    "Qwen original"
)

# ============================================================
# 10. CHARGEMENT DE L'ADAPTATEUR LORA
# ============================================================

print("\nChargement de l'adaptateur GRPO...")

grpo_model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR)
)

grpo_model.eval()

# ============================================================
# 11. ÉVALUATION APRÈS GRPO
# ============================================================

trained = evaluate(
    grpo_model,
    "Qwen + GRPO"
)

# ============================================================
# 12. COMPARAISON DES RÉSULTATS
# ============================================================

comparison = []

for example in dataset:

    example_id = example["id"]

    before = baseline[example_id]
    after = trained[example_id]

    comparison.append({
        "id": example_id,
        "domain": example["domain"],
        "question": example["question"],
        "expected": example["answer"],

        "base_prediction": before["predicted"],
        "grpo_prediction": after["predicted"],

        "base_correct": before["correct"],
        "grpo_correct": after["correct"],

        "base_tokens": before["tokens"],
        "grpo_tokens": after["tokens"],

        "base_response": before["response"],
        "grpo_response": after["response"]
    })

# ============================================================
# 13. CALCUL DES MÉTRIQUES
# ============================================================

total = len(comparison)

base_correct = sum(
    row["base_correct"]
    for row in comparison
)

grpo_correct = sum(
    row["grpo_correct"]
    for row in comparison
)

improved = sum(
    not row["base_correct"]
    and row["grpo_correct"]
    for row in comparison
)

regressed = sum(
    row["base_correct"]
    and not row["grpo_correct"]
    for row in comparison
)

changed = sum(
    row["base_response"] != row["grpo_response"]
    for row in comparison
)

base_accuracy = 100 * base_correct / total
grpo_accuracy = 100 * grpo_correct / total

base_tokens = sum(
    row["base_tokens"]
    for row in comparison
)

grpo_tokens = sum(
    row["grpo_tokens"]
    for row in comparison
)

summary = {
    "model": MODEL_NAME,
    "adapter": str(ADAPTER_DIR),
    "domain": args.domain,
    "questions": total,
    "base_accuracy": base_accuracy,
    "grpo_accuracy": grpo_accuracy,
    "accuracy_delta": (
        grpo_accuracy - base_accuracy
    ),
    "improved_questions": improved,
    "regressed_questions": regressed,
    "changed_responses": changed,
    "base_tokens": base_tokens,
    "grpo_tokens": grpo_tokens
}

# ============================================================
# 14. AFFICHAGE
# ============================================================

print("\n" + "=" * 60)
print("COMPARAISON FINALE")
print("=" * 60)

print(
    f"Qwen original : "
    f"{base_accuracy:.2f}% "
    f"({base_correct}/{total})"
)

print(
    f"Qwen + GRPO   : "
    f"{grpo_accuracy:.2f}% "
    f"({grpo_correct}/{total})"
)

print(
    f"Différence : "
    f"{summary['accuracy_delta']:+.2f} points"
)

print(f"Questions améliorées : {improved}")
print(f"Questions dégradées : {regressed}")
print(f"Réponses modifiées : {changed}")

print(f"Tokens avant : {base_tokens}")
print(f"Tokens après : {grpo_tokens}")

# ============================================================
# 15. SAUVEGARDE
# ============================================================

suffix = args.domain

if args.limit is not None:
    suffix += f"_limit{args.limit}"

csv_path = (
    RESULTS_DIR / f"grpo_smoke_compare_{suffix}.csv"
)

json_path = (
    RESULTS_DIR / f"grpo_smoke_compare_{suffix}.json"
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

print("\nComparaison terminée.")
print(f"CSV : {csv_path}")
print(f"JSON : {json_path}")

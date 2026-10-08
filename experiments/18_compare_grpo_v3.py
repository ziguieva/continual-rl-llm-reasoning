
"""Compare Qwen original, GRPO V2 et GRPO V3 avec l'évaluateur figé V16."""

import csv
import importlib.util
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_FILE = ROOT / "data/processed/validation.jsonl"

V2_CSV = ROOT / "results/grpo_v2_neutral_validation.csv"

V3_ADAPTER = ROOT / "models/grpo_maths_v3"

EVALUATOR_FILE = ROOT / "experiments/16_finalize_validation.py"

RESULTS_DIR = ROOT / "results"

MAX_NEW_TOKENS = 256

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Même prompt et même génération que l'étape 13.
SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

# ============================================================
# 2. CHARGEMENT DE L'ÉVALUATEUR V16 SANS LE MODIFIER
# ============================================================

spec = importlib.util.spec_from_file_location(
    "audit_v16",
    EVALUATOR_FILE
)

if spec is None or spec.loader is None:
    raise FileNotFoundError(
        f"Évaluateur introuvable : {EVALUATOR_FILE}"
    )

audit = importlib.util.module_from_spec(spec)

spec.loader.exec_module(audit)

audit.run_tests()

# ============================================================
# 3. DONNÉES DE VALIDATION ET RÉSULTATS DÉJÀ ENREGISTRÉS
# ============================================================

with DATA_FILE.open(
    "r",
    encoding="utf-8"
) as stream:

    dataset = [
        json.loads(line)
        for line in stream
        if line.strip()
    ]

with V2_CSV.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as stream:

    previous = list(
        csv.DictReader(stream)
    )

old_by_id = {
    row["id"]: row
    for row in previous
}

ids = [
    row["id"]
    for row in dataset
]

if (
    len(ids) != len(set(ids))
    or len(previous) != len(old_by_id)
):
    raise ValueError(
        "Identifiants dupliqués dans les données."
    )

if set(ids) != set(old_by_id):
    raise ValueError(
        "Les questions ne correspondent pas à "
        "l'évaluation V2."
    )

if not (
    V3_ADAPTER / "adapter_config.json"
).is_file():

    raise FileNotFoundError(
        f"Adaptateur V3 introuvable : {V3_ADAPTER}"
    )

for example in dataset:

    old = old_by_id[example["id"]]

    if (
        old["question"] != example["question"]
        or int(old["expected"]) != int(example["answer"])
    ):

        raise ValueError(
            f"Question modifiée : {example['id']}"
        )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nCONTINUAL RL — COMPARAISON BASE / V2 / V3")

print(
    f"Questions de validation : {len(dataset)}"
)

print(f"Appareil : {DEVICE}")

# ============================================================
# 4. CHARGEMENT DE QWEN + ADAPTATEUR V3
# ============================================================

print("\nChargement de Qwen et de GRPO V3...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
).to(DEVICE)

model = PeftModel.from_pretrained(
    base_model,
    str(V3_ADAPTER)
)

model.to(DEVICE)

model.eval()

# ============================================================
# 5. GÉNÉRATION V3 (INFÉRENCE DÉTERMINISTE)
# ============================================================

def generate_v3(question):

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

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    tokens = output[0][input_length:]

    response = tokenizer.decode(
        tokens,
        skip_special_tokens=True
    ).strip()

    return response, len(tokens)

# ============================================================
# 6. MÊME AUDIT POUR LES TROIS MODÈLES
# ============================================================

rows = []

for index, example in enumerate(
    dataset,
    start=1
):

    old = old_by_id[example["id"]]

    answer = int(example["answer"])

    response_v3, tokens_v3 = generate_v3(
        example["question"]
    )

    responses = {
        "base": old["base_response"],
        "v2": old["grpo_response"],
        "v3": response_v3
    }

    record = {
        "id": example["id"],
        "domain": example["domain"],
        "question": example["question"],
        "expected": answer
    }

    for label, response in responses.items():

        result = audit.analyze_response(
            response,
            example["domain"],
            answer
        )

        record[f"{label}_response"] = response

        record[f"{label}_prediction"] = (
            result["prediction"]
        )

        record[f"{label}_status"] = (
            result["status"]
        )

        record[f"{label}_correct"] = (
            result["strict_correct"]
        )

        record[f"{label}_format_ok"] = (
            result["format_ok"]
        )

        record[f"{label}_end_to_end"] = (
            result["end_to_end"]
        )

        record[f"{label}_conclusion_correct"] = (
            result["conclusion_correct"]
        )

    record["base_tokens"] = int(
        old["base_tokens"]
    )

    record["v2_tokens"] = int(
        old["grpo_tokens"]
    )

    record["v3_tokens"] = tokens_v3

    rows.append(record)

    if index % 10 == 0 or index == len(dataset):

        print(
            f"Évaluation V3 : {index}/{len(dataset)}"
        )

# ============================================================
# 7. SCORES STRICTS ET CHANGEMENTS QUESTION PAR QUESTION
# ============================================================

def metrics(subset):

    total = len(subset)

    output = {
        "questions": total
    }

    for label in ("base", "v2", "v3"):

        correct = sum(
            bool(row[f"{label}_correct"])
            for row in subset
        )

        output[label] = {
            "correct": correct,

            "accuracy": (
                round(100 * correct / total, 2)
                if total
                else 0.0
            ),

            "contradictions": sum(
                row[f"{label}_status"] == "contradictory"
                for row in subset
            ),

            "missing": sum(
                row[f"{label}_status"] == "missing"
                for row in subset
            ),

            "correct_and_formatted": sum(
                bool(row[f"{label}_end_to_end"])
                for row in subset
            ),

            "tokens": sum(
                row[f"{label}_tokens"]
                for row in subset
            )
        }

    for before, after in (
        ("base", "v3"),
        ("v2", "v3")
    ):

        output[f"{before}_to_{after}"] = {

            "improved": sum(
                not row[f"{before}_correct"]
                and row[f"{after}_correct"]
                for row in subset
            ),

            "regressed": sum(
                row[f"{before}_correct"]
                and not row[f"{after}_correct"]
                for row in subset
            )
        }

    return output

# ============================================================
# 8. RÉSUMÉ
# ============================================================

summary = {
    "model": MODEL_NAME,

    "adapter_v3": str(V3_ADAPTER),

    "validation_file": str(DATA_FILE),

    "evaluator": "16_finalize_validation.py",

    "system_prompt": SYSTEM_PROMPT,

    "max_new_tokens": MAX_NEW_TOKENS,

    "do_sample": False,

    "domains": {},

    "global": metrics(rows)
}

# ============================================================
# 9. COMPARAISON PAR DOMAINE
# ============================================================

print("\n" + "=" * 62)

print(
    "COMPARAISON PAR DOMAINE — "
    "EXACTITUDE NON CONTRADICTOIRE"
)

print("=" * 62)

for domain in (
    "maths",
    "algebre",
    "code"
):

    group = [
        row
        for row in rows
        if row["domain"] == domain
    ]

    result = metrics(group)

    summary["domains"][domain] = result

    print(f"\n{domain.upper()}")

    print(
        " | ".join(
            f"{label.upper()} : "
            f"{result[label]['accuracy']:.2f}%"
            for label in ("base", "v2", "v3")
        )
    )

    changes = result["v2_to_v3"]

    print(
        f"V2 -> V3 : "
        f"{changes['improved']} améliorées / "
        f"{changes['regressed']} dégradées"
    )

    print(
        "Contradictions : "
        + " / ".join(
            str(result[label]["contradictions"])
            for label in ("base", "v2", "v3")
        )
    )

# ============================================================
# 10. BILAN GLOBAL
# ============================================================

print("\n" + "=" * 62)

print(
    "BILAN GLOBAL — BASE / GRPO V2 / GRPO V3"
)

print("=" * 62)

for label in ("base", "v2", "v3"):

    result = summary["global"][label]

    print(
        f"{label.upper():4s} : "
        f"{result['accuracy']:.2f}% "
        f"({result['correct']}/{len(rows)}) | "
        f"contradictions : "
        f"{result['contradictions']} | "
        f"exactitude + format : "
        f"{result['correct_and_formatted']}"
    )

changes = summary["global"]["v2_to_v3"]

print(
    f"V2 -> V3 : "
    f"{changes['improved']} améliorées / "
    f"{changes['regressed']} dégradées"
)

# ============================================================
# 11. EXPORT DES RÉSULTATS
# ============================================================

csv_path = (
    RESULTS_DIR / "grpo_v3_comparison_validation.csv"
)

json_path = (
    RESULTS_DIR / "grpo_v3_comparison_validation.json"
)

review_path = (
    RESULTS_DIR / "grpo_v3_manual_review.csv"
)

# Sauvegarde détaillée CSV
with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as stream:

    writer = csv.DictWriter(
        stream,
        fieldnames=list(rows[0])
    )

    writer.writeheader()

    writer.writerows(rows)

# Sauvegarde du résumé JSON
with json_path.open(
    "w",
    encoding="utf-8"
) as stream:

    json.dump(
        summary,
        stream,
        indent=2,
        ensure_ascii=False
    )

# Identification des cas à vérifier
review = [
    row
    for row in rows
    if any(
        row[f"{label}_status"] in {
            "contradictory",
            "missing",
            "non_integer"
        }
        for label in ("base", "v2", "v3")
    )
]

# Sauvegarde des cas nécessitant un audit
with review_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as stream:

    writer = csv.DictWriter(
        stream,
        fieldnames=list(rows[0])
    )

    writer.writeheader()

    writer.writerows(review)

# ============================================================
# 12. FIN
# ============================================================

print(f"\nCSV : {csv_path}")

print(f"Résumé : {json_path}")

print(
    f"Cas à vérifier : "
    f"{review_path} ({len(review)})"
)

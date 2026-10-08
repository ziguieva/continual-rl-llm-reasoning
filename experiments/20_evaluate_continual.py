
import csv
import json
import importlib.util
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

M1_ADAPTER = ROOT / "models/grpo_maths_v2"

M2_ADAPTER = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

DATA_FILE = ROOT / "data/processed/validation.jsonl"

M1_RESULTS = (
    ROOT / "results/grpo_v2_neutral_validation.csv"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 256

DOMAINS = ["maths", "algebre", "code"]

# Même prompt que notre évaluation précédente.
SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

print("=" * 60)
print("CONTINUAL RL - SEQUENTIAL LEARNING")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"M1 : {M1_ADAPTER}")
print(f"M2 : {M2_ADAPTER}")

# ============================================================
# 2. VÉRIFICATION DES FICHIERS
# ============================================================

for path in [
    DATA_FILE,
    M1_RESULTS,
    EVALUATOR_FILE,
    M1_ADAPTER / "adapter_config.json",
    M2_ADAPTER / "adapter_config.json"
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

# ============================================================
# 3. CHARGEMENT DE L'ÉVALUATEUR V16
# ============================================================

spec = importlib.util.spec_from_file_location(
    "audit_v16",
    EVALUATOR_FILE
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Impossible de charger l'évaluateur V16."
    )

audit = importlib.util.module_from_spec(spec)

spec.loader.exec_module(audit)

audit.run_tests()

print("Évaluateur V16 : OK")

# ============================================================
# 4. CHARGEMENT DES DONNÉES
# ============================================================

with DATA_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

with M1_RESULTS.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:

    previous = list(csv.DictReader(file))

previous_by_id = {
    row["id"]: row
    for row in previous
}

if len(previous_by_id) != len(previous):
    raise ValueError(
        "Identifiants dupliqués dans les résultats M1."
    )

dataset_ids = {
    row["id"]
    for row in dataset
}

if dataset_ids != set(previous_by_id):
    raise ValueError(
        "Les données de validation ont changé."
    )

for example in dataset:

    old = previous_by_id[example["id"]]

    if (
        old["question"] != example["question"]
        or int(old["expected"]) != int(example["answer"])
    ):

        raise ValueError(
            f"Incohérence : {example['id']}"
        )

print(f"Questions : {len(dataset)}")

# ============================================================
# 5. CHARGEMENT DU MODÈLE M2
# ============================================================

print("\nChargement de Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
).to(DEVICE)

print("Chargement de l'adaptateur M2...")

# M2 contient les poids LoRA après les deux
# entraînements successifs : GRPO puis SFT.

model = PeftModel.from_pretrained(
    base_model,
    str(M2_ADAPTER)
)

model.to(DEVICE)

model.eval()

print("Modèle M2 chargé.")

# ============================================================
# 6. GÉNÉRATION DES RÉPONSES
# ============================================================

def generate_response(question):

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

    generated_tokens = output[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response, len(generated_tokens)

# ============================================================
# 7. ÉVALUATION DES DEUX MODÈLES
# ============================================================

results = []

print("\nDébut de l'évaluation...\n")

for index, example in enumerate(
    dataset,
    start=1
):

    example_id = example["id"]

    expected = int(example["answer"])

    old = previous_by_id[example_id]

    # Réponse de M1 déjà enregistrée.
    m1_response = old["grpo_response"]

    # Nouvelle réponse de M2.
    m2_response, m2_tokens = generate_response(
        example["question"]
    )

    # Même évaluateur pour les deux modèles.
    m1 = audit.analyze_response(
        m1_response,
        example["domain"],
        expected
    )

    m2 = audit.analyze_response(
        m2_response,
        example["domain"],
        expected
    )

    record = {
        "id": example_id,
        "domain": example["domain"],
        "question": example["question"],
        "expected": expected,

        "m1_prediction": m1["prediction"],
        "m2_prediction": m2["prediction"],

        "m1_status": m1["status"],
        "m2_status": m2["status"],

        "m1_correct": m1["strict_correct"],
        "m2_correct": m2["strict_correct"],

        "m1_format_ok": m1["format_ok"],
        "m2_format_ok": m2["format_ok"],

        "m1_end_to_end": m1["end_to_end"],
        "m2_end_to_end": m2["end_to_end"],

        "m1_tokens": int(old["grpo_tokens"]),
        "m2_tokens": m2_tokens,

        "m1_response": m1_response,
        "m2_response": m2_response
    }

    results.append(record)

    if index % 10 == 0 or index == len(dataset):

        print(
            f"Progression : {index}/{len(dataset)}"
        )

# ============================================================
# 8. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows):

    total = len(rows)

    if total == 0:
        return None

    m1_correct = sum(
        bool(row["m1_correct"])
        for row in rows
    )

    m2_correct = sum(
        bool(row["m2_correct"])
        for row in rows
    )

    m1_accuracy = 100 * m1_correct / total

    m2_accuracy = 100 * m2_correct / total

    improved = sum(
        not row["m1_correct"]
        and row["m2_correct"]
        for row in rows
    )

    regressed = sum(
        row["m1_correct"]
        and not row["m2_correct"]
        for row in rows
    )

    m1_contradictions = sum(
        row["m1_status"] == "contradictory"
        for row in rows
    )

    m2_contradictions = sum(
        row["m2_status"] == "contradictory"
        for row in rows
    )

    return {
        "questions": total,

        "m1_correct": m1_correct,
        "m2_correct": m2_correct,

        "m1_accuracy": round(
            m1_accuracy, 2
        ),

        "m2_accuracy": round(
            m2_accuracy, 2
        ),

        "accuracy_delta": round(
            m2_accuracy - m1_accuracy,
            2
        ),

        "improved": improved,
        "regressed": regressed,

        "m1_contradictions": m1_contradictions,
        "m2_contradictions": m2_contradictions,

        "m1_end_to_end": sum(
            bool(row["m1_end_to_end"])
            for row in rows
        ),

        "m2_end_to_end": sum(
            bool(row["m2_end_to_end"])
            for row in rows
        ),

        "m1_tokens": sum(
            row["m1_tokens"]
            for row in rows
        ),

        "m2_tokens": sum(
            row["m2_tokens"]
            for row in rows
        )
    }

# ============================================================
# 9. COMPARAISON PAR DOMAINE
# ============================================================

summary = {
    "model": MODEL_NAME,
    "m1": str(M1_ADAPTER),
    "m2": str(M2_ADAPTER),
    "dataset": "validation",
    "domains": {},
    "global": calculate_metrics(results)
}

print("\n" + "=" * 60)
print("COMPARAISON PAR DOMAINE")
print("=" * 60)

for domain in DOMAINS:

    subset = [
        row
        for row in results
        if row["domain"] == domain
    ]

    metrics = calculate_metrics(subset)

    summary["domains"][domain] = metrics

    print(f"\n{domain.upper()}")

    print(
        f"M1 : {metrics['m1_accuracy']:.2f}%"
    )

    print(
        f"M2 : {metrics['m2_accuracy']:.2f}%"
    )

    print(
        f"Différence : "
        f"{metrics['accuracy_delta']:+.2f} points"
    )

    print(
        f"Améliorées / dégradées : "
        f"{metrics['improved']} / "
        f"{metrics['regressed']}"
    )

    print(
        f"Contradictions : "
        f"{metrics['m1_contradictions']} -> "
        f"{metrics['m2_contradictions']}"
    )

# ============================================================
# 10. MESURE DE L'OUBLI EN MATHÉMATIQUES
# ============================================================

maths_metrics = summary["domains"]["maths"]

forgetting = max(
    0.0,
    maths_metrics["m1_accuracy"]
    - maths_metrics["m2_accuracy"]
)

summary["maths_forgetting_points"] = round(
    forgetting, 2
)

print("\n" + "=" * 60)
print("CONTINUAL LEARNING")
print("=" * 60)

print(
    f"Oubli en mathématiques : "
    f"{forgetting:.2f} points"
)

# ============================================================
# 11. BILAN GLOBAL
# ============================================================

global_metrics = summary["global"]

print("\n" + "=" * 60)
print("BILAN GLOBAL")
print("=" * 60)

print(
    f"M1 : {global_metrics['m1_accuracy']:.2f}%"
)

print(
    f"M2 : {global_metrics['m2_accuracy']:.2f}%"
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
    f"Contradictions : "
    f"{global_metrics['m1_contradictions']} -> "
    f"{global_metrics['m2_contradictions']}"
)

# ============================================================
# 12. SAUVEGARDE DES RÉSULTATS
# ============================================================

csv_path = (
    RESULTS_DIR / "continual_m1_m2.csv"
)

json_path = (
    RESULTS_DIR / "continual_m1_m2.json"
)

review_path = (
    RESULTS_DIR / "continual_m1_m2_review.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

with json_path.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )

# Réponses contradictoires ou manquantes.
review = [
    row
    for row in results
    if (
        row["m1_status"] in {
            "contradictory",
            "missing",
            "non_integer"
        }
        or row["m2_status"] in {
            "contradictory",
            "missing",
            "non_integer"
        }
    )
]

with review_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(review)

# ============================================================
# 13. FIN
# ============================================================

print("\nÉvaluation terminée.")

print(f"CSV : {csv_path}")
print(f"JSON : {json_path}")
print(f"Audit : {review_path}")

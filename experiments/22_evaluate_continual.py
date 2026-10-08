
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

M1_ADAPTER = (
    ROOT / "models/grpo_maths_v2"
)

M2_ADAPTER = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

M3_ADAPTER = (
    ROOT
    / "models/grpo_maths_v2_then_algebra_sft_then_code_sft"
)

DATA_FILE = (
    ROOT / "data/processed/validation.jsonl"
)

PREVIOUS_RESULTS = (
    ROOT / "results/continual_m1_m2.csv"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DTYPE = torch.float32

MAX_NEW_TOKENS = 256

DOMAINS = [
    "maths",
    "algebre",
    "code"
]

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

print("=" * 60)
print("CONTINUAL RL - COMPLETE SEQUENTIAL EVALUATION")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Modèle : {MODEL_NAME}")

# ============================================================
# 2. VÉRIFICATION DES FICHIERS
# ============================================================

required_files = [
    DATA_FILE,
    PREVIOUS_RESULTS,
    EVALUATOR_FILE,
    M1_ADAPTER / "adapter_config.json",
    M2_ADAPTER / "adapter_config.json",
    M3_ADAPTER / "adapter_config.json"
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

print("Fichiers : OK")

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

audit = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(audit)

audit.run_tests()

print("Évaluateur : OK")

# ============================================================
# 4. CHARGEMENT DU DATASET
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

with PREVIOUS_RESULTS.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:

    previous = list(
        csv.DictReader(file)
    )

previous_by_id = {
    row["id"]: row
    for row in previous
}

dataset_ids = [
    row["id"]
    for row in dataset
]

if len(dataset_ids) != len(set(dataset_ids)):

    raise ValueError(
        "Identifiants dupliqués dans le dataset."
    )

if len(previous) != len(previous_by_id):

    raise ValueError(
        "Identifiants dupliqués dans les résultats."
    )

if set(dataset_ids) != set(previous_by_id):

    raise ValueError(
        "Les datasets ne correspondent pas."
    )

for example in dataset:

    old = previous_by_id[
        example["id"]
    ]

    if (
        old["question"] != example["question"]
        or int(old["expected"])
        != int(example["answer"])
    ):

        raise ValueError(
            f"Incohérence : {example['id']}"
        )

print(
    f"Questions de validation : {len(dataset)}"
)

# ============================================================
# 5. CHARGEMENT DU MODÈLE M3
# ============================================================

print("\nChargement de Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE
).to(DEVICE)

print("Chargement de l'adaptateur M3...")

model = PeftModel.from_pretrained(
    base_model,
    str(M3_ADAPTER)
)

model.to(DEVICE)

model.eval()

print("Modèle M3 chargé.")

# ============================================================
# 6. GÉNÉRATION DES RÉPONSES M3
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

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated = (
        outputs[0][input_length:]
    )

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return (
        response,
        len(generated)
    )

# ============================================================
# 7. ÉVALUATION DES TROIS MODÈLES
# ============================================================

results = []

print("\nDébut de l'évaluation M3...\n")

for index, example in enumerate(
    dataset,
    start=1
):

    example_id = example["id"]

    domain = example["domain"]

    expected = int(
        example["answer"]
    )

    old = previous_by_id[
        example_id
    ]

    # M1 et M2 : réponses déjà sauvegardées.
    responses = {
        "m1": old["m1_response"],
        "m2": old["m2_response"]
    }

    # M3 : nouvelle génération.
    m3_response, m3_tokens = (
        generate_response(
            example["question"]
        )
    )

    responses["m3"] = m3_response

    record = {
        "id": example_id,
        "domain": domain,
        "question": example["question"],
        "expected": expected
    }

    # Même évaluateur pour M1, M2 et M3.
    for label, response in responses.items():

        evaluation = audit.analyze_response(
            response,
            domain,
            expected
        )

        record[f"{label}_prediction"] = (
            evaluation["prediction"]
        )

        record[f"{label}_status"] = (
            evaluation["status"]
        )

        record[f"{label}_correct"] = bool(
            evaluation["strict_correct"]
        )

        record[f"{label}_format_ok"] = bool(
            evaluation["format_ok"]
        )

        record[f"{label}_end_to_end"] = bool(
            evaluation["end_to_end"]
        )

        record[f"{label}_response"] = (
            response
        )

    # Nombre de tokens.
    record["m1_tokens"] = int(
        old["m1_tokens"]
    )

    record["m2_tokens"] = int(
        old["m2_tokens"]
    )

    record["m3_tokens"] = (
        m3_tokens
    )

    results.append(record)

    if (
        index % 10 == 0
        or index == len(dataset)
    ):

        print(
            f"Progression : "
            f"{index}/{len(dataset)}"
        )

# ============================================================
# 8. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows):

    total = len(rows)

    if total == 0:
        return None

    metrics = {
        "questions": total,
        "models": {},
        "transitions": {}
    }

    # Résultats individuels.
    for label in ["m1", "m2", "m3"]:

        correct = sum(
            row[f"{label}_correct"]
            for row in rows
        )

        contradictions = sum(
            row[f"{label}_status"]
            == "contradictory"
            for row in rows
        )

        missing = sum(
            row[f"{label}_status"]
            == "missing"
            for row in rows
        )

        end_to_end = sum(
            row[f"{label}_end_to_end"]
            for row in rows
        )

        tokens = sum(
            row[f"{label}_tokens"]
            for row in rows
        )

        accuracy = (
            100 * correct / total
        )

        metrics["models"][label] = {
            "correct": correct,
            "accuracy": round(
                accuracy, 2
            ),
            "contradictions": contradictions,
            "missing": missing,
            "end_to_end": end_to_end,
            "tokens": tokens
        }

    # Comparaisons entre les étapes.
    transitions = [
        ("m1", "m2"),
        ("m2", "m3"),
        ("m1", "m3")
    ]

    for before, after in transitions:

        improved = sum(
            not row[f"{before}_correct"]
            and row[f"{after}_correct"]
            for row in rows
        )

        regressed = sum(
            row[f"{before}_correct"]
            and not row[f"{after}_correct"]
            for row in rows
        )

        changed = sum(
            row[f"{before}_response"]
            != row[f"{after}_response"]
            for row in rows
        )

        before_accuracy = (
            metrics["models"][before]["accuracy"]
        )

        after_accuracy = (
            metrics["models"][after]["accuracy"]
        )

        metrics["transitions"][
            f"{before}_to_{after}"
        ] = {
            "improved": improved,
            "regressed": regressed,
            "changed_responses": changed,
            "accuracy_delta": round(
                after_accuracy - before_accuracy,
                2
            )
        }

    return metrics

# ============================================================
# 9. COMPARAISON PAR DOMAINE
# ============================================================

summary = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "dataset": "validation",
    "max_new_tokens": MAX_NEW_TOKENS,
    "system_prompt": SYSTEM_PROMPT,
    "adapters": {
        "m1": str(M1_ADAPTER),
        "m2": str(M2_ADAPTER),
        "m3": str(M3_ADAPTER)
    },
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

    metrics = calculate_metrics(
        subset
    )

    summary["domains"][domain] = (
        metrics
    )

    print(
        f"\n{domain.upper()}"
    )

    for label in ["m1", "m2", "m3"]:

        model_metrics = (
            metrics["models"][label]
        )

        print(
            f"{label.upper()} : "
            f"{model_metrics['accuracy']:.2f}% "
            f"({model_metrics['correct']}/{len(subset)})"
        )

    transition = (
        metrics["transitions"]["m2_to_m3"]
    )

    print(
        f"M2 -> M3 : "
        f"{transition['accuracy_delta']:+.2f} points"
    )

    print(
        "Améliorées / dégradées : "
        f"{transition['improved']} / "
        f"{transition['regressed']}"
    )

    contradictions = [
        metrics["models"][label][
            "contradictions"
        ]
        for label in ["m1", "m2", "m3"]
    ]

    print(
        "Contradictions M1/M2/M3 : "
        + " / ".join(
            str(value)
            for value in contradictions
        )
    )

# ============================================================
# 10. MESURE DE L'OUBLI
# ============================================================

print("\n" + "=" * 60)
print("CONTINUAL LEARNING - FORGETTING")
print("=" * 60)

maths = summary["domains"]["maths"][
    "models"
]

algebra = summary["domains"]["algebre"][
    "models"
]

code = summary["domains"]["code"][
    "models"
]

# Meilleure performance mathématique
# mesurée avant l'apprentissage du code.
maths_best_before = max(
    maths["m1"]["accuracy"],
    maths["m2"]["accuracy"]
)

maths_forgetting = max(
    0.0,
    maths_best_before
    - maths["m3"]["accuracy"]
)

# L'algèbre est apprise au stade M2.
algebra_forgetting = max(
    0.0,
    algebra["m2"]["accuracy"]
    - algebra["m3"]["accuracy"]
)

# Gain de la nouvelle compétence.
code_gain = (
    code["m3"]["accuracy"]
    - code["m2"]["accuracy"]
)

summary["continual_learning"] = {
    "maths_forgetting_points": round(
        maths_forgetting, 2
    ),
    "algebra_forgetting_points": round(
        algebra_forgetting, 2
    ),
    "code_gain_points": round(
        code_gain, 2
    )
}

print(
    f"Oubli mathématiques : "
    f"{maths_forgetting:.2f} points"
)

print(
    f"Oubli algèbre : "
    f"{algebra_forgetting:.2f} points"
)

print(
    f"Gain programmation : "
    f"{code_gain:+.2f} points"
)

# ============================================================
# 11. BILAN GLOBAL
# ============================================================

print("\n" + "=" * 60)
print("BILAN GLOBAL M1 / M2 / M3")
print("=" * 60)

global_metrics = (
    summary["global"]
)

for label in ["m1", "m2", "m3"]:

    metrics = (
        global_metrics["models"][label]
    )

    print(
        f"{label.upper()} : "
        f"{metrics['accuracy']:.2f}% "
        f"({metrics['correct']}/{len(results)})"
    )

    print(
        f"  Contradictions : "
        f"{metrics['contradictions']}"
    )

    print(
        f"  Exactitude + format : "
        f"{metrics['end_to_end']}"
    )

    print(
        f"  Tokens générés : "
        f"{metrics['tokens']}"
    )

transition = (
    global_metrics["transitions"]["m2_to_m3"]
)

print("-" * 60)

print(
    f"M2 -> M3 : "
    f"{transition['accuracy_delta']:+.2f} points"
)

print(
    f"Questions améliorées : "
    f"{transition['improved']}"
)

print(
    f"Questions dégradées : "
    f"{transition['regressed']}"
)

# ============================================================
# 12. SAUVEGARDE CSV
# ============================================================

csv_path = (
    RESULTS_DIR / "continual_m1_m2_m3.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        results
    )

# ============================================================
# 13. SAUVEGARDE JSON
# ============================================================

json_path = (
    RESULTS_DIR / "continual_m1_m2_m3.json"
)

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

# ============================================================
# 14. EXPORT DES CAS À AUDITER
# ============================================================

review_path = (
    RESULTS_DIR / "continual_m1_m2_m3_review.csv"
)

review = [
    row
    for row in results
    if (
        row["m3_status"] in {
            "contradictory",
            "missing",
            "non_integer"
        }
        or row["m2_correct"]
        != row["m3_correct"]
    )
]

with review_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        review
    )

# ============================================================
# 15. FIN
# ============================================================

print("\n" + "=" * 60)
print("ÉVALUATION TERMINÉE")
print("=" * 60)

print(f"CSV : {csv_path}")
print(f"JSON : {json_path}")
print(f"Audit : {review_path}")


import csv
import importlib.util
import json
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

REPLAY_ADAPTER = (
    ROOT / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_replay20"
)

DATA_FILE = (
    ROOT / "data/processed/validation.jsonl"
)

PREVIOUS_RESULTS = (
    ROOT / "results/continual_m1_m2_m3.csv"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

RESULTS_DIR = ROOT / "results"

MAX_NEW_TOKENS = 256

DOMAINS = [
    "maths",
    "algebre",
    "code"
]

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

torch.manual_seed(42)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("CONTINUAL LEARNING - EXPERIENCE REPLAY")
print("=" * 60)

print(f"Device : {DEVICE}")

# ============================================================
# 2. VÉRIFICATION DES FICHIERS
# ============================================================

required_files = [
    DATA_FILE,
    PREVIOUS_RESULTS,
    EVALUATOR_FILE,
    REPLAY_ADAPTER / "adapter_config.json"
]

for path in required_files:

    if not path.is_file():

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

spec.loader.exec_module(
    audit
)

audit.run_tests()

print("Évaluateur V16 : OK")

# ============================================================
# 4. DONNÉES ET RÉSULTATS EXISTANTS
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
        "Les données de validation ont changé."
    )

for example in dataset:

    old = previous_by_id[
        example["id"]
    ]

    if (
        old["question"] != example["question"]
        or old["domain"] != example["domain"]
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
# 5. CHARGEMENT DU MODÈLE AVEC REPLAY
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
    dtype=torch.float32,
    attn_implementation="eager"
).to(DEVICE)

print("Chargement de l'adaptateur M3-R...")

model = PeftModel.from_pretrained(
    base_model,
    str(REPLAY_ADAPTER)
)

model.to(DEVICE)

model.eval()

print("Modèle M3-R chargé.")

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

    tokens = len(generated)

    truncated = (
        tokens >= MAX_NEW_TOKENS
        and int(generated[-1])
        not in {
            tokenizer.eos_token_id,
            tokenizer.pad_token_id
        }
    )

    return response, tokens, truncated

# ============================================================
# 7. ÉVALUATION DE M2, M3 ET M3-R
# ============================================================

results = []

print("\nDébut de l'évaluation...\n")

for index, example in enumerate(
    dataset,
    start=1
):

    old = previous_by_id[
        example["id"]
    ]

    domain = example["domain"]

    expected = int(
        example["answer"]
    )

    # Réponses déjà générées lors de l'étape 22.
    responses = {
        "m2": old["m2_response"],
        "m3": old["m3_response"]
    }

    # Seule cette réponse nécessite une inférence.
    replay_response, replay_tokens, truncated = (
        generate_response(
            example["question"]
        )
    )

    responses["m3r"] = (
        replay_response
    )

    record = {
        "id": example["id"],
        "domain": domain,
        "question": example["question"],
        "expected": expected
    }

    # Même évaluateur pour les trois modèles.
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

    record["m2_tokens"] = int(
        old["m2_tokens"]
    )

    record["m3_tokens"] = int(
        old["m3_tokens"]
    )

    record["m3r_tokens"] = (
        replay_tokens
    )

    record["m3r_truncated"] = (
        truncated
    )

    results.append(
        record
    )

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

        raise ValueError(
            "Aucune question à évaluer."
        )

    output = {
        "questions": total,
        "models": {},
        "transitions": {}
    }

    for label in [
        "m2",
        "m3",
        "m3r"
    ]:

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

        output["models"][label] = {
            "correct": correct,
            "accuracy": round(
                100 * correct / total,
                2
            ),
            "contradictions": contradictions,
            "missing": missing,
            "end_to_end": end_to_end,
            "tokens": tokens
        }

    for before, after in [
        ("m2", "m3"),
        ("m2", "m3r"),
        ("m3", "m3r")
    ]:

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

        before_accuracy = (
            output["models"][before]["accuracy"]
        )

        after_accuracy = (
            output["models"][after]["accuracy"]
        )

        output["transitions"][
            f"{before}_to_{after}"
        ] = {
            "improved": improved,
            "regressed": regressed,
            "accuracy_delta": round(
                after_accuracy - before_accuracy,
                2
            )
        }

    return output

# ============================================================
# 9. COMPARAISON PAR DOMAINE
# ============================================================

summary = {
    "model": MODEL_NAME,
    "dataset": "validation",
    "replay_ratio": 0.20,
    "adapter_replay": str(REPLAY_ADAPTER),
    "max_new_tokens": MAX_NEW_TOKENS,
    "system_prompt": SYSTEM_PROMPT,
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

    for label in [
        "m2",
        "m3",
        "m3r"
    ]:

        model_metrics = (
            metrics["models"][label]
        )

        print(
            f"{label.upper()} : "
            f"{model_metrics['accuracy']:.2f}% "
            f"({model_metrics['correct']}/{len(subset)})"
        )

    changes = (
        metrics["transitions"]["m3_to_m3r"]
    )

    print(
        f"M3 -> M3-R : "
        f"{changes['accuracy_delta']:+.2f} points"
    )

    print(
        "Améliorées / dégradées : "
        f"{changes['improved']} / "
        f"{changes['regressed']}"
    )

    contradictions = [
        metrics["models"][label][
            "contradictions"
        ]
        for label in [
            "m2",
            "m3",
            "m3r"
        ]
    ]

    print(
        "Contradictions M2/M3/M3-R : "
        + " / ".join(
            str(value)
            for value in contradictions
        )
    )

# ============================================================
# 10. MESURE DE L'OUBLI ET DE LA RÉTENTION
# ============================================================

maths = summary["domains"]["maths"][
    "models"
]

algebra = summary["domains"]["algebre"][
    "models"
]

code = summary["domains"]["code"][
    "models"
]

continual = {}

for label in [
    "m3",
    "m3r"
]:

    # Perte éventuelle en mathématiques
    # par rapport au modèle M2.
    maths_forgetting = max(
        0.0,
        maths["m2"]["accuracy"]
        - maths[label]["accuracy"]
    )

    # Perte éventuelle en algèbre.
    algebra_forgetting = max(
        0.0,
        algebra["m2"]["accuracy"]
        - algebra[label]["accuracy"]
    )

    # Progression en programmation.
    code_gain = (
        code[label]["accuracy"]
        - code["m2"]["accuracy"]
    )

    continual[label] = {
        "maths_forgetting_points": round(
            maths_forgetting,
            2
        ),
        "algebra_forgetting_points": round(
            algebra_forgetting,
            2
        ),
        "code_gain_points": round(
            code_gain,
            2
        )
    }

summary["continual_learning"] = (
    continual
)

print("\n" + "=" * 60)
print("OUBLI ET RÉTENTION")
print("=" * 60)

for label in [
    "m3",
    "m3r"
]:

    metrics = continual[label]

    print(
        f"\n{label.upper()}"
    )

    print(
        f"Oubli mathématiques : "
        f"{metrics['maths_forgetting_points']:.2f} points"
    )

    print(
        f"Oubli algèbre : "
        f"{metrics['algebra_forgetting_points']:.2f} points"
    )

    print(
        f"Gain programmation : "
        f"{metrics['code_gain_points']:+.2f} points"
    )

# ============================================================
# 11. BILAN GLOBAL
# ============================================================

print("\n" + "=" * 60)
print("BILAN GLOBAL - SANS REPLAY VS AVEC REPLAY")
print("=" * 60)

global_metrics = (
    summary["global"]
)

for label in [
    "m2",
    "m3",
    "m3r"
]:

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

changes = (
    global_metrics["transitions"]["m3_to_m3r"]
)

print("-" * 60)

print(
    f"M3 -> M3-R : "
    f"{changes['accuracy_delta']:+.2f} points"
)

print(
    f"Questions améliorées : "
    f"{changes['improved']}"
)

print(
    f"Questions dégradées : "
    f"{changes['regressed']}"
)

truncated_count = sum(
    row["m3r_truncated"]
    for row in results
)

print(
    f"Réponses M3-R tronquées : "
    f"{truncated_count}"
)

# ============================================================
# 12. SAUVEGARDE CSV ET JSON
# ============================================================

csv_path = (
    RESULTS_DIR / "continual_replay20_comparison.csv"
)

json_path = (
    RESULTS_DIR / "continual_replay20_comparison.json"
)

review_path = (
    RESULTS_DIR / "continual_replay20_review.csv"
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

# ============================================================
# 13. QUESTIONS À EXAMINER
# ============================================================

review = [
    row
    for row in results
    if (
        row["m3_correct"]
        != row["m3r_correct"]
        or row["m3r_status"] in {
            "contradictory",
            "missing",
            "non_integer"
        }
        or row["m3r_truncated"]
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
# 14. FIN
# ============================================================

print("\n" + "=" * 60)
print("ÉVALUATION TERMINÉE")
print("=" * 60)

print(f"CSV : {csv_path}")
print(f"JSON : {json_path}")
print(f"Audit : {review_path}")

print(
    f"Questions à examiner : "
    f"{len(review)}"
)

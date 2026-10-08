
import csv
import json
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUT_FILE = RESULTS_DIR / "grpo_v2_compare_all.csv"

OUTPUT_FILE = RESULTS_DIR / "grpo_v2_audit.json"

DOMAINS = ["maths", "algebre", "code"]

# ============================================================
# 2. CHARGEMENT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("Le fichier CSV est vide.")

print("=" * 60)
print("CONTINUAL RL - GRPO V2 AUDIT")
print("=" * 60)

print(f"Questions : {len(rows)}")

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def to_bool(value):
    return str(value).strip().lower() == "true"


def to_int(value):
    return int(value)


def missing_prediction(value):
    return str(value).strip() in ("", "None")


def calculate_stats(subset):

    total = len(subset)

    base_correct = sum(
        to_bool(row["base_correct"])
        for row in subset
    )

    grpo_correct = sum(
        to_bool(row["grpo_correct"])
        for row in subset
    )

    base_tokens = sum(
        to_int(row["base_tokens"])
        for row in subset
    )

    grpo_tokens = sum(
        to_int(row["grpo_tokens"])
        for row in subset
    )

    base_missing = sum(
        missing_prediction(row["base_prediction"])
        for row in subset
    )

    grpo_missing = sum(
        missing_prediction(row["grpo_prediction"])
        for row in subset
    )

    base_empty = sum(
        not row["base_response"].strip()
        for row in subset
    )

    grpo_empty = sum(
        not row["grpo_response"].strip()
        for row in subset
    )

    return {
        "total": total,
        "base_correct": base_correct,
        "grpo_correct": grpo_correct,
        "base_accuracy": round(
            100 * base_correct / total, 2
        ) if total else 0,
        "grpo_accuracy": round(
            100 * grpo_correct / total, 2
        ) if total else 0,
        "base_tokens": base_tokens,
        "grpo_tokens": grpo_tokens,
        "base_missing": base_missing,
        "grpo_missing": grpo_missing,
        "base_empty": base_empty,
        "grpo_empty": grpo_empty
    }

# ============================================================
# 4. ANALYSE PAR DOMAINE
# ============================================================

audit = {
    "domains": {},
    "global": {}
}

for domain in DOMAINS:

    subset = [
        row
        for row in rows
        if row["domain"] == domain
    ]

    stats = calculate_stats(subset)

    audit["domains"][domain] = stats

    print("\n" + "-" * 60)
    print(domain.upper())
    print("-" * 60)

    print(
        f"Avant : {stats['base_accuracy']}%"
    )

    print(
        f"Après : {stats['grpo_accuracy']}%"
    )

    print(
        f"Tokens avant : {stats['base_tokens']}"
    )

    print(
        f"Tokens après : {stats['grpo_tokens']}"
    )

    print(
        f"Prédictions manquantes avant : "
        f"{stats['base_missing']}"
    )

    print(
        f"Prédictions manquantes après : "
        f"{stats['grpo_missing']}"
    )

    print(
        f"Réponses vides avant : "
        f"{stats['base_empty']}"
    )

    print(
        f"Réponses vides après : "
        f"{stats['grpo_empty']}"
    )

# ============================================================
# 5. ANALYSE GLOBALE
# ============================================================

global_stats = calculate_stats(rows)

audit["global"] = global_stats

print("\n" + "=" * 60)
print("VÉRIFICATION GLOBALE")
print("=" * 60)

print(
    f"Accuracy avant : "
    f"{global_stats['base_accuracy']}%"
)

print(
    f"Accuracy après : "
    f"{global_stats['grpo_accuracy']}%"
)

print(
    f"Tokens avant : "
    f"{global_stats['base_tokens']}"
)

print(
    f"Tokens après : "
    f"{global_stats['grpo_tokens']}"
)

# ============================================================
# 6. VÉRIFICATION DES COMPTEURS
# ============================================================

for model_key in ["base", "grpo"]:

    token_key = f"{model_key}_tokens"

    invalid = [
        row["id"]
        for row in rows
        if to_int(row[token_key]) < 1
    ]

    if invalid:

        print(
            f"\nATTENTION : {model_key} possède "
            f"{len(invalid)} compteurs invalides."
        )

        print(f"IDs : {invalid[:10]}")

    else:

        print(
            f"\nCompteurs {model_key} : OK"
        )

# ============================================================
# 7. INSPECTION DES RÉPONSES EN ALGÈBRE
# ============================================================

print("\n" + "=" * 60)
print("EXEMPLES DE RÉPONSES EN ALGÈBRE")
print("=" * 60)

algebra = [
    row
    for row in rows
    if row["domain"] == "algebre"
]

for row in algebra[:5]:

    print("\n" + "-" * 60)

    print(f"ID : {row['id']}")
    print(f"Question : {row['question']}")
    print(f"Attendu : {row['expected']}")

    print("\nAVANT GRPO :")
    print(row["base_response"])

    print("\nAPRÈS GRPO :")
    print(row["grpo_response"])

# ============================================================
# 8. SAUVEGARDE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        audit,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("AUDIT TERMINÉ")
print("=" * 60)

print(f"Résultats : {OUTPUT_FILE}")

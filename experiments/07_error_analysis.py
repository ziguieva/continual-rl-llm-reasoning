
import csv
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "baseline_validation_audited.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR / "error_analysis.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "error_analysis.json"
)

# ============================================================
# 2. CLASSIFICATION DES EXERCICES
# ============================================================

def classify_exercise(domain, question):

    q = question.strip()

    # Mathématiques
    if domain == "maths":

        expression = q.replace("Calcule :", "").strip()

        if "(" in expression:
            return "operations_combinees"

        if "*" in expression:
            return "multiplication"

        if "/" in expression:
            return "division"

        if "+" in expression:
            return "addition"

        if "-" in expression:
            return "soustraction"

    # Algèbre
    elif domain == "algebre":

        equation = q.split(":", 1)[-1].strip()

        if "(" in equation:
            return "distributivite"

        if equation.startswith("x/"):
            return "equation_fraction"

        if "x +" in equation:
            return "equation_addition"

        if "x -" in equation:
            return "equation_soustraction"

    # Programmation
    elif domain == "code":

        if "**" in q:
            return "puissance"

        if "//" in q:
            return "division_entiere"

        if "sum(" in q:
            return "somme_liste"

        if "len(" in q:
            return "longueur_liste"

        if "%" in q:
            return "modulo"

    return "autre"

# ============================================================
# 3. CHARGEMENT DES RÉSULTATS
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}"
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    results = list(csv.DictReader(file))

print("=" * 60)
print("CONTINUAL RL - ERROR ANALYSIS")
print("=" * 60)

print(f"Questions analysées : {len(results)}")

# ============================================================
# 4. REGROUPEMENT PAR COMPÉTENCE
# ============================================================

statistics = defaultdict(
    lambda: {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "errors": [],
        "manual_reviewed": 0
    }
)

for row in results:

    domain = row["domain"]

    question = row["question"]

    exercise_type = classify_exercise(
        domain,
        question
    )

    key = (domain, exercise_type)

    correct = (
        str(row["audited_correct"]).lower()
        == "true"
    )

    stats = statistics[key]

    stats["total"] += 1

    if row["review_status"].startswith("manual_"):
        stats["manual_reviewed"] += 1

    if correct:

        stats["correct"] += 1

    else:

        stats["incorrect"] += 1

        stats["errors"].append({
            "id": row["id"],
            "question": question,
            "expected": row["expected"],
            "predicted": row["audited_prediction"],
            "review_status": row["review_status"]
        })

# ============================================================
# 5. CALCUL DES MÉTRIQUES
# ============================================================

analysis = []

for (domain, exercise_type), stats in statistics.items():

    total = stats["total"]

    correct = stats["correct"]

    accuracy = (
        100 * correct / total
        if total > 0
        else 0
    )

    analysis.append({
        "domain": domain,
        "exercise_type": exercise_type,
        "total": total,
        "correct": correct,
        "incorrect": stats["incorrect"],
        "accuracy": round(accuracy, 2),
        "manual_reviewed": stats["manual_reviewed"],
        "errors": stats["errors"]
    })

# Classement par domaine puis accuracy
analysis.sort(
    key=lambda item: (
        item["domain"],
        item["accuracy"]
    )
)

# ============================================================
# 6. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\nPERFORMANCES PAR COMPÉTENCE")

print("-" * 60)

current_domain = None

for item in analysis:

    domain = item["domain"]

    if domain != current_domain:

        current_domain = domain

        print(f"\n{domain.upper()}")

    print(
        f"{item['exercise_type']:25s} | "
        f"{item['accuracy']:6.2f}% | "
        f"{item['correct']}/{item['total']}"
    )

# ============================================================
# 7. SAUVEGARDE CSV
# ============================================================

csv_fields = [
    "domain",
    "exercise_type",
    "total",
    "correct",
    "incorrect",
    "accuracy",
    "manual_reviewed"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for item in analysis:

        writer.writerow({
            field: item[field]
            for field in csv_fields
        })

# ============================================================
# 8. SAUVEGARDE JSON
# ============================================================

output = {
    "source": str(INPUT_FILE),
    "questions": len(results),
    "analysis": analysis
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 9. VÉRIFICATION FINALE
# ============================================================

total_analyzed = sum(
    item["total"]
    for item in analysis
)

assert total_analyzed == len(results), (
    "Certaines questions n'ont pas été analysées."
)

print("\n" + "=" * 60)

print("ANALYSE TERMINÉE")

print("=" * 60)

print(f"Questions : {total_analyzed}")
print(f"CSV : {OUTPUT_CSV}")
print(f"JSON : {OUTPUT_JSON}")

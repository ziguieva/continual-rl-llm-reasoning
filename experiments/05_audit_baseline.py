
import csv
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

INPUT_FILE = RESULTS_DIR / "baseline_validation.csv"

OUTPUT_FILE = RESULTS_DIR / "baseline_format_review.csv"

# ============================================================
# 2. CHARGEMENT DES RÉSULTATS
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
print("CONTINUAL RL - BASELINE ERROR AUDIT")
print("=" * 60)

print(f"\nQuestions analysées : {len(results)}")

# ============================================================
# 3. CLASSIFICATION DES RÉSULTATS
# ============================================================

correct = []
incorrect = []
format_errors = []

for result in results:

    is_correct = (
        result["correct"].lower() == "true"
    )

    is_format_error = (
        result["format_error"].lower() == "true"
    )

    if is_format_error:
        format_errors.append(result)

    elif is_correct:
        correct.append(result)

    else:
        incorrect.append(result)

# ============================================================
# 4. BILAN GÉNÉRAL
# ============================================================

print("\nBILAN")

print(f"Réponses correctes : {len(correct)}")

print(f"Réponses incorrectes : {len(incorrect)}")

print(f"Erreurs de format : {len(format_errors)}")

# ============================================================
# 5. ANALYSE DES ERREURS DE FORMAT
# ============================================================

print("\n" + "=" * 60)
print("RÉPONSES À VÉRIFIER")
print("=" * 60)

review = []

for index, result in enumerate(
    format_errors,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Erreur {index}/{len(format_errors)}"
    )

    print(f"ID : {result['id']}")

    print(f"Domaine : {result['domain']}")

    print(f"Question : {result['question']}")

    print(f"Réponse attendue : {result['expected']}")

    print("\nRéponse complète du modèle :")

    print(result["full_response"])

    review.append({
        "id": result["id"],
        "domain": result["domain"],
        "question": result["question"],
        "expected": result["expected"],
        "full_response": result["full_response"],
        "manual_prediction": "",
        "manual_correct": ""
    })

# ============================================================
# 6. EXPORT POUR VÉRIFICATION MANUELLE
# ============================================================

if review:

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=review[0].keys()
        )

        writer.writeheader()

        writer.writerows(review)

# ============================================================
# 7. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 60)

print("AUDIT TERMINÉ")

print("=" * 60)

print(f"Total : {len(results)}")

print(f"Correctes : {len(correct)}")

print(f"Incorrectes : {len(incorrect)}")

print(f"À vérifier : {len(format_errors)}")

if review:
    print(f"\nFichier généré : {OUTPUT_FILE}")

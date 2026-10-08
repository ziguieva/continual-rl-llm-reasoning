
import csv
import json
from collections import Counter
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUT_FILE = RESULTS_DIR / "baseline_validation.csv"

OUTPUT_CSV = (
    RESULTS_DIR / "baseline_validation_audited.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "baseline_validation_audited.json"
)

DOMAINS = ["maths", "algebre", "code"]

# ============================================================
# 2. RÉSULTATS DE L'AUDIT MANUEL
# ============================================================

# Format :
# ID : (réponse identifiée, statut)
#
# Statuts :
# correct    : réponse finale correcte
# incorrect  : réponse ou raisonnement erroné
# incomplete : aucune réponse finale exploitable

REVIEW = {
    "maths_validation_0023": (40, "incorrect"),
    "algebre_validation_0008": (102, "correct"),
    "algebre_validation_0027": (None, "incorrect"),
    "algebre_validation_0010": (120, "correct"),
    "algebre_validation_0001": (154, "correct"),
    "algebre_validation_0017": (18, "correct"),
    "algebre_validation_0015": (120, "correct"),
    "maths_validation_0013": (418, "correct"),
    "algebre_validation_0020": (120, "correct"),
    "algebre_validation_0007": (136, "correct"),
    "code_validation_0024": (69, "incorrect"),
    "maths_validation_0028": (211, "correct"),
    "maths_validation_0009": (82, "correct"),
    "algebre_validation_0016": (4, "correct"),
    "algebre_validation_0029": (None, "incomplete"),
    "algebre_validation_0018": (None, "incomplete"),
    "algebre_validation_0006": (42, "correct"),
    "code_validation_0006": (64, "correct"),
    "maths_validation_0012": (225, "correct"),
    "algebre_validation_0019": (56, "correct"),
    "algebre_validation_0024": (None, "incomplete"),
    "code_validation_0003": (205, "incorrect"),
}

# ============================================================
# 3. CHARGEMENT DU CSV ORIGINAL
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

    reader = csv.DictReader(file)

    original_fields = reader.fieldnames
    rows = list(reader)

if not rows:
    raise ValueError("Le fichier CSV est vide.")

# ============================================================
# 4. VÉRIFICATION DE L'AUDIT
# ============================================================

unrecognized_ids = {
    row["id"]
    for row in rows
    if row["format_error"].strip().lower() == "true"
}

review_ids = set(REVIEW)

if unrecognized_ids != review_ids:

    missing = unrecognized_ids - review_ids
    extra = review_ids - unrecognized_ids

    raise ValueError(
        "L'audit ne correspond pas au CSV.\n"
        f"ID manquants : {sorted(missing)}\n"
        f"ID supplémentaires : {sorted(extra)}"
    )

print("=" * 60)
print("CONTINUAL RL - BASELINE RECONCILIATION")
print("=" * 60)

print(f"Questions : {len(rows)}")
print(f"Réponses auditées : {len(REVIEW)}")

# ============================================================
# 5. APPLICATION DES CORRECTIONS
# ============================================================

audited_results = []

for row in rows:

    record = dict(row)

    expected = int(row["expected"])

    original_correct = (
        row["correct"].strip().lower() == "true"
    )

    if row["id"] in REVIEW:

        prediction, status = REVIEW[row["id"]]

        if status not in {
            "correct",
            "incorrect",
            "incomplete"
        }:
            raise ValueError(
                f"Statut inconnu : {status}"
            )

        if status == "correct":
            if prediction != expected:
                raise ValueError(
                    f"Correction incohérente : {row['id']}"
                )

        if status == "incorrect":
            if prediction == expected:
                raise ValueError(
                    f"Réponse pourtant correcte : {row['id']}"
                )

        audited_correct = status == "correct"
        review_status = f"manual_{status}"

    else:

        prediction = row["predicted"]

        if prediction in ("", "None"):
            prediction = None

        audited_correct = original_correct

        review_status = (
            "auto_correct"
            if original_correct
            else "auto_incorrect"
        )

    record["audited_prediction"] = (
        prediction
        if prediction is not None
        else ""
    )

    record["audited_correct"] = audited_correct

    record["review_status"] = review_status

    audited_results.append(record)

# ============================================================
# 6. SAUVEGARDE DU CSV CORRIGÉ
# ============================================================

fields = original_fields + [
    "audited_prediction",
    "audited_correct",
    "review_status"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(audited_results)

# ============================================================
# 7. CALCUL DES SCORES
# ============================================================

summary = {
    "source": str(INPUT_FILE),
    "total": len(audited_results),
    "manual_review_count": len(REVIEW),
    "domains": {},
    "global": {}
}

print("\nRÉSULTATS APRÈS AUDIT")
print("-" * 60)

for domain in DOMAINS:

    subset = [
        row
        for row in audited_results
        if row["domain"] == domain
    ]

    total = len(subset)

    correct = sum(
        row["audited_correct"]
        for row in subset
    )

    accuracy = (
        100 * correct / total
        if total
        else 0.0
    )

    summary["domains"][domain] = {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 2)
    }

    print(
        f"{domain.upper():10s} : "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

# ============================================================
# 8. SCORE GLOBAL
# ============================================================

total = len(audited_results)

correct = sum(
    row["audited_correct"]
    for row in audited_results
)

accuracy = 100 * correct / total

statuses = Counter(
    row["review_status"]
    for row in audited_results
)

summary["global"] = {
    "correct": correct,
    "total": total,
    "accuracy": round(accuracy, 2)
}

summary["review_status_counts"] = dict(statuses)

print("-" * 60)

print(f"SCORE GLOBAL : {accuracy:.2f}%")

print("\nDÉTAIL DE L'AUDIT")

for status, count in sorted(statuses.items()):
    print(f"{status}: {count}")

# ============================================================
# 9. SAUVEGARDE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nConsolidation terminée.")
print(f"CSV : {OUTPUT_CSV}")
print(f"JSON : {OUTPUT_JSON}")

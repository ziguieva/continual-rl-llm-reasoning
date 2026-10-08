
import csv
import json
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "grpo_v2_neutral_validation.csv"
)

REPORT_FILE = (
    RESULTS_DIR / "neutral_audit_report.txt"
)

SUMMARY_FILE = (
    RESULTS_DIR / "neutral_audit_summary.json"
)

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
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("Aucun résultat à analyser.")

print("=" * 60)
print("CONTINUAL RL - NEUTRAL EVALUATION AUDIT")
print("=" * 60)

print(f"Questions : {len(rows)}")

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def to_bool(value):
    return str(value).strip().lower() == "true"


def is_missing(value):
    return str(value).strip() in {
        "",
        "None",
        "null"
    }


def get_status(row):

    before = to_bool(row["base_correct"])
    after = to_bool(row["grpo_correct"])

    if not before and after:
        return "improved"

    if before and not after:
        return "regressed"

    if before and after:
        return "correct_both"

    return "incorrect_both"


def format_example(row):

    return (
        "\n" + "-" * 60 + "\n"
        f"ID : {row['id']}\n"
        f"Domaine : {row['domain']}\n"
        f"Question : {row['question']}\n"
        f"Attendu : {row['expected']}\n"
        f"Statut : {get_status(row)}\n"
        "\nAVANT GRPO\n"
        f"Prédiction : {row['base_prediction']}\n"
        f"Tokens : {row['base_tokens']}\n"
        f"Réponse :\n{row['base_response']}\n"
        "\nAPRÈS GRPO\n"
        f"Prédiction : {row['grpo_prediction']}\n"
        f"Tokens : {row['grpo_tokens']}\n"
        f"Réponse :\n{row['grpo_response']}\n"
    )

# ============================================================
# 4. REGROUPEMENT
# ============================================================

maths = [
    row for row in rows
    if row["domain"] == "maths"
]

algebra = [
    row for row in rows
    if row["domain"] == "algebre"
]

code = [
    row for row in rows
    if row["domain"] == "code"
]

maths_improved = [
    row for row in maths
    if get_status(row) == "improved"
]

maths_regressed = [
    row for row in maths
    if get_status(row) == "regressed"
]

code_improved = [
    row for row in code
    if get_status(row) == "improved"
]

code_regressed = [
    row for row in code
    if get_status(row) == "regressed"
]

algebra_missing = [
    row for row in algebra
    if (
        is_missing(row["base_prediction"])
        or is_missing(row["grpo_prediction"])
    )
]

# ============================================================
# 5. AFFICHAGE DES MÉTRIQUES
# ============================================================

print("\n" + "=" * 60)
print("1. BILAN PAR DOMAINE")
print("=" * 60)

summary = {}

for domain in ["maths", "algebre", "code"]:

    subset = [
        row for row in rows
        if row["domain"] == domain
    ]

    total = len(subset)

    before = sum(
        to_bool(row["base_correct"])
        for row in subset
    )

    after = sum(
        to_bool(row["grpo_correct"])
        for row in subset
    )

    missing_before = sum(
        is_missing(row["base_prediction"])
        for row in subset
    )

    missing_after = sum(
        is_missing(row["grpo_prediction"])
        for row in subset
    )

    summary[domain] = {
        "total": total,
        "correct_before": before,
        "correct_after": after,
        "missing_before": missing_before,
        "missing_after": missing_after
    }

    print(f"\n{domain.upper()}")

    print(f"Correct avant : {before}/{total}")
    print(f"Correct après : {after}/{total}")

    print(
        f"Prédictions manquantes : "
        f"{missing_before} -> {missing_after}"
    )

# ============================================================
# 6. AUDIT DES MATHÉMATIQUES
# ============================================================

print("\n" + "=" * 60)
print("2. MATHÉMATIQUES")
print("=" * 60)

print(
    f"Questions améliorées : "
    f"{len(maths_improved)}"
)

print(
    f"Questions dégradées : "
    f"{len(maths_regressed)}"
)

for row in maths_improved:
    print(format_example(row))

# ============================================================
# 7. AUDIT DE L'ALGÈBRE
# ============================================================

print("\n" + "=" * 60)
print("3. ALGÈBRE")
print("=" * 60)

print(
    f"Prédictions non reconnues : "
    f"{len(algebra_missing)}"
)

print("\nCinq premiers exemples :")

for row in algebra[:5]:
    print(format_example(row))

# ============================================================
# 8. AUDIT DU CODE
# ============================================================

print("\n" + "=" * 60)
print("4. PROGRAMMATION")
print("=" * 60)

print(
    f"Questions améliorées : "
    f"{len(code_improved)}"
)

print(
    f"Questions dégradées : "
    f"{len(code_regressed)}"
)

for row in code_improved + code_regressed:
    print(format_example(row))

# ============================================================
# 9. RAPPORT COMPLET
# ============================================================

report = []

report.append(
    "CONTINUAL RL - AUDIT COMPLET\n"
)

for domain, subset in [
    ("MATHS", maths),
    ("ALGEBRE", algebra),
    ("CODE", code)
]:

    report.append(
        f"\n{'=' * 60}\n"
        f"{domain}\n"
        f"{'=' * 60}\n"
    )

    for row in subset:
        report.append(format_example(row))

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# 10. SAUVEGARDE DU RÉSUMÉ
# ============================================================

summary["audit"] = {
    "maths_improved": len(maths_improved),
    "maths_regressed": len(maths_regressed),
    "code_improved": len(code_improved),
    "code_regressed": len(code_regressed),
    "algebra_missing": len(algebra_missing)
}

with open(
    SUMMARY_FILE,
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
# 11. FIN
# ============================================================

print("\n" + "=" * 60)
print("AUDIT TERMINÉ")
print("=" * 60)

print(f"Rapport : {REPORT_FILE}")
print(f"Résumé : {SUMMARY_FILE}")

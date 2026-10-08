
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "grpo_v2_neutral_validation.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR / "neutral_reconciled.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "neutral_reconciled.json"
)

REVIEW_CSV = (
    RESULTS_DIR / "neutral_manual_review.csv"
)

DOMAINS = ["maths", "algebre", "code"]

# ============================================================
# 2. RECONNAISSANCE DES RÉPONSES
# ============================================================

NUMBER = r"(-?\d+(?:\.\d+)?)"

LABEL = (
    r"(?:R[ÉE]PONSE|RESPONSE|"
    r"R[ÉE]SULTAT|RESULTAT|"
    r"REPMESSE(?:MENT)?)"
)

LABEL_PATTERN = re.compile(
    rf"^{LABEL}\s*:\s*"
    rf"(?:x\s*=\s*)?{NUMBER}\s*[.!]?$",
    re.IGNORECASE
)

NUMERIC_PATTERN = re.compile(
    rf"^{NUMBER}\s*[.!]?$"
)

ALGEBRA_PATTERN = re.compile(
    rf"^x\s*=\s*{NUMBER}\s*[.!]?$",
    re.IGNORECASE
)

FINAL_PHRASE_PATTERN = re.compile(
    rf"(?:est|is|vaut|affiche|donne)"
    rf"\s+(?:x\s*=\s*)?"
    rf"[`*]*{NUMBER}[`*]*[.!]?$",
    re.IGNORECASE
)

MATH_EXPRESSION_PATTERN = re.compile(
    rf"^[\d\s()+*/%.-]+"
    rf"=\s*{NUMBER}\s*[.!]?$"
)

# ============================================================
# 3. CONVERSION NUMÉRIQUE
# ============================================================

def parse_integer(value):

    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError):
        return None

    if number != number.to_integral_value():
        return None

    return int(number)

# ============================================================
# 4. EXTRACTION D'UNE LIGNE FINALE
# ============================================================

def extract_final_line(line, domain):

    line = line.strip().strip(" *`")

    patterns = [
        LABEL_PATTERN,
        NUMERIC_PATTERN,
        ALGEBRA_PATTERN,
        FINAL_PHRASE_PATTERN
    ]

    if domain == "maths":
        patterns.append(
            MATH_EXPRESSION_PATTERN
        )

    for pattern in patterns:

        match = pattern.search(line)

        if match:
            return parse_integer(
                match.group(1)
            )

    return None

# ============================================================
# 5. ANALYSE D'UNE RÉPONSE COMPLÈTE
# ============================================================

def analyze_response(response, domain, expected):

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if not lines:

        return {
            "prediction": None,
            "status": "missing",
            "labels": [],
            "final_answer": None
        }

    # Recherche des réponses explicitement annoncées.
    labels = []

    invalid_labels = False

    for line in lines:

        clean = line.strip().strip(" *`")

        match = LABEL_PATTERN.fullmatch(clean)

        if match:

            number = parse_integer(
                match.group(1)
            )

            if number is None:
                invalid_labels = True

            labels.append(number)

    # Extraction de la dernière ligne.
    final_answer = extract_final_line(
        lines[-1],
        domain
    )

    valid_labels = [
        value
        for value in labels
        if value is not None
    ]

    # La conclusion est prioritaire pour proposer
    # une prédiction, mais une contradiction
    # impose une vérification manuelle.
    if final_answer is not None:

        prediction = final_answer

    elif valid_labels:

        prediction = valid_labels[-1]

    else:

        prediction = None

    # Détection des contradictions.
    conflict = invalid_labels

    if len(set(valid_labels)) > 1:
        conflict = True

    if final_answer is not None:

        for value in valid_labels:

            if value != final_answer:
                conflict = True

    # Classification.
    if conflict:

        status = "ambiguous"

    elif prediction is None:

        status = "missing"

    elif prediction == expected:

        status = "correct"

    else:

        status = "incorrect"

    return {
        "prediction": prediction,
        "status": status,
        "labels": labels,
        "final_answer": final_answer
    }

# ============================================================
# 6. TESTS UNITAIRES
# ============================================================

def run_tests():

    test = analyze_response(
        "REPMESSE: 39",
        "maths",
        39
    )

    assert test["status"] == "correct"

    test = analyze_response(
        "REPONSE: nombre.",
        "maths",
        39
    )

    assert test["status"] == "missing"

    test = analyze_response(
        "REPONSE: 103\n"
        "Donc, la solution est x = 29.",
        "algebre",
        29
    )

    assert test["status"] == "ambiguous"

    test = analyze_response(
        "REPMESSEMENT: 3\n"
        "Le résultat est 4.",
        "code",
        4
    )

    assert test["status"] == "ambiguous"

    test = analyze_response(
        "22 * 19 = 418",
        "maths",
        418
    )

    assert test["status"] == "correct"

    test = analyze_response(
        "REPONSE: 204.5\n"
        "Donc, la solution est x = 23.",
        "algebre",
        23
    )

    assert test["status"] == "ambiguous"

    print("Tests unitaires : OK")

# ============================================================
# 7. CHARGEMENT DES DONNÉES
# ============================================================

def load_results():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))

# ============================================================
# 8. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows, prefix):

    statuses = [
        row[f"{prefix}_audit_status"]
        for row in rows
    ]

    total = len(statuses)

    correct = statuses.count("correct")

    incorrect = statuses.count("incorrect")

    ambiguous = statuses.count("ambiguous")

    missing = statuses.count("missing")

    resolved = correct + incorrect

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "ambiguous": ambiguous,
        "missing": missing,
        "resolved": resolved,
        "accuracy_on_resolved": round(
            100 * correct / resolved,
            2
        ) if resolved else None,
        "coverage": round(
            100 * resolved / total,
            2
        ) if total else 0
    }

# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("CONTINUAL RL - NEUTRAL RECONCILIATION")
    print("=" * 60)

    run_tests()

    rows = load_results()

    print(f"Questions : {len(rows)}")

    reconciled = []
    manual_review = []

    for row in rows:

        expected = int(row["expected"])

        before = analyze_response(
            row["base_response"],
            row["domain"],
            expected
        )

        after = analyze_response(
            row["grpo_response"],
            row["domain"],
            expected
        )

        record = dict(row)

        # Résultats avant GRPO.
        record["base_audit_prediction"] = (
            before["prediction"]
        )

        record["base_audit_status"] = (
            before["status"]
        )

        record["base_final_answer"] = (
            before["final_answer"]
        )

        record["base_labels"] = json.dumps(
            before["labels"]
        )

        # Résultats après GRPO.
        record["grpo_audit_prediction"] = (
            after["prediction"]
        )

        record["grpo_audit_status"] = (
            after["status"]
        )

        record["grpo_final_answer"] = (
            after["final_answer"]
        )

        record["grpo_labels"] = json.dumps(
            after["labels"]
        )

        reconciled.append(record)

        # Les réponses ambiguës ou introuvables
        # doivent être examinées manuellement.
        if (
            before["status"]
            in {"ambiguous", "missing"}
            or after["status"]
            in {"ambiguous", "missing"}
        ):

            manual_review.append(record)

    # ========================================================
    # 10. SAUVEGARDE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=reconciled[0].keys()
        )

        writer.writeheader()
        writer.writerows(reconciled)

    if manual_review:

        with open(
            REVIEW_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=manual_review[0].keys()
            )

            writer.writeheader()
            writer.writerows(manual_review)

    # ========================================================
    # 11. RÉSULTATS PAR DOMAINE
    # ========================================================

    summary = {
        "source": str(INPUT_FILE),
        "domains": {},
        "global": {},
        "manual_review_count": len(manual_review)
    }

    print("\n" + "=" * 60)
    print("RÉSULTATS APRÈS RÉCONCILIATION")
    print("=" * 60)

    for domain in DOMAINS:

        subset = [
            row
            for row in reconciled
            if row["domain"] == domain
        ]

        base = calculate_metrics(
            subset,
            "base"
        )

        grpo = calculate_metrics(
            subset,
            "grpo"
        )

        summary["domains"][domain] = {
            "before": base,
            "after": grpo
        }

        print(f"\n{domain.upper()}")

        print(
            f"Avant : {base['correct']} correctes, "
            f"{base['ambiguous']} ambiguës, "
            f"{base['missing']} manquantes"
        )

        print(
            f"Après : {grpo['correct']} correctes, "
            f"{grpo['ambiguous']} ambiguës, "
            f"{grpo['missing']} manquantes"
        )

        print(
            f"Couverture : {base['coverage']}%"
            f" -> {grpo['coverage']}%"
        )

    # ========================================================
    # 12. COMPARAISON SUR LES PAIRES VÉRIFIABLES
    # ========================================================

    resolved_statuses = {
        "correct",
        "incorrect"
    }

    paired = [
        row
        for row in reconciled
        if (
            row["base_audit_status"]
            in resolved_statuses
            and row["grpo_audit_status"]
            in resolved_statuses
        )
    ]

    improved = sum(
        row["base_audit_status"] == "incorrect"
        and row["grpo_audit_status"] == "correct"
        for row in paired
    )

    regressed = sum(
        row["base_audit_status"] == "correct"
        and row["grpo_audit_status"] == "incorrect"
        for row in paired
    )

    summary["global"] = {
        "before": calculate_metrics(
            reconciled,
            "base"
        ),
        "after": calculate_metrics(
            reconciled,
            "grpo"
        ),
        "paired_resolved": len(paired),
        "paired_improved": improved,
        "paired_regressed": regressed
    }

    # ========================================================
    # 13. AFFICHAGE FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("BILAN GLOBAL")
    print("=" * 60)

    print(
        f"Paires évaluables : "
        f"{len(paired)}/{len(reconciled)}"
    )

    print(
        f"Questions améliorées : {improved}"
    )

    print(
        f"Questions dégradées : {regressed}"
    )

    print(
        f"Questions à vérifier : "
        f"{len(manual_review)}"
    )

    # ========================================================
    # 14. SAUVEGARDE JSON
    # ========================================================

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

    print("\nAnalyse terminée.")

    print(f"CSV : {OUTPUT_CSV}")
    print(f"JSON : {OUTPUT_JSON}")

    if manual_review:
        print(
            f"Vérification manuelle : {REVIEW_CSV}"
        )


if __name__ == "__main__":
    main()

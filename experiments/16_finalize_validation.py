"""Audit reproductible de Qwen avant/après GRPO sur la validation.

Ne réexécute pas les modèles. Ne sélectionne jamais une réponse
sur la base de la valeur attendue.
"""

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "grpo_v2_neutral_validation.csv"
DEFAULT_OUTPUT = ROOT / "results"
DOMAINS = ("maths", "algebre", "code")

NUMBER = r"-?\d+(?:\.\d+)?"
LABEL = r"(?:R[ÉE]PONSE|RESPONSE|R[ÉE]SULTAT|RESULTAT|REPMESSE(?:MENT)?)"
LABEL_RE = re.compile(
    rf"^{LABEL}\s*:\s*(?:x\s*=\s*)?({NUMBER})\s*[.!]?\s*$",
    re.IGNORECASE,
)
CANONICAL_RE = re.compile(rf"^REPONSE:\s*({NUMBER})$", re.IGNORECASE)
NUMBER_RE = re.compile(rf"^({NUMBER})\s*[.!]?\s*$")
ALGEBRA_RE = re.compile(rf"^x\s*=\s*({NUMBER})\s*[.!]?\s*$", re.IGNORECASE)
FINAL_PHRASE_RE = re.compile(
    rf"(?:\best\b|\bis\b|\bvaut\b|\baffiche\b|\bdonne\b|\bqui\s+est\b)"
    rf"\s+(?:x\s*=\s*)?[`*]*({NUMBER})[`*]*[.!]?\s*$",
    re.IGNORECASE,
)
MATH_RESULT_RE = re.compile(
    rf"^[\d\s()+*/%.,-]+=\s*({NUMBER})\s*[.!]?\s*$"
)


def integer_or_none(raw):
    if raw is None:
        return None
    try:
        number = Decimal(str(raw))
    except (InvalidOperation, TypeError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)


def clean_line(text):
    return text.strip().strip(" *`").strip()


def extract_conclusion(last_line, domain):
    patterns = [LABEL_RE, NUMBER_RE, ALGEBRA_RE, FINAL_PHRASE_RE]
    if domain == "maths":
        patterns.append(MATH_RESULT_RE)
    for pattern in patterns:
        match = pattern.fullmatch(last_line) if pattern != FINAL_PHRASE_RE else pattern.search(last_line)
        if match:
            return integer_or_none(match.group(1))
    return None


def analyze_response(response, domain, expected):
    lines = [clean_line(line) for line in response.splitlines() if clean_line(line)]
    if not lines:
        return {
            "status": "missing", "prediction": None, "declared": [],
            "conclusion": None, "conclusion_correct": False,
            "strict_correct": False, "format_ok": False, "end_to_end": False,
        }

    # On relève les réponses déclarées sans consulter la valeur attendue.
    declared = []
    non_integer_declared = False
    for line in lines:
        match = LABEL_RE.fullmatch(line)
        if match:
            value = integer_or_none(match.group(1))
            if value is None:
                non_integer_declared = True
            else:
                declared.append(value)

    conclusion = extract_conclusion(lines[-1], domain)
    canonical_match = CANONICAL_RE.fullmatch(lines[-1])
    format_ok = bool(
        canonical_match and integer_or_none(canonical_match.group(1)) is not None
    )

    conflict = len(set(declared)) > 1 or (
        conclusion is not None
        and any(value != conclusion for value in declared)
    )

    if conflict:
        status = "contradictory"
        prediction = None
    elif non_integer_declared:
        status = "non_integer"
        prediction = None
    else:
        prediction = conclusion if conclusion is not None else (
            declared[-1] if declared else None
        )
        if prediction is None:
            status = "missing"
        elif prediction == expected:
            status = "correct"
        else:
            status = "incorrect"

    strict_correct = status == "correct"
    return {
        "status": status,
        "prediction": prediction,
        "declared": declared,
        "conclusion": conclusion,
        "conclusion_correct": conclusion is not None and conclusion == expected,
        "strict_correct": strict_correct,
        "format_ok": format_ok,
        "end_to_end": strict_correct and format_ok,
    }


def run_tests():
    cases = [
        ("REPONSE: 114", "maths", 114, "correct", True),
        ("REPMESSE: nombre.", "maths", 114, "missing", False),
        ("REPMESSE: 39", "maths", 39, "correct", False),
        ("22 * 19 = 418", "maths", 418, "correct", False),
        ("REPONSE: 103\nDonc, la solution est x = 29.", "algebre", 29,
         "contradictory", False),
        ("REPONSE: 10\nDonc, la solution est x = 10.", "algebre", 10,
         "correct", False),
        ("REPMESSEMENT: 3\nLe résultat est 4.", "code", 4,
         "contradictory", False),
        ("REPONSE: 0.8909090909090909", "algebre", 154,
         "non_integer", False),
        ("REPONSE: 806", "maths", 798, "incorrect", True),
    ]
    for response, domain, expected, status, format_ok in cases:
        result = analyze_response(response, domain, expected)
        assert result["status"] == status, (response, result)
        assert result["format_ok"] == format_ok, (response, result)
    print(f"Tests de l'évaluateur : {len(cases)}/{len(cases)} OK")


def load_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "domain", "expected", "base_response", "grpo_response"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Colonnes manquantes : {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Le fichier d'entrée est vide.")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants dupliqués dans le fichier d'entrée.")
    return rows


def summarize(rows, name):
    total = len(rows)
    base_correct = sum(row["strict_base_correct"] for row in rows)
    after_correct = sum(row["strict_grpo_correct"] for row in rows)
    improved = sum(
        not row["strict_base_correct"] and row["strict_grpo_correct"]
        for row in rows
    )
    regressed = sum(
        row["strict_base_correct"] and not row["strict_grpo_correct"]
        for row in rows
    )
    result = {
        "questions": total,
        "base_correct": base_correct,
        "grpo_correct": after_correct,
        "base_accuracy": round(100 * base_correct / total, 2) if total else 0,
        "grpo_accuracy": round(100 * after_correct / total, 2) if total else 0,
        "delta_points": round(100 * (after_correct - base_correct) / total, 2)
        if total else 0,
        "improved": improved,
        "regressed": regressed,
        "base_contradictory": sum(row["base_status"] == "contradictory" for row in rows),
        "grpo_contradictory": sum(row["grpo_status"] == "contradictory" for row in rows),
        "base_missing": sum(row["base_status"] == "missing" for row in rows),
        "grpo_missing": sum(row["grpo_status"] == "missing" for row in rows),
        "base_format_compliant": sum(row["base_format_ok"] for row in rows),
        "grpo_format_compliant": sum(row["grpo_format_ok"] for row in rows),
        "base_end_to_end": sum(row["base_end_to_end"] for row in rows),
        "grpo_end_to_end": sum(row["grpo_end_to_end"] for row in rows),
        "base_conclusion_correct": sum(row["base_conclusion_correct"] for row in rows),
        "grpo_conclusion_correct": sum(row["grpo_conclusion_correct"] for row in rows),
    }
    print(f"\n{name} : {total} questions")
    print(f"  Réponses non contradictoires correctes : {base_correct} -> {after_correct}")
    print(f"  Accuracy : {result['base_accuracy']:.2f}% -> {result['grpo_accuracy']:.2f}% "
          f"({result['delta_points']:+.2f} points)")
    print(f"  Améliorées / dégradées : {improved} / {regressed}")
    print(f"  Contradictions : {result['base_contradictory']} -> {result['grpo_contradictory']}")
    print(f"  Réponses manquantes : {result['base_missing']} -> {result['grpo_missing']}")
    print(f"  Exactitude + format : {result['base_end_to_end']} -> {result['grpo_end_to_end']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_tests()
    rows = load_rows(args.input)
    audited = []
    review = []

    for row in rows:
        expected = integer_or_none(row["expected"])
        if expected is None or row["domain"] not in DOMAINS:
            raise ValueError(f"Exemple invalide : {row['id']}")
        before = analyze_response(row["base_response"], row["domain"], expected)
        after = analyze_response(row["grpo_response"], row["domain"], expected)
        record = dict(row)
        for prefix, assessment in (("base", before), ("grpo", after)):
            record[f"{prefix}_status"] = assessment["status"]
            record[f"{prefix}_audit_prediction"] = assessment["prediction"]
            record[f"{prefix}_declared"] = json.dumps(assessment["declared"])
            record[f"{prefix}_conclusion"] = assessment["conclusion"]
            record[f"{prefix}_conclusion_correct"] = assessment["conclusion_correct"]
            record[f"strict_{prefix}_correct"] = assessment["strict_correct"]
            record[f"{prefix}_format_ok"] = assessment["format_ok"]
            record[f"{prefix}_end_to_end"] = assessment["end_to_end"]
        audited.append(record)
        if before["status"] in {"missing", "non_integer", "contradictory"} or (
            after["status"] in {"missing", "non_integer", "contradictory"}
        ):
            review.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "validation_final_v1.csv"
    review_path = args.output_dir / "validation_final_manual_review.csv"
    json_path = args.output_dir / "validation_final_v1.json"
    fields = list(audited[0])
    for path, subset in ((csv_path, audited), (review_path, review)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)

    print("\nCONTINUAL RL - AUDIT FINAL DE VALIDATION")
    summary = {
        "input": str(args.input),
        "policy": "Une réponse contradictoire est incorrecte; aucun cas n'est exclu du dénominateur.",
        "domains": {
            domain: summarize([row for row in audited if row["domain"] == domain], domain.upper())
            for domain in DOMAINS
        },
        "global": summarize(audited, "GLOBAL"),
        "manual_review_count": len(review),
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"\nCSV complet : {csv_path}")
    print(f"Résumé : {json_path}")
    print(f"Cas à examiner : {review_path}")


if __name__ == "__main__":
    main()

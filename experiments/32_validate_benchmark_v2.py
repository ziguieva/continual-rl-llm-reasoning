
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK = ROOT / "data/benchmark_v2_v1"

RESULTS = ROOT / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = RESULTS / "benchmark_v2_validator_audit.json"

LATEX_CONFIG = LatexExtractionConfig(
    boxed_match_priority=0,
)

EXPR_CONFIG = ExprExtractionConfig()

MATH_DOMAINS = {
    "maths_appliques",
    "maths_competition",
    "algebre_avancee",
}

# ============================================================
# 2. CHARGEMENT DES FICHIERS
# ============================================================

def load_jsonl(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_partition(partition):

    questions = load_jsonl(
        BENCHMARK / f"{partition}.jsonl"
    )

    answers = load_jsonl(
        BENCHMARK
        / "grading"
        / f"{partition}_keys.jsonl"
    )

    question_ids = {
        row["id"]
        for row in questions
    }

    answer_ids = {
        row["id"]
        for row in answers
    }

    if len(question_ids) != len(questions):
        raise ValueError(
            f"IDs de questions dupliqués : "
            f"{partition}"
        )

    if len(answer_ids) != len(answers):
        raise ValueError(
            f"IDs de corrections dupliqués : "
            f"{partition}"
        )

    if question_ids != answer_ids:
        raise ValueError(
            f"Questions et corrections "
            f"incohérentes : {partition}"
        )

    answers_by_id = {
        row["id"]: row
        for row in answers
    }

    return [
        (
            question,
            answers_by_id[question["id"]],
        )
        for question in questions
    ]

# ============================================================
# 3. EXTRACTION DE LA RÉPONSE FINALE
# ============================================================

FINAL_LABEL = re.compile(
    r"^(?:REPONSE|RÉPONSE|ANSWER|"
    r"FINAL ANSWER|####)"
    r"\s*[:：]?\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


def last_boxed(text):

    marker = r"\boxed{"

    start = text.rfind(
        marker
    )

    if start < 0:
        return None

    position = (
        start + len(marker)
    )

    depth = 1

    while position < len(text):

        char = text[position]

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[
                    start:position + 1
                ]

        position += 1

    return None


def extract_final(text):

    if not isinstance(text, str):
        return None, False

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None, False

    # Le format strict exige une ligne
    # REPONSE: ... à la toute fin.

    last_line = lines[-1]

    strict_format = bool(
        re.fullmatch(
            r"REPONSE:\s*\S.*",
            last_line,
            flags=re.IGNORECASE,
        )
    )

    # On privilégie la dernière réponse
    # explicitement annoncée.

    for line in reversed(lines):

        match = FINAL_LABEL.match(
            line
        )

        if match:

            value = (
                match.group(1)
                .strip()
            )

            if value:
                return (
                    value,
                    strict_format,
                )

    # Sinon, on accepte une réponse
    # mathématique explicitement encadrée.

    boxed = last_boxed(
        text
    )

    if boxed is not None:
        return boxed, strict_format

    # Une réponse composée uniquement
    # d'une expression sur la dernière
    # ligne est également recevable,
    # mais son format n'est pas strict.

    if len(last_line) <= 128:

        if re.fullmatch(
            r"[\d\s.,+\-*/()]+",
            last_line,
        ):

            return last_line, False

        if re.fullmatch(
            r"x\s*=\s*.+",
            last_line,
            flags=re.IGNORECASE,
        ):

            return (
                last_line.split(
                    "=",
                    1,
                )[1].strip(),
                False,
            )

    return None, False

# ============================================================
# 4. PARSING MATHÉMATIQUE
# ============================================================

def normalize_numeric_text(text):

    text = str(text).strip()

    text = text.replace(
        "−",
        "-",
    )

    # Séparateurs de milliers :
    # 12,000 -> 12000.

    text = re.sub(
        r"(?<=\d),(?=\d{3}(?:\D|$))",
        "",
        text,
    )

    return text


def parse_math_answer(text):

    if text is None:
        return []

    value = normalize_numeric_text(
        text
    )

    if not value:
        return []

    # Pour les réponses numériques
    # simples, utiliser Expr.

    if re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?"
        r"(?:\s*/\s*[-+]?\d+)?",
        value,
    ):

        parsed = parse(
            value,
            extraction_config=[
                EXPR_CONFIG,
            ],
            fallback_mode="no_fallback",
        )

        if parsed:
            return parsed

    # Les réponses MATH peuvent être
    # des fractions, ensembles ou
    # expressions LaTeX.

    if r"\boxed{" in value:

        latex = value

    else:

        latex = (
            r"\boxed{"
            + value
            + "}"
        )

    parsed = parse(
        latex,
        extraction_config=[
            LATEX_CONFIG,
        ],
        fallback_mode="no_fallback",
    )

    if parsed:
        return parsed

    return parse(
        value,
        extraction_config=[
            EXPR_CONFIG,
            LATEX_CONFIG,
        ],
        fallback_mode="no_fallback",
    )

# ============================================================
# 5. VÉRIFICATEUR MATHÉMATIQUE
# ============================================================

def grade_math(
    expected,
    response,
):

    gold = parse_math_answer(
        expected
    )

    if not gold:

        return {
            "status": "unsupported_gold",
            "correct": False,
            "format_ok": False,
            "final_answer": None,
            "needs_review": True,
        }

    final, format_ok = extract_final(
        response
    )

    if final is None:

        return {
            "status": "missing_final",
            "correct": False,
            "format_ok": format_ok,
            "final_answer": None,
            "needs_review": True,
        }

    prediction = parse_math_answer(
        final
    )

    if not prediction:

        return {
            "status": "unparsed_prediction",
            "correct": False,
            "format_ok": format_ok,
            "final_answer": final,
            "needs_review": True,
        }

    try:

        correct = bool(
            verify(
                gold,
                prediction,
            )
        )

    except Exception:

        return {
            "status": "verification_error",
            "correct": False,
            "format_ok": format_ok,
            "final_answer": final,
            "needs_review": True,
        }

    return {
        "status": (
            "correct"
            if correct
            else "incorrect"
        ),
        "correct": correct,
        "format_ok": format_ok,
        "final_answer": final,
        "needs_review": False,
    }

# ============================================================
# 6. VÉRIFICATEUR DE LOGIQUE
# ============================================================

def normalize_choice(text):

    if text is None:
        return None

    value = str(text).strip()

    match = re.fullmatch(
        r"\(?\s*([A-E])\s*\)?[.]?",
        value,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return (
        match.group(1).upper()
    )


def grade_logic(
    expected,
    response,
):

    gold = normalize_choice(
        expected
    )

    if gold is None:

        return {
            "status": "unsupported_gold",
            "correct": False,
            "format_ok": False,
            "final_answer": None,
            "needs_review": True,
        }

    final, format_ok = extract_final(
        response
    )

    # Réponse de type (C) ou C,
    # éventuellement sur la
    # dernière ligne.

    if final is None:

        lines = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        if lines:

            final = lines[-1]

    if final is not None:

        final = final.strip().strip(
            r"$"
        )

        if (
            final.startswith("(")
            and final.endswith(")")
        ):

            final = final

    prediction = normalize_choice(
        final
    )

    if prediction is None:

        return {
            "status": "missing_choice",
            "correct": False,
            "format_ok": format_ok,
            "final_answer": final,
            "needs_review": True,
        }

    correct = (
        prediction == gold
    )

    return {
        "status": (
            "correct"
            if correct
            else "incorrect"
        ),
        "correct": correct,
        "format_ok": format_ok,
        "final_answer": prediction,
        "needs_review": False,
    }

# ============================================================
# 7. AUDIT DES TESTS PYTHON
# ============================================================

def audit_python_tests(
    key,
    domain,
):

    tests = key.get(
        "tests"
    ) or {}

    if domain == "programmation":

        public_tests = tests.get(
            "public"
        ) or []

        hidden_tests = tests.get(
            "hidden"
        ) or []

        challenge_tests = tests.get(
            "challenge"
        ) or []

        setup = tests.get(
            "setup"
        ) or ""

        all_tests = (
            public_tests
            + hidden_tests
            + challenge_tests
        )

        if not all_tests:

            return {
                "status": "missing_tests",
                "test_count": 0,
            }

        try:

            if setup:
                ast.parse(setup)

            for test in all_tests:
                ast.parse(test)

        except SyntaxError:

            return {
                "status": "invalid_test_syntax",
                "test_count": len(
                    all_tests
                ),
            }

        return {
            "status": "ready_for_sandbox",
            "test_count": len(
                all_tests
            ),
        }

    if domain == "programmation_avancee":

        # Ce contrôle examine seulement
        # le schéma. Les tests EvalPlus
        # seront intégrés à l'étape
        # d'exécution isolée.

        base_tests = tests.get(
            "test"
        ) or ""

        return {
            "status": (
                "base_tests_present"
                if base_tests
                else "missing_base_tests"
            ),
            "test_count": (
                1
                if base_tests
                else 0
            ),
        }

    raise ValueError(
        f"Domaine Python inconnu : {domain}"
    )

# ============================================================
# 8. TESTS SYNTHÉTIQUES DES VÉRIFICATEURS
# ============================================================

def run_tests():

    cases = [
        (
            "fraction équivalente",
            grade_math(
                "1/2",
                "REPONSE: 0.5",
            )["correct"],
            True,
        ),
        (
            "réponse numérique correcte",
            grade_math(
                "42",
                "Le résultat est 42.\n"
                "REPONSE: 42",
            )["correct"],
            True,
        ),
        (
            "réponse numérique incorrecte",
            grade_math(
                "42",
                "REPONSE: 41",
            )["correct"],
            False,
        ),
        (
            "équivalence LaTeX",
            grade_math(
                r"\frac{3}{4}",
                r"REPONSE: \frac{6}{8}",
            )["correct"],
            True,
        ),
        (
            "choix logique correct",
            grade_logic(
                "(C)",
                "REPONSE: (C)",
            )["correct"],
            True,
        ),
        (
            "choix logique incorrect",
            grade_logic(
                "(C)",
                "REPONSE: (A)",
            )["correct"],
            False,
        ),
    ]

    passed = 0

    for name, actual, expected in cases:

        if actual != expected:

            raise AssertionError(
                f"Test échoué : {name}"
            )

        passed += 1

    print(
        f"Tests des vérificateurs : "
        f"{passed}/{len(cases)} OK"
    )

# ============================================================
# 9. AUDIT DU JEU DE VALIDATION
# ============================================================

def audit_validation():

    records = load_partition(
        "validation"
    )

    stats = Counter()

    review_ids = []

    for question, key in records:

        domain = question[
            "domain"
        ]

        stats[
            f"{domain}_total"
        ] += 1

        if domain in MATH_DOMAINS:

            gold = parse_math_answer(
                key.get("expected")
            )

            if gold:

                stats[
                    f"{domain}_gold_parsed"
                ] += 1

            else:

                stats[
                    f"{domain}_gold_unparsed"
                ] += 1

                review_ids.append(
                    question["id"]
                )

        elif domain == "programmation":

            check = audit_python_tests(
                key,
                domain,
            )

            stats[
                f"programmation_{check['status']}"
            ] += 1

            if (
                check["status"]
                != "ready_for_sandbox"
            ):

                review_ids.append(
                    question["id"]
                )

    return {
        "questions": len(records),
        "counts": dict(stats),
        "manual_review_ids": review_ids,
    }

# ============================================================
# 10. CONTRÔLE DU SCHÉMA DU TEST
# ============================================================

def audit_test_schema():

    # Nous ne lançons aucune inférence.
    # Nous ne corrigeons aucune réponse
    # du jeu de test.
    #
    # Ce contrôle sert uniquement
    # à détecter les vérificateurs
    # incomplets avant l'évaluation.

    records = load_partition(
        "test"
    )

    stats = Counter()

    for question, key in records:

        domain = question[
            "domain"
        ]

        stats[
            f"{domain}_total"
        ] += 1

        if domain == "programmation":

            result = audit_python_tests(
                key,
                domain,
            )

            stats[
                f"programmation_{result['status']}"
            ] += 1

        elif domain == "programmation_avancee":

            result = audit_python_tests(
                key,
                domain,
            )

            stats[
                f"programmation_avancee_{result['status']}"
            ] += 1

    return {
        "questions": len(records),
        "counts": dict(stats),
    }

# ============================================================
# 11. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-test-schema",
        action="store_true",
        help=(
            "Ne pas examiner la structure "
            "des tests indépendants."
        ),
    )

    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK V2 - AUDIT DES VÉRIFICATEURS")
    print("=" * 65)

    run_tests()

    validation = audit_validation()

    print("\nVALIDATION")

    print(
        f"Questions : "
        f"{validation['questions']}"
    )

    for name, count in sorted(
        validation["counts"].items()
    ):

        print(
            f"{name}: {count}"
        )

    print(
        "Questions nécessitant "
        "une revue : "
        f"{len(validation['manual_review_ids'])}"
    )

    if args.skip_test_schema:

        test_schema = {
            "status": "not_checked",
        }

    else:

        test_schema = (
            audit_test_schema()
        )

        print("\nSCHÉMA DU TEST")

        print(
            f"Questions : "
            f"{test_schema['questions']}"
        )

        for name, count in sorted(
            test_schema["counts"].items()
        ):

            print(
                f"{name}: {count}"
            )

    report = {
        "benchmark": "V2",
        "version": "validator_v1",
        "validation": validation,
        "test_schema": test_schema,
        "code_execution_performed": False,
        "model_inference_performed": False,
    }

    with REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("AUDIT TERMINÉ")
    print("=" * 65)

    print(
        f"Rapport : {REPORT}"
    )

    print(
        "Aucun programme généré "
        "n'a été exécuté."
    )


if __name__ == "__main__":
    main()

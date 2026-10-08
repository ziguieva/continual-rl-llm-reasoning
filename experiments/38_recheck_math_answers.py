
import hashlib
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

RESULTS = (
    ROOT
    / "results"
    / "benchmark_v2_math_qwen_baseline"
)

KEYS_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "grading"
    / "validation_keys.jsonl"
)

GENERATIONS_FILE = RESULTS / "generations.jsonl"
GRADES_FILE = RESULTS / "grades.jsonl"

REPORT_FILE = RESULTS / "math_recheck_v1.json"

DETAIL_FILE = (
    RESULTS / "math_recheck_details_v1.jsonl"
)

EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        boxed_match_priority=0,
    ),
    ExprExtractionConfig(),
]

# Exiger un marqueur explicite
# de conclusion sur la dernière ligne.

CONCLUSION = re.compile(
    r"^\s*(?:"
    r"therefore|thus|hence|so|finally|"
    r"in conclusion|"
    r"the final answer is|"
    r"the answer is"
    r")\b\s*[,:\s]*",
    flags=re.IGNORECASE,
)

NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?\d[\d,]*"
    r"(?:\.\d+)?"
    r"(?:/\d+)?"
    r"(?![A-Za-z0-9_])"
)

# ============================================================
# 2. CHARGEMENT
# ============================================================

def load_jsonl(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    indexed = {
        row["id"]: row
        for row in rows
    }

    if len(indexed) != len(rows):

        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return indexed


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()

# ============================================================
# 3. EXTRACTION CONSERVATRICE
# ============================================================

def extract_conclusion(response):

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if not lines:

        return {
            "category": "empty_response",
            "candidate": None,
            "last_line": "",
        }

    last_line = lines[-1]

    match = CONCLUSION.match(
        last_line
    )

    if not match:

        return {
            "category": "no_explicit_conclusion",
            "candidate": None,
            "last_line": last_line,
        }

    conclusion = last_line[
        match.end():
    ]

    numbers = NUMBER.findall(
        conclusion
    )

    if len(numbers) == 0:

        return {
            "category": "no_numeric_candidate",
            "candidate": None,
            "last_line": last_line,
        }

    if len(numbers) > 1:

        return {
            "category": "ambiguous_multiple_numbers",
            "candidate": None,
            "numbers": numbers,
            "last_line": last_line,
        }

    return {
        "category": "unique_numeric_candidate",
        "candidate": numbers[0],
        "last_line": last_line,
    }

# ============================================================
# 4. COMPARAISON MATHÉMATIQUE
# ============================================================

def parse_answer(answer):

    candidates = [
        r"\boxed{" + str(answer) + "}",
        str(answer),
    ]

    for candidate in candidates:

        try:

            parsed = parse(
                candidate,
                extraction_config=(
                    EXTRACTION_CONFIG
                ),
                fallback_mode="no_fallback",
            )

        except Exception:
            continue

        if parsed:
            return parsed

    return []


def check_candidate(candidate, expected):

    predicted = parse_answer(
        candidate
    )

    gold = parse_answer(
        expected
    )

    if not predicted or not gold:

        return "verification_unavailable"

    try:

        correct = bool(
            verify(
                gold,
                predicted,
            )
        )

    except Exception:

        return "verification_error"

    return (
        "rescued_correct"
        if correct
        else "candidate_incorrect"
    )

# ============================================================
# 5. TESTS LOCAUX DE L'EXTRACTEUR
# ============================================================

def run_tests():

    tests = [
        (
            "Therefore, their combined "
            "experience is 70 years.",
            "70",
            "unique_numeric_candidate",
        ),
        (
            "Therefore, the plane can "
            "hold **90** more bags.",
            "90",
            "unique_numeric_candidate",
        ),
        (
            "Therefore, Pete should leave "
            "at 0850 hours to arrive "
            "by 0900 hours.",
            None,
            "ambiguous_multiple_numbers",
        ),
        (
            "Therefore, Linda initially "
            "had $10.",
            "10",
            "unique_numeric_candidate",
        ),
        (
            "So, there are 15 male "
            "alligators.",
            "15",
            "unique_numeric_candidate",
        ),
        (
            "Step 1: calculate 42.",
            None,
            "no_explicit_conclusion",
        ),
    ]

    for response, expected, category in tests:

        result = extract_conclusion(
            response
        )

        if (
            result["candidate"] != expected
            or result["category"] != category
        ):

            raise AssertionError(
                f"Test échoué : {response}"
            )

    return len(tests)

# ============================================================
# 6. AUDIT DES RÉSULTATS EXISTANTS
# ============================================================

def main():

    generations = load_jsonl(
        GENERATIONS_FILE
    )

    grades = load_jsonl(
        GRADES_FILE
    )

    keys = load_jsonl(
        KEYS_FILE
    )

    if len(generations) != 350:

        raise ValueError(
            "350 générations attendues."
        )

    if len(grades) != 350:

        raise ValueError(
            "350 corrections attendues."
        )

    if set(generations) != set(grades):

        raise ValueError(
            "Les identifiants diffèrent."
        )

    tests_passed = run_tests()

    print("=" * 65)
    print("RELECTURE MATHÉMATIQUE - BENCHMARK V2")
    print("=" * 65)

    print(
        f"Tests d'extraction : "
        f"{tests_passed}/{tests_passed} OK"
    )

    counts = Counter()

    details = []

    original_correct = sum(
        bool(
            grade.get(
                "correct",
                False,
            )
        )
        for grade in grades.values()
    )

    for example_id in sorted(grades):

        grade = grades[
            example_id
        ]

        if grade["status"] != "no_final_answer":
            continue

        generation = generations[
            example_id
        ]

        key = keys[
            example_id
        ]

        domain = grade[
            "domain"
        ]

        truncated = bool(
            generation.get(
                "truncated",
                False,
            )
        )

        response = generation[
            "response"
        ]

        detail = {
            "id": example_id,
            "domain": domain,
            "truncated": truncated,
            "expected": key["expected"],
            "candidate": None,
            "last_line": "",
        }

        # Ne pas attribuer une nouvelle
        # réponse à un texte interrompu.

        if truncated:

            category = "truncated"

        # La récupération automatique
        # concerne uniquement GSM8K.

        elif domain != "maths_appliques":

            category = (
                "manual_review_other_domain"
            )

        # Une réponse encadrée déjà
        # rejetée mérite une revue manuelle.

        elif re.search(
            r"\\boxed\s*\{",
            response,
        ):

            category = (
                "manual_review_boxed"
            )

        else:

            extraction = extract_conclusion(
                response
            )

            category = extraction[
                "category"
            ]

            detail["candidate"] = (
                extraction["candidate"]
            )

            detail["last_line"] = (
                extraction["last_line"]
            )

            if (
                category
                == "unique_numeric_candidate"
            ):

                category = check_candidate(
                    extraction["candidate"],
                    key["expected"],
                )

        detail["category"] = category

        if not detail["last_line"]:

            lines = [
                line.strip()
                for line in response.splitlines()
                if line.strip()
            ]

            detail["last_line"] = (
                lines[-1]
                if lines
                else ""
            )

        counts[
            category
        ] += 1

        details.append(
            detail
        )

    rescued = counts[
        "rescued_correct"
    ]

    indicative_correct = (
        original_correct + rescued
    )

    report = {
        "partition": "validation",
        "model": (
            "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        "original_correct": original_correct,
        "original_total": 350,
        "original_accuracy": round(
            100 * original_correct / 350,
            2,
        ),
        "rescued_correct": rescued,
        "indicative_correct": (
            indicative_correct
        ),
        "indicative_accuracy": round(
            100 * indicative_correct / 350,
            2,
        ),
        "categories": dict(
            counts
        ),
        "source_hashes": {
            "generations": sha256_file(
                GENERATIONS_FILE
            ),
            "grades": sha256_file(
                GRADES_FILE
            ),
        },
        "method": (
            "Analyse exploratoire après "
            "l'évaluation initiale. "
            "Le score de référence "
            "reste inchangé."
        ),
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with DETAIL_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for detail in details:

            file.write(
                json.dumps(
                    detail,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\nCATÉGORIES"
    )

    for category, count in counts.most_common():

        print(
            f"{category}: {count}"
        )

    print(
        "\nRÉSULTATS"
    )

    print(
        f"Score initial : "
        f"{original_correct}/350"
    )

    print(
        f"Réponses correctes récupérées : "
        f"{rescued}"
    )

    print(
        "Score exploratoire après "
        "récupération : "
        f"{indicative_correct}/350"
    )

    print(
        "\nCAS À EXAMINER MANUELLEMENT"
    )

    shown = 0

    for detail in details:

        if detail["category"] in {
            "truncated",
            "rescued_correct",
            "candidate_incorrect",
        }:
            continue

        print(
            f"\n{detail['id']}"
        )

        print(
            f"Catégorie : "
            f"{detail['category']}"
        )

        print(
            f"Attendu : "
            f"{detail['expected']}"
        )

        print(
            f"Fin : "
            f"{detail['last_line'][:250]}"
        )

        shown += 1

        if shown >= 8:
            break

    print(
        "\nRapport : "
        f"{REPORT_FILE}"
    )

    print(
        "Détails : "
        f"{DETAIL_FILE}"
    )


if __name__ == "__main__":
    main()

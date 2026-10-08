
import json
import re

from collections import Counter
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "results"
    / "benchmark_v2_math_qwen_baseline"
)

BENCHMARK = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
)

GENERATIONS_FILE = RESULTS / "generations.jsonl"
GRADES_FILE = RESULTS / "grades.jsonl"

QUESTIONS_FILE = BENCHMARK / "validation.jsonl"

KEYS_FILE = (
    BENCHMARK
    / "grading"
    / "validation_keys.jsonl"
)

AUDIT_FILE = RESULTS / "math_audit_v1.json"

REVIEW_FILE = (
    RESULTS / "math_manual_review_v1.jsonl"
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

    ids = [
        row["id"]
        for row in rows
    ]

    if len(ids) != len(set(ids)):

        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return {
        row["id"]: row
        for row in rows
    }


generations = load_jsonl(
    GENERATIONS_FILE
)

grades = load_jsonl(
    GRADES_FILE
)

questions = load_jsonl(
    QUESTIONS_FILE
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
        "Les générations et les "
        "corrections ne correspondent pas."
    )

# ============================================================
# 3. ANALYSE
# ============================================================

statuses = Counter()

truncation_by_status = Counter()

no_final_categories = Counter()

boxed_by_status = Counter()

domains = {}

review_rows = []

for example_id in sorted(grades):

    generation = generations[
        example_id
    ]

    grade = grades[
        example_id
    ]

    response = generation[
        "response"
    ]

    status = grade[
        "status"
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

    has_boxed = bool(
        re.search(
            r"\\boxed\s*\{",
            response,
        )
    )

    has_final_marker = bool(
        re.search(
            r"(?im)^\s*"
            r"(?:REPONSE|RÉPONSE|"
            r"FINAL_ANSWER|FINAL ANSWER)"
            r"\s*:",
            response,
        )
    )

    has_answer_phrase = bool(
        re.search(
            r"(?i)\b"
            r"(?:final answer is|"
            r"answer is|"
            r"therefore the answer is)"
            r"\b",
            response,
        )
    )

    statuses[
        status
    ] += 1

    if truncated:

        truncation_by_status[
            status
        ] += 1

    if has_boxed:

        boxed_by_status[
            status
        ] += 1

    if domain not in domains:

        domains[
            domain
        ] = {
            "total": 0,
            "correct": 0,
            "no_final_answer": 0,
            "truncated": 0,
            "format_ok": 0,
        }

    stats = domains[
        domain
    ]

    stats["total"] += 1

    stats["correct"] += int(
        bool(
            grade.get(
                "correct",
                False,
            )
        )
    )

    stats["no_final_answer"] += int(
        status == "no_final_answer"
    )

    stats["truncated"] += int(
        truncated
    )

    stats["format_ok"] += int(
        bool(
            grade.get(
                "format_ok",
                False,
            )
        )
    )

    # --------------------------------------------------------
    # Cas nécessitant une revue :
    # réponse finale non reconnue.
    # --------------------------------------------------------

    if status == "no_final_answer":

        if truncated:

            category = (
                "no_final_truncated"
            )

        else:

            category = (
                "no_final_not_truncated"
            )

        no_final_categories[
            category
        ] += 1

        if has_boxed:

            no_final_categories[
                "no_final_with_boxed"
            ] += 1

        if has_final_marker:

            no_final_categories[
                "no_final_with_final_marker"
            ] += 1

        if has_answer_phrase:

            no_final_categories[
                "no_final_with_answer_phrase"
            ] += 1

        review_rows.append({
            "id": example_id,
            "domain": domain,
            "status": status,
            "category": category,
            "truncated": truncated,
            "tokens": generation.get(
                "tokens",
                0,
            ),
            "has_boxed": has_boxed,
            "has_final_marker": has_final_marker,
            "has_answer_phrase": has_answer_phrase,
            "question": questions[
                example_id
            ]["question"],
            "expected": keys[
                example_id
            ]["expected"],
            "response": response,
        })

# ============================================================
# 4. ENREGISTREMENT
# ============================================================

total_truncated = sum(
    truncation_by_status.values()
)

report = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "partition": "validation",
    "evaluated": len(grades),
    "status_counts": dict(
        statuses
    ),
    "truncated_total": total_truncated,
    "truncation_by_status": dict(
        truncation_by_status
    ),
    "boxed_by_status": dict(
        boxed_by_status
    ),
    "no_final_categories": dict(
        no_final_categories
    ),
    "domains": domains,
    "note": (
        "Audit diagnostique uniquement. "
        "Les scores initiaux ne sont "
        "pas modifiés."
    ),
}

with AUDIT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )

with REVIEW_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    for row in review_rows:

        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

# ============================================================
# 5. AFFICHAGE
# ============================================================

print("=" * 65)
print("AUDIT MATHÉMATIQUE - BENCHMARK V2")
print("=" * 65)

print(
    f"Problèmes évalués : {len(grades)}"
)

print("\nSTATUTS")

for status, count in sorted(
    statuses.items()
):

    print(
        f"{status}: {count}"
    )

print("\nTRONCATURES PAR STATUT")

for status, count in sorted(
    truncation_by_status.items()
):

    print(
        f"{status}: {count}"
    )

print(
    f"Total tronqué : {total_truncated}"
)

print("\nAUDIT DES RÉPONSES FINALES ABSENTES")

for category, count in sorted(
    no_final_categories.items()
):

    print(
        f"{category}: {count}"
    )

print("\nRÉPONSES AVEC BOXED PAR STATUT")

for status, count in sorted(
    boxed_by_status.items()
):

    print(
        f"{status}: {count}"
    )

print("\nPAR DOMAINE")

for domain, stats in domains.items():

    print(
        f"{domain}: "
        f"{stats['correct']}/{stats['total']} "
        f"correctes, "
        f"{stats['no_final_answer']} "
        f"sans réponse reconnue, "
        f"{stats['truncated']} tronquées"
    )

print("\nEXEMPLES NON TRONQUÉS SANS RÉPONSE RECONNUE")

shown = 0

for row in review_rows:

    if row["truncated"]:
        continue

    print("\n" + "-" * 60)

    print(
        f"ID : {row['id']}"
    )

    print(
        f"Réponse attendue : {row['expected']}"
    )

    print(
        "Fin de la génération :"
    )

    print(
        row["response"][-500:]
    )

    shown += 1

    if shown >= 5:
        break

print("\n" + "=" * 65)
print("AUDIT TERMINÉ")
print("=" * 65)

print(
    f"Résumé : {AUDIT_FILE}"
)

print(
    f"Revue détaillée : {REVIEW_FILE}"
)

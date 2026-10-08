
import json
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "results"
    / "benchmark_v2_mbpp_qwen_baseline"
)

GENERATIONS = RESULTS / "generations.jsonl"
GRADES = RESULTS / "grades.jsonl"
SUMMARY = RESULTS / "error_audit.json"
REPORT = RESULTS / "error_audit.md"

# ============================================================
# 2. CHARGEMENT
# ============================================================

def load_jsonl(path):

    if not path.is_file():
        raise FileNotFoundError(path)

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


generations = load_jsonl(GENERATIONS)
grades = load_jsonl(GRADES)

if len(generations) != 90 or len(grades) != 90:
    raise ValueError(
        "90 générations et 90 corrections attendues."
    )

if set(generations) != set(grades):
    raise ValueError(
        "Les générations et corrections diffèrent."
    )

# ============================================================
# 3. CLASSIFICATION DES ERREURS
# ============================================================

def classify_error(grade):

    status = grade["status"]
    error = grade.get("error") or ""

    if status == "passed":
        return "passed"

    if status == "failed_assertion":
        return "incorrect_result"

    if status == "syntax_error":
        return "syntax_error"

    if status == "missing_dependency":
        return "missing_dependency"

    if status == "execution_error":

        exception = error.split(
            ":",
            1,
        )[0].strip()

        if exception == "NameError":
            return "NameError"

        if exception == "TypeError":
            return "TypeError"

        if exception == "IndexError":
            return "IndexError"

        if exception == "KeyError":
            return "KeyError"

        if exception == "ValueError":
            return "ValueError"

        if exception == "ImportError":
            return "ImportError"

        if exception == "ModuleNotFoundError":
            return "ModuleNotFoundError"

        return exception or "other_execution_error"

    return status


# ============================================================
# 4. CONSTRUCTION DU RAPPORT
# ============================================================

status_counts = Counter()
error_counts = Counter()
truncated_counts = Counter()

error_examples = defaultdict(list)

records = []

for example_id in sorted(grades):

    grade = grades[example_id]
    generation = generations[example_id]

    category = classify_error(grade)

    truncated = bool(
        generation.get("truncated", False)
    )

    record = {
        "id": example_id,
        "status": grade["status"],
        "category": category,
        "tests_passed": grade["tests_passed"],
        "tests_total": grade["tests_total"],
        "error": grade.get("error") or "",
        "truncated": truncated,
        "tokens": generation.get("tokens", 0),
        "code": generation.get("code") or "",
    }

    records.append(record)

    status_counts[grade["status"]] += 1
    error_counts[category] += 1

    if truncated:
        truncated_counts[category] += 1

    if category != "passed":
        error_examples[category].append(record)

# ============================================================
# 5. BILAN NUMÉRIQUE
# ============================================================

passed_exercises = status_counts["passed"]

passed_tests = sum(
    row["tests_passed"]
    for row in records
)

total_tests = sum(
    row["tests_total"]
    for row in records
)

truncated_total = sum(
    row["truncated"]
    for row in records
)

summary = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "MBPP validation",
    "total_exercises": len(records),
    "passed_exercises": passed_exercises,
    "pass_rate": round(
        100 * passed_exercises / len(records),
        2,
    ),
    "tests_passed": passed_tests,
    "tests_total": total_tests,
    "status_counts": dict(status_counts),
    "error_categories": dict(error_counts),
    "truncated_total": truncated_total,
    "truncated_by_category": dict(truncated_counts),
    "records": records,
}

with SUMMARY.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )

# ============================================================
# 6. RAPPORT MARKDOWN
# ============================================================

lines = [
    "# Audit des erreurs MBPP - Qwen",
    "",
    f"Exercices : {len(records)}",
    f"Réussis : {passed_exercises}",
    f"Tests réussis : {passed_tests}/{total_tests}",
    f"Générations tronquées : {truncated_total}",
    "",
    "## Catégories d'erreurs",
    "",
]

for category, count in error_counts.most_common():

    lines.append(
        f"- {category} : {count}"
    )

lines.extend([
    "",
    "## Détail des programmes incorrects",
    "",
])

for category, examples in error_examples.items():

    lines.extend([
        f"### {category}",
        "",
    ])

    for row in examples:

        lines.extend([
            f"#### {row['id']}",
            "",
            f"Statut : {row['status']}",
            "",
            (
                f"Tests : {row['tests_passed']}"
                f"/{row['tests_total']}"
            ),
            "",
            f"Tronqué : {row['truncated']}",
            "",
            f"Erreur : {row['error']}",
            "",
            "```python",
            row["code"],
            "```",
            "",
        ])

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

# ============================================================
# 7. AFFICHAGE
# ============================================================

print("=" * 65)
print("AUDIT DES ERREURS - MBPP")
print("=" * 65)

print(f"Exercices : {len(records)}")
print(f"Réussis : {passed_exercises}")
print(
    f"Tests réussis : {passed_tests}/{total_tests}"
)
print(
    f"Générations tronquées : {truncated_total}"
)

print("\nCATÉGORIES D'ERREURS")

for category, count in error_counts.most_common():

    print(
        f"{category}: {count}"
    )

print("\nEXEMPLES D'ERREURS D'EXÉCUTION")

shown = 0

for row in records:

    if row["status"] not in {
        "execution_error",
        "missing_dependency",
    }:
        continue

    print(
        f"{row['id']} : {row['error']}"
    )

    shown += 1

    if shown == 12:
        break

print("\n" + "=" * 65)
print("AUDIT TERMINÉ")
print("=" * 65)

print(f"Résumé : {SUMMARY}")
print(f"Rapport détaillé : {REPORT}")

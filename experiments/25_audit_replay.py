
import csv
import json
from math import comb
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results/continual_replay20_comparison.csv"
)

REPORT_FILE = (
    ROOT / "results/replay20_audit_report.md"
)

SUMMARY_FILE = (
    ROOT / "results/replay20_audit_summary.json"
)

DOMAINS = ["maths", "algebre", "code"]

MODELS = ["m2", "m3", "m3r"]

# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def as_bool(value):
    return str(value).strip().lower() == "true"


def is_correct(row, model):
    return as_bool(row[f"{model}_correct"])


def is_formatted(row, model):
    return as_bool(row[f"{model}_end_to_end"])


def count_changes(rows, before, after):

    improved = []
    regressed = []

    for row in rows:

        old = is_correct(row, before)
        new = is_correct(row, after)

        if not old and new:
            improved.append(row)

        elif old and not new:
            regressed.append(row)

    return improved, regressed


def exact_mcnemar_p(improved, regressed):

    # Test exact bilatéral sur les paires
    # dont le résultat a changé.
    n = improved + regressed

    if n == 0:
        return 1.0

    k = min(improved, regressed)

    probability = sum(
        comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2 * probability)


def model_metrics(rows, model):

    total = len(rows)

    correct = sum(
        is_correct(row, model)
        for row in rows
    )

    formatted = sum(
        is_formatted(row, model)
        for row in rows
    )

    contradictions = sum(
        row[f"{model}_status"] == "contradictory"
        for row in rows
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(
            100 * correct / total, 2
        ) if total else 0.0,
        "exact_and_formatted": formatted,
        "contradictions": contradictions
    }


def transition_metrics(rows, before, after):

    improved, regressed = count_changes(
        rows, before, after
    )

    return {
        "improved": len(improved),
        "regressed": len(regressed),
        "net_gain": len(improved) - len(regressed),
        "mcnemar_exact_p": exact_mcnemar_p(
            len(improved),
            len(regressed)
        )
    }

# ============================================================
# 3. CHARGEMENT ET VÉRIFICATIONS
# ============================================================

if not INPUT_FILE.is_file():
    raise FileNotFoundError(INPUT_FILE)

with INPUT_FILE.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:

    rows = list(csv.DictReader(file))

required_columns = {
    "id",
    "domain",
    "question",
    "expected"
}

for model in MODELS:

    required_columns.update({
        f"{model}_prediction",
        f"{model}_status",
        f"{model}_correct",
        f"{model}_end_to_end",
        f"{model}_response"
    })

if not rows:
    raise ValueError("Le CSV est vide.")

missing_columns = (
    required_columns - set(rows[0])
)

if missing_columns:
    raise ValueError(
        f"Colonnes absentes : {missing_columns}"
    )

ids = [row["id"] for row in rows]

if len(ids) != len(set(ids)):
    raise ValueError(
        "Identifiants dupliqués."
    )

print("=" * 60)
print("CONTINUAL LEARNING - AUDIT EXPERIENCE REPLAY")
print("=" * 60)

print(f"Questions chargées : {len(rows)}")

# ============================================================
# 4. BILAN PAR DOMAINE
# ============================================================

summary = {
    "source": str(INPUT_FILE),
    "domains": {},
    "global": {}
}

for domain in DOMAINS:

    subset = [
        row for row in rows
        if row["domain"] == domain
    ]

    metrics = {
        "models": {
            model: model_metrics(
                subset, model
            )
            for model in MODELS
        },

        "m2_to_m3r": transition_metrics(
            subset, "m2", "m3r"
        ),

        "m3_to_m3r": transition_metrics(
            subset, "m3", "m3r"
        )
    }

    summary["domains"][domain] = metrics

    print("\n" + "-" * 60)
    print(domain.upper())
    print("-" * 60)

    for model in MODELS:

        result = metrics["models"][model]

        print(
            f"{model.upper()} : "
            f"{result['correct']}/{result['total']} "
            f"({result['accuracy']:.2f}%)"
        )

    for transition in [
        "m2_to_m3r",
        "m3_to_m3r"
    ]:

        result = metrics[transition]

        print(
            f"{transition.upper()} : "
            f"{result['improved']} améliorées / "
            f"{result['regressed']} dégradées "
            f"| p exact = "
            f"{result['mcnemar_exact_p']:.4f}"
        )

# ============================================================
# 5. BILAN GLOBAL
# ============================================================

summary["global"] = {
    "models": {
        model: model_metrics(
            rows, model
        )
        for model in MODELS
    },

    "m2_to_m3r": transition_metrics(
        rows, "m2", "m3r"
    ),

    "m3_to_m3r": transition_metrics(
        rows, "m3", "m3r"
    )
}

improved, regressed = count_changes(
    rows, "m3", "m3r"
)

changed_rows = improved + regressed

print("\n" + "=" * 60)
print("BILAN GLOBAL")
print("=" * 60)

for model in MODELS:

    result = summary["global"]["models"][model]

    print(
        f"{model.upper()} : "
        f"{result['correct']}/{result['total']} "
        f"({result['accuracy']:.2f}%) "
        f"| Exactitude + format : "
        f"{result['exact_and_formatted']}"
    )

print(
    f"\nM3 -> M3-R : "
    f"{len(improved)} améliorées / "
    f"{len(regressed)} dégradées"
)

print(
    f"Questions dont l'exactitude change : "
    f"{len(changed_rows)}"
)

# ============================================================
# 6. AUDIT DES CHANGEMENTS
# ============================================================

report = [
    "# Audit de l'Experience Replay à 20 %",
    "",
    "Comparaison M2 / M3 / M3-R.",
    "",
    "Une réponse contradictoire est considérée "
    "comme incorrecte par l'évaluateur V16.",
    ""
]

for domain in DOMAINS:

    report.extend([
        f"## {domain.upper()}",
        ""
    ])

    domain_changes = [
        row
        for row in changed_rows
        if row["domain"] == domain
    ]

    if not domain_changes:

        report.extend([
            "Aucun changement d'exactitude.",
            ""
        ])

        continue

    for row in domain_changes:

        m3_correct = is_correct(
            row, "m3"
        )

        m3r_correct = is_correct(
            row, "m3r"
        )

        status = (
            "AMÉLIORATION"
            if not m3_correct and m3r_correct
            else "RÉGRESSION"
        )

        report.extend([
            f"### {row['id']} — {status}",
            "",
            f"Question : {row['question']}",
            "",
            f"Attendu : {row['expected']}",
            ""
        ])

        for model in MODELS:

            report.extend([
                f"**{model.upper()}**",
                "",
                f"- Prédiction : "
                f"{row[f'{model}_prediction']}",
                f"- Statut : "
                f"{row[f'{model}_status']}",
                f"- Correct : "
                f"{is_correct(row, model)}",
                f"- Exactitude + format : "
                f"{is_formatted(row, model)}",
                "",
                "````text",
                row[f"{model}_response"],
                "````",
                ""
            ])

        print(
            f"{status:12s} | "
            f"{row['id']} | "
            f"Attendu : {row['expected']} | "
            f"M3 : {row['m3_prediction']} | "
            f"M3-R : {row['m3r_prediction']}"
        )

# ============================================================
# 7. EXPORT DU RAPPORT
# ============================================================

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

summary["changed_ids"] = {
    "improved": [
        row["id"]
        for row in improved
    ],
    "regressed": [
        row["id"]
        for row in regressed
    ]
}

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("AUDIT TERMINÉ")
print("=" * 60)

print(f"Rapport : {REPORT_FILE}")
print(f"Résumé : {SUMMARY_FILE}")

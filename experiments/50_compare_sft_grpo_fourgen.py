
import json
import math
import random

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SFT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_sft_stage0"
)

GRPO_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_grpo_fourgen_v1"
)

REPORT_FILE = (
    GRPO_DIR
    / "paired_comparison_sft_fourgen.json"
)

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
)

DOMAIN_SIZES = {
    "maths_appliques": 150,
    "algebre_avancee": 100,
    "maths_competition": 100,
}

PROTOCOL_FIELDS = (
    "partition",
    "domains",
    "seed",
    "device",
    "max_new_tokens",
    "do_sample",
    "system_prompt",
    "questions_hash",
    "grading_hash",
    "evaluator_hash",
    "base_model",
)

SEED = 42
BOOTSTRAP_REPETITIONS = 5000


# ============================================================
# 2. LECTURE DES RÉSULTATS
# ============================================================

def read_json(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
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


# ============================================================
# 3. VÉRIFICATION DES PROTOCOLES
# ============================================================

def verify_protocol():

    sft = read_json(
        SFT_DIR / "metadata.json"
    )

    grpo = read_json(
        GRPO_DIR / "metadata.json"
    )

    for field in PROTOCOL_FIELDS:

        if field not in sft or field not in grpo:
            raise RuntimeError(
                f"Métadonnée absente : {field}"
            )

        if sft[field] != grpo[field]:
            raise RuntimeError(
                f"Protocole différent : {field}"
            )

    if sft["partition"] != "validation":
        raise RuntimeError(
            "Seule la validation est autorisée."
        )

    print(
        "Protocoles identiques : OUI"
    )

    return {
        field: sft[field]
        for field in PROTOCOL_FIELDS
    }


# ============================================================
# 4. CONSTRUIRE LES PAIRES
# ============================================================

def build_records():

    sft_grades = read_jsonl(
        SFT_DIR / "grades.jsonl"
    )

    grpo_grades = read_jsonl(
        GRPO_DIR / "grades.jsonl"
    )

    sft_generations = read_jsonl(
        SFT_DIR / "generations.jsonl"
    )

    grpo_generations = read_jsonl(
        GRPO_DIR / "generations.jsonl"
    )

    ids = set(sft_grades)

    if len(ids) != 350:
        raise RuntimeError(
            "350 résultats SFT attendus."
        )

    if not (
        ids
        == set(grpo_grades)
        == set(sft_generations)
        == set(grpo_generations)
    ):
        raise RuntimeError(
            "Les identifiants des "
            "évaluations ne correspondent pas."
        )

    records = []

    for example_id in sorted(ids):

        a = sft_grades[example_id]
        b = grpo_grades[example_id]

        domain = a["domain"]

        if domain != b["domain"]:
            raise RuntimeError(
                f"Domaine différent : {example_id}"
            )

        if domain not in DOMAINS:
            raise RuntimeError(
                f"Domaine inconnu : {domain}"
            )

        ag = sft_generations[example_id]
        bg = grpo_generations[example_id]

        records.append({
            "id": example_id,
            "domain": domain,

            "sft_correct": bool(
                a["correct"]
            ),
            "grpo_correct": bool(
                b["correct"]
            ),

            "sft_format": bool(
                a["format_ok"]
            ),
            "grpo_format": bool(
                b["format_ok"]
            ),

            "sft_status": a["status"],
            "grpo_status": b["status"],

            "sft_truncated": bool(
                ag["truncated"]
            ),
            "grpo_truncated": bool(
                bg["truncated"]
            ),

            "sft_tokens": int(
                ag["tokens"]
            ),
            "grpo_tokens": int(
                bg["tokens"]
            ),
        })

    for domain, expected in DOMAIN_SIZES.items():

        actual = sum(
            row["domain"] == domain
            for row in records
        )

        if actual != expected:
            raise RuntimeError(
                f"{domain} : "
                f"{actual} au lieu de {expected}"
            )

    return records


# ============================================================
# 5. INTERVALLE BOOTSTRAP APPARIÉ
# ============================================================

def bootstrap_ci(rows):

    changes = [
        int(row["grpo_correct"])
        - int(row["sft_correct"])
        for row in rows
    ]

    n = len(changes)

    rng = random.Random(SEED)

    estimates = []

    for _ in range(BOOTSTRAP_REPETITIONS):

        change = sum(
            changes[rng.randrange(n)]
            for _ in range(n)
        )

        estimates.append(
            100 * change / n
        )

    estimates.sort()

    low = int(
        0.025 * (len(estimates) - 1)
    )

    high = int(
        0.975 * (len(estimates) - 1)
    )

    return [
        estimates[low],
        estimates[high],
    ]


# ============================================================
# 6. TEST DE MCNEMAR EXACT
# ============================================================

def mcnemar_exact(gained, lost):

    discordant = gained + lost

    if discordant == 0:
        return 1.0

    smaller = min(gained, lost)

    probability = (
        2
        * sum(
            math.comb(discordant, k)
            for k in range(smaller + 1)
        )
        / (2 ** discordant)
    )

    return min(
        1.0,
        probability,
    )


# ============================================================
# 7. COMPARAISON D'UN GROUPE
# ============================================================

def compare(rows):

    n = len(rows)

    sft_correct = sum(
        row["sft_correct"]
        for row in rows
    )

    grpo_correct = sum(
        row["grpo_correct"]
        for row in rows
    )

    gained_ids = [
        row["id"]
        for row in rows
        if (
            not row["sft_correct"]
            and row["grpo_correct"]
        )
    ]

    lost_ids = [
        row["id"]
        for row in rows
        if (
            row["sft_correct"]
            and not row["grpo_correct"]
        )
    ]

    return {
        "total": n,

        "sft_correct": sft_correct,

        "grpo_correct": grpo_correct,

        "sft_accuracy": (
            100 * sft_correct / n
        ),

        "grpo_accuracy": (
            100 * grpo_correct / n
        ),

        "delta_points": (
            100
            * (grpo_correct - sft_correct)
            / n
        ),

        "gained": len(gained_ids),
        "lost": len(lost_ids),

        "gained_ids": gained_ids,
        "lost_ids": lost_ids,

        "both_correct": sum(
            row["sft_correct"]
            and row["grpo_correct"]
            for row in rows
        ),

        "both_incorrect": sum(
            not row["sft_correct"]
            and not row["grpo_correct"]
            for row in rows
        ),

        "sft_format": sum(
            row["sft_format"]
            for row in rows
        ),

        "grpo_format": sum(
            row["grpo_format"]
            for row in rows
        ),

        "sft_truncated": sum(
            row["sft_truncated"]
            for row in rows
        ),

        "grpo_truncated": sum(
            row["grpo_truncated"]
            for row in rows
        ),

        "sft_tokens": sum(
            row["sft_tokens"]
            for row in rows
        ),

        "grpo_tokens": sum(
            row["grpo_tokens"]
            for row in rows
        ),

        "bootstrap_ci95": (
            bootstrap_ci(rows)
        ),

        "mcnemar_exact_p": (
            mcnemar_exact(
                len(gained_ids),
                len(lost_ids),
            )
        ),
    }


# ============================================================
# 8. AFFICHAGE
# ============================================================

def display(title, result):

    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)

    print(
        "SFT : "
        f"{result['sft_correct']}"
        f"/{result['total']} "
        f"({result['sft_accuracy']:.2f} %)"
    )

    print(
        "GRPO 48 étapes : "
        f"{result['grpo_correct']}"
        f"/{result['total']} "
        f"({result['grpo_accuracy']:.2f} %)"
    )

    print(
        "Évolution : "
        f"{result['delta_points']:+.2f} points"
    )

    print(
        "Exercices gagnés : "
        f"{result['gained']}"
    )

    print(
        "Exercices perdus : "
        f"{result['lost']}"
    )

    print(
        "Corrects dans les deux : "
        f"{result['both_correct']}"
    )

    print(
        "Incorrects dans les deux : "
        f"{result['both_incorrect']}"
    )

    low, high = result["bootstrap_ci95"]

    print(
        "IC bootstrap exploratoire à 95 % : "
        f"[{low:+.2f}, {high:+.2f}] points"
    )

    print(
        "McNemar exact exploratoire : "
        f"p = {result['mcnemar_exact_p']:.4f}"
    )


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print(
        "BENCHMARK V2 - SFT VS GRPO 48 ÉTAPES"
    )
    print("=" * 65)

    protocol = verify_protocol()

    records = build_records()

    overall = compare(records)

    display(
        "ENSEMBLE DES 350 PROBLÈMES",
        overall,
    )

    domain_results = {}

    for domain in DOMAINS:

        subset = [
            row
            for row in records
            if row["domain"] == domain
        ]

        result = compare(subset)

        domain_results[domain] = result

        display(
            domain,
            result,
        )

    print("\n" + "=" * 65)
    print("FORMAT ET LONGUEUR")
    print("=" * 65)

    print(
        "Format reconnu : "
        f"{overall['sft_format']} → "
        f"{overall['grpo_format']}"
    )

    print(
        "Générations tronquées : "
        f"{overall['sft_truncated']} → "
        f"{overall['grpo_truncated']}"
    )

    print(
        "Tokens moyens : "
        f"{overall['sft_tokens'] / 350:.1f} → "
        f"{overall['grpo_tokens'] / 350:.1f}"
    )

    print("\n" + "=" * 65)
    print("PROBLÈMES AYANT CHANGÉ DE STATUT")
    print("=" * 65)

    print(
        "Gagnés :",
        overall["gained_ids"],
    )

    print(
        "Perdus :",
        overall["lost_ids"],
    )

    report = {
        "experiment": (
            "sft_vs_grpo_fourgen_48steps"
        ),
        "partition": "validation",
        "protocol": protocol,
        "overall": overall,
        "domains": domain_results,
        "records": records,
        "test_evaluated": False,
    }

    GRPO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("COMPARAISON TERMINÉE")
    print("=" * 65)

    print(
        "Rapport :",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()

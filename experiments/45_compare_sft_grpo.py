
import json
import math
import random
import statistics

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

SFT_DIR = (
    RESULTS / "benchmark_v2_math_sft_stage0"
)

GRPO_DIR = (
    RESULTS / "benchmark_v2_math_grpo_stage0_pilot"
)

OUTPUT_FILE = (
    GRPO_DIR / "paired_comparison_sft_grpo.json"
)

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
)

SEED = 42

BOOTSTRAP_REPETITIONS = 3000


# ============================================================
# 2. CHARGEMENT
# ============================================================

def load_json(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


# ============================================================
# 3. VÉRIFICATION DU PROTOCOLE
# ============================================================

def verify_protocol(sft, grpo):

    fields = (
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

    mismatches = [
        field
        for field in fields
        if sft.get(field) != grpo.get(field)
    ]

    if mismatches:
        raise RuntimeError(
            "Protocoles incompatibles : "
            + ", ".join(mismatches)
        )


# ============================================================
# 4. STATISTIQUES APPARIÉES
# ============================================================

def compare(records):

    total = len(records)

    if not total:
        raise ValueError("Groupe vide.")

    sft_correct = sum(
        row["sft_correct"]
        for row in records
    )

    grpo_correct = sum(
        row["grpo_correct"]
        for row in records
    )

    gained = [
        row["id"]
        for row in records
        if (
            not row["sft_correct"]
            and row["grpo_correct"]
        )
    ]

    lost = [
        row["id"]
        for row in records
        if (
            row["sft_correct"]
            and not row["grpo_correct"]
        )
    ]

    sft_tokens = [
        row["sft_tokens"]
        for row in records
    ]

    grpo_tokens = [
        row["grpo_tokens"]
        for row in records
    ]

    return {
        "total": total,
        "sft_correct": sft_correct,
        "grpo_correct": grpo_correct,
        "sft_accuracy": (
            100 * sft_correct / total
        ),
        "grpo_accuracy": (
            100 * grpo_correct / total
        ),
        "difference_points": (
            100
            * (grpo_correct - sft_correct)
            / total
        ),
        "gained": len(gained),
        "lost": len(lost),
        "gained_ids": gained,
        "lost_ids": lost,
        "both_correct": sum(
            row["sft_correct"]
            and row["grpo_correct"]
            for row in records
        ),
        "both_incorrect": sum(
            not row["sft_correct"]
            and not row["grpo_correct"]
            for row in records
        ),
        "sft_format": sum(
            row["sft_format"]
            for row in records
        ),
        "grpo_format": sum(
            row["grpo_format"]
            for row in records
        ),
        "sft_truncated": sum(
            row["sft_truncated"]
            for row in records
        ),
        "grpo_truncated": sum(
            row["grpo_truncated"]
            for row in records
        ),
        "sft_mean_tokens": statistics.mean(
            sft_tokens
        ),
        "grpo_mean_tokens": statistics.mean(
            grpo_tokens
        ),
        "sft_median_tokens": statistics.median(
            sft_tokens
        ),
        "grpo_median_tokens": statistics.median(
            grpo_tokens
        ),
        "sft_total_tokens": sum(
            sft_tokens
        ),
        "grpo_total_tokens": sum(
            grpo_tokens
        ),
    }


# ============================================================
# 5. INTERVALLE BOOTSTRAP APPARIÉ
# ============================================================

def bootstrap_interval(records):

    differences = [
        int(row["grpo_correct"])
        - int(row["sft_correct"])
        for row in records
    ]

    n = len(differences)

    rng = random.Random(SEED)

    estimates = []

    for _ in range(BOOTSTRAP_REPETITIONS):

        sample = [
            differences[rng.randrange(n)]
            for _ in range(n)
        ]

        estimates.append(
            100 * sum(sample) / n
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

def mcnemar_pvalue(gained, lost):

    n = gained + lost

    if n == 0:
        return 1.0

    smaller = min(gained, lost)

    probability = (
        2
        * sum(
            math.comb(n, index)
            for index in range(smaller + 1)
        )
        / (2 ** n)
    )

    return min(1.0, probability)


# ============================================================
# 7. AFFICHAGE
# ============================================================

def display(name, stats, interval):

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)

    print(
        f"SFT : {stats['sft_correct']}"
        f"/{stats['total']} "
        f"({stats['sft_accuracy']:.2f} %)"
    )

    print(
        f"GRPO : {stats['grpo_correct']}"
        f"/{stats['total']} "
        f"({stats['grpo_accuracy']:.2f} %)"
    )

    print(
        "Évolution : "
        f"{stats['difference_points']:+.2f} "
        "points"
    )

    print(
        f"Exercices gagnés : "
        f"{stats['gained']}"
    )

    print(
        f"Exercices perdus : "
        f"{stats['lost']}"
    )

    print(
        "Corrects dans les deux modèles : "
        f"{stats['both_correct']}"
    )

    print(
        "Incorrects dans les deux modèles : "
        f"{stats['both_incorrect']}"
    )

    print(
        "Intervalle bootstrap "
        "exploratoire à 95 % : "
        f"[{interval[0]:+.2f}, "
        f"{interval[1]:+.2f}] points"
    )


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("BENCHMARK V2 - SFT VS GRPO")
    print("=" * 65)

    sft_metadata = load_json(
        SFT_DIR / "metadata.json"
    )

    grpo_metadata = load_json(
        GRPO_DIR / "metadata.json"
    )

    verify_protocol(
        sft_metadata,
        grpo_metadata,
    )

    print("Protocoles comparables : OUI")

    sft_grades = load_jsonl(
        SFT_DIR / "grades.jsonl"
    )

    grpo_grades = load_jsonl(
        GRPO_DIR / "grades.jsonl"
    )

    sft_generations = load_jsonl(
        SFT_DIR / "generations.jsonl"
    )

    grpo_generations = load_jsonl(
        GRPO_DIR / "generations.jsonl"
    )

    identifiers = set(sft_grades)

    if len(identifiers) != 350:
        raise ValueError(
            "350 résultats SFT attendus."
        )

    if not (
        identifiers
        == set(grpo_grades)
        == set(sft_generations)
        == set(grpo_generations)
    ):
        raise ValueError(
            "Les identifiants ne correspondent pas."
        )

    records = []

    for example_id in sorted(identifiers):

        sft = sft_grades[example_id]
        grpo = grpo_grades[example_id]

        domain = sft["domain"]

        if domain != grpo["domain"]:
            raise ValueError(
                f"Domaine incohérent : {example_id}"
            )

        if domain not in DOMAINS:
            raise ValueError(
                f"Domaine inattendu : {domain}"
            )

        sft_gen = sft_generations[example_id]
        grpo_gen = grpo_generations[example_id]

        records.append({
            "id": example_id,
            "domain": domain,
            "sft_correct": bool(
                sft["correct"]
            ),
            "grpo_correct": bool(
                grpo["correct"]
            ),
            "sft_status": sft["status"],
            "grpo_status": grpo["status"],
            "sft_format": bool(
                sft["format_ok"]
            ),
            "grpo_format": bool(
                grpo["format_ok"]
            ),
            "sft_truncated": bool(
                sft_gen["truncated"]
            ),
            "grpo_truncated": bool(
                grpo_gen["truncated"]
            ),
            "sft_tokens": sft_gen["tokens"],
            "grpo_tokens": grpo_gen["tokens"],
        })

    overall = compare(records)

    overall_interval = bootstrap_interval(
        records
    )

    display(
        "ENSEMBLE DES 350 PROBLÈMES",
        overall,
        overall_interval,
    )

    domains_report = {}

    for domain in DOMAINS:

        subset = [
            row
            for row in records
            if row["domain"] == domain
        ]

        stats = compare(subset)

        interval = bootstrap_interval(
            subset
        )

        domains_report[domain] = {
            "statistics": stats,
            "bootstrap_ci95": interval,
        }

        display(
            domain,
            stats,
            interval,
        )

    p_value = mcnemar_pvalue(
        overall["gained"],
        overall["lost"],
    )

    print("\n" + "=" * 65)
    print("FORMAT ET EFFICACITÉ")
    print("=" * 65)

    print(
        "Format reconnu : "
        f"{overall['sft_format']} "
        f"→ {overall['grpo_format']}"
    )

    print(
        "Générations tronquées : "
        f"{overall['sft_truncated']} "
        f"→ {overall['grpo_truncated']}"
    )

    print(
        "Tokens moyens par réponse : "
        f"{overall['sft_mean_tokens']:.1f} "
        f"→ {overall['grpo_mean_tokens']:.1f}"
    )

    print(
        "Tokens médians par réponse : "
        f"{overall['sft_median_tokens']:.1f} "
        f"→ {overall['grpo_median_tokens']:.1f}"
    )

    print(
        "Tokens générés au total : "
        f"{overall['sft_total_tokens']} "
        f"→ {overall['grpo_total_tokens']}"
    )

    print(
        "McNemar exact exploratoire : "
        f"p = {p_value:.4f}"
    )

    report = {
        "experiment": "sft_vs_grpo_stage0_pilot",
        "partition": "validation",
        "protocol_comparable": True,
        "overall": overall,
        "overall_bootstrap_ci95": (
            overall_interval
        ),
        "mcnemar_exact_pvalue": p_value,
        "domains": domains_report,
        "records": records,
        "test_evaluated": False,
        "note": (
            "Analyse exploratoire appariée. "
            "Les scores historiques "
            "ne sont pas modifiés."
        ),
    }

    with OUTPUT_FILE.open(
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
    print("COMPARAISON TERMINÉE")
    print("=" * 65)

    print(f"Rapport : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

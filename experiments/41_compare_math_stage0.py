
import json
import math
import random

from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_qwen_baseline"
)

SFT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_sft_stage0"
)

OUTPUT_FILE = (
    SFT_DIR
    / "paired_comparison_v1.json"
)

SEED = 42

BOOTSTRAP_REPETITIONS = 3000

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
)


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
# 3. VÉRIFIER LE PROTOCOLE
# ============================================================

def verify_protocol(base, sft):

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
    )

    differences = [
        field
        for field in fields
        if base.get(field) != sft.get(field)
    ]

    if differences:

        raise RuntimeError(
            "Protocoles différents : "
            + ", ".join(differences)
        )


# ============================================================
# 4. ANALYSE APPARIÉE
# ============================================================

def compare_group(records):

    total = len(records)

    if not total:

        raise ValueError(
            "Groupe vide."
        )

    base_correct = sum(
        row["base_correct"]
        for row in records
    )

    sft_correct = sum(
        row["sft_correct"]
        for row in records
    )

    gained = sum(
        not row["base_correct"]
        and row["sft_correct"]
        for row in records
    )

    lost = sum(
        row["base_correct"]
        and not row["sft_correct"]
        for row in records
    )

    both_correct = sum(
        row["base_correct"]
        and row["sft_correct"]
        for row in records
    )

    both_incorrect = sum(
        not row["base_correct"]
        and not row["sft_correct"]
        for row in records
    )

    base_format = sum(
        row["base_format"]
        for row in records
    )

    sft_format = sum(
        row["sft_format"]
        for row in records
    )

    sft_correct_with_format = sum(
        row["sft_correct"]
        and row["sft_format"]
        for row in records
    )

    base_truncated = sum(
        row["base_truncated"]
        for row in records
    )

    sft_truncated = sum(
        row["sft_truncated"]
        for row in records
    )

    base_no_final = sum(
        row["base_status"]
        == "no_final_answer"
        for row in records
    )

    sft_no_final = sum(
        row["sft_status"]
        == "no_final_answer"
        for row in records
    )

    return {
        "total": total,
        "base_correct": base_correct,
        "sft_correct": sft_correct,
        "base_accuracy": (
            100 * base_correct / total
        ),
        "sft_accuracy": (
            100 * sft_correct / total
        ),
        "difference_points": (
            100
            * (sft_correct - base_correct)
            / total
        ),
        "gained": gained,
        "lost": lost,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "base_format": base_format,
        "sft_format": sft_format,
        "sft_correct_with_format": (
            sft_correct_with_format
        ),
        "base_truncated": base_truncated,
        "sft_truncated": sft_truncated,
        "base_no_final": base_no_final,
        "sft_no_final": sft_no_final,
    }


# ============================================================
# 5. INTERVALLE BOOTSTRAP APPARIÉ
# ============================================================

def paired_bootstrap(records):

    differences = [
        int(row["sft_correct"])
        - int(row["base_correct"])
        for row in records
    ]

    n = len(differences)

    rng = random.Random(SEED)

    estimates = []

    for _ in range(
        BOOTSTRAP_REPETITIONS
    ):

        sample = [
            differences[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        estimates.append(
            100 * sum(sample) / n
        )

    estimates.sort()

    lower_index = int(
        0.025
        * (BOOTSTRAP_REPETITIONS - 1)
    )

    upper_index = int(
        0.975
        * (BOOTSTRAP_REPETITIONS - 1)
    )

    return [
        estimates[lower_index],
        estimates[upper_index],
    ]


# ============================================================
# 6. TEST APPARIÉ EXPLORATOIRE
# ============================================================

def mcnemar_exact_pvalue(gained, lost):

    discordant = (
        gained + lost
    )

    if discordant == 0:

        return 1.0

    smaller = min(
        gained,
        lost,
    )

    probability = (
        2
        * sum(
            math.comb(
                discordant,
                index,
            )
            for index in range(
                smaller + 1
            )
        )
        / (2 ** discordant)
    )

    return min(
        1.0,
        probability,
    )


# ============================================================
# 7. AFFICHAGE
# ============================================================

def display_group(name, stats, interval):

    print("\n" + "-" * 65)

    print(name)

    print("-" * 65)

    print(
        "Qwen initial : "
        f"{stats['base_correct']}"
        f"/{stats['total']} "
        f"({stats['base_accuracy']:.2f} %)"
    )

    print(
        "Qwen + SFT : "
        f"{stats['sft_correct']}"
        f"/{stats['total']} "
        f"({stats['sft_accuracy']:.2f} %)"
    )

    print(
        "Évolution : "
        f"{stats['difference_points']:+.2f} "
        "points"
    )

    print(
        "Exercices gagnés : "
        f"{stats['gained']}"
    )

    print(
        "Exercices perdus : "
        f"{stats['lost']}"
    )

    print(
        "Corrects avec les deux : "
        f"{stats['both_correct']}"
    )

    print(
        "Incorrects avec les deux : "
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

    print(
        "BENCHMARK V2 - COMPARAISON APPARIÉE"
    )

    print("=" * 65)

    base_metadata = load_json(
        BASE_DIR / "metadata.json"
    )

    sft_metadata = load_json(
        SFT_DIR / "metadata.json"
    )

    verify_protocol(
        base_metadata,
        sft_metadata,
    )

    base_grades = load_jsonl(
        BASE_DIR / "grades.jsonl"
    )

    sft_grades = load_jsonl(
        SFT_DIR / "grades.jsonl"
    )

    base_generations = load_jsonl(
        BASE_DIR / "generations.jsonl"
    )

    sft_generations = load_jsonl(
        SFT_DIR / "generations.jsonl"
    )

    identifiers = set(
        base_grades
    )

    if len(identifiers) != 350:

        raise ValueError(
            "350 résultats initiaux attendus."
        )

    if not (
        identifiers
        == set(sft_grades)
        == set(base_generations)
        == set(sft_generations)
    ):

        raise ValueError(
            "Les identifiants des "
            "évaluations ne correspondent pas."
        )

    records = []

    for example_id in sorted(identifiers):

        base = base_grades[
            example_id
        ]

        sft = sft_grades[
            example_id
        ]

        if base["domain"] != sft["domain"]:

            raise ValueError(
                "Domaines incohérents : "
                + example_id
            )

        domain = base[
            "domain"
        ]

        if domain not in DOMAINS:

            raise ValueError(
                "Domaine inattendu : "
                + domain
            )

        row = {
            "id": example_id,
            "domain": domain,

            "base_correct": bool(
                base["correct"]
            ),

            "sft_correct": bool(
                sft["correct"]
            ),

            "base_status": base[
                "status"
            ],

            "sft_status": sft[
                "status"
            ],

            "base_format": bool(
                base["format_ok"]
            ),

            "sft_format": bool(
                sft["format_ok"]
            ),

            "base_truncated": bool(
                base_generations[
                    example_id
                ]["truncated"]
            ),

            "sft_truncated": bool(
                sft_generations[
                    example_id
                ]["truncated"]
            ),

            "base_tokens": (
                base_generations[
                    example_id
                ]["tokens"]
            ),

            "sft_tokens": (
                sft_generations[
                    example_id
                ]["tokens"]
            ),
        }

        records.append(
            row
        )

    all_stats = compare_group(
        records
    )

    all_interval = paired_bootstrap(
        records
    )

    domain_reports = {}

    display_group(
        "ENSEMBLE DES 350 PROBLÈMES",
        all_stats,
        all_interval,
    )

    for domain in DOMAINS:

        subset = [
            row
            for row in records
            if row["domain"] == domain
        ]

        stats = compare_group(
            subset
        )

        interval = paired_bootstrap(
            subset
        )

        domain_reports[
            domain
        ] = {
            "statistics": stats,
            "bootstrap_ci95": interval,
        }

        display_group(
            domain,
            stats,
            interval,
        )

    p_value = mcnemar_exact_pvalue(
        all_stats["gained"],
        all_stats["lost"],
    )

    print("\n" + "=" * 65)

    print(
        "FORMAT ET LONGUEUR DES RÉPONSES"
    )

    print("=" * 65)

    print(
        "Format reconnu : "
        f"{all_stats['base_format']} "
        "→ "
        f"{all_stats['sft_format']}"
    )

    print(
        "Réponses SFT correctes "
        "avec le format demandé : "
        f"{all_stats['sft_correct_with_format']}"
    )

    print(
        "Générations tronquées : "
        f"{all_stats['base_truncated']} "
        "→ "
        f"{all_stats['sft_truncated']}"
    )

    print(
        "Réponses finales non reconnues : "
        f"{all_stats['base_no_final']} "
        "→ "
        f"{all_stats['sft_no_final']}"
    )

    print(
        "Test de McNemar exact "
        f"(exploratoire), p = {p_value:.4f}"
    )

    gained_ids = [
        row["id"]
        for row in records
        if (
            not row["base_correct"]
            and row["sft_correct"]
        )
    ]

    lost_ids = [
        row["id"]
        for row in records
        if (
            row["base_correct"]
            and not row["sft_correct"]
        )
    ]

    report = {
        "protocol_comparable": True,
        "partition": "validation",
        "overall": all_stats,
        "overall_bootstrap_ci95": (
            all_interval
        ),
        "mcnemar_exact_pvalue_exploratory": (
            p_value
        ),
        "domains": domain_reports,
        "gained_ids": gained_ids,
        "lost_ids": lost_ids,
        "records": records,
        "note": (
            "Comparaison appariée exploratoire "
            "sur la validation. "
            "Les scores de référence "
            "ne sont pas modifiés."
        ),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print(
        "COMPARAISON TERMINÉE"
    )

    print("=" * 65)

    print(
        f"Rapport : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

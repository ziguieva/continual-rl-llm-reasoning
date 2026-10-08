
import csv
import gc
import hashlib
import importlib.util
import json
import random
import re
import statistics

from pathlib import Path

import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MODELS_DIR = (
    ROOT / "models/replay_multiseed_v1"
)

DATA_FILE = (
    ROOT / "data/processed/validation.jsonl"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

RESULTS_DIR = (
    ROOT / "results/replay_multiseed_validation_v1"
)

SEEDS = (42, 123, 2026)

RATIOS = (0, 10, 20)

DOMAINS = ("maths", "algebre", "code")

MAX_NEW_TOKENS = 256

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

# Référence M2 déjà mesurée sur la validation.
M2_CORRECT = {
    "maths": 23,
    "algebre": 27,
    "code": 6,
}

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

random.seed(42)
torch.manual_seed(42)

print("=" * 65)
print("CONTINUAL LEARNING - MULTI-SEED EVALUATION")
print("=" * 65)

print(f"Device : {DEVICE}")
print(f"Seeds : {SEEDS}")
print(f"Ratios : {RATIOS}")

# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def as_bool(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() == "true"


def file_hash(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def adapter_signature(adapter_dir):

    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]

    weights = next(
        (
            path
            for path in candidates
            if path.is_file()
        ),
        None,
    )

    if weights is None:

        raise FileNotFoundError(
            f"Poids LoRA introuvables : "
            f"{adapter_dir}"
        )

    stat = weights.stat()

    return {
        "config_hash": file_hash(
            adapter_dir / "adapter_config.json"
        ),
        "weights_name": weights.name,
        "weights_size": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
    }


def save_csv(path, records):

    if not records:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def mean_std(values):

    if not values:
        return 0.0, 0.0

    average = statistics.mean(
        values
    )

    deviation = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return (
        round(average, 2),
        round(deviation, 2),
    )

# ============================================================
# 3. VÉRIFICATION DES NEUF ADAPTATEURS
# ============================================================

required_files = [
    DATA_FILE,
    EVALUATOR_FILE,
]

for path in required_files:

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

adapter_paths = {}

missing_adapters = []

for seed in SEEDS:

    for ratio in RATIOS:

        adapter_dir = (
            MODELS_DIR
            / f"seed_{seed}"
            / f"replay_{ratio}"
        )

        adapter_paths[
            (seed, ratio)
        ] = adapter_dir

        config_file = (
            adapter_dir / "adapter_config.json"
        )

        if not config_file.is_file():

            missing_adapters.append(
                str(adapter_dir)
            )

if missing_adapters:

    print("\nAdaptateurs manquants :")

    for path in missing_adapters:
        print(path)

    raise FileNotFoundError(
        "Les neuf entraînements doivent "
        "être terminés avant l'évaluation."
    )

print("Neuf adaptateurs détectés.")

# ============================================================
# 4. CHARGEMENT DE L'ÉVALUATEUR V16
# ============================================================

spec = importlib.util.spec_from_file_location(
    "audit_v16",
    EVALUATOR_FILE,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Impossible de charger l'évaluateur V16."
    )

audit = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    audit
)

audit.run_tests()

print("Évaluateur V16 : OK")

# ============================================================
# 5. CHARGEMENT DE LA VALIDATION
# ============================================================

with DATA_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(dataset) != 90:

    raise ValueError(
        "90 questions de validation attendues."
    )

ids = [
    row["id"]
    for row in dataset
]

if len(ids) != len(set(ids)):

    raise ValueError(
        "Identifiants dupliqués."
    )

for domain in DOMAINS:

    count = sum(
        row["domain"] == domain
        for row in dataset
    )

    if count != 30:

        raise ValueError(
            f"{domain} : 30 questions attendues."
        )

print(
    f"Questions de validation : {len(dataset)}"
)

# ============================================================
# 6. CLASSIFICATION DES TYPES D'EXERCICES
# ============================================================

def classify_subtype(question, domain):

    if domain == "maths":

        expression = (
            question.split(":", 1)[-1]
            .strip()
        )

        if "(" in expression:
            return "operations_combinees"

        if "/" in expression:
            return "division"

        if "*" in expression:
            return "multiplication"

        if "-" in expression:
            return "soustraction"

        return "addition"

    if domain == "algebre":

        equation = (
            question.split(":", 1)[-1]
            .strip()
        )

        if "x/" in equation:
            return "equation_fraction"

        if "(x" in equation:
            return "distributivite"

        if re.search(
            r"\dx\s*-\s*\d+",
            equation,
        ):
            return "equation_soustraction"

        return "equation_addition"

    if domain == "code":

        expression = (
            question.strip()
            .splitlines()[-1]
        )

        if "**" in expression:
            return "puissance"

        if "//" in expression:
            return "division_entiere"

        if "sum(" in expression:
            return "somme_liste"

        if "len(" in expression:
            return "longueur_liste"

        if "%" in expression:
            return "modulo"

        return "autre"

    raise ValueError(
        f"Domaine inconnu : {domain}"
    )

# ============================================================
# 7. EMPREINTE DE L'ÉVALUATION
# ============================================================

DATA_HASH = file_hash(
    DATA_FILE
)

EVALUATOR_HASH = file_hash(
    EVALUATOR_FILE
)

def build_metadata(seed, ratio):

    adapter_dir = adapter_paths[
        (seed, ratio)
    ]

    return {
        "seed": seed,
        "ratio": ratio,
        "model_name": MODEL_NAME,
        "adapter": str(adapter_dir.resolve()),
        "adapter_signature": adapter_signature(
            adapter_dir
        ),
        "dataset_hash": DATA_HASH,
        "evaluator_hash": EVALUATOR_HASH,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "dtype": "float32",
        "device": DEVICE,
    }

# ============================================================
# 8. REPRISE DES ÉVALUATIONS EXISTANTES
# ============================================================

def run_directory(seed, ratio):

    path = (
        RESULTS_DIR
        / f"seed_{seed}"
        / f"replay_{ratio}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def load_cached_run(seed, ratio):

    output_dir = run_directory(
        seed,
        ratio,
    )

    csv_file = (
        output_dir / "predictions.csv"
    )

    metadata_file = (
        output_dir / "metadata.json"
    )

    if (
        not csv_file.is_file()
        or not metadata_file.is_file()
    ):

        return None

    try:

        with metadata_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            saved_metadata = json.load(
                file
            )

        current_metadata = build_metadata(
            seed,
            ratio,
        )

        if saved_metadata != current_metadata:

            return None

        with csv_file.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            records = list(
                csv.DictReader(file)
            )

        if len(records) != len(dataset):

            return None

        if [
            row["id"]
            for row in records
        ] != ids:

            return None

        required = {
            "id",
            "domain",
            "subtype",
            "expected",
            "prediction",
            "status",
            "strict_correct",
            "format_ok",
            "end_to_end",
            "conclusion_correct",
            "tokens",
            "truncated",
            "response",
        }

        if not required.issubset(
            records[0].keys()
        ):

            return None

        return records

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):

        return None

# ============================================================
# 9. TOKENIZER COMMUN
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

# ============================================================
# 10. GÉNÉRATION D'UNE RÉPONSE
# ============================================================

def generate_response(model, question):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated = (
        output[0][input_length:]
    )

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    token_count = len(
        generated
    )

    last_token = (
        int(generated[-1])
        if token_count
        else None
    )

    stop_tokens = {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }

    truncated = (
        token_count >= MAX_NEW_TOKENS
        and last_token not in stop_tokens
    )

    return (
        response,
        token_count,
        truncated,
    )

# ============================================================
# 11. ÉVALUATION D'UN MODÈLE
# ============================================================

def evaluate_model(seed, ratio):

    cached = load_cached_run(
        seed,
        ratio,
    )

    if cached is not None:

        print(
            f"\nSEED {seed} / REPLAY {ratio}% : "
            "résultats déjà enregistrés."
        )

        return cached

    adapter_dir = adapter_paths[
        (seed, ratio)
    ]

    print("\n" + "=" * 65)

    print(
        f"ÉVALUATION SEED {seed} "
        f"- REPLAY {ratio}%"
    )

    print("=" * 65)

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to(DEVICE)
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
    )

    model.to(DEVICE)
    model.eval()

    records = []

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        question = example["question"]

        domain = example["domain"]

        expected = int(
            example["answer"]
        )

        response, tokens, truncated = (
            generate_response(
                model,
                question,
            )
        )

        evaluation = audit.analyze_response(
            response,
            domain,
            expected,
        )

        record = {
            "id": example["id"],
            "seed": seed,
            "ratio": ratio,
            "domain": domain,
            "subtype": classify_subtype(
                question,
                domain,
            ),
            "question": question,
            "expected": expected,
            "prediction": (
                evaluation["prediction"]
            ),
            "status": (
                evaluation["status"]
            ),
            "strict_correct": bool(
                evaluation["strict_correct"]
            ),
            "format_ok": bool(
                evaluation["format_ok"]
            ),
            "end_to_end": bool(
                evaluation["end_to_end"]
            ),
            "conclusion_correct": bool(
                evaluation.get(
                    "conclusion_correct",
                    False,
                )
            ),
            "tokens": tokens,
            "truncated": truncated,
            "response": response,
        }

        records.append(
            record
        )

        if (
            index % 15 == 0
            or index == len(dataset)
        ):

            print(
                f"Progression : "
                f"{index}/{len(dataset)}"
            )

    output_dir = run_directory(
        seed,
        ratio,
    )

    save_csv(
        output_dir / "predictions.csv",
        records,
    )

    save_json(
        output_dir / "metadata.json",
        build_metadata(
            seed,
            ratio,
        ),
    )

    del model
    del base_model

    gc.collect()

    if DEVICE == "mps":

        torch.mps.empty_cache()

    return records

# ============================================================
# 12. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(records):

    total = len(
        records
    )

    if total == 0:

        raise ValueError(
            "Aucune réponse à évaluer."
        )

    correct = sum(
        as_bool(
            row["strict_correct"]
        )
        for row in records
    )

    formatted = sum(
        as_bool(
            row["end_to_end"]
        )
        for row in records
    )

    conclusion_correct = sum(
        as_bool(
            row["conclusion_correct"]
        )
        for row in records
    )

    contradictions = sum(
        row["status"] == "contradictory"
        for row in records
    )

    missing = sum(
        row["status"] == "missing"
        for row in records
    )

    truncated = sum(
        as_bool(
            row["truncated"]
        )
        for row in records
    )

    total_tokens = sum(
        int(
            row["tokens"]
        )
        for row in records
    )

    return {
        "questions": total,

        "correct": correct,

        "accuracy": round(
            100 * correct / total,
            2,
        ),

        "formatted_correct": formatted,

        "formatted_accuracy": round(
            100 * formatted / total,
            2,
        ),

        "conclusion_correct": (
            conclusion_correct
        ),

        "conclusion_accuracy": round(
            100 * conclusion_correct / total,
            2,
        ),

        "contradictions": contradictions,

        "missing": missing,

        "truncated": truncated,

        "mean_tokens": round(
            total_tokens / total,
            2,
        ),
    }

# ============================================================
# 13. MÉTRIQUES PAR MODÈLE
# ============================================================

def build_run_metrics(
    records,
    seed,
    ratio,
):

    output = []

    groups = [
        (
            "global",
            "global",
            "all",
            records,
        )
    ]

    for domain in DOMAINS:

        domain_records = [
            row
            for row in records
            if row["domain"] == domain
        ]

        groups.append(
            (
                "domain",
                domain,
                "all",
                domain_records,
            )
        )

        subtypes = sorted({
            row["subtype"]
            for row in domain_records
        })

        for subtype in subtypes:

            subset = [
                row
                for row in domain_records
                if row["subtype"] == subtype
            ]

            groups.append(
                (
                    "subtype",
                    domain,
                    subtype,
                    subset,
                )
            )

    for (
        scope,
        domain,
        subtype,
        subset,
    ) in groups:

        metrics = calculate_metrics(
            subset
        )

        output.append({
            "seed": seed,
            "ratio": ratio,
            "scope": scope,
            "domain": domain,
            "subtype": subtype,
            **metrics,
        })

    return output

# ============================================================
# 14. MOYENNE ET ÉCART-TYPE
# ============================================================

def aggregate_metrics(per_run):

    output = []

    keys = sorted({
        (
            row["ratio"],
            row["scope"],
            row["domain"],
            row["subtype"],
        )
        for row in per_run
    })

    for (
        ratio,
        scope,
        domain,
        subtype,
    ) in keys:

        group = [
            row
            for row in per_run
            if (
                row["ratio"] == ratio
                and row["scope"] == scope
                and row["domain"] == domain
                and row["subtype"] == subtype
            )
        ]

        if len(group) != len(SEEDS):

            raise ValueError(
                "Il manque des résultats "
                f"pour {ratio}% / {domain} "
                f"/ {subtype}."
            )

        accuracy_mean, accuracy_std = (
            mean_std([
                row["accuracy"]
                for row in group
            ])
        )

        format_mean, format_std = (
            mean_std([
                row["formatted_accuracy"]
                for row in group
            ])
        )

        conclusion_mean, conclusion_std = (
            mean_std([
                row["conclusion_accuracy"]
                for row in group
            ])
        )

        contradiction_mean, _ = (
            mean_std([
                row["contradictions"]
                for row in group
            ])
        )

        missing_mean, _ = (
            mean_std([
                row["missing"]
                for row in group
            ])
        )

        tokens_mean, _ = (
            mean_std([
                row["mean_tokens"]
                for row in group
            ])
        )

        output.append({
            "ratio": ratio,
            "scope": scope,
            "domain": domain,
            "subtype": subtype,
            "questions_per_seed": (
                group[0]["questions"]
            ),
            "seeds": len(group),
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "formatted_mean": format_mean,
            "formatted_std": format_std,
            "conclusion_mean": conclusion_mean,
            "conclusion_std": conclusion_std,
            "contradictions_mean": (
                contradiction_mean
            ),
            "missing_mean": missing_mean,
            "mean_tokens": tokens_mean,
        })

    return output

# ============================================================
# 15. COMPARAISONS APPARIÉES ENTRE RATIOS
# ============================================================

def build_paired_comparisons(
    all_predictions,
):

    comparisons = []

    for seed in SEEDS:

        baseline = {
            row["id"]: row
            for row in all_predictions[
                (seed, 0)
            ]
        }

        for ratio in (10, 20):

            candidate = {
                row["id"]: row
                for row in all_predictions[
                    (seed, ratio)
                ]
            }

            for domain in (
                "global",
                *DOMAINS,
            ):

                if domain == "global":

                    selected_ids = ids

                else:

                    selected_ids = [
                        row["id"]
                        for row in dataset
                        if row["domain"] == domain
                    ]

                improved = 0
                regressed = 0

                format_before = 0
                format_after = 0

                for example_id in selected_ids:

                    before = baseline[
                        example_id
                    ]

                    after = candidate[
                        example_id
                    ]

                    before_correct = as_bool(
                        before["strict_correct"]
                    )

                    after_correct = as_bool(
                        after["strict_correct"]
                    )

                    if (
                        not before_correct
                        and after_correct
                    ):
                        improved += 1

                    if (
                        before_correct
                        and not after_correct
                    ):
                        regressed += 1

                    format_before += as_bool(
                        before["end_to_end"]
                    )

                    format_after += as_bool(
                        after["end_to_end"]
                    )

                total = len(
                    selected_ids
                )

                accuracy_delta = (
                    100
                    * (improved - regressed)
                    / total
                )

                format_delta = (
                    100
                    * (
                        format_after
                        - format_before
                    )
                    / total
                )

                comparisons.append({
                    "seed": seed,
                    "ratio_before": 0,
                    "ratio_after": ratio,
                    "domain": domain,
                    "questions": total,
                    "improved": improved,
                    "regressed": regressed,
                    "accuracy_delta": round(
                        accuracy_delta,
                        2,
                    ),
                    "format_delta": round(
                        format_delta,
                        2,
                    ),
                })

    return comparisons


def aggregate_paired_comparisons(
    comparisons,
):

    output = []

    for ratio in (10, 20):

        for domain in (
            "global",
            *DOMAINS,
        ):

            group = [
                row
                for row in comparisons
                if (
                    row["ratio_after"] == ratio
                    and row["domain"] == domain
                )
            ]

            accuracy_mean, accuracy_std = (
                mean_std([
                    row["accuracy_delta"]
                    for row in group
                ])
            )

            format_mean, format_std = (
                mean_std([
                    row["format_delta"]
                    for row in group
                ])
            )

            output.append({
                "ratio_before": 0,
                "ratio_after": ratio,
                "domain": domain,
                "seeds": len(group),
                "accuracy_delta_mean": (
                    accuracy_mean
                ),
                "accuracy_delta_std": (
                    accuracy_std
                ),
                "format_delta_mean": (
                    format_mean
                ),
                "format_delta_std": (
                    format_std
                ),
            })

    return output

# ============================================================
# 16. PROGRAMME PRINCIPAL
# ============================================================

def main():

    all_predictions = {}

    per_run = []

    # --------------------------------------------------------
    # ÉVALUATION DES NEUF MODÈLES
    # --------------------------------------------------------

    for seed in SEEDS:

        for ratio in RATIOS:

            records = evaluate_model(
                seed,
                ratio,
            )

            all_predictions[
                (seed, ratio)
            ] = records

            run_metrics = build_run_metrics(
                records,
                seed,
                ratio,
            )

            per_run.extend(
                run_metrics
            )

    # --------------------------------------------------------
    # AGRÉGATION
    # --------------------------------------------------------

    aggregate = aggregate_metrics(
        per_run
    )

    paired = build_paired_comparisons(
        all_predictions
    )

    paired_summary = (
        aggregate_paired_comparisons(
            paired
        )
    )

    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    save_csv(
        RESULTS_DIR / "summary_per_run.csv",
        per_run,
    )

    save_csv(
        RESULTS_DIR / "summary_by_ratio.csv",
        aggregate,
    )

    save_csv(
        RESULTS_DIR / "paired_per_seed.csv",
        paired,
    )

    save_csv(
        RESULTS_DIR / "paired_summary.csv",
        paired_summary,
    )

    save_json(
        RESULTS_DIR / "summary.json",
        {
            "model": MODEL_NAME,
            "dataset": "validation",
            "seeds": list(SEEDS),
            "ratios": list(RATIOS),
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "m2_reference_correct": M2_CORRECT,
            "per_run": per_run,
            "aggregated": aggregate,
            "paired_per_seed": paired,
            "paired_summary": paired_summary,
        },
    )

    # ========================================================
    # 17. AFFICHAGE : SCORES PAR GRAINE
    # ========================================================

    print("\n" + "=" * 65)
    print("RÉSULTATS PAR GRAINE")
    print("=" * 65)

    for seed in SEEDS:

        print(
            f"\nSEED {seed}"
        )

        for ratio in RATIOS:

            row = next(
                item
                for item in per_run
                if (
                    item["seed"] == seed
                    and item["ratio"] == ratio
                    and item["scope"] == "global"
                )
            )

            print(
                f"Replay {ratio:2d}% : "
                f"accuracy {row['accuracy']:.2f}% | "
                f"exactitude + format "
                f"{row['formatted_accuracy']:.2f}% | "
                f"contradictions "
                f"{row['contradictions']}"
            )

    # ========================================================
    # 18. MOYENNES PAR DOMAINE
    # ========================================================

    print("\n" + "=" * 65)
    print("MOYENNES SUR LES TROIS GRAINES")
    print("=" * 65)

    for domain in (
        "global",
        *DOMAINS,
    ):

        print(
            f"\n{domain.upper()}"
        )

        for ratio in RATIOS:

            row = next(
                item
                for item in aggregate
                if (
                    item["ratio"] == ratio
                    and item["domain"] == domain
                    and item["subtype"] == "all"
                )
            )

            print(
                f"Replay {ratio:2d}% : "
                f"{row['accuracy_mean']:.2f}% "
                f"± {row['accuracy_std']:.2f} | "
                f"format : "
                f"{row['formatted_mean']:.2f}% "
                f"± {row['formatted_std']:.2f}"
            )

    # ========================================================
    # 19. OUBLI EN ALGÈBRE ET GAIN EN CODE
    # ========================================================

    print("\n" + "=" * 65)
    print("OUBLI ET ACQUISITION")
    print("=" * 65)

    m2_maths = (
        100 * M2_CORRECT["maths"] / 30
    )

    m2_algebra = (
        100 * M2_CORRECT["algebre"] / 30
    )

    m2_code = (
        100 * M2_CORRECT["code"] / 30
    )

    for ratio in RATIOS:

        domain_scores = {}

        for domain in DOMAINS:

            row = next(
                item
                for item in aggregate
                if (
                    item["ratio"] == ratio
                    and item["scope"] == "domain"
                    and item["domain"] == domain
                )
            )

            domain_scores[
                domain
            ] = row["accuracy_mean"]

        maths_forgetting = max(
            0.0,
            m2_maths - domain_scores["maths"],
        )

        algebra_forgetting = max(
            0.0,
            m2_algebra - domain_scores["algebre"],
        )

        code_gain = (
            domain_scores["code"]
            - m2_code
        )

        print(
            f"\nReplay {ratio}%"
        )

        print(
            f"Oubli maths : "
            f"{maths_forgetting:.2f} points"
        )

        print(
            f"Oubli algèbre : "
            f"{algebra_forgetting:.2f} points"
        )

        print(
            f"Gain code : "
            f"{code_gain:+.2f} points"
        )

    # ========================================================
    # 20. IMPACT MOYEN DU REPLAY
    # ========================================================

    print("\n" + "=" * 65)
    print("IMPACT DU REPLAY PAR RAPPORT À 0%")
    print("=" * 65)

    for ratio in (10, 20):

        print(
            f"\nREPLAY {ratio}%"
        )

        for domain in (
            "global",
            *DOMAINS,
        ):

            row = next(
                item
                for item in paired_summary
                if (
                    item["ratio_after"] == ratio
                    and item["domain"] == domain
                )
            )

            print(
                f"{domain.upper():8s} : "
                f"{row['accuracy_delta_mean']:+.2f} "
                f"± {row['accuracy_delta_std']:.2f} "
                "points"
            )

    # ========================================================
    # 21. TYPES D'EXERCICES PYTHON
    # ========================================================

    print("\n" + "=" * 65)
    print("PROGRAMMATION - ANALYSE PAR OPÉRATION")
    print("=" * 65)

    code_subtypes = sorted({
        row["subtype"]
        for row in aggregate
        if (
            row["scope"] == "subtype"
            and row["domain"] == "code"
        )
    })

    for subtype in code_subtypes:

        print(
            f"\n{subtype.upper()}"
        )

        for ratio in RATIOS:

            row = next(
                item
                for item in aggregate
                if (
                    item["ratio"] == ratio
                    and item["domain"] == "code"
                    and item["subtype"] == subtype
                )
            )

            print(
                f"Replay {ratio:2d}% : "
                f"{row['accuracy_mean']:.2f}% "
                f"± {row['accuracy_std']:.2f}"
            )

    # ========================================================
    # 22. FIN
    # ========================================================

    print("\n" + "=" * 65)
    print("ÉVALUATION MULTI-SEED TERMINÉE")
    print("=" * 65)

    print(
        f"Résultats : {RESULTS_DIR}"
    )

    print(
        "Résumé : summary.json"
    )

    print(
        "Détails : summary_per_run.csv"
    )

    print(
        "Moyennes : summary_by_ratio.csv"
    )

    print(
        "Comparaisons : paired_summary.csv"
    )


if __name__ == "__main__":

    main()

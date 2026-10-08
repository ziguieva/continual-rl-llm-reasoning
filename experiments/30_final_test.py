import csv
import gc
import hashlib
import importlib.util
import json
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
# 1. CONFIGURATION FINALE - NE PLUS MODIFIER APRÈS TEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEST_FILE = (
    ROOT / "data/processed/test.jsonl"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

M2_ADAPTER = (
    ROOT / "models/grpo_maths_v2_then_algebra_sft"
)

MULTISEED_DIR = (
    ROOT / "models/replay_multiseed_v1"
)

RESULTS_DIR = (
    ROOT / "results/final_test_v1"
)

SEEDS = (
    42,
    123,
    2026,
)

# Le choix a été effectué exclusivement
# sur le jeu de validation.
RATIOS = (
    0,
    20,
)

DOMAINS = (
    "maths",
    "algebre",
    "code",
)

MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé en mathématiques, "
    "en algèbre et en programmation Python. "
    "Résous le problème avec précision. "
    "Tu peux expliquer brièvement ton raisonnement. "
    "Termine par une ligne au format REPONSE: nombre. "
    "N'ajoute aucun texte après cette ligne."
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("CONTINUAL RL - FINAL HELD-OUT TEST")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Seeds : {SEEDS}")
print(f"Ratios évalués : {RATIOS}")
print("Configuration sélectionnée : Replay 20 %")

# ============================================================
# 2. UTILITAIRES
# ============================================================

def as_bool(value):

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def mean_std(values):

    if not values:
        return 0.0, 0.0

    mean = statistics.mean(
        values
    )

    if len(values) > 1:

        std = statistics.stdev(
            values
        )

    else:

        std = 0.0

    return (
        round(mean, 2),
        round(std, 2),
    )


def save_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


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

# ============================================================
# 3. VÉRIFICATIONS
# ============================================================

required_files = [
    TEST_FILE,
    EVALUATOR_FILE,
    M2_ADAPTER / "adapter_config.json",
]

for seed in SEEDS:

    for ratio in RATIOS:

        adapter = (
            MULTISEED_DIR
            / f"seed_{seed}"
            / f"replay_{ratio}"
            / "adapter_config.json"
        )

        required_files.append(
            adapter
        )

for path in required_files:

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

print("Fichiers : OK")

# ============================================================
# 4. CHARGEMENT DE L'ÉVALUATEUR FIGÉ V16
# ============================================================

spec = importlib.util.spec_from_file_location(
    "audit_v16",
    EVALUATOR_FILE,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Impossible de charger "
        "l'évaluateur V16."
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
# 5. CHARGEMENT DU TEST HELD-OUT
# ============================================================

with TEST_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    dataset = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(dataset) != 150:

    raise ValueError(
        f"150 questions attendues, "
        f"{len(dataset)} trouvées."
    )

ids = [
    row["id"]
    for row in dataset
]

if len(ids) != len(set(ids)):

    raise ValueError(
        "Identifiants dupliqués "
        "dans le test."
    )

for domain in DOMAINS:

    count = sum(
        row["domain"] == domain
        for row in dataset
    )

    if count != 50:

        raise ValueError(
            f"{domain} : "
            f"50 questions attendues, "
            f"{count} trouvées."
        )

TEST_HASH = sha256_file(
    TEST_FILE
)

EVALUATOR_HASH = sha256_file(
    EVALUATOR_FILE
)

print(
    f"Questions de test : {len(dataset)}"
)

print(
    f"SHA256 test : {TEST_HASH}"
)

# ============================================================
# 6. SOUS-TYPES
# ============================================================

def classify_subtype(
    question,
    domain,
):

    if domain == "maths":

        expression = (
            question
            .split(":", 1)[-1]
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
            question
            .split(":", 1)[-1]
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
            question
            .strip()
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
# 7. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

# ============================================================
# 8. GÉNÉRATION DÉTERMINISTE
# ============================================================

def generate_response(
    model,
    question,
):

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
            pad_token_id=(
                tokenizer.pad_token_id
            ),
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
# 9. ÉVALUATION D'UN ADAPTATEUR
# ============================================================

def evaluate_adapter(
    label,
    adapter_dir,
    seed=None,
    ratio=None,
):

    run_dir = (
        RESULTS_DIR / label
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        run_dir / "predictions.csv"
    )

    metadata_file = (
        run_dir / "metadata.json"
    )

    # --------------------------------------------------------
    # Cache : on évite une seconde génération
    # accidentelle sur le test.
    # --------------------------------------------------------

    if (
        predictions_file.is_file()
        and metadata_file.is_file()
    ):

        print(
            f"\n{label} : "
            "résultats du test déjà présents."
        )

        with predictions_file.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            rows = list(
                csv.DictReader(file)
            )

        if len(rows) != 150:

            raise ValueError(
                f"Cache invalide pour {label}."
            )

        return rows

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"TEST FINAL : {label}"
    )

    print("=" * 70)

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

    predictions = []

    # --------------------------------------------------------
    # Inférence
    # --------------------------------------------------------

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        expected = int(
            example["answer"]
        )

        response, tokens, truncated = (
            generate_response(
                model,
                example["question"],
            )
        )

        evaluation = (
            audit.analyze_response(
                response,
                example["domain"],
                expected,
            )
        )

        predictions.append({
            "id": example["id"],

            "label": label,

            "seed": (
                seed
                if seed is not None
                else ""
            ),

            "ratio": (
                ratio
                if ratio is not None
                else ""
            ),

            "domain": (
                example["domain"]
            ),

            "subtype": classify_subtype(
                example["question"],
                example["domain"],
            ),

            "question": (
                example["question"]
            ),

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
        })

        if (
            index % 25 == 0
            or index == len(dataset)
        ):

            print(
                f"Progression : "
                f"{index}/{len(dataset)}"
            )

    # --------------------------------------------------------
    # Sauvegarde immédiate
    # --------------------------------------------------------

    save_csv(
        predictions_file,
        predictions,
    )

    metadata = {
        "label": label,
        "seed": seed,
        "ratio": ratio,
        "model": MODEL_NAME,
        "adapter": str(
            adapter_dir.resolve()
        ),
        "test_file": str(
            TEST_FILE.resolve()
        ),
        "test_hash": TEST_HASH,
        "evaluator_hash": (
            EVALUATOR_HASH
        ),
        "system_prompt": (
            SYSTEM_PROMPT
        ),
        "max_new_tokens": (
            MAX_NEW_TOKENS
        ),
        "do_sample": False,
        "dtype": "float32",
    }

    save_json(
        metadata_file,
        metadata,
    )

    del model
    del base_model

    gc.collect()

    if DEVICE == "mps":

        torch.mps.empty_cache()

    return predictions

# ============================================================
# 10. MÉTRIQUES
# ============================================================

def calculate_metrics(
    records,
):

    total = len(records)

    if total == 0:

        raise ValueError(
            "Aucune réponse."
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
        row["status"]
        == "contradictory"
        for row in records
    )

    missing = sum(
        row["status"]
        == "missing"
        for row in records
    )

    truncated = sum(
        as_bool(
            row["truncated"]
        )
        for row in records
    )

    total_tokens = sum(
        int(row["tokens"])
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
            100
            * conclusion_correct
            / total,
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
# 11. MÉTRIQUES PAR DOMAINE / SOUS-TYPE
# ============================================================

def build_metrics(
    records,
    label,
    seed=None,
    ratio=None,
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

        domain_rows = [
            row
            for row in records
            if row["domain"] == domain
        ]

        groups.append(
            (
                "domain",
                domain,
                "all",
                domain_rows,
            )
        )

        subtypes = sorted({
            row["subtype"]
            for row in domain_rows
        })

        for subtype in subtypes:

            subset = [
                row
                for row in domain_rows
                if row["subtype"]
                == subtype
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

        output.append({
            "label": label,
            "seed": (
                seed
                if seed is not None
                else ""
            ),
            "ratio": (
                ratio
                if ratio is not None
                else ""
            ),
            "scope": scope,
            "domain": domain,
            "subtype": subtype,
            **calculate_metrics(
                subset
            ),
        })

    return output

# ============================================================
# 12. ÉVALUATION M2
# ============================================================

all_predictions = {}

all_predictions["m2"] = (
    evaluate_adapter(
        label="m2",
        adapter_dir=M2_ADAPTER,
    )
)

# ============================================================
# 13. ÉVALUATION 0 % ET 20 % POUR LES TROIS SEEDS
# ============================================================

for ratio in RATIOS:

    for seed in SEEDS:

        label = (
            f"replay_{ratio}_seed_{seed}"
        )

        adapter_dir = (
            MULTISEED_DIR
            / f"seed_{seed}"
            / f"replay_{ratio}"
        )

        all_predictions[label] = (
            evaluate_adapter(
                label=label,
                adapter_dir=adapter_dir,
                seed=seed,
                ratio=ratio,
            )
        )

# ============================================================
# 14. RÉSUMÉ INDIVIDUEL
# ============================================================

per_run = []

per_run.extend(
    build_metrics(
        all_predictions["m2"],
        "m2",
    )
)

for ratio in RATIOS:

    for seed in SEEDS:

        label = (
            f"replay_{ratio}_seed_{seed}"
        )

        per_run.extend(
            build_metrics(
                all_predictions[label],
                label,
                seed,
                ratio,
            )
        )

# ============================================================
# 15. AGRÉGATION MULTI-SEED
# ============================================================

aggregated = []

for ratio in RATIOS:

    ratio_rows = [
        row
        for row in per_run
        if row["ratio"] == ratio
    ]

    group_keys = sorted({
        (
            row["scope"],
            row["domain"],
            row["subtype"],
        )
        for row in ratio_rows
    })

    for (
        scope,
        domain,
        subtype,
    ) in group_keys:

        group = [
            row
            for row in ratio_rows
            if (
                row["scope"] == scope
                and row["domain"] == domain
                and row["subtype"] == subtype
            )
        ]

        if len(group) != 3:

            raise ValueError(
                f"Trois seeds attendues : "
                f"{ratio} / {domain} / {subtype}"
            )

        accuracy_mean, accuracy_std = (
            mean_std([
                row["accuracy"]
                for row in group
            ])
        )

        formatted_mean, formatted_std = (
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

        aggregated.append({
            "ratio": ratio,
            "scope": scope,
            "domain": domain,
            "subtype": subtype,

            "questions_per_seed": (
                group[0]["questions"]
            ),

            "accuracy_mean": (
                accuracy_mean
            ),

            "accuracy_std": (
                accuracy_std
            ),

            "formatted_mean": (
                formatted_mean
            ),

            "formatted_std": (
                formatted_std
            ),

            "conclusion_mean": (
                conclusion_mean
            ),

            "conclusion_std": (
                conclusion_std
            ),
        })

# ============================================================
# 16. M2 COMME RÉFÉRENCE POUR L'OUBLI
# ============================================================

m2_domain_metrics = {}

for domain in DOMAINS:

    row = next(
        item
        for item in per_run
        if (
            item["label"] == "m2"
            and item["scope"] == "domain"
            and item["domain"] == domain
        )
    )

    m2_domain_metrics[
        domain
    ] = row

# ============================================================
# 17. RÉSULTATS PAR SEED
# ============================================================

print("\n" + "=" * 70)
print("RÉSULTATS TEST PAR GRAINE")
print("=" * 70)

for ratio in RATIOS:

    print(
        f"\nREPLAY {ratio}%"
    )

    for seed in SEEDS:

        label = (
            f"replay_{ratio}_seed_{seed}"
        )

        row = next(
            item
            for item in per_run
            if (
                item["label"] == label
                and item["scope"] == "global"
            )
        )

        print(
            f"Seed {seed:4d} : "
            f"{row['accuracy']:.2f}% | "
            f"format "
            f"{row['formatted_accuracy']:.2f}%"
        )

# ============================================================
# 18. RÉSULTATS FINAUX MOYENS
# ============================================================

print("\n" + "=" * 70)
print("RÉSULTATS FINAUX - TEST HELD-OUT")
print("=" * 70)

for domain in (
    "global",
    *DOMAINS,
):

    print(
        f"\n{domain.upper()}"
    )

    if domain != "global":

        m2 = (
            m2_domain_metrics[
                domain
            ]
        )

        print(
            f"M2        : "
            f"{m2['accuracy']:.2f}%"
        )

    for ratio in RATIOS:

        row = next(
            item
            for item in aggregated
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

# ============================================================
# 19. OUBLI ET ACQUISITION FINAUX
# ============================================================

print("\n" + "=" * 70)
print("FORGETTING / ACQUISITION - TEST FINAL")
print("=" * 70)

continual_results = []

for ratio in RATIOS:

    scores = {}

    for domain in DOMAINS:

        row = next(
            item
            for item in aggregated
            if (
                item["ratio"] == ratio
                and item["scope"] == "domain"
                and item["domain"] == domain
            )
        )

        scores[
            domain
        ] = row["accuracy_mean"]

    maths_forgetting = max(
        0.0,
        m2_domain_metrics[
            "maths"
        ]["accuracy"]
        - scores["maths"],
    )

    algebra_forgetting = max(
        0.0,
        m2_domain_metrics[
            "algebre"
        ]["accuracy"]
        - scores["algebre"],
    )

    code_gain = (
        scores["code"]
        - m2_domain_metrics[
            "code"
        ]["accuracy"]
    )

    continual_results.append({
        "ratio": ratio,
        "maths_forgetting": round(
            maths_forgetting,
            2,
        ),
        "algebra_forgetting": round(
            algebra_forgetting,
            2,
        ),
        "code_gain": round(
            code_gain,
            2,
        ),
    })

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

# ============================================================
# 20. EFFET DU REPLAY 20 % SUR TEST
# ============================================================

print("\n" + "=" * 70)
print("EFFET FINAL DU REPLAY 20% VS 0%")
print("=" * 70)

final_deltas = []

for domain in (
    "global",
    *DOMAINS,
):

    zero = next(
        item
        for item in aggregated
        if (
            item["ratio"] == 0
            and item["domain"] == domain
            and item["subtype"] == "all"
        )
    )

    replay = next(
        item
        for item in aggregated
        if (
            item["ratio"] == 20
            and item["domain"] == domain
            and item["subtype"] == "all"
        )
    )

    accuracy_delta = (
        replay["accuracy_mean"]
        - zero["accuracy_mean"]
    )

    format_delta = (
        replay["formatted_mean"]
        - zero["formatted_mean"]
    )

    final_deltas.append({
        "domain": domain,
        "accuracy_delta": round(
            accuracy_delta,
            2,
        ),
        "format_delta": round(
            format_delta,
            2,
        ),
    })

    print(
        f"{domain.upper():8s} : "
        f"accuracy "
        f"{accuracy_delta:+.2f} points | "
        f"format "
        f"{format_delta:+.2f} points"
    )

# ============================================================
# 21. PROGRAMMATION PAR OPÉRATION
# ============================================================

print("\n" + "=" * 70)
print("PROGRAMMATION - TEST FINAL PAR OPÉRATION")
print("=" * 70)

code_subtypes = sorted({
    row["subtype"]
    for row in aggregated
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
            for item in aggregated
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

# ============================================================
# 22. SAUVEGARDE DU RÉSUMÉ FINAL
# ============================================================

save_csv(
    RESULTS_DIR / "metrics_per_run.csv",
    per_run,
)

save_csv(
    RESULTS_DIR / "metrics_aggregated.csv",
    aggregated,
)

save_csv(
    RESULTS_DIR / "continual_metrics.csv",
    continual_results,
)

save_csv(
    RESULTS_DIR / "replay20_vs_0.csv",
    final_deltas,
)

final_summary = {
    "protocol_status": (
        "FINAL_HELD_OUT_TEST"
    ),

    "selection_rule": (
        "Replay 20% selected exclusively "
        "from validation multi-seed results."
    ),

    "test_file": str(
        TEST_FILE.resolve()
    ),

    "test_hash": TEST_HASH,

    "evaluator_hash": (
        EVALUATOR_HASH
    ),

    "model": MODEL_NAME,

    "seeds": list(
        SEEDS
    ),

    "ratios_evaluated": list(
        RATIOS
    ),

    "system_prompt": (
        SYSTEM_PROMPT
    ),

    "per_run": per_run,

    "aggregated": aggregated,

    "continual_learning": (
        continual_results
    ),

    "replay20_vs_0": (
        final_deltas
    ),
}

save_json(
    RESULTS_DIR / "final_summary.json",
    final_summary,
)

# ============================================================
# 23. FIN
# ============================================================

print("\n" + "=" * 70)
print("TEST FINAL TERMINÉ")
print("=" * 70)

print(
    f"Résultats : {RESULTS_DIR}"
)

print(
    "Résumé : final_summary.json"
)

print(
    "\nIMPORTANT : "
    "les résultats du test sont désormais ouverts. "
    "Ne pas modifier les hyperparamètres en fonction "
    "de ces résultats puis réévaluer sur ce même test."
)
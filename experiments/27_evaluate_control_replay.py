
import csv
import importlib.util
import json
from math import comb
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

CONTROL_ADAPTER = (
    ROOT
    / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_control96"
)

CONTROL_MANIFEST = (
    CONTROL_ADAPTER / "control_manifest.json"
)

REPLAY_MANIFEST = (
    ROOT
    / "models"
    / "grpo_maths_v2_then_algebra_sft_then_code_replay20"
    / "replay_manifest.json"
)

DATA_FILE = (
    ROOT / "data/processed/validation.jsonl"
)

TRAIN_FILE = (
    ROOT / "data/processed/train.jsonl"
)

PREVIOUS_RESULTS = (
    ROOT / "results/continual_replay20_comparison.csv"
)

EVALUATOR_FILE = (
    ROOT / "experiments/16_finalize_validation.py"
)

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULTS_DIR / "continual_control96_vs_replay20.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "continual_control96_vs_replay20.json"
)

REVIEW_CSV = (
    RESULTS_DIR / "continual_control96_vs_replay20_review.csv"
)

SEED = 42
MAX_NEW_TOKENS = 256

DOMAINS = [
    "maths",
    "algebre",
    "code",
]

MODELS = [
    "m2",
    "m3",
    "m3c",
    "m3r",
]

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

torch.manual_seed(SEED)

print("=" * 65)
print("CONTINUAL LEARNING - CONTROL VS EXPERIENCE REPLAY")
print("=" * 65)

print(f"Device : {DEVICE}")
print(f"Modèle témoin : {CONTROL_ADAPTER}")

# ============================================================
# 2. VÉRIFICATION DES FICHIERS
# ============================================================

required_files = [
    DATA_FILE,
    TRAIN_FILE,
    PREVIOUS_RESULTS,
    EVALUATOR_FILE,
    CONTROL_MANIFEST,
    REPLAY_MANIFEST,
    CONTROL_ADAPTER / "adapter_config.json",
]

for path in required_files:

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

print("Fichiers : OK")

# ============================================================
# 3. VÉRIFICATION DU PROTOCOLE EXPÉRIMENTAL
# ============================================================

with CONTROL_MANIFEST.open(
    "r",
    encoding="utf-8",
) as file:

    control_manifest = json.load(file)

with REPLAY_MANIFEST.open(
    "r",
    encoding="utf-8",
) as file:

    replay_manifest = json.load(file)

with TRAIN_FILE.open(
    "r",
    encoding="utf-8",
) as file:

    train_records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

train_by_id = {
    row["id"]: row
    for row in train_records
}

if len(train_by_id) != len(train_records):

    raise ValueError(
        "Identifiants dupliqués dans le train."
    )

control_ids = control_manifest["selected_ids"]
replay_ids = replay_manifest["selected_ids"]

if (
    len(control_ids) != 120
    or len(replay_ids) != 120
):

    raise ValueError(
        "Chaque expérience doit présenter "
        "120 exemples."
    )

if any(
    example_id not in train_by_id
    for example_id in control_ids + replay_ids
):

    raise ValueError(
        "Identifiants introuvables dans le train."
    )

control_code_ids = {
    example_id
    for example_id in control_ids
    if train_by_id[example_id]["domain"] == "code"
}

replay_code_ids = {
    example_id
    for example_id in replay_ids
    if train_by_id[example_id]["domain"] == "code"
}

if (
    control_code_ids != replay_code_ids
    or len(control_code_ids) != 96
):

    raise ValueError(
        "M3-C et M3-R n'utilisent pas "
        "les mêmes 96 exercices de code."
    )

if any(
    train_by_id[example_id]["domain"] != "code"
    for example_id in control_ids
):

    raise ValueError(
        "M3-C doit contenir uniquement du code."
    )

replay_old_count = sum(
    train_by_id[example_id]["domain"] != "code"
    for example_id in replay_ids
)

if replay_old_count != 24:

    raise ValueError(
        "M3-R doit contenir 24 anciens exercices."
    )

print(
    "Protocole vérifié : "
    "96 exercices de code identiques."
)

print(
    "M3-C : 24 répétitions de code."
)

print(
    "M3-R : 24 exercices de replay."
)

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
# 5. CHARGEMENT DES QUESTIONS ET DES RÉSULTATS
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

with PREVIOUS_RESULTS.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:

    previous = list(
        csv.DictReader(file)
    )

previous_by_id = {
    row["id"]: row
    for row in previous
}

dataset_ids = [
    row["id"]
    for row in dataset
]

if len(dataset_ids) != len(set(dataset_ids)):

    raise ValueError(
        "Identifiants dupliqués dans la validation."
    )

if len(previous) != len(previous_by_id):

    raise ValueError(
        "Identifiants dupliqués dans les résultats."
    )

if set(dataset_ids) != set(previous_by_id):

    raise ValueError(
        "Les questions de validation ont changé."
    )

for example in dataset:

    old = previous_by_id[
        example["id"]
    ]

    if (
        old["question"] != example["question"]
        or old["domain"] != example["domain"]
        or int(old["expected"])
        != int(example["answer"])
    ):

        raise ValueError(
            f"Question incohérente : {example['id']}"
        )

if len(dataset) != 90:

    raise ValueError(
        "90 questions de validation attendues."
    )

for domain in DOMAINS:

    count = sum(
        row["domain"] == domain
        for row in dataset
    )

    if count != 30:

        raise ValueError(
            f"{domain} : 30 questions attendues, "
            f"{count} trouvées."
        )

print(
    f"Questions de validation : {len(dataset)}"
)

# ============================================================
# 6. CHARGEMENT DE QWEN ET DE M3-C
# ============================================================

print("\nChargement de Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    attn_implementation="eager",
).to(DEVICE)

print("Chargement de M3-C...")

model = PeftModel.from_pretrained(
    base_model,
    str(CONTROL_ADAPTER),
)

model.to(DEVICE)

model.eval()

print("Modèle témoin chargé.")

# ============================================================
# 7. GÉNÉRATION DES RÉPONSES M3-C
# ============================================================

def generate_response(question):

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

    generated = output[0][input_length:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    token_count = len(generated)

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
# 8. ÉVALUATION DES QUATRE MODÈLES
# ============================================================

results = []

print("\nDébut de l'évaluation M3-C...\n")

for index, example in enumerate(
    dataset,
    start=1,
):

    example_id = example["id"]

    old = previous_by_id[
        example_id
    ]

    domain = example["domain"]

    expected = int(
        example["answer"]
    )

    control_response, control_tokens, truncated = (
        generate_response(
            example["question"]
        )
    )

    # Trois réponses sont déjà enregistrées.
    # Seule la réponse de M3-C est nouvelle.

    responses = {
        "m2": old["m2_response"],
        "m3": old["m3_response"],
        "m3c": control_response,
        "m3r": old["m3r_response"],
    }

    record = {
        "id": example_id,
        "domain": domain,
        "question": example["question"],
        "expected": expected,
    }

    for label, response in responses.items():

        evaluation = audit.analyze_response(
            response,
            domain,
            expected,
        )

        record[f"{label}_prediction"] = (
            evaluation["prediction"]
        )

        record[f"{label}_status"] = (
            evaluation["status"]
        )

        record[f"{label}_correct"] = bool(
            evaluation["strict_correct"]
        )

        record[f"{label}_format_ok"] = bool(
            evaluation["format_ok"]
        )

        record[f"{label}_end_to_end"] = bool(
            evaluation["end_to_end"]
        )

        # Cet indicateur reste distinct du score
        # strict et ne remplace pas l'audit humain.

        record[f"{label}_conclusion_correct"] = bool(
            evaluation.get(
                "conclusion_correct",
                False,
            )
        )

        record[f"{label}_response"] = (
            response
        )

    record["m2_tokens"] = int(
        old["m2_tokens"]
    )

    record["m3_tokens"] = int(
        old["m3_tokens"]
    )

    record["m3c_tokens"] = (
        control_tokens
    )

    record["m3r_tokens"] = int(
        old["m3r_tokens"]
    )

    record["m3c_truncated"] = (
        truncated
    )

    # Vérifie que le réexamen avec V16 donne
    # les mêmes scores pour les modèles
    # précédemment évalués.

    for label in [
        "m2",
        "m3",
        "m3r",
    ]:

        previous_correct = (
            str(
                old[f"{label}_correct"]
            ).strip().lower() == "true"
        )

        if (
            previous_correct
            != record[f"{label}_correct"]
        ):

            raise ValueError(
                f"Score historique différent "
                f"pour {example_id} / {label}."
            )

    results.append(record)

    if (
        index % 10 == 0
        or index == len(dataset)
    ):

        print(
            f"Progression : "
            f"{index}/{len(dataset)}"
        )

# ============================================================
# 9. TEST EXACT DE MCNEMAR
# ============================================================

def exact_mcnemar_p(improved, regressed):

    changed = improved + regressed

    if changed == 0:
        return 1.0

    k = min(
        improved,
        regressed,
    )

    probability = sum(
        comb(changed, i)
        for i in range(k + 1)
    ) / (2 ** changed)

    return min(
        1.0,
        2 * probability,
    )

# ============================================================
# 10. MÉTRIQUES INDIVIDUELLES
# ============================================================

def model_metrics(rows, label):

    total = len(rows)

    correct = sum(
        row[f"{label}_correct"]
        for row in rows
    )

    end_to_end = sum(
        row[f"{label}_end_to_end"]
        for row in rows
    )

    conclusions = sum(
        row[f"{label}_conclusion_correct"]
        for row in rows
    )

    contradictions = sum(
        row[f"{label}_status"]
        == "contradictory"
        for row in rows
    )

    missing = sum(
        row[f"{label}_status"]
        == "missing"
        for row in rows
    )

    tokens = sum(
        row[f"{label}_tokens"]
        for row in rows
    )

    return {
        "questions": total,

        "correct": correct,

        "accuracy": round(
            100 * correct / total,
            2,
        ) if total else 0.0,

        "correct_and_formatted": end_to_end,

        "conclusions_recognized_as_correct": (
            conclusions
        ),

        "contradictions": contradictions,

        "missing": missing,

        "tokens": tokens,
    }

# ============================================================
# 11. MÉTRIQUES APPARIÉES
# ============================================================

def transition_metrics(
    rows,
    before,
    after,
):

    improved = sum(
        not row[f"{before}_correct"]
        and row[f"{after}_correct"]
        for row in rows
    )

    regressed = sum(
        row[f"{before}_correct"]
        and not row[f"{after}_correct"]
        for row in rows
    )

    before_correct = sum(
        row[f"{before}_correct"]
        for row in rows
    )

    after_correct = sum(
        row[f"{after}_correct"]
        for row in rows
    )

    total = len(rows)

    delta = (
        100
        * (after_correct - before_correct)
        / total
        if total
        else 0.0
    )

    return {
        "improved": improved,

        "regressed": regressed,

        "accuracy_delta_points": round(
            delta,
            2,
        ),

        "mcnemar_exact_p": round(
            exact_mcnemar_p(
                improved,
                regressed,
            ),
            6,
        ),
    }

# ============================================================
# 12. RÉSUMÉ DES MÉTRIQUES
# ============================================================

def calculate_metrics(rows):

    return {
        "questions": len(rows),

        "models": {
            label: model_metrics(
                rows,
                label,
            )
            for label in MODELS
        },

        "comparisons": {
            "m2_to_m3c": transition_metrics(
                rows,
                "m2",
                "m3c",
            ),

            "m2_to_m3r": transition_metrics(
                rows,
                "m2",
                "m3r",
            ),

            "m3_to_m3c": transition_metrics(
                rows,
                "m3",
                "m3c",
            ),

            "m3c_to_m3r": transition_metrics(
                rows,
                "m3c",
                "m3r",
            ),
        },
    }

summary = {
    "model": MODEL_NAME,

    "dataset": "validation",

    "evaluator": "V16",

    "seed": SEED,

    "max_new_tokens": MAX_NEW_TOKENS,

    "system_prompt": SYSTEM_PROMPT,

    "controlled_comparison": {
        "control_model": "m3c",

        "replay_model": "m3r",

        "common_unique_code_examples": 96,

        "control_code_repetitions": 24,

        "replay_old_examples": 24,

        "training_steps": 60,
    },

    "domains": {},

    "global": calculate_metrics(
        results
    ),
}

# ============================================================
# 13. COMPARAISON PAR DOMAINE
# ============================================================

print("\n" + "=" * 65)

print(
    "COMPARAISON PAR DOMAINE"
)

print("=" * 65)

for domain in DOMAINS:

    subset = [
        row
        for row in results
        if row["domain"] == domain
    ]

    metrics = calculate_metrics(
        subset
    )

    summary["domains"][domain] = (
        metrics
    )

    print(
        f"\n{domain.upper()}"
    )

    for label in MODELS:

        current = (
            metrics["models"][label]
        )

        print(
            f"{label.upper():4s} : "
            f"{current['accuracy']:.2f}% "
            f"({current['correct']}/{len(subset)})"
        )

    comparison = (
        metrics["comparisons"]["m3c_to_m3r"]
    )

    print(
        f"M3-C -> M3-R : "
        f"{comparison['accuracy_delta_points']:+.2f} "
        "points"
    )

    print(
        "Améliorées / dégradées : "
        f"{comparison['improved']} / "
        f"{comparison['regressed']}"
    )

    print(
        "Test McNemar exact : "
        f"p = {comparison['mcnemar_exact_p']:.4f}"
    )

    print(
        "Exactitude + format M3-C/M3-R : "
        f"{metrics['models']['m3c']['correct_and_formatted']}"
        " / "
        f"{metrics['models']['m3r']['correct_and_formatted']}"
    )

# ============================================================
# 14. OUBLI ET ACQUISITION DES COMPÉTENCES
# ============================================================

maths = summary["domains"]["maths"][
    "models"
]

algebra = summary["domains"]["algebre"][
    "models"
]

code = summary["domains"]["code"][
    "models"
]

continual = {}

for label in [
    "m3",
    "m3c",
    "m3r",
]:

    maths_forgetting = max(
        0.0,
        maths["m2"]["accuracy"]
        - maths[label]["accuracy"],
    )

    algebra_forgetting = max(
        0.0,
        algebra["m2"]["accuracy"]
        - algebra[label]["accuracy"],
    )

    code_gain = (
        code[label]["accuracy"]
        - code["m2"]["accuracy"]
    )

    continual[label] = {
        "maths_forgetting_points": round(
            maths_forgetting,
            2,
        ),

        "algebra_forgetting_points": round(
            algebra_forgetting,
            2,
        ),

        "code_gain_points": round(
            code_gain,
            2,
        ),
    }

summary["continual_learning"] = (
    continual
)

print("\n" + "=" * 65)

print(
    "OUBLI ET ACQUISITION"
)

print("=" * 65)

for label in [
    "m3",
    "m3c",
    "m3r",
]:

    metrics = continual[label]

    print(
        f"\n{label.upper()}"
    )

    print(
        f"Oubli maths : "
        f"{metrics['maths_forgetting_points']:.2f} points"
    )

    print(
        f"Oubli algèbre : "
        f"{metrics['algebra_forgetting_points']:.2f} points"
    )

    print(
        f"Gain code : "
        f"{metrics['code_gain_points']:+.2f} points"
    )

# ============================================================
# 15. BILAN GLOBAL
# ============================================================

global_metrics = (
    summary["global"]
)

print("\n" + "=" * 65)

print(
    "BILAN GLOBAL - CONTRÔLE VS REPLAY"
)

print("=" * 65)

for label in MODELS:

    metrics = (
        global_metrics["models"][label]
    )

    print(
        f"\n{label.upper()} : "
        f"{metrics['accuracy']:.2f}% "
        f"({metrics['correct']}/{len(results)})"
    )

    print(
        f"  Exactitude + format : "
        f"{metrics['correct_and_formatted']}"
    )

    print(
        f"  Contradictions : "
        f"{metrics['contradictions']}"
    )

    print(
        f"  Réponses manquantes : "
        f"{metrics['missing']}"
    )

    print(
        f"  Tokens générés : "
        f"{metrics['tokens']}"
    )

comparison = (
    global_metrics["comparisons"]["m3c_to_m3r"]
)

print("\n" + "-" * 65)

print(
    f"M3-C -> M3-R : "
    f"{comparison['accuracy_delta_points']:+.2f} points"
)

print(
    f"Améliorées / dégradées : "
    f"{comparison['improved']} / "
    f"{comparison['regressed']}"
)

print(
    f"Test de McNemar exact : "
    f"p = {comparison['mcnemar_exact_p']:.4f}"
)

truncated_count = sum(
    row["m3c_truncated"]
    for row in results
)

print(
    f"Réponses M3-C tronquées : "
    f"{truncated_count}"
)

# ============================================================
# 16. SAUVEGARDE DES RÉSULTATS
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        results
    )

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False,
    )

# ============================================================
# 17. CAS À EXAMINER
# ============================================================

review = [
    row
    for row in results
    if (
        row["m3c_correct"]
        != row["m3r_correct"]

        or row["m3c_end_to_end"]
        != row["m3r_end_to_end"]

        or row["m3c_status"] in {
            "contradictory",
            "missing",
            "non_integer",
        }

        or row["m3r_status"] in {
            "contradictory",
            "missing",
            "non_integer",
        }

        or row["m3c_truncated"]
    )
]

with REVIEW_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        review
    )

# ============================================================
# 18. FIN
# ============================================================

print("\n" + "=" * 65)

print(
    "ÉVALUATION CONTRÔLÉE TERMINÉE"
)

print("=" * 65)

print(
    f"CSV : {OUTPUT_CSV}"
)

print(
    f"JSON : {OUTPUT_JSON}"
)

print(
    f"Audit : {REVIEW_CSV}"
)

print(
    f"Questions à vérifier : "
    f"{len(review)}"
)


import argparse
import hashlib
import importlib.util
import json
import random

from collections import Counter
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

ADAPTER_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "final_adapter"
)

TRAIN_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "train.jsonl"
)

KEYS_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "grading"
    / "train_keys.jsonl"
)

EVALUATOR_FILE = (
    ROOT
    / "experiments"
    / "36_evaluate_math_qwen.py"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_grpo_prescreen"
)

METADATA_FILE = OUTPUT_DIR / "metadata.json"

GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"

SUMMARY_FILE = OUTPUT_DIR / "summary.json"

SEED = 42

POOL_SIZE = 96

NUM_GENERATIONS = 4

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.9

TOP_P = 0.9

DEVICE = "mps"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# 2. UTILITAIRES
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

    identifiers = [
        row["id"]
        for row in rows
    ]

    if len(identifiers) != len(set(identifiers)):

        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return rows


def index_by_id(rows):

    return {
        row["id"]: row
        for row in rows
    }


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(block)

    return digest.hexdigest()


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


def append_jsonl(path, record):

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

# ============================================================
# 3. CHARGER LE VÉRIFICATEUR EXISTANT
# ============================================================

def load_evaluator():

    spec = importlib.util.spec_from_file_location(
        "math_evaluator_v2",
        EVALUATOR_FILE,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Impossible de charger le vérificateur."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module

# ============================================================
# 4. PRÉPARER LE POOL D'ENTRAÎNEMENT
# ============================================================

def build_pool():

    questions = load_jsonl(
        TRAIN_FILE
    )

    corrections = index_by_id(
        load_jsonl(KEYS_FILE)
    )

    gsm8k = [
        row
        for row in questions
        if row["domain"] == "maths_appliques"
    ]

    if len(gsm8k) != 600:

        raise ValueError(
            "600 exemples GSM8K attendus."
        )

    rng = random.Random(
        SEED
    )

    selected = rng.sample(
        gsm8k,
        POOL_SIZE,
    )

    pool = []

    for row in selected:

        example_id = row["id"]

        correction = corrections[
            example_id
        ]

        pool.append({
            "id": example_id,
            "question": row["question"],
            "expected": str(
                correction["expected"]
            ),
        })

    return pool

# ============================================================
# 5. CHARGEMENT DU MODÈLE SFT
# ============================================================

def load_model():

    print(
        "\nChargement de Qwen + SFT..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR),
        is_trainable=False,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    model.config.use_cache = True

    return model, tokenizer

# ============================================================
# 6. QUATRE GÉNÉRATIONS PAR PROBLÈME
# ============================================================

def evaluate_question(
    model,
    tokenizer,
    evaluator,
    row,
    question_index,
):

    # Chaque problème possède sa propre
    # graine afin de permettre une reprise
    # sans dépendre des précédents.

    torch.manual_seed(
        SEED + question_index
    )

    messages = [
        {
            "role": "system",
            "content": evaluator.SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": row["question"],
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

        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=NUM_GENERATIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    gold = evaluator.parse_gold(
        row["expected"]
    )

    if not gold:

        raise RuntimeError(
            "Correction non interprétable : "
            + row["id"]
        )

    generations = []

    for output in outputs:

        generated = output[
            prompt_length:
        ]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        grade = evaluator.grade_math(
            response,
            gold,
        )

        tokens = len(
            generated
        )

        last_token = (
            int(generated[-1].item())
            if tokens
            else None
        )

        truncated = (
            tokens >= MAX_NEW_TOKENS
            and last_token not in {
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            }
        )

        correct = bool(
            grade["correct"]
        )

        format_ok = bool(
            grade["format_ok"]
        )

        reward = (
            float(correct)
            + 0.02 * float(format_ok)
        )

        generations.append({
            "response": response,
            "tokens": tokens,
            "truncated": truncated,
            "correct": correct,
            "format_ok": format_ok,
            "status": grade["status"],
            "reward": reward,
            "extracted_answer": (
                grade["extracted_answer"]
            ),
        })

    correct_count = sum(
        int(item["correct"])
        for item in generations
    )

    rewards = [
        item["reward"]
        for item in generations
    ]

    # Signal de raisonnement :
    # au moins une réponse correcte
    # et au moins une réponse incorrecte.

    correctness_signal = (
        0 < correct_count < NUM_GENERATIONS
    )

    return {
        "id": row["id"],
        "question": row["question"],
        "expected": row["expected"],
        "correct_count": correct_count,
        "correctness_signal": correctness_signal,
        "reward_variation": (
            len(set(rewards)) > 1
        ),
        "generations": generations,
    }

# ============================================================
# 7. BILAN
# ============================================================

def summarize(records):

    total_groups = len(
        records
    )

    total_generations = (
        total_groups * NUM_GENERATIONS
    )

    correct_generations = sum(
        row["correct_count"]
        for row in records
    )

    correctness_signal = sum(
        row["correctness_signal"]
        for row in records
    )

    no_correct = sum(
        row["correct_count"] == 0
        for row in records
    )

    all_correct = sum(
        row["correct_count"] == NUM_GENERATIONS
        for row in records
    )

    format_only_signal = sum(
        row["correct_count"] == 0
        and row["reward_variation"]
        for row in records
    )

    truncated = sum(
        item["truncated"]
        for row in records
        for item in row["generations"]
    )

    distribution = Counter(
        row["correct_count"]
        for row in records
    )

    candidate_ids = [
        row["id"]
        for row in records
        if row["correctness_signal"]
    ]

    summary = {
        "partition": "train",
        "groups": total_groups,
        "generations": total_generations,
        "correct_generations": (
            correct_generations
        ),
        "correctness_signal_groups": (
            correctness_signal
        ),
        "no_correct_groups": no_correct,
        "all_correct_groups": all_correct,
        "format_only_signal_groups": (
            format_only_signal
        ),
        "truncated_generations": truncated,
        "correct_per_group": dict(
            distribution
        ),
        "candidate_ids": candidate_ids,
        "training_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )

    print("\n" + "=" * 65)
    print("BILAN DU PRÉ-DIAGNOSTIC GRPO")
    print("=" * 65)

    print(
        f"Problèmes analysés : "
        f"{total_groups}"
    )

    print(
        f"Générations : "
        f"{total_generations}"
    )

    print(
        "Réponses correctes : "
        f"{correct_generations}"
    )

    print(
        "Groupes avec signal de justesse : "
        f"{correctness_signal}/{total_groups}"
    )

    print(
        "Groupes sans réponse correcte : "
        f"{no_correct}"
    )

    print(
        "Groupes entièrement corrects : "
        f"{all_correct}"
    )

    print(
        "Groupes avec signal de format "
        "uniquement : "
        f"{format_only_signal}"
    )

    print(
        "Générations tronquées : "
        f"{truncated}"
    )

    print(
        "Distribution des réponses "
        "correctes par groupe : "
        f"{dict(sorted(distribution.items()))}"
    )

    print(
        f"\nRésumé : {SUMMARY_FILE}"
    )

# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Nombre de problèmes à analyser.",
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= POOL_SIZE:

        raise ValueError(
            f"--limit doit être entre "
            f"1 et {POOL_SIZE}."
        )

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "GPU MPS indisponible."
        )

    print("=" * 65)
    print("BENCHMARK V2 - PRÉ-DIAGNOSTIC GRPO")
    print("=" * 65)

    evaluator = load_evaluator()

    pool = build_pool()

    selected = pool[
        :args.limit
    ]

    metadata = {
        "model": MODEL_NAME,
        "adapter_sha256": sha256_file(
            ADAPTER_DIR
            / "adapter_model.safetensors"
        ),
        "evaluator_sha256": sha256_file(
            EVALUATOR_FILE
        ),
        "train_sha256": sha256_file(
            TRAIN_FILE
        ),
        "keys_sha256": sha256_file(
            KEYS_FILE
        ),
        "seed": SEED,
        "pool_ids": [
            row["id"]
            for row in pool
        ],
        "num_generations": NUM_GENERATIONS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "device": DEVICE,
    }

    if METADATA_FILE.exists():

        previous = json.loads(
            METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )

        if previous != metadata:

            raise RuntimeError(
                "Configuration différente. "
                "Ne mélange pas les expériences."
            )

    else:

        if GENERATIONS_FILE.exists():

            raise RuntimeError(
                "Générations existantes "
                "sans métadonnées."
            )

        save_json(
            METADATA_FILE,
            metadata,
        )

    if GENERATIONS_FILE.exists():

        cached_rows = load_jsonl(
            GENERATIONS_FILE
        )

    else:

        cached_rows = []

    cached = index_by_id(
        cached_rows
    )

    missing = [
        row
        for row in selected
        if row["id"] not in cached
    ]

    if missing:

        model, tokenizer = (
            load_model()
        )

    else:

        model = None
        tokenizer = None

    for index, row in enumerate(
        selected
    ):

        example_id = row["id"]

        if example_id not in cached:

            result = evaluate_question(
                model,
                tokenizer,
                evaluator,
                row,
                index,
            )

            append_jsonl(
                GENERATIONS_FILE,
                result,
            )

            cached[
                example_id
            ] = result

        result = cached[
            example_id
        ]

        print(
            f"[{index + 1}/{len(selected)}] "
            f"{example_id} : "
            f"{result['correct_count']}/4 "
            "correctes | "
            "signal="
            f"{result['correctness_signal']}"
        )

    records = [
        cached[row["id"]]
        for row in selected
    ]

    summarize(
        records
    )


if __name__ == "__main__":

    main()

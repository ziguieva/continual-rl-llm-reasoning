
import importlib.util
import json

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

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = (
    ROOT
    / "models/benchmark_v2_sft_stage0/final_adapter"
)

SMOKE_CONFIG = (
    ROOT
    / "models/benchmark_v2_grpo_fourgen_smoke/run_config.json"
)

TRAIN_FILE = (
    ROOT / "data/benchmark_v2_v1/train.jsonl"
)

KEYS_FILE = (
    ROOT
    / "data/benchmark_v2_v1/grading/train_keys.jsonl"
)

EVALUATOR_FILE = (
    ROOT / "experiments/36_evaluate_math_qwen.py"
)

OUTPUT_FILE = (
    ROOT
    / "results/grpo_fourgen_smoke_diagnostic.json"
)

DEVICE = "mps"
SEED = 42

torch.manual_seed(SEED)


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

def load_jsonl(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return {
            row["id"]: row
            for line in file
            if (row := json.loads(line))
        }


def load_evaluator():

    spec = importlib.util.spec_from_file_location(
        "math_evaluator",
        EVALUATOR_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Vérificateur introuvable."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. CHARGEMENT DE QWEN + SFT
# ============================================================

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR),
        is_trainable=False,
    )

    model = model.to(DEVICE)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer


# ============================================================
# 4. GÉNÉRATION ET CORRECTION
# ============================================================

def generate(
    model,
    tokenizer,
    evaluator,
    question,
    expected,
    sampled,
):

    messages = [
        {
            "role": "system",
            "content": evaluator.SYSTEM_PROMPT,
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

    max_tokens = 256 if sampled else 512

    count = 4 if sampled else 1

    generation_options = {
        "max_new_tokens": max_tokens,
        "num_return_sequences": count,
        "do_sample": sampled,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if sampled:
        generation_options.update({
            "temperature": 0.9,
            "top_p": 0.9,
        })

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            **generation_options,
        )

    prompt_length = inputs[
        "input_ids"
    ].shape[1]

    gold = evaluator.parse_gold(
        expected
    )

    records = []

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

        token_count = len(generated)

        final_token = (
            int(generated[-1].item())
            if token_count
            else None
        )

        truncated = (
            token_count >= max_tokens
            and final_token
            not in {
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            }
        )

        records.append({
            "response": response,
            "tokens": token_count,
            "truncated": truncated,
            "status": grade["status"],
            "correct": grade["correct"],
            "format_ok": grade["format_ok"],
            "extracted_answer": grade[
                "extracted_answer"
            ],
            "reward": (
                float(grade["correct"])
                + 0.02 * float(
                    grade["format_ok"]
                )
            ),
        })

    return records


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("DIAGNOSTIC GRPO - QUATRE GÉNÉRATIONS")
    print("=" * 65)

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    metadata = json.loads(
        SMOKE_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    selected_ids = metadata[
        "selected_ids"
    ]

    if len(selected_ids) != 2:
        raise RuntimeError(
            "Deux questions attendues."
        )

    questions = load_jsonl(
        TRAIN_FILE
    )

    keys = load_jsonl(
        KEYS_FILE
    )

    evaluator = load_evaluator()

    model, tokenizer = load_model()

    report = {
        "model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "source": "train uniquement",
        "seed": SEED,
        "questions": [],
    }

    for example_id in selected_ids:

        question = questions[
            example_id
        ]["question"]

        expected = str(
            keys[example_id]["expected"]
        )

        target = questions[
            example_id
        ]["target"]

        print("\n" + "-" * 65)
        print("ID :", example_id)
        print("QUESTION :", question)
        print("ATTENDU :", expected)

        reference_tokens = len(
            tokenizer.encode(
                target,
                add_special_tokens=False,
            )
        )

        print(
            "Tokens de la solution officielle :",
            reference_tokens,
        )

        sampled = generate(
            model,
            tokenizer,
            evaluator,
            question,
            expected,
            sampled=True,
        )

        greedy = generate(
            model,
            tokenizer,
            evaluator,
            question,
            expected,
            sampled=False,
        )

        print("\nQUATRE GÉNÉRATIONS ÉCHANTILLONNÉES")

        for index, row in enumerate(
            sampled,
            start=1,
        ):

            print(
                f"{index}. "
                f"statut={row['status']}, "
                f"récompense={row['reward']:.2f}, "
                f"tokens={row['tokens']}, "
                f"tronquée={row['truncated']}"
            )

            print(
                "Réponse extraite :",
                row["extracted_answer"],
            )

            print(
                "Fin de génération :",
                row["response"][-350:],
            )

        print("\nGÉNÉRATION DÉTERMINISTE À 512 TOKENS")

        print(
            f"Statut : {greedy[0]['status']}"
        )

        print(
            f"Tokens : {greedy[0]['tokens']}"
        )

        print(
            f"Tronquée : {greedy[0]['truncated']}"
        )

        print(
            "Réponse extraite :",
            greedy[0]["extracted_answer"],
        )

        print(
            "Fin de génération :",
            greedy[0]["response"][-350:],
        )

        report["questions"].append({
            "id": example_id,
            "question": question,
            "expected": expected,
            "reference_solution_tokens": (
                reference_tokens
            ),
            "sampled_256": sampled,
            "greedy_512": greedy[0],
        })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("DIAGNOSTIC TERMINÉ")
    print("=" * 65)

    print(
        "Rapport :",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

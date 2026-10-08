
import hashlib
import importlib.util
import inspect
import json
import random

from collections import Counter
from pathlib import Path

import torch

from datasets import Dataset

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "final_adapter"
)

SFT_COMPLETION = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "completed.json"
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
    / "models"
    / "benchmark_v2_grpo_stage0_pilot"
)

FINAL_ADAPTER = (
    OUTPUT_DIR
    / "final_adapter"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "run_config.json"
)

COMPLETION_FILE = (
    OUTPUT_DIR
    / "completed.json"
)

REWARD_REPORT = (
    OUTPUT_DIR
    / "reward_report.json"
)

TRAINING_LOG = (
    OUTPUT_DIR
    / "training_log.json"
)

SEED = 42

NUM_QUESTIONS = 16

NUM_GENERATIONS = 2

MAX_PROMPT_TOKENS = 384

MAX_COMPLETION_TOKENS = 256

MAX_STEPS = 4

LEARNING_RATE = 1e-5

DEVICE = "mps"

REWARD_CORRECT = 1.0

REWARD_FORMAT = 0.02

reward_history = []

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. UTILITAIRES
# ============================================================

def load_jsonl(path):

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier absent : {path}"
        )

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

    if len(identifiers) != len(
        set(identifiers)
    ):

        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return rows


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


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. IMPORTER LE VÉRIFICATEUR DE L'ÉTAPE 36
# ============================================================

def load_evaluator():

    spec = (
        importlib.util.spec_from_file_location(
            "math_evaluator_v2",
            EVALUATOR_FILE,
        )
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Impossible de charger "
            "le vérificateur mathématique."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


evaluator = load_evaluator()


# ============================================================
# 4. VÉRIFICATIONS PRÉALABLES
# ============================================================

def check_environment():

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    required = [
        TRAIN_FILE,
        KEYS_FILE,
        EVALUATOR_FILE,
        SFT_COMPLETION,
        ADAPTER_DIR / "adapter_config.json",
        ADAPTER_DIR / "adapter_model.safetensors",
    ]

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    with SFT_COMPLETION.open(
        "r",
        encoding="utf-8",
    ) as file:

        completion = json.load(
            file
        )

    if completion.get("mode") != "all":

        raise RuntimeError(
            "L'adaptateur SFT complet "
            "est requis."
        )

    if completion.get("trained_examples") != 600:

        raise RuntimeError(
            "L'adaptateur doit provenir "
            "des 600 exemples GSM8K."
        )

    if completion.get("global_step") != 75:

        raise RuntimeError(
            "75 étapes SFT attendues."
        )

    tests_passed = (
        evaluator.run_verifier_tests()
    )

    print(
        "Tests du vérificateur : "
        f"{tests_passed}/{tests_passed} OK"
    )


# ============================================================
# 5. PRÉPARATION DU DATASET GRPO
# ============================================================

def build_dataset(tokenizer):

    questions = load_jsonl(
        TRAIN_FILE
    )

    corrections = {
        row["id"]: row
        for row in load_jsonl(
            KEYS_FILE
        )
    }

    selected = [
        row
        for row in questions
        if row["domain"] == "maths_appliques"
    ]

    if len(selected) != 600:

        raise ValueError(
            "600 problèmes GSM8K attendus."
        )

    eligible = []

    excluded = Counter()

    for row in selected:

        example_id = row["id"]

        if example_id not in corrections:

            raise ValueError(
                f"Correction absente : {example_id}"
            )

        expected = str(
            corrections[
                example_id
            ]["expected"]
        ).strip()

        question = row[
            "question"
        ]

        gold = evaluator.parse_gold(
            expected
        )

        if not gold:

            excluded[
                "gold_unparsed"
            ] += 1

            continue

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

        # Le prompt est déjà au format
        # conversationnel Qwen.
        #
        # TRL reçoit une chaîne afin
        # de ne pas appliquer deux fois
        # le template.

        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(prompt_ids) > MAX_PROMPT_TOKENS:

            excluded[
                "prompt_too_long"
            ] += 1

            continue

        eligible.append({
            "id": example_id,
            "prompt": prompt,
            "expected": expected,
        })

    if len(eligible) < NUM_QUESTIONS:

        raise RuntimeError(
            "Nombre insuffisant "
            "de questions exploitables."
        )

    # Échantillonnage reproductible,
    # uniquement dans le train officiel.

    selected_pilot = random.Random(
        SEED
    ).sample(
        eligible,
        NUM_QUESTIONS,
    )

    print(
        f"Questions GSM8K disponibles : "
        f"{len(selected)}"
    )

    print(
        f"Questions exploitables : "
        f"{len(eligible)}"
    )

    print(
        f"Questions du pilote : "
        f"{len(selected_pilot)}"
    )

    print(
        f"Exclusions : "
        f"{dict(excluded)}"
    )

    dataset = Dataset.from_list(
        selected_pilot
    )

    return (
        dataset,
        [
            row["id"]
            for row in selected_pilot
        ],
    )


# ============================================================
# 6. RÉCOMPENSE MATHÉMATIQUE VÉRIFIABLE
# ============================================================

def grade_completion(
    response,
    expected,
):

    gold = evaluator.parse_gold(
        str(expected)
    )

    if not gold:

        raise RuntimeError(
            "Réponse officielle "
            "non interprétable."
        )

    result = evaluator.grade_math(
        response,
        gold,
    )

    correct = bool(
        result["correct"]
    )

    format_ok = bool(
        result["format_ok"]
    )

    # La justesse est prioritaire.
    # Le format ne représente
    # que 2 % de la récompense.

    reward = (
        REWARD_CORRECT * int(correct)
        + REWARD_FORMAT * int(format_ok)
    )

    return {
        "reward": reward,
        "correct": correct,
        "format_ok": format_ok,
        "status": result["status"],
    }


def reward_function(
    completions,
    expected,
    **kwargs,
):

    if len(completions) != len(expected):

        raise ValueError(
            "Nombre de générations "
            "différent du nombre "
            "de corrections."
        )

    rewards = []

    correct_count = 0

    format_count = 0

    statuses = Counter()

    for response, gold in zip(
        completions,
        expected,
    ):

        # Les prompts du dataset sont
        # des chaînes, donc les
        # complétions doivent être
        # des chaînes également.

        if not isinstance(
            response,
            str,
        ):

            raise TypeError(
                "Complétion texte attendue."
            )

        result = grade_completion(
            response,
            gold,
        )

        rewards.append(
            result["reward"]
        )

        correct_count += int(
            result["correct"]
        )

        format_count += int(
            result["format_ok"]
        )

        statuses[
            result["status"]
        ] += 1

    record = {
        "completions": len(completions),
        "correct": correct_count,
        "format_ok": format_count,
        "statuses": dict(statuses),
        "rewards": rewards,
    }

    reward_history.append(
        record
    )

    print(
        "\nRécompenses GRPO : "
        f"{rewards} | "
        f"correctes : "
        f"{correct_count}/{len(completions)}"
    )

    return rewards


# ============================================================
# 7. TESTS DE LA RÉCOMPENSE
# ============================================================

def test_reward():

    tests = [
        (
            "REPONSE: \\boxed{42}",
            "42",
            1.02,
        ),
        (
            "REPONSE: \\boxed{41}",
            "42",
            0.02,
        ),
        (
            "The result is \\boxed{42}.",
            "42",
            1.0,
        ),
        (
            "No final answer.",
            "42",
            0.0,
        ),
    ]

    passed = 0

    for response, expected, target in tests:

        result = grade_completion(
            response,
            expected,
        )

        if abs(
            result["reward"] - target
        ) > 1e-6:

            raise AssertionError(
                "Récompense incorrecte : "
                f"{response}"
            )

        passed += 1

    return passed


# ============================================================
# 8. CHARGEMENT DE L'ADAPTATEUR SFT
# ============================================================

def load_model():

    print(
        "\nChargement du modèle "
        "et de l'adaptateur SFT..."
    )

    base = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    )

    base.config.use_cache = False

    # Reprendre l'adaptateur SFT
    # avec des paramètres LoRA
    # à nouveau entraînables.

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR),
        is_trainable=True,
    )

    model.config.use_cache = False

    model = model.to(
        DEVICE
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Paramètres entraînables : "
        f"{trainable:,}"
    )

    if trainable != 540672:

        raise RuntimeError(
            "Nombre inattendu "
            "de paramètres LoRA."
        )

    return model


# ============================================================
# 9. CONFIGURATION GRPO COMPATIBLE
# ============================================================

def build_grpo_config():

    requested = {
        "output_dir": str(OUTPUT_DIR),

        "max_steps": MAX_STEPS,

        "per_device_train_batch_size": 1,

        "gradient_accumulation_steps": 2,

        "generation_batch_size": 2,

        "num_generations": (
            NUM_GENERATIONS
        ),

        "max_completion_length": (
            MAX_COMPLETION_TOKENS
        ),

        "temperature": 0.9,

        "top_p": 0.9,

        "learning_rate": LEARNING_RATE,

        "lr_scheduler_type": "constant",

        "warmup_steps": 0,

        "weight_decay": 0.0,

        "optim": "adamw_torch",

        "max_grad_norm": 1.0,

        "gradient_checkpointing": True,

        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },

        "fp16": False,

        "bf16": False,

        # Pas de modèle de référence
        # supplémentaire en mémoire.

        "beta": 0.0,

        # Formulation DAPO de la loss,
        # avec avantages calculés
        # au niveau du groupe.

        "loss_type": "dapo",

        "scale_rewards": "group",

        "num_iterations": 1,

        # Génération locale classique.
        # Aucun serveur vLLM.

        "use_vllm": False,

        "remove_unused_columns": False,

        "logging_strategy": "steps",

        "logging_steps": 1,

        "save_strategy": "no",

        "eval_strategy": "no",

        "report_to": "none",

        "push_to_hub": False,

        "dataloader_num_workers": 0,

        "dataloader_pin_memory": False,

        "seed": SEED,

        "data_seed": SEED,
    }

    supported = set(
        inspect.signature(
            GRPOConfig
        ).parameters
    )

    unsupported = (
        set(requested)
        - supported
    )

    if unsupported:

        raise RuntimeError(
            "Options GRPO non reconnues "
            "par la version installée : "
            + ", ".join(
                sorted(unsupported)
            )
        )

    return GRPOConfig(
        **requested
    )


# ============================================================
# 10. MÉTADONNÉES EXPÉRIMENTALES
# ============================================================

def build_metadata(selected_ids):

    return {
        "experiment": "grpo_v2_stage0_pilot",
        "base_model": BASE_MODEL,
        "initial_adapter": str(ADAPTER_DIR),
        "seed": SEED,
        "selected_ids": selected_ids,
        "num_questions": NUM_QUESTIONS,
        "num_generations": NUM_GENERATIONS,
        "max_prompt_tokens": (
            MAX_PROMPT_TOKENS
        ),
        "max_completion_tokens": (
            MAX_COMPLETION_TOKENS
        ),
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "reward_correct": REWARD_CORRECT,
        "reward_format": REWARD_FORMAT,
        "beta": 0.0,
        "loss_type": "dapo",
        "device": DEVICE,
        "train_hash": sha256_file(
            TRAIN_FILE
        ),
        "keys_hash": sha256_file(
            KEYS_FILE
        ),
        "evaluator_hash": sha256_file(
            EVALUATOR_FILE
        ),
        "initial_adapter_hash": sha256_file(
            ADAPTER_DIR
            / "adapter_model.safetensors"
        ),
        "script_hash": sha256_file(
            Path(__file__).resolve()
        ),
    }


# ============================================================
# 11. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("BENCHMARK V2 - PREMIER PILOTE GRPO")
    print("=" * 65)

    set_seed(
        SEED
    )

    check_environment()

    reward_tests = test_reward()

    print(
        "Tests de récompense : "
        f"{reward_tests}/{reward_tests} OK"
    )

    if COMPLETION_FILE.exists():

        print(
            "\nLe pilote est déjà terminé."
        )

        print(
            f"Adaptateur : {FINAL_ADAPTER}"
        )

        return

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    dataset, selected_ids = (
        build_dataset(
            tokenizer
        )
    )

    # Vérifier la compatibilité
    # avant de charger le modèle.

    config = build_grpo_config()

    print(
        "Configuration GRPO : OK"
    )

    metadata = build_metadata(
        selected_ids
    )

    if METADATA_FILE.exists():

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            previous = json.load(
                file
            )

        if previous != metadata:

            raise RuntimeError(
                "La configuration du pilote "
                "a changé. Ne mélange pas "
                "deux expériences."
            )

    else:

        existing_training = (
            FINAL_ADAPTER.exists()
            or TRAINING_LOG.exists()
            or REWARD_REPORT.exists()
        )

        if existing_training:

            raise RuntimeError(
                "Résultats présents "
                "sans configuration."
            )

        save_json(
            METADATA_FILE,
            metadata,
        )

    model = load_model()

    trainer_parameters = set(
        inspect.signature(
            GRPOTrainer.__init__
        ).parameters
    )

    if "processing_class" in trainer_parameters:

        tokenizer_argument = {
            "processing_class": tokenizer
        }

    elif "tokenizer" in trainer_parameters:

        tokenizer_argument = {
            "tokenizer": tokenizer
        }

    else:

        raise RuntimeError(
            "API de GRPOTrainer "
            "non reconnue."
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
        **tokenizer_argument,
    )

    print(
        "\nDémarrage de l'entraînement GRPO..."
    )

    print(
        f"Questions du pilote : "
        f"{NUM_QUESTIONS}"
    )

    print(
        f"Générations par question : "
        f"{NUM_GENERATIONS}"
    )

    print(
        f"Étapes prévues : "
        f"{MAX_STEPS}"
    )

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    train_result = (
        trainer.train()
    )

    # --------------------------------------------------------
    # SAUVEGARDE LOCALE
    # --------------------------------------------------------

    trainer.save_model(
        str(FINAL_ADAPTER)
    )

    tokenizer.save_pretrained(
        str(FINAL_ADAPTER)
    )

    save_json(
        TRAINING_LOG,
        trainer.state.log_history,
    )

    save_json(
        REWARD_REPORT,
        {
            "calls": len(reward_history),
            "history": reward_history,
        },
    )

    completion = {
        "experiment": (
            "grpo_v2_stage0_pilot"
        ),
        "global_step": (
            trainer.state.global_step
        ),
        "training_loss": (
            train_result.training_loss
        ),
        "adapter_path": (
            str(FINAL_ADAPTER)
        ),
        "reward_calls": (
            len(reward_history)
        ),
        "validation_evaluated": False,
        "test_evaluated": False,
    }

    save_json(
        COMPLETION_FILE,
        completion,
    )

    # --------------------------------------------------------
    # BILAN
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "PILOTE GRPO - TERMINÉ"
    )

    print(
        "=" * 65
    )

    print(
        "Étapes d'optimisation : "
        f"{trainer.state.global_step}"
    )

    print(
        "Loss moyenne : "
        f"{train_result.training_loss:.4f}"
    )

    print(
        "Appels du vérificateur : "
        f"{len(reward_history)}"
    )

    print(
        "Adaptateur : "
        f"{FINAL_ADAPTER}"
    )

    print(
        "Rapport : "
        f"{COMPLETION_FILE}"
    )


if __name__ == "__main__":

    main()

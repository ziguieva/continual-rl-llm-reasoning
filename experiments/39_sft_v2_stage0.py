
import argparse
import hashlib
import inspect
import json
import random
import re

from pathlib import Path

import torch

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = (
    ROOT / "data/benchmark_v2_v1/train.jsonl"
)

KEYS_FILE = (
    ROOT
    / "data/benchmark_v2_v1/grading/train_keys.jsonl"
)

MODEL_DIR = ROOT / "models"

DOMAIN = "maths_appliques"

SEED = 42
MAX_LENGTH = 768
EPOCHS = 1

BATCH_SIZE = 1
GRAD_ACCUMULATION = 8

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0.03

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = (
    "You are solving a mathematical reasoning problem. "
    "Work through the problem carefully and show "
    "the essential reasoning steps. "
    "End your response with exactly one final line "
    "in this format: "
    "REPONSE: \\boxed{your final mathematical answer}. "
    "Put only the final answer inside the box. "
    "Do not write anything after the final line."
)

MODEL_DIR.mkdir(
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

    ids = [row["id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return rows


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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
# 3. PRÉPARATION DES RÉPONSES
# ============================================================

def prepare_reasoning(target):

    solution = str(target).strip()

    # Supprimer les annotations GSM8K
    # utilisées pour détailler les calculs.

    solution = re.sub(
        r"<<[^<>]*>>",
        "",
        solution,
    )

    # La réponse finale sera reconstruite
    # à partir de la correction officielle.

    if "####" in solution:
        solution = solution.split(
            "####",
            1,
        )[0].strip()

    return solution


def build_answer(target, expected):

    reasoning = prepare_reasoning(target)

    final_answer = (
        "REPONSE: \\boxed{"
        + str(expected).strip()
        + "}"
    )

    if reasoning:
        return (
            reasoning.rstrip()
            + "\n\n"
            + final_answer
        )

    return final_answer


def load_training_examples():

    questions = load_jsonl(TRAIN_FILE)

    corrections = {
        row["id"]: row
        for row in load_jsonl(KEYS_FILE)
    }

    selected = [
        row
        for row in questions
        if row["domain"] == DOMAIN
    ]

    if len(selected) != 600:
        raise ValueError(
            "600 exemples attendus, "
            f"{len(selected)} trouvés."
        )

    examples = []

    for row in selected:

        example_id = row["id"]

        if example_id not in corrections:
            raise ValueError(
                f"Correction absente : {example_id}"
            )

        key = corrections[example_id]

        if key["domain"] != DOMAIN:
            raise ValueError(
                f"Domaine incorrect : {example_id}"
            )

        question = str(
            row.get("question", "")
        ).strip()

        target = str(
            row.get("target", "")
        ).strip()

        expected = str(
            key.get("expected", "")
        ).strip()

        if not question or not target or not expected:
            raise ValueError(
                f"Exemple incomplet : {example_id}"
            )

        examples.append({
            "id": example_id,
            "question": question,
            "answer": build_answer(
                target,
                expected,
            ),
        })

    return examples


# ============================================================
# 4. DATASET ET TOKENISATION
# ============================================================

class SupervisedDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def tokenize_examples(tokenizer, examples):

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Le tokenizer ne possède pas de token EOS."
        )

    tokenized = []
    excluded = []
    lengths = []

    for row in examples:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        completion_ids = tokenizer.encode(
            row["answer"],
            add_special_tokens=False,
        )

        completion_ids.append(
            tokenizer.eos_token_id
        )

        input_ids = (
            prompt_ids
            + completion_ids
        )

        # La loss porte uniquement
        # sur la réponse de l'assistant.

        labels = (
            [-100] * len(prompt_ids)
            + completion_ids
        )

        # On ne coupe pas silencieusement
        # le raisonnement ou la réponse finale.

        if len(input_ids) > MAX_LENGTH:

            excluded.append({
                "id": row["id"],
                "reason": "sequence_too_long",
                "tokens": len(input_ids),
            })

            continue

        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"Labels incohérents : {row['id']}"
            )

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": (
                [1] * len(input_ids)
            ),
            "labels": labels,
        })

        lengths.append(
            len(input_ids)
        )

    if not tokenized:
        raise RuntimeError(
            "Aucun exemple exploitable."
        )

    retention = (
        len(tokenized) / len(examples)
    )

    if retention < 0.95:
        raise RuntimeError(
            "Moins de 95 % des exemples "
            "ont été conservés. "
            "Il faut revoir MAX_LENGTH."
        )

    report = {
        "selected": len(examples),
        "retained": len(tokenized),
        "excluded": len(excluded),
        "retention": retention,
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "mean_tokens": (
            sum(lengths) / len(lengths)
        ),
        "excluded_examples": excluded,
    }

    return (
        SupervisedDataset(tokenized),
        report,
    )


# ============================================================
# 5. VÉRIFICATION DE L'API TRANSFORMERS
# ============================================================

def create_training_arguments(output_dir, pilot):

    # Vérifier les paramètres réellement
    # acceptés par la version installée.

    supported = set(
        inspect.signature(
            TrainingArguments
        ).parameters
    )

    requested = {
        "output_dir": str(output_dir),

        "num_train_epochs": EPOCHS,

        "per_device_train_batch_size": BATCH_SIZE,

        "gradient_accumulation_steps": (
            GRAD_ACCUMULATION
        ),

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "lr_scheduler_type": "linear",

        "warmup_steps": WARMUP_STEPS,

        "optim": "adamw_torch",

        "max_grad_norm": 1.0,

        "gradient_checkpointing": True,

        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },

        "fp16": False,

        "bf16": False,

        "logging_strategy": "steps",

        "logging_steps": (
            1 if pilot else 5
        ),

        "save_strategy": "steps",

        "save_steps": 25,

        "save_total_limit": 2,

        "save_only_model": False,

        "eval_strategy": "no",

        "remove_unused_columns": False,

        "dataloader_num_workers": 0,

        "dataloader_pin_memory": False,

        "report_to": "none",

        "push_to_hub": False,

        "seed": SEED,

        "data_seed": SEED,
    }

    # Ces options peuvent être absentes
    # d'une version de Transformers.
    # Leur valeur par défaut convient
    # pour notre entraînement local.

    optional = {
        "eval_strategy",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "data_seed",
    }

    unsupported = (
        set(requested) - supported
    )

    critical = (
        unsupported - optional
    )

    if critical:

        raise RuntimeError(
            "Paramètres essentiels "
            "non reconnus par Transformers : "
            + ", ".join(sorted(critical))
        )

    for name in sorted(
        unsupported & optional
    ):

        print(
            "Option non disponible "
            f"(valeur par défaut utilisée) : "
            f"{name}"
        )

    compatible = {
        name: value
        for name, value in requested.items()
        if name in supported
    }

    # save_safetensors n'est PAS passé
    # à TrainingArguments.
    #
    # L'adaptateur LoRA sera sauvegardé
    # explicitement à la fin.

    return TrainingArguments(
        **compatible
    )


# ============================================================
# 6. CHARGEMENT DU MODÈLE LORA
# ============================================================

def load_lora_model():

    print("\nChargement de Qwen...")

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    )

    model.config.use_cache = False

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(
        model,
        config,
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# 7. CONFIGURATION REPRODUCTIBLE
# ============================================================

def build_metadata(mode, selected_ids):

    return {
        "experiment": "benchmark_v2_sft_stage0",
        "mode": mode,
        "model": MODEL_NAME,
        "domain": DOMAIN,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": (
            GRAD_ACCUMULATION
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_modules": [
            "q_proj",
            "v_proj",
        ],
        "dtype": "float32",
        "system_prompt": SYSTEM_PROMPT,
        "selected_ids": selected_ids,
        "train_hash": sha256_file(
            TRAIN_FILE
        ),
        "keys_hash": sha256_file(
            KEYS_FILE
        ),
        "script_hash": sha256_file(
            Path(__file__).resolve()
        ),
    }


def validate_or_recover_metadata(
    metadata_file,
    metadata,
    output_dir,
    final_adapter,
    completion_file,
    mode,
):

    if not metadata_file.exists():

        if list(output_dir.iterdir()):
            raise RuntimeError(
                "Fichiers existants sans "
                "configuration d'expérience."
            )

        save_json(
            metadata_file,
            metadata,
        )

        return

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        previous = json.load(file)

    if previous == metadata:
        return

    # Les tentatives précédentes ont
    # échoué avant trainer.train().
    #
    # Autoriser uniquement la migration
    # du pilote si aucun entraînement
    # n'a démarré et si les paramètres
    # scientifiques sont identiques.

    migration_fields = {
        "script_hash",
        "warmup_steps",
    }

    previous_core = {
        key: value
        for key, value in previous.items()
        if key not in migration_fields
    }

    current_core = {
        key: value
        for key, value in metadata.items()
        if key not in migration_fields
    }

    checkpoints = list(
        output_dir.glob(
            "checkpoint-*"
        )
    )

    can_migrate = (
        mode == "pilot"
        and previous_core == current_core
        and not checkpoints
        and not final_adapter.exists()
        and not completion_file.exists()
        and not (
            output_dir
            / "trainer_state.json"
        ).exists()
    )

    if not can_migrate:

        raise RuntimeError(
            "Configuration différente. "
            "Impossible de mélanger "
            "deux expériences."
        )

    backup = (
        output_dir
        / "run_config_before_api_fix_v2.json"
    )

    if not backup.exists():

        save_json(
            backup,
            previous,
        )

    save_json(
        metadata_file,
        metadata,
    )

    print(
        "Configuration du pilote "
        "interrompu récupérée."
    )


# ============================================================
# 8. ENTRAÎNEMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Essai sur 24 exemples.",
    )

    mode.add_argument(
        "--all",
        action="store_true",
        help="Entraînement sur 600 exemples.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK V2 - SFT STAGE 0")
    print("=" * 65)

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    set_seed(SEED)
    random.seed(SEED)

    run_mode = (
        "pilot" if args.pilot else "all"
    )

    folder = (
        "benchmark_v2_sft_stage0_pilot"
        if args.pilot
        else "benchmark_v2_sft_stage0"
    )

    output_dir = (
        MODEL_DIR / folder
    )

    final_adapter = (
        output_dir / "final_adapter"
    )

    metadata_file = (
        output_dir / "run_config.json"
    )

    data_report_file = (
        output_dir / "data_report.json"
    )

    completion_file = (
        output_dir / "completed.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Mode : {run_mode}")
    print(f"Domaine : {DOMAIN}")
    print("GPU : Apple MPS")

    examples = load_training_examples()

    if args.pilot:

        examples = random.Random(
            SEED
        ).sample(
            examples,
            24,
        )

    selected_ids = [
        row["id"]
        for row in examples
    ]

    metadata = build_metadata(
        run_mode,
        selected_ids,
    )

    validate_or_recover_metadata(
        metadata_file,
        metadata,
        output_dir,
        final_adapter,
        completion_file,
        run_mode,
    )

    if completion_file.exists():

        print(
            "\nEntraînement déjà terminé."
        )

        print(
            f"Adaptateur : {final_adapter}"
        )

        return

    print(
        f"Exemples sélectionnés : "
        f"{len(examples)}"
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    train_dataset, data_report = (
        tokenize_examples(
            tokenizer,
            examples,
        )
    )

    save_json(
        data_report_file,
        data_report,
    )

    print(
        f"Exemples conservés : "
        f"{data_report['retained']}"
    )

    print(
        f"Exemples trop longs : "
        f"{data_report['excluded']}"
    )

    print(
        f"Longueur moyenne : "
        f"{data_report['mean_tokens']:.1f} tokens"
    )

    print(
        f"Longueur maximale : "
        f"{data_report['maximum_tokens']} tokens"
    )

    # Tester d'abord l'API.
    # Cela évite de charger les poids
    # si une option est incompatible.

    training_args = (
        create_training_arguments(
            output_dir,
            args.pilot,
        )
    )

    print(
        "\nTrainingArguments : "
        "configuration acceptée."
    )

    # --------------------------------------------------------
    # MODÈLE
    # --------------------------------------------------------

    model = load_lora_model()

    # --------------------------------------------------------
    # COLLATOR
    # --------------------------------------------------------

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer_options = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }

    trainer_parameters = set(
        inspect.signature(
            Trainer.__init__
        ).parameters
    )

    if "processing_class" in trainer_parameters:

        trainer_options[
            "processing_class"
        ] = tokenizer

    elif "tokenizer" in trainer_parameters:

        trainer_options[
            "tokenizer"
        ] = tokenizer

    else:

        raise RuntimeError(
            "Le constructeur Trainer "
            "ne reconnaît pas le tokenizer."
        )

    trainer = Trainer(
        **trainer_options
    )

    latest_checkpoint = (
        get_last_checkpoint(
            str(output_dir)
        )
    )

    if latest_checkpoint:

        print(
            "\nReprise du checkpoint : "
            f"{latest_checkpoint}"
        )

    else:

        print(
            "\nNouvel entraînement."
        )

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    train_result = trainer.train(
        resume_from_checkpoint=(
            latest_checkpoint
        )
    )

    # --------------------------------------------------------
    # SAUVEGARDE FINALE
    # --------------------------------------------------------

    trainer.save_model(
        str(final_adapter)
    )

    tokenizer.save_pretrained(
        str(final_adapter)
    )

    trainer.save_state()

    result = {
        "experiment": "benchmark_v2_sft_stage0",
        "mode": run_mode,
        "domain": DOMAIN,
        "selected_examples": len(examples),
        "trained_examples": len(train_dataset),
        "global_step": trainer.state.global_step,
        "training_loss": train_result.training_loss,
        "adapter_path": str(final_adapter),
        "validation_evaluated": False,
        "test_evaluated": False,
    }

    save_json(
        completion_file,
        result,
    )

    print("\n" + "=" * 65)
    print("SFT STAGE 0 - TERMINÉ")
    print("=" * 65)

    print(
        f"Exemples entraînés : "
        f"{len(train_dataset)}"
    )

    print(
        f"Étapes d'optimisation : "
        f"{trainer.state.global_step}"
    )

    print(
        f"Loss moyenne : "
        f"{train_result.training_loss:.4f}"
    )

    print(
        f"Adaptateur : {final_adapter}"
    )

    print(
        f"Rapport : {completion_file}"
    )


if __name__ == "__main__":
    main()

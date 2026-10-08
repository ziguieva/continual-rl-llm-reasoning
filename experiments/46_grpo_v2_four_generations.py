
import argparse
import importlib.util
import inspect
import json

from collections import Counter
from importlib.metadata import version
from pathlib import Path

from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PILOT_SCRIPT = (
    ROOT / "experiments/42_grpo_v2_pilot.py"
)

MODEL_ROOT = ROOT / "models"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_GENERATIONS = 4

GRADIENT_ACCUMULATION = 4

GENERATION_BATCH_SIZE = 4

MAX_COMPLETION_TOKENS = 256

LEARNING_RATE = 1e-5

SEED = 42


# ============================================================
# 2. IMPORTER LE PIPELINE GRPO VALIDÉ
# ============================================================

def import_pilot():

    if not PILOT_SCRIPT.is_file():
        raise FileNotFoundError(PILOT_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "grpo_pilot_v2",
        PILOT_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Impossible de charger l'étape 42."
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. CONFIGURATION DE L'EXPÉRIENCE
# ============================================================

def configure_pilot(pilot, smoke):

    if smoke:

        folder = (
            "benchmark_v2_grpo_fourgen_smoke"
        )

        questions = 2
        steps = 2

    else:

        folder = (
            "benchmark_v2_grpo_fourgen_v1"
        )

        questions = 48
        steps = 48

    output = MODEL_ROOT / folder

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot.OUTPUT_DIR = output

    pilot.FINAL_ADAPTER = (
        output / "final_adapter"
    )

    pilot.METADATA_FILE = (
        output / "run_config.json"
    )

    pilot.COMPLETION_FILE = (
        output / "completed.json"
    )

    pilot.REWARD_REPORT = (
        output / "reward_report.json"
    )

    pilot.TRAINING_LOG = (
        output / "training_log.json"
    )

    pilot.NUM_QUESTIONS = questions

    pilot.NUM_GENERATIONS = NUM_GENERATIONS

    pilot.MAX_STEPS = steps

    pilot.MAX_COMPLETION_TOKENS = (
        MAX_COMPLETION_TOKENS
    )

    pilot.LEARNING_RATE = LEARNING_RATE

    pilot.reward_history.clear()

    return {
        "output": output,
        "questions": questions,
        "steps": steps,
        "smoke": smoke,
    }


# ============================================================
# 4. CONSTRUIRE LA CONFIGURATION GRPO
# ============================================================

def build_config(run):

    requested = {
        "output_dir": str(run["output"]),

        "max_steps": run["steps"],

        "per_device_train_batch_size": 1,

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION
        ),

        "generation_batch_size": (
            GENERATION_BATCH_SIZE
        ),

        "num_generations": NUM_GENERATIONS,

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

        "beta": 0.0,

        "loss_type": "dapo",

        "scale_rewards": "group",

        "num_iterations": 1,

        "use_vllm": False,

        "remove_unused_columns": False,

        "logging_strategy": "steps",

        "logging_steps": 1,

        "save_strategy": (
            "no"
            if run["smoke"]
            else "steps"
        ),

        "save_steps": 8,

        "save_total_limit": 2,

        "save_only_model": False,

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

    unsupported = set(requested) - supported

    if unsupported:

        raise RuntimeError(
            "Options GRPO incompatibles : "
            + ", ".join(sorted(unsupported))
        )

    return GRPOConfig(**requested)


# ============================================================
# 5. MÉTADONNÉES REPRODUCTIBLES
# ============================================================

def build_metadata(pilot, run, selected_ids):

    return {
        "experiment": (
            "grpo_v2_four_generations_v1"
        ),

        "mode": (
            "smoke"
            if run["smoke"]
            else "all"
        ),

        "base_model": BASE_MODEL,

        "initial_adapter": str(
            pilot.ADAPTER_DIR
        ),

        "seed": SEED,

        "selected_ids": selected_ids,

        "num_questions": run["questions"],

        "num_generations": NUM_GENERATIONS,

        "generation_batch_size": (
            GENERATION_BATCH_SIZE
        ),

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION
        ),

        "max_steps": run["steps"],

        "max_completion_tokens": (
            MAX_COMPLETION_TOKENS
        ),

        "learning_rate": LEARNING_RATE,

        "reward_correct": (
            pilot.REWARD_CORRECT
        ),

        "reward_format": (
            pilot.REWARD_FORMAT
        ),

        "beta": 0.0,

        "loss_type": "dapo",

        "trl_version": version("trl"),

        "train_hash": pilot.sha256_file(
            pilot.TRAIN_FILE
        ),

        "keys_hash": pilot.sha256_file(
            pilot.KEYS_FILE
        ),

        "evaluator_hash": pilot.sha256_file(
            pilot.EVALUATOR_FILE
        ),

        "initial_adapter_hash": (
            pilot.sha256_file(
                pilot.ADAPTER_DIR
                / "adapter_model.safetensors"
            )
        ),

        "pilot_script_hash": (
            pilot.sha256_file(
                PILOT_SCRIPT
            )
        ),

        "script_hash": pilot.sha256_file(
            Path(__file__).resolve()
        ),

        "validation_evaluated": False,

        "test_evaluated": False,
    }


def verify_metadata(pilot, metadata):

    path = pilot.METADATA_FILE

    if path.is_file():

        previous = json.loads(
            path.read_text(encoding="utf-8")
        )

        if previous != metadata:

            raise RuntimeError(
                "La configuration a changé. "
                "Ne mélange pas deux expériences."
            )

    else:

        if (
            pilot.FINAL_ADAPTER.exists()
            or pilot.COMPLETION_FILE.exists()
            or pilot.TRAINING_LOG.exists()
        ):

            raise RuntimeError(
                "Résultats existants sans "
                "configuration d'expérience."
            )

        pilot.save_json(path, metadata)


# ============================================================
# 6. JOURNAL DES RÉCOMPENSES
# ============================================================

def load_reward_events(path):

    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_reward_events(path, events):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for event in events:

            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )


def make_reward_function(
    pilot,
    reward_events_file,
    trainer_holder,
):

    def reward_function(
        completions,
        expected,
        **kwargs,
    ):

        rewards = pilot.reward_function(
            completions,
            expected,
            **kwargs,
        )

        trainer = trainer_holder[
            "trainer"
        ]

        event = {
            **pilot.reward_history[-1],
            "step_before_update": (
                trainer.state.global_step
            ),
        }

        with reward_events_file.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return rewards

    return reward_function


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Deux étapes avec quatre générations.",
    )

    mode.add_argument(
        "--all",
        action="store_true",
        help="Expérience sur 48 étapes.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK V2 - GRPO QUATRE GÉNÉRATIONS")
    print("=" * 65)

    pilot = import_pilot()

    run = configure_pilot(
        pilot,
        smoke=args.smoke,
    )

    set_seed(SEED)

    pilot.check_environment()

    reward_tests = pilot.test_reward()

    print(
        "Tests de récompense : "
        f"{reward_tests}/{reward_tests} OK"
    )

    if pilot.COMPLETION_FILE.exists():

        print(
            "Expérience déjà terminée."
        )

        print(
            f"Adaptateur : {pilot.FINAL_ADAPTER}"
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
        pilot.build_dataset(tokenizer)
    )

    # Vérifier GRPOConfig avant
    # de charger les poids Qwen.

    config = build_config(run)

    metadata = build_metadata(
        pilot,
        run,
        selected_ids,
    )

    verify_metadata(
        pilot,
        metadata,
    )

    print(
        f"Questions sélectionnées : "
        f"{run['questions']}"
    )

    print(
        f"Générations par groupe : "
        f"{NUM_GENERATIONS}"
    )

    print(
        f"Étapes prévues : "
        f"{run['steps']}"
    )

    print(
        f"Dossier : {run['output']}"
    )

    # --------------------------------------------------------
    # CHARGEMENT DU MODÈLE
    # --------------------------------------------------------

    model = pilot.load_model()

    reward_events_file = (
        run["output"]
        / "reward_events.jsonl"
    )

    trainer_holder = {}

    reward_function = make_reward_function(
        pilot,
        reward_events_file,
        trainer_holder,
    )

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
            "API GRPOTrainer non reconnue."
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
        **tokenizer_argument,
    )

    trainer_holder[
        "trainer"
    ] = trainer

    # --------------------------------------------------------
    # REPRISE ÉVENTUELLE
    # --------------------------------------------------------

    checkpoint = None

    if not run["smoke"]:

        checkpoint = get_last_checkpoint(
            str(run["output"])
        )

    if checkpoint:

        checkpoint_step = int(
            Path(checkpoint).name.split("-")[-1]
        )

        # Supprimer du journal les groupes
        # produits après le checkpoint :
        # ils seront recalculés à la reprise.

        events = load_reward_events(
            reward_events_file
        )

        events = [
            event
            for event in events
            if event["step_before_update"]
            < checkpoint_step
        ]

        save_reward_events(
            reward_events_file,
            events,
        )

        print(
            "Reprise du checkpoint : "
            f"{checkpoint}"
        )

    else:

        if reward_events_file.exists():

            raise RuntimeError(
                "Un journal de récompenses existe "
                "sans checkpoint exploitable. "
                "Ne pas relancer cette expérience "
                "dans le même dossier."
            )

        print(
            "Nouvel entraînement GRPO."
        )

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    train_result = trainer.train(
        resume_from_checkpoint=checkpoint
    )

    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    trainer.save_model(
        str(pilot.FINAL_ADAPTER)
    )

    tokenizer.save_pretrained(
        str(pilot.FINAL_ADAPTER)
    )

    trainer.save_state()

    pilot.save_json(
        pilot.TRAINING_LOG,
        trainer.state.log_history,
    )

    events = load_reward_events(
        reward_events_file
    )

    all_rewards = [
        reward
        for event in events
        for reward in event["rewards"]
    ]

    total_correct = sum(
        event["correct"]
        for event in events
    )

    groups_with_signal = sum(
        len(set(event["rewards"])) > 1
        for event in events
    )

    distribution = Counter(
        str(reward)
        for reward in all_rewards
    )

    reward_report = {
        "groups": len(events),

        "generations": len(all_rewards),

        "correct": total_correct,

        "groups_with_signal": (
            groups_with_signal
        ),

        "reward_distribution": dict(
            distribution
        ),

        "events": events,
    }

    pilot.save_json(
        pilot.REWARD_REPORT,
        reward_report,
    )

    completion = {
        "experiment": (
            "grpo_v2_four_generations_v1"
        ),

        "mode": (
            "smoke"
            if run["smoke"]
            else "all"
        ),

        "global_step": (
            trainer.state.global_step
        ),

        "training_loss": (
            train_result.training_loss
        ),

        "adapter_path": str(
            pilot.FINAL_ADAPTER
        ),

        "groups": len(events),

        "generations": len(all_rewards),

        "groups_with_signal": (
            groups_with_signal
        ),

        "validation_evaluated": False,

        "test_evaluated": False,
    }

    pilot.save_json(
        pilot.COMPLETION_FILE,
        completion,
    )

    # --------------------------------------------------------
    # BILAN
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("GRPO QUATRE GÉNÉRATIONS - TERMINÉ")
    print("=" * 65)

    print(
        "Étapes d'optimisation : "
        f"{trainer.state.global_step}"
    )

    print(
        "Loss moyenne : "
        f"{train_result.training_loss:.4f}"
    )

    print(
        f"Groupes évalués : "
        f"{len(events)}"
    )

    print(
        f"Générations : "
        f"{len(all_rewards)}"
    )

    print(
        f"Réponses correctes : "
        f"{total_correct}"
    )

    print(
        "Groupes avec signal : "
        f"{groups_with_signal}"
        f"/{len(events)}"
    )

    print(
        f"Adaptateur : "
        f"{pilot.FINAL_ADAPTER}"
    )

    print(
        f"Rapport : "
        f"{pilot.REWARD_REPORT}"
    )


if __name__ == "__main__":
    main()

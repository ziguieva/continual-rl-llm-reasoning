
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

MODEL_LABEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
    " + SFT-stage0"
    " + GRPO-fourgen-48steps"
)

STAGE40_SCRIPT = (
    ROOT
    / "experiments"
    / "40_evaluate_math_sft_stage0.py"
)

GRPO_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_fourgen_v1"
)

ADAPTER_DIR = (
    GRPO_DIR
    / "final_adapter"
)

RUN_CONFIG_FILE = (
    GRPO_DIR
    / "run_config.json"
)

COMPLETION_FILE = (
    GRPO_DIR
    / "completed.json"
)

REWARD_REPORT_FILE = (
    GRPO_DIR
    / "reward_report.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_grpo_fourgen_v1"
)


# ============================================================
# 2. CHARGER L'ÉVALUATEUR VALIDÉ
# ============================================================

def load_stage40():

    if not STAGE40_SCRIPT.is_file():

        raise FileNotFoundError(
            STAGE40_SCRIPT
        )

    spec = importlib.util.spec_from_file_location(
        "math_sft_stage0_evaluator",
        STAGE40_SCRIPT,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Impossible de charger "
            "l'évaluateur de l'étape 40."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# 3. VÉRIFIER L'EXPÉRIENCE GRPO
# ============================================================

def read_json(path):

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier absent : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def verify_grpo_training():

    adapter_config = (
        ADAPTER_DIR
        / "adapter_config.json"
    )

    adapter_weights = (
        ADAPTER_DIR
        / "adapter_model.safetensors"
    )

    for path in (
        adapter_config,
        adapter_weights,
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"Adaptateur incomplet : {path}"
            )

    run_config = read_json(
        RUN_CONFIG_FILE
    )

    completion = read_json(
        COMPLETION_FILE
    )

    reward_report = read_json(
        REWARD_REPORT_FILE
    )

    expected_experiment = (
        "grpo_v2_four_generations_v1"
    )

    if (
        run_config.get("experiment")
        != expected_experiment
    ):

        raise RuntimeError(
            "Configuration GRPO inattendue."
        )

    if (
        completion.get("experiment")
        != expected_experiment
    ):

        raise RuntimeError(
            "Rapport d'entraînement "
            "GRPO inattendu."
        )

    if (
        run_config.get("mode") != "all"
        or completion.get("mode") != "all"
    ):

        raise RuntimeError(
            "L'expérience complète "
            "est requise."
        )

    if completion.get("global_step") != 48:

        raise RuntimeError(
            "48 étapes GRPO attendues."
        )

    if run_config.get("num_generations") != 4:

        raise RuntimeError(
            "Quatre générations "
            "par groupe attendues."
        )

    if run_config.get("num_questions") != 48:

        raise RuntimeError(
            "48 problèmes d'entraînement "
            "attendus."
        )

    events = reward_report.get(
        "events",
        [],
    )

    if len(events) != 48:

        raise RuntimeError(
            "48 groupes de récompenses "
            "attendus."
        )

    generations = sum(
        len(event["rewards"])
        for event in events
    )

    if generations != 192:

        raise RuntimeError(
            "192 générations attendues."
        )

    correct = sum(
        event["correct"]
        for event in events
    )

    reward_signal = sum(
        len(set(event["rewards"])) > 1
        for event in events
    )

    # Un signal de justesse exige
    # au moins une réponse correcte
    # et une réponse incorrecte
    # dans le même groupe.

    correctness_signal = sum(
        0 < event["correct"] < 4
        for event in events
    )

    format_only_signal = sum(
        (
            event["correct"] == 0
            and len(
                set(event["rewards"])
            ) > 1
        )
        for event in events
    )

    all_correct = sum(
        event["correct"] == 4
        for event in events
    )

    print("\n" + "-" * 65)

    print(
        "AUDIT DE L'ENTRAÎNEMENT GRPO"
    )

    print("-" * 65)

    print(
        "Étapes d'optimisation : "
        f"{completion['global_step']}"
    )

    print(
        f"Générations : {generations}"
    )

    print(
        f"Réponses correctes : "
        f"{correct}/{generations}"
    )

    print(
        "Groupes avec variation "
        "de récompense : "
        f"{reward_signal}/48"
    )

    print(
        "Groupes avec signal "
        "de justesse : "
        f"{correctness_signal}/48"
    )

    print(
        "Groupes avec variation "
        "de format uniquement "
        "et aucune bonne réponse : "
        f"{format_only_signal}"
    )

    print(
        "Groupes entièrement corrects : "
        f"{all_correct}"
    )

    return (
        adapter_config,
        adapter_weights,
    )


# ============================================================
# 4. CHARGER QWEN + ADAPTATEUR GRPO
# ============================================================

def load_grpo_model(evaluator):

    print(
        "\nChargement de Qwen "
        "et de l'adaptateur "
        "GRPO à 48 étapes..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

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

    model = model.to(
        evaluator.DEVICE
    )

    model.config.use_cache = True

    model.eval()

    return (
        model,
        tokenizer,
    )


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)

    print(
        "BENCHMARK V2 - ÉVALUATION "
        "GRPO 48 ÉTAPES"
    )

    print("=" * 65)

    adapter_config, adapter_weights = (
        verify_grpo_training()
    )

    stage40 = load_stage40()

    evaluator = (
        stage40.load_evaluator()
    )

    # Vérifier que nous conservons
    # le protocole de Qwen initial.

    stage40.verify_baseline(
        evaluator
    )

    # Configurer uniquement :
    # - le nouvel adaptateur ;
    # - son rapport d'entraînement ;
    # - son nom ;
    # - son dossier de résultats.

    stage40.ADAPTER_DIR = (
        ADAPTER_DIR
    )

    stage40.COMPLETION_FILE = (
        COMPLETION_FILE
    )

    stage40.MODEL_LABEL = (
        MODEL_LABEL
    )

    stage40.OUTPUT_DIR = (
        OUTPUT_DIR
    )

    # Éviter l'erreur rencontrée
    # lors de l'étape 44 :
    # metadata.json doit pouvoir
    # être créé immédiatement.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage40.load_adapter_model = (
        load_grpo_model
    )

    stage40.configure_evaluator(
        evaluator,
        adapter_config,
        adapter_weights,
    )

    print(
        f"\nAdaptateur : "
        f"{ADAPTER_DIR}"
    )

    print(
        f"Résultats : "
        f"{OUTPUT_DIR}"
    )

    print(
        "Protocole : identique "
        "aux évaluations précédentes."
    )

    # Réutiliser les options
    # --limit et --all de
    # l'évaluateur de l'étape 36.
    #
    # Toutes les générations
    # sont enregistrées dans
    # un dossier indépendant.
    #
    # Aucun problème du test
    # indépendant n'est utilisé.

    evaluator.main()


if __name__ == "__main__":

    main()

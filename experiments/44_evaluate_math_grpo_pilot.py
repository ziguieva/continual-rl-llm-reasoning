
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
    " + GRPO-stage0-pilot"
)

STAGE40_SCRIPT = (
    ROOT
    / "experiments"
    / "40_evaluate_math_sft_stage0.py"
)

GRPO_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_stage0_pilot"
)

ADAPTER_DIR = (
    GRPO_DIR
    / "final_adapter"
)

COMPLETION_FILE = (
    GRPO_DIR
    / "completed.json"
)

REWARD_REPORT = (
    GRPO_DIR
    / "reward_report.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_grpo_stage0_pilot"
)


# ============================================================
# 2. CHARGER L'ÉVALUATEUR DE L'ÉTAPE 40
# ============================================================

def load_stage40():

    if not STAGE40_SCRIPT.is_file():

        raise FileNotFoundError(
            STAGE40_SCRIPT
        )

    spec = (
        importlib.util.spec_from_file_location(
            "math_sft_stage0_evaluator",
            STAGE40_SCRIPT,
        )
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Impossible de charger "
            "l'évaluateur de l'étape 40."
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


# ============================================================
# 3. VÉRIFIER LE PILOTE GRPO
# ============================================================

def verify_grpo_pilot():

    adapter_config = (
        ADAPTER_DIR
        / "adapter_config.json"
    )

    adapter_weights = (
        ADAPTER_DIR
        / "adapter_model.safetensors"
    )

    required = [
        adapter_config,
        adapter_weights,
        COMPLETION_FILE,
        REWARD_REPORT,
    ]

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                f"Fichier absent : {path}"
            )

    with COMPLETION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        completion = json.load(
            file
        )

    if (
        completion.get("experiment")
        != "grpo_v2_stage0_pilot"
    ):

        raise RuntimeError(
            "Cet adaptateur ne provient pas "
            "du pilote GRPO attendu."
        )

    if completion.get("global_step") != 4:

        raise RuntimeError(
            "Le pilote GRPO doit avoir "
            "effectué quatre étapes."
        )

    if completion.get("reward_calls") != 4:

        raise RuntimeError(
            "Quatre appels au vérificateur "
            "sont attendus."
        )

    return (
        adapter_config,
        adapter_weights,
    )


# ============================================================
# 4. CHARGER QWEN + L'ADAPTATEUR GRPO
# ============================================================

def load_grpo_model(evaluator):

    print(
        "\nChargement de Qwen "
        "et de l'adaptateur GRPO..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    base = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
        )
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
    print("BENCHMARK V2 - ÉVALUATION GRPO PILOTE")
    print("=" * 65)

    adapter_config, adapter_weights = (
        verify_grpo_pilot()
    )

    stage40 = load_stage40()

    evaluator = (
        stage40.load_evaluator()
    )

    # Vérifier la comparabilité
    # avec Qwen initial avant de modifier
    # la configuration de l'évaluateur.

    stage40.verify_baseline(
        evaluator
    )

    # Fournir à l'étape 40
    # notre nouvel adaptateur,
    # son rapport d'entraînement
    # et son dossier de sortie.

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

    # Remplacer uniquement
    # le chargement de l'adaptateur.
    #
    # Le prompt, les questions,
    # la génération et la correction
    # de l'étape 36 restent identiques.

    stage40.load_adapter_model = (
        load_grpo_model
    )

    stage40.configure_evaluator(
        evaluator,
        adapter_config,
        adapter_weights,
    )

    print(
        f"\nAdaptateur : {ADAPTER_DIR}"
    )

    print(
        f"Résultats : {OUTPUT_DIR}"
    )

    print(
        "Protocole : identique "
        "aux évaluations précédentes."
    )

    # La fonction main de l'étape 36
    # gère les arguments :
    # --limit 3 et --all.
    #
    # Les générations et corrections
    # sont sauvegardées progressivement.

    evaluator.main()


if __name__ == "__main__":

    main()

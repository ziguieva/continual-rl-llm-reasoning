
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
    " + SFT-stage0-LoRA"
)

ADAPTER_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "final_adapter"
)

COMPLETION_FILE = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "completed.json"
)

EVALUATOR_FILE = (
    ROOT
    / "experiments"
    / "36_evaluate_math_qwen.py"
)

BASELINE_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_qwen_baseline"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_sft_stage0"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. CHARGER LE VÉRIFICATEUR EXISTANT
# ============================================================

def load_evaluator():

    if not EVALUATOR_FILE.is_file():

        raise FileNotFoundError(
            EVALUATOR_FILE
        )

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


# ============================================================
# 3. VÉRIFIER L'ADAPTATEUR
# ============================================================

def verify_adapter():

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

    if completion.get("mode") != "all":

        raise RuntimeError(
            "L'adaptateur ne provient pas "
            "de l'entraînement complet."
        )

    if completion.get("domain") != "maths_appliques":

        raise RuntimeError(
            "Domaine d'entraînement incorrect."
        )

    if completion.get("trained_examples") != 600:

        raise RuntimeError(
            "600 exemples entraînés attendus."
        )

    if completion.get("global_step") != 75:

        raise RuntimeError(
            "75 étapes d'optimisation attendues."
        )

    return (
        adapter_config,
        adapter_weights,
    )


# ============================================================
# 4. VÉRIFIER LA COMPARABILITÉ
# ============================================================

def verify_baseline(evaluator):

    metadata_file = (
        BASELINE_DIR
        / "metadata.json"
    )

    summary_file = (
        BASELINE_DIR
        / "summary.json"
    )

    for path in (
        metadata_file,
        summary_file,
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"Référence initiale absente : {path}"
            )

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        baseline = json.load(
            file
        )

    with summary_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        summary = json.load(
            file
        )

    if summary.get("evaluated") != 350:

        raise RuntimeError(
            "La référence initiale "
            "doit contenir 350 problèmes."
        )

    if baseline.get("model") != BASE_MODEL:

        raise RuntimeError(
            "Le modèle de référence "
            "est différent."
        )

    # Construire les métadonnées à partir
    # du script 36, encore configuré
    # pour le modèle de référence.

    current = (
        evaluator.build_metadata()
    )

    comparable_fields = [
        "partition",
        "domains",
        "seed",
        "device",
        "max_new_tokens",
        "do_sample",
        "system_prompt",
        "questions_hash",
        "grading_hash",
        "evaluator_hash",
    ]

    mismatches = []

    for field in comparable_fields:

        if (
            baseline.get(field)
            != current.get(field)
        ):

            mismatches.append(
                field
            )

    if mismatches:

        raise RuntimeError(
            "Le protocole diffère de "
            "la référence initiale : "
            + ", ".join(
                mismatches
            )
        )

    if current["max_new_tokens"] != 768:

        raise RuntimeError(
            "Budget de génération inattendu."
        )

    if current["do_sample"] is not False:

        raise RuntimeError(
            "La génération doit "
            "rester déterministe."
        )

    print(
        "Référence initiale : "
        f"{summary['correct']}/350"
    )

    print(
        "Protocole identique "
        "à l'évaluation initiale."
    )


# ============================================================
# 5. CHARGEMENT DU MODÈLE AVEC LORA
# ============================================================

def load_adapter_model(evaluator):

    print(
        "\nChargement de Qwen et "
        "de l'adaptateur SFT..."
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

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    )

    model = PeftModel.from_pretrained(
        base_model,
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
# 6. CONFIGURER UNE ÉVALUATION DISTINCTE
# ============================================================

def configure_evaluator(
    evaluator,
    adapter_config,
    adapter_weights,
):

    # Sauvegarder les fonctions originales
    # avant de changer la configuration.

    original_metadata = (
        evaluator.build_metadata
    )

    # Nouveau nom de modèle.
    # Le système de génération,
    # le prompt et la correction
    # restent inchangés.

    evaluator.MODEL_NAME = (
        MODEL_LABEL
    )

    evaluator.OUTPUT_DIR = (
        OUTPUT_DIR
    )

    evaluator.GENERATIONS_FILE = (
        OUTPUT_DIR
        / "generations.jsonl"
    )

    evaluator.GRADES_FILE = (
        OUTPUT_DIR
        / "grades.jsonl"
    )

    evaluator.METADATA_FILE = (
        OUTPUT_DIR
        / "metadata.json"
    )

    evaluator.SUMMARY_FILE = (
        OUTPUT_DIR
        / "summary.json"
    )

    # Remplacer uniquement le
    # chargement du modèle.

    def load_model():

        return load_adapter_model(
            evaluator
        )

    evaluator.load_model = (
        load_model
    )

    # Conserver toutes les métadonnées
    # de l'évaluation initiale.
    #
    # Ajouter une empreinte cryptographique
    # des poids LoRA pour empêcher
    # de mélanger différents adaptateurs
    # dans un même rapport.

    def build_metadata():

        metadata = (
            original_metadata()
        )

        metadata["base_model"] = (
            BASE_MODEL
        )

        metadata["adapter_path"] = (
            str(ADAPTER_DIR)
        )

        metadata["adapter_config_sha256"] = (
            evaluator.sha256_file(
                adapter_config
            )
        )

        metadata["adapter_weights_sha256"] = (
            evaluator.sha256_file(
                adapter_weights
            )
        )

        metadata["training_completion_sha256"] = (
            evaluator.sha256_file(
                COMPLETION_FILE
            )
        )

        return metadata

    evaluator.build_metadata = (
        build_metadata
    )


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 65)

    print(
        "BENCHMARK V2 - ÉVALUATION SFT STAGE 0"
    )

    print("=" * 65)

    adapter_config, adapter_weights = (
        verify_adapter()
    )

    evaluator = (
        load_evaluator()
    )

    # Vérifier le protocole AVANT
    # de changer le nom du modèle
    # ou les dossiers de sortie.

    verify_baseline(
        evaluator
    )

    configure_evaluator(
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

    # Réutiliser entièrement la boucle
    # de génération, les 350 problèmes,
    # Math-Verify, le masquage des réponses
    # et le bilan de l'étape 36.
    #
    # Les arguments --limit et --all
    # sont traités directement
    # par le script 36.

    evaluator.main()


if __name__ == "__main__":

    main()

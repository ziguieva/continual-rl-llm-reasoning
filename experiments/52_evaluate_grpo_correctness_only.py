
import importlib.util
import json

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE49_SCRIPT = (
    ROOT
    / "experiments"
    / "49_evaluate_math_grpo_fourgen.py"
)

REFERENCE_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_fourgen_v1"
)

ABLATION_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_correctness_only_v1"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_grpo_correctness_only_v1"
)

MODEL_LABEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
    " + SFT-stage0"
    " + GRPO-correctness-only-48steps"
)


# ============================================================
# 2. CHARGEMENT
# ============================================================

def read_json(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_stage49():

    if not STAGE49_SCRIPT.is_file():
        raise FileNotFoundError(
            STAGE49_SCRIPT
        )

    spec = importlib.util.spec_from_file_location(
        "grpo_fourgen_evaluator",
        STAGE49_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Impossible de charger l'étape 49."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# 3. VÉRIFIER L'ABLATION
# ============================================================

def verify_ablation():

    reference = read_json(
        REFERENCE_DIR
        / "run_config.json"
    )

    ablation = read_json(
        ABLATION_DIR
        / "run_config.json"
    )

    completion = read_json(
        ABLATION_DIR
        / "completed.json"
    )

    comparable_fields = (
        "experiment",
        "mode",
        "base_model",
        "initial_adapter",
        "seed",
        "selected_ids",
        "num_questions",
        "num_generations",
        "generation_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "max_completion_tokens",
        "learning_rate",
        "reward_correct",
        "beta",
        "loss_type",
        "trl_version",
        "train_hash",
        "keys_hash",
        "evaluator_hash",
        "initial_adapter_hash",
        "pilot_script_hash",
        "script_hash",
    )

    for field in comparable_fields:

        if (
            field not in reference
            or field not in ablation
        ):
            raise RuntimeError(
                f"Métadonnée absente : {field}"
            )

        if reference[field] != ablation[field]:
            raise RuntimeError(
                f"Expériences différentes : {field}"
            )

    if reference["reward_format"] != 0.02:
        raise RuntimeError(
            "Récompense de référence inattendue."
        )

    if ablation["reward_format"] != 0.0:
        raise RuntimeError(
            "La récompense de format "
            "n'a pas été désactivée."
        )

    if (
        ablation.get("reward_variant")
        != "correctness_only"
    ):
        raise RuntimeError(
            "Variante de récompense incorrecte."
        )

    if (
        completion.get("reward_variant")
        != "correctness_only"
    ):
        raise RuntimeError(
            "Rapport d'ablation incorrect."
        )

    if completion.get("global_step") != 48:
        raise RuntimeError(
            "48 étapes attendues."
        )

    adapter = (
        ABLATION_DIR
        / "final_adapter"
    )

    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
    ):

        if not (
            adapter / filename
        ).is_file():

            raise FileNotFoundError(
                adapter / filename
            )

    print(
        "Même modèle initial : OUI"
    )

    print(
        "Mêmes questions "
        "d'entraînement : OUI"
    )

    print(
        "Même budget GRPO : OUI"
    )

    print(
        "Récompense de format : "
        "0.02 → 0.00"
    )


# ============================================================
# 4. ÉVALUATION
# ============================================================

def main():

    print("=" * 65)

    print(
        "BENCHMARK V2 - ÉVALUATION "
        "GRPO JUSTESSE UNIQUEMENT"
    )

    print("=" * 65)

    verify_ablation()

    evaluator_script = load_stage49()

    # Réutiliser le même programme
    # d'évaluation que l'étape 49.
    #
    # Seuls changent :
    # - le dossier de l'adaptateur ;
    # - les rapports d'entraînement ;
    # - le nom du modèle ;
    # - le dossier des résultats.

    evaluator_script.GRPO_DIR = (
        ABLATION_DIR
    )

    evaluator_script.ADAPTER_DIR = (
        ABLATION_DIR
        / "final_adapter"
    )

    evaluator_script.RUN_CONFIG_FILE = (
        ABLATION_DIR
        / "run_config.json"
    )

    evaluator_script.COMPLETION_FILE = (
        ABLATION_DIR
        / "completed.json"
    )

    evaluator_script.REWARD_REPORT_FILE = (
        ABLATION_DIR
        / "reward_report.json"
    )

    evaluator_script.OUTPUT_DIR = (
        OUTPUT_DIR
    )

    evaluator_script.MODEL_LABEL = (
        MODEL_LABEL
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nAdaptateur :",
        evaluator_script.ADAPTER_DIR,
    )

    print(
        "Résultats :",
        OUTPUT_DIR,
    )

    # L'étape 49 vérifie l'entraînement
    # puis réutilise l'évaluateur 36.
    #
    # Les options --limit et --all
    # sont conservées.
    #
    # Le test indépendant reste fermé.

    evaluator_script.main()


if __name__ == "__main__":
    main()

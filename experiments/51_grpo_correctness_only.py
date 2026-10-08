
import importlib.util
import json
import sys

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE46_SCRIPT = (
    ROOT
    / "experiments"
    / "46_grpo_v2_four_generations.py"
)

REFERENCE_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_fourgen_v1"
)

REFERENCE_CONFIG = (
    REFERENCE_DIR
    / "run_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "benchmark_v2_grpo_correctness_only_v1"
)

REWARD_CORRECT = 1.0

REWARD_FORMAT = 0.0


# ============================================================
# 2. CHARGER LE SCRIPT GRPO DÉJÀ VALIDÉ
# ============================================================

def load_stage46():

    if not STAGE46_SCRIPT.is_file():

        raise FileNotFoundError(
            STAGE46_SCRIPT
        )

    spec = importlib.util.spec_from_file_location(
        "grpo_four_generations",
        STAGE46_SCRIPT,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Impossible de charger l'étape 46."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# 3. VÉRIFIER L'EXPÉRIENCE DE RÉFÉRENCE
# ============================================================

def load_reference():

    if not REFERENCE_CONFIG.is_file():

        raise FileNotFoundError(
            "L'expérience GRPO de référence "
            "doit être terminée."
        )

    reference = json.loads(
        REFERENCE_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    if reference.get("mode") != "all":

        raise RuntimeError(
            "L'expérience de référence "
            "doit être complète."
        )

    if reference.get("num_questions") != 48:

        raise RuntimeError(
            "48 questions attendues."
        )

    if reference.get("num_generations") != 4:

        raise RuntimeError(
            "Quatre générations attendues."
        )

    if reference.get("max_steps") != 48:

        raise RuntimeError(
            "48 étapes attendues."
        )

    if reference.get("reward_correct") != 1.0:

        raise RuntimeError(
            "Récompense de justesse inattendue."
        )

    if reference.get("reward_format") != 0.02:

        raise RuntimeError(
            "Récompense de format "
            "de référence inattendue."
        )

    return reference


# ============================================================
# 4. TESTS DE LA NOUVELLE RÉCOMPENSE
# ============================================================

def test_correctness_reward(pilot):

    cases = [
        (
            r"REPONSE: \boxed{42}",
            "42",
            1.0,
        ),
        (
            r"REPONSE: \boxed{41}",
            "42",
            0.0,
        ),
        (
            r"The result is \boxed{42}.",
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

    for response, expected, target in cases:

        result = pilot.grade_completion(
            response,
            expected,
        )

        actual = result["reward"]

        if abs(actual - target) > 1e-6:

            raise AssertionError(
                "Récompense incorrecte : "
                f"{response} "
                f"({actual} au lieu de {target})"
            )

        passed += 1

    return passed


# ============================================================
# 5. REDIRIGER VERS UN DOSSIER INDÉPENDANT
# ============================================================

def configure_ablation(
    original_configure,
    pilot,
    smoke,
):

    if smoke:

        raise RuntimeError(
            "Utilise uniquement --all. "
            "La configuration mémoire "
            "à quatre générations "
            "a déjà été testée."
        )

    run = original_configure(
        pilot,
        smoke=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot.OUTPUT_DIR = OUTPUT_DIR

    pilot.FINAL_ADAPTER = (
        OUTPUT_DIR
        / "final_adapter"
    )

    pilot.METADATA_FILE = (
        OUTPUT_DIR
        / "run_config.json"
    )

    pilot.COMPLETION_FILE = (
        OUTPUT_DIR
        / "completed.json"
    )

    pilot.REWARD_REPORT = (
        OUTPUT_DIR
        / "reward_report.json"
    )

    pilot.TRAINING_LOG = (
        OUTPUT_DIR
        / "training_log.json"
    )

    run["output"] = OUTPUT_DIR

    return run


# ============================================================
# 6. VÉRIFIER LA COMPARABILITÉ
# ============================================================

def verify_same_budget(
    reference,
    ablation,
):

    fields = (
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

    differences = [
        field
        for field in fields
        if (
            reference.get(field)
            != ablation.get(field)
        )
    ]

    if differences:

        raise RuntimeError(
            "Expériences non comparables : "
            + ", ".join(differences)
        )

    if ablation.get("reward_format") != 0.0:

        raise RuntimeError(
            "La récompense de format "
            "n'a pas été désactivée."
        )


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    if sys.argv[1:] != ["--all"]:

        raise SystemExit(
            "Commande attendue :\n"
            "python "
            "experiments/"
            "51_grpo_correctness_only.py "
            "--all"
        )

    print("=" * 65)
    print(
        "BENCHMARK V2 - ABLATION "
        "DE LA RÉCOMPENSE DE FORMAT"
    )
    print("=" * 65)

    reference = load_reference()

    stage46 = load_stage46()

    pilot = stage46.import_pilot()

    # Unique changement de l'objectif :
    # la justesse vaut 1,
    # le format ne rapporte rien.

    pilot.REWARD_CORRECT = (
        REWARD_CORRECT
    )

    pilot.REWARD_FORMAT = (
        REWARD_FORMAT
    )

    # Remplacer les tests de l'étape 42 :
    # ils attendaient une récompense
    # supplémentaire de 0,02.

    pilot.test_reward = lambda: (
        test_correctness_reward(
            pilot
        )
    )

    # Réutiliser exactement
    # le même pipeline GRPO.

    original_configure = (
        stage46.configure_pilot
    )

    original_build_metadata = (
        stage46.build_metadata
    )

    stage46.import_pilot = lambda: pilot

    def new_configure(
        pilot_module,
        smoke,
    ):

        return configure_ablation(
            original_configure,
            pilot_module,
            smoke,
        )

    stage46.configure_pilot = (
        new_configure
    )

    def new_build_metadata(
        pilot_module,
        run,
        selected_ids,
    ):

        metadata = original_build_metadata(
            pilot_module,
            run,
            selected_ids,
        )

        verify_same_budget(
            reference,
            metadata,
        )

        metadata["reward_variant"] = (
            "correctness_only"
        )

        metadata["ablation_script_hash"] = (
            pilot.sha256_file(
                Path(__file__).resolve()
            )
        )

        metadata[
            "reference_run_config_hash"
        ] = pilot.sha256_file(
            REFERENCE_CONFIG
        )

        return metadata

    stage46.build_metadata = (
        new_build_metadata
    )

    print(
        "\nRécompense de justesse : "
        f"{pilot.REWARD_CORRECT}"
    )

    print(
        "Récompense de format : "
        f"{pilot.REWARD_FORMAT}"
    )

    print(
        "Même modèle initial : OUI"
    )

    print(
        "Même budget : OUI"
    )

    print(
        f"Dossier indépendant : "
        f"{OUTPUT_DIR}"
    )

    # Le script 46 gère :
    # - le chargement du modèle SFT ;
    # - les 48 étapes GRPO ;
    # - les checkpoints ;
    # - la reprise éventuelle ;
    # - les journaux de récompense ;
    # - la sauvegarde de l'adaptateur.

    stage46.main()

    completion_path = (
        pilot.COMPLETION_FILE
    )

    if completion_path.is_file():

        completion = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )

        completion["reward_variant"] = (
            "correctness_only"
        )

        completion["reference_experiment"] = (
            "benchmark_v2_grpo_fourgen_v1"
        )

        pilot.save_json(
            completion_path,
            completion,
        )

        print(
            "\nABLATION TERMINÉE"
        )

        print(
            "Récompense : "
            "justesse uniquement"
        )

        print(
            "Adaptateur : "
            f"{pilot.FINAL_ADAPTER}"
        )


if __name__ == "__main__":

    main()

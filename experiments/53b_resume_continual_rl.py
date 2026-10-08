import importlib.util
import sys

from pathlib import Path


# ============================================================
# 1. CHARGER LE PIPELINE 53 EXISTANT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_FILE = (
    ROOT
    / "experiments"
    / "53_continual_rl_pipeline.py"
)


def load_pipeline():

    if not PIPELINE_FILE.is_file():

        raise FileNotFoundError(
            PIPELINE_FILE
        )

    spec = importlib.util.spec_from_file_location(
        "continual_rl_pipeline",
        PIPELINE_FILE,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Impossible de charger "
            "53_continual_rl_pipeline.py"
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
# 2. NORMALISATION ROBUSTE DES TESTS MBPP
# ============================================================

TEST_KEYS = (
    "test_list",
    "tests",
    "assertions",
    "test_cases",
    "unit_tests",
    "cases",
    "test",
)

SETUP_KEYS = (
    "setup",
    "setup_code",
    "test_setup",
)


def find_setup(obj):

    if not isinstance(
        obj,
        dict,
    ):

        return ""

    for key in SETUP_KEYS:

        value = obj.get(
            key
        )

        if isinstance(
            value,
            str,
        ):

            return value

    for value in obj.values():

        if isinstance(
            value,
            dict,
        ):

            result = find_setup(
                value
            )

            if result:

                return result

    return ""


def normalize_test_list(value):

    # --------------------------------------------------------
    # Cas 1 :
    # une assertion unique sous forme de chaîne.
    # --------------------------------------------------------

    if isinstance(
        value,
        str,
    ):

        return [
            value
        ]

    # --------------------------------------------------------
    # Cas 2 :
    # liste classique MBPP :
    #
    # [
    #   "assert ...",
    #   "assert ...",
    #   "assert ..."
    # ]
    # --------------------------------------------------------

    if isinstance(
        value,
        list,
    ):

        if all(
            isinstance(
                item,
                str,
            )
            for item in value
        ):

            return value

        # ----------------------------------------------------
        # Certains formats stockent :
        #
        # [
        #   {"test": "assert ..."},
        #   {"code": "assert ..."}
        # ]
        # ----------------------------------------------------

        extracted = []

        for item in value:

            if isinstance(
                item,
                str,
            ):

                extracted.append(
                    item
                )

                continue

            if not isinstance(
                item,
                dict,
            ):

                continue

            for key in (
                "test",
                "assertion",
                "code",
                "source",
            ):

                candidate = item.get(
                    key
                )

                if isinstance(
                    candidate,
                    str,
                ):

                    extracted.append(
                        candidate
                    )

                    break

        if extracted:

            return extracted

        return None

    # --------------------------------------------------------
    # Cas 3 :
    # dictionnaire imbriqué :
    #
    # "tests": {
    #     "test_list": [...]
    # }
    # --------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        for key in TEST_KEYS:

            if key not in value:

                continue

            result = normalize_test_list(
                value[
                    key
                ]
            )

            if result:

                return result

        # ----------------------------------------------------
        # Autre possibilité :
        #
        # {
        #   "test_1": "assert ...",
        #   "test_2": "assert ..."
        # }
        # ----------------------------------------------------

        string_values = [
            item
            for item in value.values()
            if isinstance(
                item,
                str,
            )
            and (
                "assert "
                in item
                or "assert("
                in item
            )
        ]

        if string_values:

            return string_values

        # Recherche récursive.

        for nested in value.values():

            if isinstance(
                nested,
                (
                    dict,
                    list,
                ),
            ):

                result = normalize_test_list(
                    nested
                )

                if result:

                    return result

    return None


def find_tests(obj):

    if not isinstance(
        obj,
        dict,
    ):

        return None

    # --------------------------------------------------------
    # Priorité aux noms officiels / usuels MBPP.
    # --------------------------------------------------------

    for key in TEST_KEYS:

        if key not in obj:

            continue

        tests = normalize_test_list(
            obj[
                key
            ]
        )

        if tests:

            return tests

    # --------------------------------------------------------
    # Puis recherche récursive dans :
    # grading / reference / sandbox / etc.
    # --------------------------------------------------------

    for value in obj.values():

        if isinstance(
            value,
            dict,
        ):

            tests = find_tests(
                value
            )

            if tests:

                return tests

    return None


def normalize_mbpp_grading(
    grading,
):

    if not isinstance(
        grading,
        dict,
    ):

        raise RuntimeError(
            "Correction MBPP invalide : "
            "un dictionnaire JSON était attendu."
        )

    tests = find_tests(
        grading
    )

    if not tests:

        raise RuntimeError(
            "Tests MBPP introuvables.\n"
            "Clés racine disponibles : "
            + ", ".join(
                sorted(
                    grading.keys()
                )
            )
            + "\nContenu reçu : "
            + repr(
                grading
            )[:1500]
        )

    setup = find_setup(
        grading
    )

    # --------------------------------------------------------
    # Sécurité :
    # notre benchmark a été construit avec plusieurs tests
    # par problème.
    # --------------------------------------------------------

    tests = [
        test.strip()
        for test in tests
        if isinstance(
            test,
            str,
        )
        and test.strip()
    ]

    if not tests:

        raise RuntimeError(
            "La liste des tests MBPP "
            "est vide après normalisation."
        )

    return {
        "setup": setup,
        "tests": tests,
    }


# ============================================================
# 3. MAIN
# ============================================================

def main():

    pipeline = load_pipeline()

    # --------------------------------------------------------
    # On remplace uniquement la fonction responsable
    # du crash MBPP.
    #
    # Toutes les autres fonctions du pipeline restent
    # strictement celles de l'expérience 53.
    # --------------------------------------------------------

    pipeline.normalize_mbpp_grading = (
        normalize_mbpp_grading
    )

    print(
        "=" * 72
    )

    print(
        "REPRISE CONTINUAL RL V2"
    )

    print(
        "=" * 72
    )

    print(
        "Correctif actif : "
        "normalisation MBPP imbriquée"
    )

    print(
        "Stages déjà terminés : "
        "réutilisés automatiquement"
    )

    print(
        "Validation/test : "
        "toujours non utilisés"
    )

    pipeline.main()


if __name__ == "__main__":

    main()
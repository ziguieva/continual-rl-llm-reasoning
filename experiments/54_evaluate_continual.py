import argparse
import gc
import importlib.util
import json
import math
import re

from collections import Counter
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

SEED_DEFAULT = 42

VALIDATION_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "validation.jsonl"
)

VALIDATION_KEYS_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "grading"
    / "validation_keys.jsonl"
)

MATH_EVALUATOR_FILE = (
    ROOT
    / "experiments"
    / "36_evaluate_math_qwen.py"
)

MBPP_SANDBOX_FILE = (
    ROOT
    / "experiments"
    / "33_audit_mbpp_sandbox.py"
)

MBPP_EVALUATOR_FILE = (
    ROOT
    / "experiments"
    / "34_evaluate_mbpp_qwen.py"
)

SFT_STAGE0_ADAPTER = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "final_adapter"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "continual_rl_v2"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "continual_rl_v2"
)

DEVICE = "mps"

MATH_MAX_NEW_TOKENS = 768

MBPP_MAX_NEW_TOKENS = 512


# ============================================================
# 2. DOMAINES
# ============================================================

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
    "programmation",
)

DOMAIN_LABELS = {
    "maths_appliques":
        "Maths appliquées / GSM8K",

    "algebre_avancee":
        "Algèbre avancée",

    "maths_competition":
        "Maths compétition",

    "programmation":
        "Programmation / MBPP",
}

DOMAIN_STAGE = {
    "maths_appliques": 0,
    "algebre_avancee": 1,
    "maths_competition": 2,
    "programmation": 3,
}

EXPECTED_VALIDATION_COUNTS = {
    "maths_appliques": 150,
    "algebre_avancee": 100,
    "maths_competition": 100,
    "programmation": 90,
}

STRATEGIES = (
    "no_replay",
    "fixed_replay",
    "adaptive_replay",
)

STAGE_NAMES = {
    0: "stage0_gsm8k",
    1: "stage1_algebre_avancee",
    2: "stage2_maths_competition",
    3: "stage3_programmation",
}


# ============================================================
# 3. PROMPT CODE
# ============================================================

DEFAULT_CODE_SYSTEM_PROMPT = (
    "You are solving a Python programming problem. "
    "Return a complete Python solution defining the requested "
    "function. Output Python code only. "
    "Do not use Markdown fences and do not add explanations."
)


# ============================================================
# 4. JSON / JSONL
# ============================================================

def load_json(path):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_jsonl(path):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def load_jsonl_if_exists(path):

    if not path.is_file():

        return []

    return load_jsonl(
        path
    )


def append_jsonl(
    path,
    row,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# 5. IMPORT DYNAMIQUE
# ============================================================

def import_module(
    name,
    path,
):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Impossible de charger : {path}"
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
# 6. ENVIRONNEMENT
# ============================================================

def check_environment(
    seed,
):

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "MPS indisponible."
        )

    required_files = (
        VALIDATION_FILE,
        VALIDATION_KEYS_FILE,
        MATH_EVALUATOR_FILE,
        MBPP_SANDBOX_FILE,
        SFT_STAGE0_ADAPTER
        / "adapter_config.json",
        SFT_STAGE0_ADAPTER
        / "adapter_model.safetensors",
    )

    for path in required_files:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    for strategy in STRATEGIES:

        root = (
            MODEL_ROOT
            / f"seed_{seed}"
            / strategy
        )

        adapters = (
            root
            / "stage_1_algebre_avancee"
            / "final_adapter",

            root
            / "stage_2_maths_competition"
            / "final_adapter",

            root
            / "stage_3_programmation"
            / "final_adapter",
        )

        for adapter in adapters:

            if not (
                adapter
                / "adapter_model.safetensors"
            ).is_file():

                raise FileNotFoundError(
                    adapter
                    / "adapter_model.safetensors"
                )


# ============================================================
# 7. DATASET
# ============================================================

def load_validation():

    rows = load_jsonl(
        VALIDATION_FILE
    )

    key_rows = load_jsonl(
        VALIDATION_KEYS_FILE
    )

    rows_by_id = {
        row["id"]: row
        for row in rows
    }

    keys_by_id = {
        row["id"]: row
        for row in key_rows
    }

    if len(
        rows_by_id
    ) != len(
        rows
    ):

        raise RuntimeError(
            "IDs validation dupliqués."
        )

    if len(
        keys_by_id
    ) != len(
        key_rows
    ):

        raise RuntimeError(
            "IDs grading dupliqués."
        )

    missing = (
        set(
            rows_by_id
        )
        - set(
            keys_by_id
        )
    )

    if missing:

        raise RuntimeError(
            "Corrections manquantes : "
            + repr(
                sorted(
                    missing
                )[:10]
            )
        )

    by_domain = {
        domain: []
        for domain in DOMAINS
    }

    for row in rows:

        domain = row[
            "domain"
        ]

        if domain in by_domain:

            by_domain[
                domain
            ].append(
                row
            )

    for (
        domain,
        expected,
    ) in EXPECTED_VALIDATION_COUNTS.items():

        actual = len(
            by_domain[
                domain
            ]
        )

        if actual != expected:

            raise RuntimeError(
                f"{domain}: "
                f"{actual} au lieu de "
                f"{expected}"
            )

    if len(
        rows
    ) != 440:

        raise RuntimeError(
            f"440 problèmes attendus, "
            f"{len(rows)} trouvés."
        )

    return (
        rows_by_id,
        keys_by_id,
        by_domain,
    )


# ============================================================
# 8. MATH EVALUATOR
# ============================================================

def load_math_evaluator():

    module = import_module(
        "continual_math_eval",
        MATH_EVALUATOR_FILE,
    )

    required = (
        "SYSTEM_PROMPT",
        "parse_gold",
        "grade_math",
        "run_verifier_tests",
    )

    for name in required:

        if not hasattr(
            module,
            name,
        ):

            raise RuntimeError(
                f"Fonction absente "
                f"dans étape 36 : {name}"
            )

    result = (
        module.run_verifier_tests()
    )

    print(
        "Vérificateur mathématique : "
        f"{result}/{result} OK"
    )

    return module


# ============================================================
# 9. MBPP SANDBOX
# ============================================================

def load_mbpp_runner():

    module = import_module(
        "continual_mbpp_sandbox",
        MBPP_SANDBOX_FILE,
    )

    direct = getattr(
        module,
        "run_case",
        None,
    )

    if callable(
        direct
    ):

        return direct

    sandbox = getattr(
        module,
        "sandbox",
        None,
    )

    if sandbox is not None:

        run_case = getattr(
            sandbox,
            "run_case",
            None,
        )

        if callable(
            run_case
        ):

            return run_case

    raise RuntimeError(
        "run_case introuvable "
        "dans l'étape 33."
    )


# ============================================================
# 10. CONFIG MBPP
# ============================================================

def load_mbpp_configuration():

    system_prompt = (
        DEFAULT_CODE_SYSTEM_PROMPT
    )

    max_tokens = (
        MBPP_MAX_NEW_TOKENS
    )

    if MBPP_EVALUATOR_FILE.is_file():

        try:

            module = import_module(
                "continual_mbpp_eval",
                MBPP_EVALUATOR_FILE,
            )

            candidate_prompt = getattr(
                module,
                "SYSTEM_PROMPT",
                None,
            )

            if isinstance(
                candidate_prompt,
                str,
            ):

                system_prompt = (
                    candidate_prompt
                )

            for attr in (
                "MAX_NEW_TOKENS",
                "MAX_COMPLETION_TOKENS",
            ):

                candidate = getattr(
                    module,
                    attr,
                    None,
                )

                if isinstance(
                    candidate,
                    int,
                ):

                    max_tokens = (
                        candidate
                    )

                    break

        except Exception as error:

            print(
                "Avertissement : "
                "configuration étape 34 "
                "non importée :",
                error,
            )

    return {
        "system_prompt":
            system_prompt,

        "max_new_tokens":
            max_tokens,
    }


# ============================================================
# 11. NORMALISATION MBPP
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

    if isinstance(
        value,
        str,
    ):

        return [
            value
        ]

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

                candidate = (
                    item.get(
                        key
                    )
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

    if isinstance(
        value,
        dict,
    ):

        for key in TEST_KEYS:

            if key not in value:

                continue

            result = (
                normalize_test_list(
                    value[
                        key
                    ]
                )
            )

            if result:

                return result

        string_values = [
            item
            for item
            in value.values()
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

        for nested in value.values():

            if isinstance(
                nested,
                (
                    dict,
                    list,
                ),
            ):

                result = (
                    normalize_test_list(
                        nested
                    )
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

    tests = find_tests(
        grading
    )

    if not tests:

        raise RuntimeError(
            "Tests MBPP introuvables."
        )

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
            "Tests MBPP vides."
        )

    return {
        "setup":
            find_setup(
                grading
            ),

        "tests":
            tests,
    }


# ============================================================
# 12. EXTRACTION CODE
# ============================================================

def extract_python_code(text):

    text = text.strip()

    fenced = re.search(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if fenced:

        return (
            fenced
            .group(1)
            .strip()
        )

    text = re.sub(
        r"^```(?:python)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    lines = (
        text.splitlines()
    )

    first_code = None

    for index, line in enumerate(
        lines
    ):

        stripped = (
            line.lstrip()
        )

        if (
            stripped.startswith(
                "def "
            )
            or stripped.startswith(
                "async def "
            )
            or stripped.startswith(
                "import "
            )
            or stripped.startswith(
                "from "
            )
            or stripped.startswith(
                "@"
            )
        ):

            first_code = index

            break

    if first_code is not None:

        text = "\n".join(
            lines[
                first_code:
            ]
        ).strip()

    return text


# ============================================================
# 13. PROMPTS
# ============================================================

def build_prompt(
    tokenizer,
    math_evaluator,
    mbpp_config,
    row,
):

    if (
        row["domain"]
        == "programmation"
    ):

        system_prompt = (
            mbpp_config[
                "system_prompt"
            ]
        )

    else:

        system_prompt = (
            math_evaluator
            .SYSTEM_PROMPT
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": row[
                "question"
            ],
        },
    ]

    return (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


# ============================================================
# 14. CHARGEMENT MODÈLE
# ============================================================

def load_model(
    adapter_dir,
):

    print(
        "\nChargement modèle : "
        f"{adapter_dir}"
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    )

    model = (
        PeftModel.from_pretrained(
            base,
            str(
                adapter_dir
            ),
            is_trainable=False,
        )
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    model.config.use_cache = (
        True
    )

    return model


def release_model(
    model,
):

    del model

    gc.collect()

    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# ============================================================
# 15. GÉNÉRATION GREEDY
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(
        DEVICE
    )

    prompt_tokens = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=(
                max_new_tokens
            ),
            pad_token_id=(
                tokenizer.pad_token_id
            ),
            eos_token_id=(
                tokenizer.eos_token_id
            ),
        )

    generated = output[0][
        prompt_tokens:
    ]

    new_tokens = int(
        generated.shape[0]
    )

    response = (
        tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
        .strip()
    )

    truncated = (
        new_tokens
        >= max_new_tokens
    )

    return {
        "response":
            response,

        "tokens":
            new_tokens,

        "truncated":
            truncated,
    }


# ============================================================
# 16. GRADING MATH
# ============================================================

def grade_math_example(
    evaluator,
    grading,
    response,
):

    if "expected" not in grading:

        raise RuntimeError(
            "Champ expected absent "
            "pour un problème math."
        )

    expected = str(
        grading[
            "expected"
        ]
    )

    gold = (
        evaluator.parse_gold(
            expected
        )
    )

    if not gold:

        raise RuntimeError(
            "Gold math non interprétable."
        )

    result = (
        evaluator.grade_math(
            response,
            gold,
        )
    )

    return {
        "correct":
            bool(
                result[
                    "correct"
                ]
            ),

        "status":
            result[
                "status"
            ],

        "format_ok":
            bool(
                result.get(
                    "format_ok",
                    False,
                )
            ),
    }


# ============================================================
# 17. GRADING MBPP
# ============================================================

def grade_mbpp_example(
    run_case,
    example_id,
    grading,
    response,
):

    normalized = (
        normalize_mbpp_grading(
            grading
        )
    )

    code = extract_python_code(
        response
    )

    case = {
        "id":
            example_id,

        "setup":
            normalized[
                "setup"
            ],

        "solution":
            code,

        "tests":
            normalized[
                "tests"
            ],
    }

    result = run_case(
        case
    )

    status = result.get(
        "status",
        "unknown",
    )

    return {
        "correct":
            status == "passed",

        "status":
            status,

        "format_ok":
            None,
    }


# ============================================================
# 18. ADAPTERS PAR STAGE
# ============================================================

def get_adapter(
    seed,
    strategy,
    stage,
):

    if stage == 0:

        return (
            SFT_STAGE0_ADAPTER
        )

    root = (
        MODEL_ROOT
        / f"seed_{seed}"
        / strategy
    )

    if stage == 1:

        return (
            root
            / "stage_1_algebre_avancee"
            / "final_adapter"
        )

    if stage == 2:

        return (
            root
            / "stage_2_maths_competition"
            / "final_adapter"
        )

    if stage == 3:

        return (
            root
            / "stage_3_programmation"
            / "final_adapter"
        )

    raise ValueError(
        stage
    )


# ============================================================
# 19. DOMAINES À ÉVALUER À CHAQUE STAGE
# ============================================================

def domains_for_stage(
    stage,
):

    return list(
        DOMAINS[
            :stage + 1
        ]
    )


# ============================================================
# 20. DOSSIER RÉSULTATS
# ============================================================

def checkpoint_result_dir(
    seed,
    strategy,
    stage,
):

    if stage == 0:

        return (
            RESULT_ROOT
            / f"seed_{seed}"
            / "common"
            / STAGE_NAMES[
                stage
            ]
        )

    return (
        RESULT_ROOT
        / f"seed_{seed}"
        / strategy
        / STAGE_NAMES[
            stage
        ]
    )


# ============================================================
# 21. RÉSUMÉ D'UN CHECKPOINT
# ============================================================

def summarize_records(
    records,
    expected_domains,
):

    by_domain = {}

    for domain in expected_domains:

        subset = [
            record
            for record in records
            if record[
                "domain"
            ] == domain
        ]

        total = len(
            subset
        )

        correct = sum(
            record[
                "correct"
            ]
            for record in subset
        )

        truncated = sum(
            record[
                "truncated"
            ]
            for record in subset
        )

        statuses = Counter(
            record[
                "status"
            ]
            for record in subset
        )

        by_domain[
            domain
        ] = {
            "total":
                total,

            "correct":
                correct,

            "accuracy":
                (
                    correct
                    / total
                    if total
                    else None
                ),

            "truncated":
                truncated,

            "statuses":
                dict(
                    statuses
                ),
        }

    total = len(
        records
    )

    correct = sum(
        record[
            "correct"
        ]
        for record in records
    )

    domain_accuracies = [
        by_domain[
            domain
        ][
            "accuracy"
        ]
        for domain
        in expected_domains
        if by_domain[
            domain
        ][
            "accuracy"
        ]
        is not None
    ]

    return {
        "total":
            total,

        "correct":
            correct,

        "micro_accuracy":
            (
                correct
                / total
                if total
                else None
            ),

        "average_domain_accuracy":
            (
                sum(
                    domain_accuracies
                )
                / len(
                    domain_accuracies
                )
                if domain_accuracies
                else None
            ),

        "domains":
            by_domain,
    }


# ============================================================
# 22. ÉVALUER UN CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    seed,
    strategy,
    stage,
    adapter_dir,
    domains,
    rows_by_id,
    keys_by_id,
    by_domain,
    tokenizer,
    math_evaluator,
    mbpp_config,
    mbpp_run_case,
):

    output_dir = (
        checkpoint_result_dir(
            seed,
            strategy,
            stage,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records_file = (
        output_dir
        / "records.jsonl"
    )

    summary_file = (
        output_dir
        / "summary.json"
    )

    complete_file = (
        output_dir
        / "completed.json"
    )

    expected_rows = []

    for domain in domains:

        expected_rows.extend(
            by_domain[
                domain
            ]
        )

    expected_ids = {
        row["id"]
        for row in expected_rows
    }

    if complete_file.is_file():

        summary = load_json(
            summary_file
        )

        if (
            summary[
                "total"
            ]
            != len(
                expected_rows
            )
        ):

            raise RuntimeError(
                "Résumé complet incohérent."
            )

        print(
            "\nDéjà évalué : "
            f"{strategy} / stage {stage}"
        )

        return summary

    existing = (
        load_jsonl_if_exists(
            records_file
        )
    )

    existing_by_id = {
        row["id"]: row
        for row in existing
    }

    unexpected = (
        set(
            existing_by_id
        )
        - expected_ids
    )

    if unexpected:

        raise RuntimeError(
            "Résultats inattendus dans "
            f"{records_file}"
        )

    remaining = [
        row
        for row in expected_rows
        if row[
            "id"
        ]
        not in existing_by_id
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ÉVALUATION CONTINUAL"
    )

    print(
        "=" * 72
    )

    print(
        f"Stratégie : {strategy}"
    )

    print(
        f"Stage : {stage}"
    )

    print(
        "Domaines : "
        + ", ".join(
            domains
        )
    )

    print(
        "Déjà évalués : "
        f"{len(existing_by_id)}"
    )

    print(
        "Restants : "
        f"{len(remaining)}"
    )

    if not remaining:

        records = list(
            existing_by_id.values()
        )

        summary = summarize_records(
            records,
            domains,
        )

        save_json(
            summary_file,
            summary,
        )

        save_json(
            complete_file,
            {
                "complete": True,
                "total":
                    len(records),
            },
        )

        return summary

    model = load_model(
        adapter_dir
    )

    try:

        total_expected = len(
            expected_rows
        )

        offset = len(
            existing_by_id
        )

        for local_index, row in enumerate(
            remaining,
            start=1,
        ):

            example_id = row[
                "id"
            ]

            domain = row[
                "domain"
            ]

            grading = (
                keys_by_id[
                    example_id
                ]
            )

            prompt = build_prompt(
                tokenizer=tokenizer,
                math_evaluator=(
                    math_evaluator
                ),
                mbpp_config=(
                    mbpp_config
                ),
                row=row,
            )

            if domain == "programmation":

                max_tokens = (
                    mbpp_config[
                        "max_new_tokens"
                    ]
                )

            else:

                max_tokens = (
                    MATH_MAX_NEW_TOKENS
                )

            generation = (
                generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=(
                        max_tokens
                    ),
                )
            )

            if domain == "programmation":

                grade = (
                    grade_mbpp_example(
                        run_case=(
                            mbpp_run_case
                        ),
                        example_id=(
                            example_id
                        ),
                        grading=grading,
                        response=(
                            generation[
                                "response"
                            ]
                        ),
                    )
                )

            else:

                grade = (
                    grade_math_example(
                        evaluator=(
                            math_evaluator
                        ),
                        grading=grading,
                        response=(
                            generation[
                                "response"
                            ]
                        ),
                    )
                )

            record = {
                "id":
                    example_id,

                "domain":
                    domain,

                "strategy":
                    strategy,

                "stage":
                    stage,

                "correct":
                    grade[
                        "correct"
                    ],

                "status":
                    grade[
                        "status"
                    ],

                "format_ok":
                    grade[
                        "format_ok"
                    ],

                "tokens":
                    generation[
                        "tokens"
                    ],

                "truncated":
                    generation[
                        "truncated"
                    ],

                "response":
                    generation[
                        "response"
                    ],
            }

            append_jsonl(
                records_file,
                record,
            )

            global_index = (
                offset
                + local_index
            )

            print(
                f"[{global_index}/"
                f"{total_expected}] "
                f"{domain} : "
                f"{grade['status']} "
                f"({generation['tokens']} tokens)"
            )

    finally:

        release_model(
            model
        )

    records = load_jsonl(
        records_file
    )

    if len(
        records
    ) != len(
        expected_rows
    ):

        raise RuntimeError(
            "Évaluation incomplète : "
            f"{len(records)}/"
            f"{len(expected_rows)}"
        )

    summary = summarize_records(
        records,
        domains,
    )

    save_json(
        summary_file,
        summary,
    )

    save_json(
        complete_file,
        {
            "complete": True,
            "total":
                len(records),
            "validation_used":
                True,
            "test_used":
                False,
        },
    )

    print_checkpoint_summary(
        strategy,
        stage,
        summary,
    )

    return summary


# ============================================================
# 23. AFFICHAGE CHECKPOINT
# ============================================================

def print_checkpoint_summary(
    strategy,
    stage,
    summary,
):

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"BILAN {strategy} "
        f"- STAGE {stage}"
    )

    print(
        "-" * 72
    )

    for (
        domain,
        result,
    ) in summary[
        "domains"
    ].items():

        print(
            f"{domain}: "
            f"{result['correct']}/"
            f"{result['total']} "
            f"("
            f"{100 * result['accuracy']:.2f}%"
            f")"
        )

    print(
        "Average Domain Accuracy : "
        f"{100 * summary['average_domain_accuracy']:.2f}%"
    )


# ============================================================
# 24. CONSTRUIRE LA MATRICE A[t,d]
# ============================================================

def build_accuracy_matrix(
    seed,
    strategy,
    checkpoint_summaries,
):

    matrix = {}

    for stage in range(
        4
    ):

        if stage == 0:

            summary = (
                checkpoint_summaries[
                    "common_stage0"
                ]
            )

        else:

            summary = (
                checkpoint_summaries[
                    strategy
                ][stage]
            )

        matrix[
            str(stage)
        ] = {}

        for domain in domains_for_stage(
            stage
        ):

            matrix[
                str(stage)
            ][domain] = (
                summary[
                    "domains"
                ][domain][
                    "accuracy"
                ]
            )

    return matrix


# ============================================================
# 25. METRIQUES CONTINUAL
# ============================================================

def compute_continual_metrics(
    matrix,
):

    final_stage = 3

    acquisition = {}

    final_accuracy = {}

    forgetting = {}

    retention_ratio = {}

    backward_transfer = {}

    for domain in DOMAINS:

        learned_stage = (
            DOMAIN_STAGE[
                domain
            ]
        )

        acquisition_accuracy = (
            matrix[
                str(
                    learned_stage
                )
            ][domain]
        )

        final = (
            matrix[
                str(
                    final_stage
                )
            ][domain]
        )

        acquisition[
            domain
        ] = (
            acquisition_accuracy
        )

        final_accuracy[
            domain
        ] = final

        history = []

        for stage in range(
            learned_stage,
            final_stage + 1,
        ):

            if (
                domain
                in matrix[
                    str(stage)
                ]
            ):

                history.append(
                    matrix[
                        str(stage)
                    ][domain]
                )

        best_previous = max(
            history
        )

        forgetting[
            domain
        ] = max(
            0.0,
            best_previous
            - final,
        )

        if (
            acquisition_accuracy
            > 0
        ):

            retention_ratio[
                domain
            ] = (
                final
                / acquisition_accuracy
            )

        else:

            retention_ratio[
                domain
            ] = None

        backward_transfer[
            domain
        ] = (
            final
            - acquisition_accuracy
        )

    average_accuracy_by_stage = {}

    for stage in range(
        4
    ):

        values = list(
            matrix[
                str(stage)
            ].values()
        )

        average_accuracy_by_stage[
            str(stage)
        ] = (
            sum(values)
            / len(values)
        )

    old_domains = (
        "maths_appliques",
        "algebre_avancee",
        "maths_competition",
    )

    average_forgetting_final = (
        sum(
            forgetting[
                domain
            ]
            for domain
            in old_domains
        )
        / len(
            old_domains
        )
    )

    average_bwt_final = (
        sum(
            backward_transfer[
                domain
            ]
            for domain
            in old_domains
        )
        / len(
            old_domains
        )
    )

    final_average_accuracy = (
        sum(
            final_accuracy.values()
        )
        / len(
            final_accuracy
        )
    )

    return {
        "acquisition_accuracy":
            acquisition,

        "final_accuracy":
            final_accuracy,

        "forgetting_final":
            forgetting,

        "retention_ratio":
            retention_ratio,

        "backward_transfer":
            backward_transfer,

        "average_accuracy_by_stage":
            average_accuracy_by_stage,

        "final_average_accuracy":
            final_average_accuracy,

        "average_forgetting_final_old_domains":
            average_forgetting_final,

        "average_backward_transfer_old_domains":
            average_bwt_final,
    }


# ============================================================
# 26. CSV
# ============================================================

def write_matrix_csv(
    path,
    matrices,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        (
            "strategy,stage,"
            "maths_appliques,"
            "algebre_avancee,"
            "maths_competition,"
            "programmation"
        )
    ]

    for strategy in STRATEGIES:

        matrix = matrices[
            strategy
        ]

        for stage in range(
            4
        ):

            values = []

            for domain in DOMAINS:

                value = (
                    matrix[
                        str(stage)
                    ].get(
                        domain
                    )
                )

                if value is None:

                    values.append(
                        ""
                    )

                else:

                    values.append(
                        f"{100 * value:.6f}"
                    )

            lines.append(
                ",".join(
                    [
                        strategy,
                        str(stage),
                        *values,
                    ]
                )
            )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# 27. RAPPORT MARKDOWN
# ============================================================

def write_markdown_report(
    path,
    matrices,
    metrics,
    seed,
):

    lines = []

    lines.append(
        "# Continual RL V2 — Validation Report"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Seed: `{seed}`"
    )

    lines.append(
        ""
    )

    lines.append(
        "Independent test evaluated: **NO**"
    )

    lines.append(
        ""
    )

    lines.append(
        "Validation is used only for "
        "retrospective model assessment."
    )

    lines.append(
        ""
    )

    for strategy in STRATEGIES:

        lines.append(
            f"## {strategy}"
        )

        lines.append(
            ""
        )

        lines.append(
            "| Stage | GSM8K | Algebra | "
            "Competition | MBPP | Avg seen |"
        )

        lines.append(
            "|---:|---:|---:|---:|---:|---:|"
        )

        matrix = matrices[
            strategy
        ]

        for stage in range(
            4
        ):

            row = []

            for domain in DOMAINS:

                value = (
                    matrix[
                        str(stage)
                    ].get(
                        domain
                    )
                )

                if value is None:

                    row.append(
                        "—"
                    )

                else:

                    row.append(
                        f"{100 * value:.2f}%"
                    )

            avg = (
                metrics[
                    strategy
                ][
                    "average_accuracy_by_stage"
                ][
                    str(stage)
                ]
            )

            lines.append(
                f"| {stage} | "
                + " | ".join(
                    row
                )
                + f" | {100 * avg:.2f}% |"
            )

        lines.append(
            ""
        )

        final_avg = (
            metrics[
                strategy
            ][
                "final_average_accuracy"
            ]
        )

        avg_forgetting = (
            metrics[
                strategy
            ][
                "average_forgetting_final_old_domains"
            ]
        )

        bwt = (
            metrics[
                strategy
            ][
                "average_backward_transfer_old_domains"
            ]
        )

        lines.append(
            f"- Final average accuracy: "
            f"**{100 * final_avg:.2f}%**"
        )

        lines.append(
            f"- Final average forgetting: "
            f"**{100 * avg_forgetting:.2f} points**"
        )

        lines.append(
            f"- Average backward transfer: "
            f"**{100 * bwt:+.2f} points**"
        )

        lines.append(
            ""
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# 28. AFFICHAGE FINAL
# ============================================================

def print_final_comparison(
    matrices,
    metrics,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONTINUAL RL V2 - "
        "VALIDATION TERMINÉE"
    )

    print(
        "=" * 80
    )

    print(
        "\nMATRICE FINALE APRÈS STAGE 3"
    )

    print(
        "-" * 80
    )

    header = (
        f"{'Stratégie':<20}"
        f"{'GSM8K':>10}"
        f"{'Algèbre':>10}"
        f"{'Compét.':>10}"
        f"{'MBPP':>10}"
        f"{'Moyenne':>10}"
        f"{'Forget':>10}"
    )

    print(
        header
    )

    for strategy in STRATEGIES:

        final = (
            metrics[
                strategy
            ][
                "final_accuracy"
            ]
        )

        avg = (
            metrics[
                strategy
            ][
                "final_average_accuracy"
            ]
        )

        forgetting = (
            metrics[
                strategy
            ][
                "average_forgetting_final_old_domains"
            ]
        )

        print(
            f"{strategy:<20}"
            f"{100 * final['maths_appliques']:>9.2f}%"
            f"{100 * final['algebre_avancee']:>9.2f}%"
            f"{100 * final['maths_competition']:>9.2f}%"
            f"{100 * final['programmation']:>9.2f}%"
            f"{100 * avg:>9.2f}%"
            f"{100 * forgetting:>9.2f}"
        )

    print(
        "\nBACKWARD TRANSFER FINAL "
        "(anciens domaines)"
    )

    print(
        "-" * 80
    )

    for strategy in STRATEGIES:

        bwt = (
            metrics[
                strategy
            ][
                "average_backward_transfer_old_domains"
            ]
        )

        print(
            f"{strategy:<20}: "
            f"{100 * bwt:+.2f} points"
        )

    print(
        "\nLe test indépendant de "
        "714 problèmes reste fermé."
    )


# ============================================================
# 29. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            SEED_DEFAULT
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=(
            "all",
            *STRATEGIES,
        ),
        default="all",
    )

    args = parser.parse_args()

    print(
        "=" * 80
    )

    print(
        "CONTINUAL RL V2 "
        "- RETROSPECTIVE VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Seed : {args.seed}"
    )

    print(
        "Validation : 440 problèmes"
    )

    print(
        "Test indépendant : NON"
    )

    check_environment(
        args.seed
    )

    math_evaluator = (
        load_math_evaluator()
    )

    mbpp_run_case = (
        load_mbpp_runner()
    )

    mbpp_config = (
        load_mbpp_configuration()
    )

    print(
        "MBPP max_new_tokens : "
        f"{mbpp_config['max_new_tokens']}"
    )

    (
        rows_by_id,
        keys_by_id,
        by_domain,
    ) = load_validation()

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "left"
    )

    if (
        args.strategy
        == "all"
    ):

        selected_strategies = list(
            STRATEGIES
        )

    else:

        selected_strategies = [
            args.strategy
        ]

    checkpoint_summaries = {
        "common_stage0":
            None,
    }

    for strategy in STRATEGIES:

        checkpoint_summaries[
            strategy
        ] = {}

    # --------------------------------------------------------
    # Stage 0 commun.
    # Seulement GSM8K est déjà appris.
    # --------------------------------------------------------

    checkpoint_summaries[
        "common_stage0"
    ] = evaluate_checkpoint(
        seed=(
            args.seed
        ),
        strategy=(
            "common"
        ),
        stage=0,
        adapter_dir=(
            SFT_STAGE0_ADAPTER
        ),
        domains=[
            "maths_appliques"
        ],
        rows_by_id=(
            rows_by_id
        ),
        keys_by_id=(
            keys_by_id
        ),
        by_domain=(
            by_domain
        ),
        tokenizer=(
            tokenizer
        ),
        math_evaluator=(
            math_evaluator
        ),
        mbpp_config=(
            mbpp_config
        ),
        mbpp_run_case=(
            mbpp_run_case
        ),
    )

    # --------------------------------------------------------
    # Stage 1, 2, 3 pour chaque stratégie.
    #
    # On n'évalue que les compétences déjà vues.
    # --------------------------------------------------------

    for strategy in selected_strategies:

        for stage in (
            1,
            2,
            3,
        ):

            adapter = get_adapter(
                seed=(
                    args.seed
                ),
                strategy=(
                    strategy
                ),
                stage=stage,
            )

            domains = domains_for_stage(
                stage
            )

            checkpoint_summaries[
                strategy
            ][stage] = (
                evaluate_checkpoint(
                    seed=(
                        args.seed
                    ),
                    strategy=(
                        strategy
                    ),
                    stage=stage,
                    adapter_dir=(
                        adapter
                    ),
                    domains=(
                        domains
                    ),
                    rows_by_id=(
                        rows_by_id
                    ),
                    keys_by_id=(
                        keys_by_id
                    ),
                    by_domain=(
                        by_domain
                    ),
                    tokenizer=(
                        tokenizer
                    ),
                    math_evaluator=(
                        math_evaluator
                    ),
                    mbpp_config=(
                        mbpp_config
                    ),
                    mbpp_run_case=(
                        mbpp_run_case
                    ),
                )
            )

    # --------------------------------------------------------
    # Si une seule stratégie a été demandée,
    # on s'arrête avec son bilan.
    # --------------------------------------------------------

    if (
        args.strategy
        != "all"
    ):

        strategy = (
            args.strategy
        )

        matrix = build_accuracy_matrix(
            seed=(
                args.seed
            ),
            strategy=(
                strategy
            ),
            checkpoint_summaries=(
                checkpoint_summaries
            ),
        )

        metrics = (
            compute_continual_metrics(
                matrix
            )
        )

        result_dir = (
            RESULT_ROOT
            / f"seed_{args.seed}"
            / strategy
        )

        save_json(
            result_dir
            / "accuracy_matrix.json",
            matrix,
        )

        save_json(
            result_dir
            / "continual_metrics.json",
            metrics,
        )

        print(
            "\nÉvaluation de "
            f"{strategy} terminée."
        )

        return

    # --------------------------------------------------------
    # Agrégation des trois stratégies.
    # --------------------------------------------------------

    matrices = {}

    metrics = {}

    for strategy in STRATEGIES:

        matrices[
            strategy
        ] = build_accuracy_matrix(
            seed=(
                args.seed
            ),
            strategy=(
                strategy
            ),
            checkpoint_summaries=(
                checkpoint_summaries
            ),
        )

        metrics[
            strategy
        ] = (
            compute_continual_metrics(
                matrices[
                    strategy
                ]
            )
        )

    aggregate_dir = (
        RESULT_ROOT
        / f"seed_{args.seed}"
        / "aggregate"
    )

    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        aggregate_dir
        / "accuracy_matrices.json",
        matrices,
    )

    save_json(
        aggregate_dir
        / "continual_metrics.json",
        metrics,
    )

    write_matrix_csv(
        aggregate_dir
        / "accuracy_matrix.csv",
        matrices,
    )

    write_markdown_report(
        aggregate_dir
        / "validation_report.md",
        matrices,
        metrics,
        args.seed,
    )

    print_final_comparison(
        matrices,
        metrics,
    )

    print(
        "\nRapports :"
    )

    print(
        aggregate_dir
        / "accuracy_matrices.json"
    )

    print(
        aggregate_dir
        / "continual_metrics.json"
    )

    print(
        aggregate_dir
        / "accuracy_matrix.csv"
    )

    print(
        aggregate_dir
        / "validation_report.md"
    )


if __name__ == "__main__":

    main()
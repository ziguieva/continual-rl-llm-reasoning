import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import random
import re

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

from transformers.trainer_utils import (
    get_last_checkpoint,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_ADAPTER = (
    ROOT
    / "models"
    / "benchmark_v2_sft_stage0"
    / "final_adapter"
)

BASE_COMPLETION = (
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

TRAIN_KEYS_FILE = (
    ROOT
    / "data"
    / "benchmark_v2_v1"
    / "grading"
    / "train_keys.jsonl"
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

OUTPUT_ROOT = (
    ROOT
    / "models"
    / "continual_rl_v2"
)

DEVICE = "mps"

SEED_DEFAULT = 42

NUM_GENERATIONS = 4
MAX_COMPLETION_TOKENS = 256

# ------------------------------------------------------------
# 40 groupes par nouveau stage.
#
# No replay :
#   40 nouveaux problèmes.
#
# Fixed / Adaptive :
#   32 nouveaux problèmes
#   8 problèmes de replay
#
# => exactement 20 % de replay.
# ------------------------------------------------------------

STAGE_STEPS = 40

REPLAY_GROUPS = 8

CURRENT_GROUPS_WITH_REPLAY = (
    STAGE_STEPS - REPLAY_GROUPS
)

GRADIENT_ACCUMULATION_STEPS = 4

GENERATION_BATCH_SIZE = 4

LEARNING_RATE = 1e-5

SAVE_STEPS = 8

# ------------------------------------------------------------
# Probes internes à TRAIN.
# Ils servent uniquement au contrôleur adaptatif.
# ------------------------------------------------------------

PROBE_SIZE = 8

PROBE_MAX_NEW_TOKENS = 512

ADAPTIVE_EPSILON = 0.05

# ------------------------------------------------------------
# Mémoire épisodique bornée.
# ------------------------------------------------------------

MEMORY_PER_DOMAIN = 24


# ============================================================
# 2. DOMAINES
# ============================================================

DOMAIN_ORDER = (
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
        "Maths de compétition",

    "programmation":
        "Programmation / MBPP",
}

EXPECTED_DOMAIN_COUNTS = {
    "maths_appliques": 600,
    "algebre_avancee": 300,
    "maths_competition": 300,
    "programmation": 374,
}

STAGES = (
    {
        "index": 1,
        "name": "algebre_avancee",
        "domain": "algebre_avancee",
    },
    {
        "index": 2,
        "name": "maths_competition",
        "domain": "maths_competition",
    },
    {
        "index": 3,
        "name": "programmation",
        "domain": "programmation",
    },
)

STRATEGIES = (
    "no_replay",
    "fixed_replay",
    "adaptive_replay",
)


# ============================================================
# 3. PROMPT PROGRAMMATION
# ============================================================

CODE_SYSTEM_PROMPT = (
    "You are solving a Python programming problem. "
    "Return a complete Python solution defining the requested "
    "function. Output Python code only. "
    "Do not use Markdown fences and do not add explanations."
)


# ============================================================
# 4. UTILITAIRES
# ============================================================

def stable_offset(text):

    return int(
        hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


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

    ids = [
        row["id"]
        for row in rows
    ]

    if len(ids) != len(
        set(ids)
    ):

        raise RuntimeError(
            f"Identifiants dupliqués : {path}"
        )

    return rows


def load_jsonl_if_exists(path):

    if not path.is_file():

        return []

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


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


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
            f"Impossible de charger {path}"
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


def release_model(model):

    del model

    gc.collect()

    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# ============================================================
# 5. ENVIRONNEMENT
# ============================================================

def check_environment():

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    required = (
        TRAIN_FILE,
        TRAIN_KEYS_FILE,
        MATH_EVALUATOR_FILE,
        MBPP_SANDBOX_FILE,
        BASE_COMPLETION,
        BASE_ADAPTER
        / "adapter_config.json",
        BASE_ADAPTER
        / "adapter_model.safetensors",
    )

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    completion = load_json(
        BASE_COMPLETION
    )

    if completion.get(
        "mode"
    ) != "all":

        raise RuntimeError(
            "Le checkpoint SFT complet "
            "est requis."
        )

    if completion.get(
        "trained_examples"
    ) != 600:

        raise RuntimeError(
            "Le SFT stage0 doit provenir "
            "des 600 exemples GSM8K."
        )

    if completion.get(
        "global_step"
    ) != 75:

        raise RuntimeError(
            "75 étapes SFT attendues."
        )


# ============================================================
# 6. MODULES DE CORRECTION EXISTANTS
# ============================================================

def load_math_evaluator():

    evaluator = import_module(
        "continual_math_evaluator",
        MATH_EVALUATOR_FILE,
    )

    passed = (
        evaluator.run_verifier_tests()
    )

    print(
        "Tests vérificateur math : "
        f"{passed}/{passed} OK"
    )

    return evaluator


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
        "Impossible de trouver "
        "sandbox.run_case dans l'étape 33."
    )


# ============================================================
# 7. CHARGER LE BENCHMARK TRAIN
# ============================================================

def load_benchmark():

    rows = load_jsonl(
        TRAIN_FILE
    )

    key_rows = load_jsonl(
        TRAIN_KEYS_FILE
    )

    keys = {
        row["id"]: row
        for row in key_rows
    }

    by_domain = {
        domain: []
        for domain in DOMAIN_ORDER
    }

    for row in rows:

        domain = row["domain"]

        if domain in by_domain:

            by_domain[
                domain
            ].append(
                row
            )

        if row["id"] not in keys:

            raise RuntimeError(
                "Correction absente : "
                + row["id"]
            )

    for (
        domain,
        expected_count,
    ) in EXPECTED_DOMAIN_COUNTS.items():

        actual = len(
            by_domain[
                domain
            ]
        )

        if actual != expected_count:

            raise RuntimeError(
                f"{domain}: "
                f"{actual} exemples "
                f"au lieu de "
                f"{expected_count}."
            )

    return (
        rows,
        keys,
        by_domain,
    )


# ============================================================
# 8. PROBES ET POOLS
# ============================================================

def build_probe_ids(
    by_domain,
    seed,
):

    result = {}

    for index, domain in enumerate(
        DOMAIN_ORDER
    ):

        ids = [
            row["id"]
            for row in by_domain[
                domain
            ]
        ]

        rng = random.Random(
            seed
            + 10_000
            + index * 1003
        )

        result[domain] = (
            rng.sample(
                ids,
                PROBE_SIZE,
            )
        )

    return result


def build_training_pools(
    by_domain,
    probe_ids,
    seed,
):

    pools = {}

    for index, domain in enumerate(
        DOMAIN_ORDER
    ):

        forbidden = set(
            probe_ids[
                domain
            ]
        )

        ids = [
            row["id"]
            for row in by_domain[
                domain
            ]
            if row["id"]
            not in forbidden
        ]

        rng = random.Random(
            seed
            + 20_000
            + index * 2003
        )

        rng.shuffle(
            ids
        )

        pools[
            domain
        ] = ids

    return pools


# ============================================================
# 9. CONSTRUCTION DES PROMPTS
# ============================================================

def build_prompt(
    tokenizer,
    evaluator,
    row,
):

    if (
        row["domain"]
        == "programmation"
    ):

        system_prompt = (
            CODE_SYSTEM_PROMPT
        )

    else:

        system_prompt = (
            evaluator.SYSTEM_PROMPT
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": row["question"],
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
# 10. MBPP
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
            fenced.group(1)
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

    lines = text.splitlines()

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


def normalize_mbpp_grading(
    grading,
):

    payload = grading

    for key in (
        "case",
        "grading",
        "sandbox_case",
    ):

        nested = payload.get(
            key
        )

        if isinstance(
            nested,
            dict,
        ):

            payload = nested

            break

    setup = payload.get(
        "setup",
        "",
    )

    tests = payload.get(
        "tests"
    )

    if tests is None:

        for alias in (
            "test_list",
            "assertions",
            "test_cases",
        ):

            if alias in payload:

                tests = payload[
                    alias
                ]

                break

    if tests is None:

        raise RuntimeError(
            "Tests MBPP introuvables. "
            "Clés disponibles : "
            + ", ".join(
                sorted(
                    payload.keys()
                )
            )
        )

    if isinstance(
        tests,
        str,
    ):

        tests = [
            tests
        ]

    if not isinstance(
        tests,
        list,
    ):

        raise RuntimeError(
            "Format des tests MBPP "
            "non reconnu."
        )

    return {
        "setup": setup,
        "tests": tests,
    }


# ============================================================
# 11. MODÈLE
# ============================================================

def load_model(
    adapter_dir,
    trainable,
):

    print(
        "\nChargement : "
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
            is_trainable=trainable,
        )
    )

    model = model.to(
        DEVICE
    )

    model.config.use_cache = (
        not trainable
    )

    if not trainable:

        model.eval()

    return model


# ============================================================
# 12. RÉCOMPENSE
# ============================================================

def make_reward_function(
    evaluator,
    mbpp_run_case,
    events_file,
    trainer_holder,
):

    def reward_function(
        completions,
        expected,
        domain,
        grading_json,
        id,
        source_type,
        **kwargs,
    ):

        rewards = []

        correct_count = 0

        statuses = Counter()

        for (
            response,
            gold_text,
            task_domain,
            grading_text,
            example_id,
        ) in zip(
            completions,
            expected,
            domain,
            grading_json,
            id,
        ):

            if task_domain == "programmation":

                grading = json.loads(
                    grading_text
                )

                grading = (
                    normalize_mbpp_grading(
                        grading
                    )
                )

                code = extract_python_code(
                    response
                )

                case = {
                    "id": example_id,
                    "setup": grading[
                        "setup"
                    ],
                    "solution": code,
                    "tests": grading[
                        "tests"
                    ],
                }

                result = (
                    mbpp_run_case(
                        case
                    )
                )

                status = result.get(
                    "status",
                    "unknown",
                )

                correct = (
                    status == "passed"
                )

            else:

                gold = evaluator.parse_gold(
                    str(
                        gold_text
                    )
                )

                if not gold:

                    raise RuntimeError(
                        "Gold non interprétable : "
                        + example_id
                    )

                result = (
                    evaluator.grade_math(
                        response,
                        gold,
                    )
                )

                status = (
                    result["status"]
                )

                correct = bool(
                    result[
                        "correct"
                    ]
                )

            reward = float(
                correct
            )

            rewards.append(
                reward
            )

            correct_count += int(
                correct
            )

            statuses[
                status
            ] += 1

        trainer = (
            trainer_holder.get(
                "trainer"
            )
        )

        event = {
            "step_before_update": (
                trainer.state.global_step
                if trainer is not None
                else None
            ),

            "ids": list(
                dict.fromkeys(
                    id
                )
            ),

            "domains": list(
                dict.fromkeys(
                    domain
                )
            ),

            "source_types": list(
                dict.fromkeys(
                    source_type
                )
            ),

            "completions": len(
                completions
            ),

            "correct": (
                correct_count
            ),

            "rewards": rewards,

            "statuses": dict(
                statuses
            ),
        }

        append_jsonl(
            events_file,
            event,
        )

        print(
            "\nRécompenses : "
            f"{rewards} | "
            f"{correct_count}/"
            f"{len(completions)} correctes"
        )

        return rewards

    return reward_function


# ============================================================
# 13. PROBES POUR LE REPLAY ADAPTATIF
# ============================================================

def evaluate_math_probes(
    adapter_dir,
    domains,
    probe_ids,
    rows_by_id,
    keys_by_id,
    evaluator,
    tokenizer,
):

    model = load_model(
        adapter_dir,
        trainable=False,
    )

    scores = {}

    try:

        for domain in domains:

            correct = 0

            ids = probe_ids[
                domain
            ]

            for example_id in ids:

                row = rows_by_id[
                    example_id
                ]

                expected = str(
                    keys_by_id[
                        example_id
                    ][
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
                        "Gold probe invalide : "
                        + example_id
                    )

                prompt = build_prompt(
                    tokenizer,
                    evaluator,
                    row,
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(
                    DEVICE
                )

                with torch.inference_mode():

                    output = (
                        model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=(
                                PROBE_MAX_NEW_TOKENS
                            ),
                            pad_token_id=(
                                tokenizer.pad_token_id
                            ),
                            eos_token_id=(
                                tokenizer.eos_token_id
                            ),
                        )
                    )

                prompt_length = (
                    inputs[
                        "input_ids"
                    ].shape[1]
                )

                response = (
                    tokenizer.decode(
                        output[0][
                            prompt_length:
                        ],
                        skip_special_tokens=True,
                    )
                    .strip()
                )

                result = (
                    evaluator.grade_math(
                        response,
                        gold,
                    )
                )

                correct += int(
                    result[
                        "correct"
                    ]
                )

            scores[
                domain
            ] = (
                correct
                / len(ids)
            )

            print(
                "Probe "
                f"{DOMAIN_LABELS[domain]} : "
                f"{correct}/{len(ids)} "
                f"({100 * scores[domain]:.1f} %)"
            )

    finally:

        release_model(
            model
        )

    return scores


# ============================================================
# 14. MÉMOIRE DE REPLAY
# ============================================================

def load_or_create_memory(
    path,
    training_pools,
):

    if path.is_file():

        return load_json(
            path
        )

    memory = {
        "maths_appliques":
            training_pools[
                "maths_appliques"
            ][
                :MEMORY_PER_DOMAIN
            ]
    }

    save_json(
        path,
        memory,
    )

    return memory


def update_memory(
    memory,
    domain,
    plan,
    memory_file,
):

    current_ids = []

    for item in plan[
        "examples"
    ]:

        if (
            item["domain"]
            == domain
            and item[
                "source_type"
            ]
            == "current"
        ):

            if (
                item["id"]
                not in current_ids
            ):

                current_ids.append(
                    item["id"]
                )

    memory[
        domain
    ] = current_ids[
        :MEMORY_PER_DOMAIN
    ]

    save_json(
        memory_file,
        memory,
    )


# ============================================================
# 15. ALLOCATION ENTIÈRE DES SLOTS
# ============================================================

def allocate_slots(
    weights,
    total,
):

    domains = list(
        weights.keys()
    )

    if not domains:

        return {}

    total_weight = sum(
        max(
            0.0,
            weights[d],
        )
        for d in domains
    )

    if total_weight == 0:

        normalized = {
            d:
                1.0
                / len(domains)
            for d in domains
        }

    else:

        normalized = {
            d:
                max(
                    0.0,
                    weights[d],
                )
                / total_weight
            for d in domains
        }

    raw = {
        d:
            normalized[d]
            * total
        for d in domains
    }

    allocation = {
        d:
            math.floor(
                raw[d]
            )
        for d in domains
    }

    missing = (
        total
        - sum(
            allocation.values()
        )
    )

    order = sorted(
        domains,
        key=lambda d: (
            -(
                raw[d]
                - allocation[d]
            ),
            DOMAIN_ORDER.index(
                d
            ),
        ),
    )

    for domain in order[
        :missing
    ]:

        allocation[
            domain
        ] += 1

    return allocation


# ============================================================
# 16. FIXED REPLAY
# ============================================================

def fixed_allocation(
    old_domains,
):

    return allocate_slots(
        {
            domain: 1.0
            for domain in old_domains
        },
        REPLAY_GROUPS,
    )


# ============================================================
# 17. ADAPTIVE REPLAY
# ============================================================

def load_adaptive_state(path):

    if path.is_file():

        return load_json(
            path
        )

    return {
        "best_probe_accuracy": {},
        "history": [],
    }


def initialize_adaptive_reference(
    state,
    state_file,
    probe_ids,
    rows_by_id,
    keys_by_id,
    evaluator,
    tokenizer,
):

    if (
        "maths_appliques"
        in state[
            "best_probe_accuracy"
        ]
    ):

        return

    print(
        "\nRéférence initiale de "
        "rétention GSM8K..."
    )

    scores = evaluate_math_probes(
        adapter_dir=BASE_ADAPTER,
        domains=[
            "maths_appliques"
        ],
        probe_ids=probe_ids,
        rows_by_id=rows_by_id,
        keys_by_id=keys_by_id,
        evaluator=evaluator,
        tokenizer=tokenizer,
    )

    state[
        "best_probe_accuracy"
    ][
        "maths_appliques"
    ] = scores[
        "maths_appliques"
    ]

    state[
        "history"
    ].append({
        "event":
            "stage0_reference",

        "scores":
            scores,
    })

    save_json(
        state_file,
        state,
    )


def adaptive_allocation(
    state,
    state_file,
    current_adapter,
    old_domains,
    stage_index,
    probe_ids,
    rows_by_id,
    keys_by_id,
    evaluator,
    tokenizer,
):

    # --------------------------------------------------------
    # Stage 1 :
    # une seule ancienne compétence.
    # Aucun probe supplémentaire nécessaire.
    # --------------------------------------------------------

    if len(
        old_domains
    ) == 1:

        domain = old_domains[0]

        allocation = {
            domain:
                REPLAY_GROUPS
        }

        return {
            "scores": {
                domain:
                    state[
                        "best_probe_accuracy"
                    ][domain]
            },

            "best_before": dict(
                state[
                    "best_probe_accuracy"
                ]
            ),

            "forgetting": {
                domain: 0.0
            },

            "priorities": {
                domain: 1.0
            },

            "allocation":
                allocation,
        }

    print(
        "\nMesure de l'oubli avant "
        f"le stage {stage_index}..."
    )

    scores = evaluate_math_probes(
        adapter_dir=current_adapter,
        domains=old_domains,
        probe_ids=probe_ids,
        rows_by_id=rows_by_id,
        keys_by_id=keys_by_id,
        evaluator=evaluator,
        tokenizer=tokenizer,
    )

    best_before = dict(
        state[
            "best_probe_accuracy"
        ]
    )

    # Une compétence nouvellement apprise
    # prend son score courant comme référence.

    for domain in old_domains:

        if domain not in best_before:

            best_before[
                domain
            ] = scores[
                domain
            ]

    forgetting = {}

    priorities = {}

    for domain in old_domains:

        deficit = max(
            0.0,
            best_before[
                domain
            ]
            - scores[
                domain
            ],
        )

        forgetting[
            domain
        ] = deficit

        priorities[
            domain
        ] = (
            ADAPTIVE_EPSILON
            + deficit
        )

    allocation = allocate_slots(
        priorities,
        REPLAY_GROUPS,
    )

    # Mise à jour du meilleur score connu.

    for domain in old_domains:

        previous = state[
            "best_probe_accuracy"
        ].get(
            domain,
            scores[
                domain
            ],
        )

        state[
            "best_probe_accuracy"
        ][domain] = max(
            previous,
            scores[
                domain
            ],
        )

    state[
        "history"
    ].append({
        "event":
            "adaptive_allocation",

        "stage_index":
            stage_index,

        "scores":
            scores,

        "best_before":
            best_before,

        "forgetting":
            forgetting,

        "priorities":
            priorities,

        "allocation":
            allocation,
    })

    save_json(
        state_file,
        state,
    )

    return {
        "scores":
            scores,

        "best_before":
            best_before,

        "forgetting":
            forgetting,

        "priorities":
            priorities,

        "allocation":
            allocation,
    }


# ============================================================
# 18. ÉCHANTILLONNAGE DU REPLAY
# ============================================================

def sample_replay(
    memory,
    allocation,
    seed,
    stage_index,
):

    result = []

    for domain in DOMAIN_ORDER:

        count = allocation.get(
            domain,
            0,
        )

        if count == 0:

            continue

        available = memory.get(
            domain,
            [],
        )

        if len(
            available
        ) < count:

            raise RuntimeError(
                f"Mémoire insuffisante "
                f"pour {domain}: "
                f"{len(available)} < {count}"
            )

        rng = random.Random(
            seed
            + stage_index * 997
            + stable_offset(
                domain
            )
        )

        selected = rng.sample(
            available,
            count,
        )

        for example_id in selected:

            result.append({
                "id":
                    example_id,

                "domain":
                    domain,

                "source_type":
                    "replay",
            })

    return result


# ============================================================
# 19. PLAN D'ENTRAÎNEMENT D'UN STAGE
# ============================================================

def build_stage_plan(
    strategy,
    stage,
    training_pools,
    memory,
    seed,
    adaptive_info=None,
):

    index = stage[
        "index"
    ]

    current_domain = stage[
        "domain"
    ]

    if strategy == "no_replay":

        current_count = (
            STAGE_STEPS
        )

        allocation = {}

    else:

        current_count = (
            CURRENT_GROUPS_WITH_REPLAY
        )

        old_domains = list(
            DOMAIN_ORDER[
                :index
            ]
        )

        if strategy == "fixed_replay":

            allocation = (
                fixed_allocation(
                    old_domains
                )
            )

        elif strategy == "adaptive_replay":

            if adaptive_info is None:

                raise RuntimeError(
                    "Allocation adaptative absente."
                )

            allocation = (
                adaptive_info[
                    "allocation"
                ]
            )

        else:

            raise ValueError(
                strategy
            )

    current_ids = (
        training_pools[
            current_domain
        ][
            :current_count
        ]
    )

    examples = [
        {
            "id":
                example_id,

            "domain":
                current_domain,

            "source_type":
                "current",
        }
        for example_id in current_ids
    ]

    if allocation:

        examples.extend(
            sample_replay(
                memory=memory,
                allocation=allocation,
                seed=seed,
                stage_index=index,
            )
        )

    if len(
        examples
    ) != STAGE_STEPS:

        raise RuntimeError(
            f"{len(examples)} groupes "
            f"au lieu de {STAGE_STEPS}."
        )

    rng = random.Random(
        seed
        + index * 101
        + stable_offset(
            strategy
        )
    )

    rng.shuffle(
        examples
    )

    return {
        "strategy":
            strategy,

        "stage_index":
            index,

        "stage_name":
            stage["name"],

        "current_domain":
            current_domain,

        "current_groups":
            current_count,

        "replay_groups":
            STAGE_STEPS
            - current_count,

        "replay_allocation":
            allocation,

        "adaptive_info":
            adaptive_info,

        "examples":
            examples,
    }


# ============================================================
# 20. DATASET TRL
# ============================================================

def build_dataset(
    plan,
    rows_by_id,
    keys_by_id,
    tokenizer,
    evaluator,
):

    records = []

    for item in plan[
        "examples"
    ]:

        example_id = item[
            "id"
        ]

        row = rows_by_id[
            example_id
        ]

        grading = keys_by_id[
            example_id
        ]

        domain = row[
            "domain"
        ]

        if domain == "programmation":

            normalize_mbpp_grading(
                grading
            )

            expected = ""

        else:

            if "expected" not in grading:

                raise RuntimeError(
                    "expected absent : "
                    + example_id
                )

            expected = str(
                grading[
                    "expected"
                ]
            )

            if not evaluator.parse_gold(
                expected
            ):

                raise RuntimeError(
                    "Gold math invalide : "
                    + example_id
                )

        records.append({
            "id":
                example_id,

            "domain":
                domain,

            "source_type":
                item[
                    "source_type"
                ],

            "prompt":
                build_prompt(
                    tokenizer,
                    evaluator,
                    row,
                ),

            "expected":
                expected,

            "grading_json":
                json.dumps(
                    grading,
                    ensure_ascii=False,
                ),
        })

    return Dataset.from_list(
        records
    )


# ============================================================
# 21. CONFIGURATION GRPO
# ============================================================

def build_grpo_config(
    output_dir,
    seed,
):

    requested = {
        "output_dir":
            str(
                output_dir
            ),

        "max_steps":
            STAGE_STEPS,

        "per_device_train_batch_size":
            1,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "generation_batch_size":
            GENERATION_BATCH_SIZE,

        "num_generations":
            NUM_GENERATIONS,

        "max_completion_length":
            MAX_COMPLETION_TOKENS,

        "temperature":
            0.9,

        "top_p":
            0.9,

        "learning_rate":
            LEARNING_RATE,

        "lr_scheduler_type":
            "constant",

        "warmup_steps":
            0,

        "weight_decay":
            0.0,

        "optim":
            "adamw_torch",

        "max_grad_norm":
            1.0,

        "gradient_checkpointing":
            True,

        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },

        "fp16":
            False,

        "bf16":
            False,

        "beta":
            0.0,

        "loss_type":
            "dapo",

        "scale_rewards":
            "group",

        "num_iterations":
            1,

        "use_vllm":
            False,

        "remove_unused_columns":
            False,

        "logging_strategy":
            "steps",

        "logging_steps":
            1,

        "save_strategy":
            "steps",

        "save_steps":
            SAVE_STEPS,

        "save_total_limit":
            2,

        "save_only_model":
            False,

        "eval_strategy":
            "no",

        "report_to":
            "none",

        "push_to_hub":
            False,

        "dataloader_num_workers":
            0,

        "dataloader_pin_memory":
            False,

        "seed":
            seed,

        "data_seed":
            seed,
    }

    supported = set(
        inspect.signature(
            GRPOConfig
        ).parameters
    )

    unsupported = (
        set(
            requested
        )
        - supported
    )

    if unsupported:

        raise RuntimeError(
            "Options GRPO non reconnues : "
            + ", ".join(
                sorted(
                    unsupported
                )
            )
        )

    return GRPOConfig(
        **requested
    )


# ============================================================
# 22. MÉTADONNÉES
# ============================================================

def build_metadata(
    strategy,
    stage,
    seed,
    source_adapter,
    plan,
    probe_ids,
):

    return {
        "experiment":
            "continual_rl_v2",

        "strategy":
            strategy,

        "seed":
            seed,

        "stage_index":
            stage["index"],

        "stage_name":
            stage["name"],

        "current_domain":
            stage["domain"],

        "base_model":
            BASE_MODEL,

        "source_adapter":
            str(
                source_adapter
            ),

        "source_adapter_hash":
            sha256_file(
                source_adapter
                / "adapter_model.safetensors"
            ),

        "train_hash":
            sha256_file(
                TRAIN_FILE
            ),

        "keys_hash":
            sha256_file(
                TRAIN_KEYS_FILE
            ),

        "num_generations":
            NUM_GENERATIONS,

        "stage_steps":
            STAGE_STEPS,

        "max_completion_tokens":
            MAX_COMPLETION_TOKENS,

        "learning_rate":
            LEARNING_RATE,

        "reward":
            "correctness_only",

        "beta":
            0.0,

        "loss_type":
            "dapo",

        "current_groups":
            plan[
                "current_groups"
            ],

        "replay_groups":
            plan[
                "replay_groups"
            ],

        "replay_allocation":
            plan[
                "replay_allocation"
            ],

        "training_ids": [
            item["id"]
            for item in plan[
                "examples"
            ]
        ],

        "probe_ids":
            probe_ids,

        "validation_used":
            False,

        "test_used":
            False,
    }


# ============================================================
# 23. ENTRAÎNER UN STAGE
# ============================================================

def train_stage(
    strategy,
    stage,
    seed,
    source_adapter,
    stage_dir,
    plan,
    rows_by_id,
    keys_by_id,
    tokenizer,
    evaluator,
    mbpp_run_case,
    probe_ids,
):

    stage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_adapter = (
        stage_dir
        / "final_adapter"
    )

    completed_file = (
        stage_dir
        / "completed.json"
    )

    metadata_file = (
        stage_dir
        / "metadata.json"
    )

    plan_file = (
        stage_dir
        / "stage_plan.json"
    )

    events_file = (
        stage_dir
        / "reward_events.jsonl"
    )

    reward_report_file = (
        stage_dir
        / "reward_report.json"
    )

    training_log_file = (
        stage_dir
        / "training_log.json"
    )

    if completed_file.is_file():

        completion = load_json(
            completed_file
        )

        if (
            completion.get(
                "global_step"
            )
            != STAGE_STEPS
        ):

            raise RuntimeError(
                "Stage terminé avec un "
                "nombre d'étapes incorrect."
            )

        if not (
            final_adapter
            / "adapter_model.safetensors"
        ).is_file():

            raise RuntimeError(
                "Adaptateur final manquant."
            )

        print(
            "\nStage déjà terminé : "
            f"{stage_dir}"
        )

        return final_adapter

    if plan_file.is_file():

        previous_plan = load_json(
            plan_file
        )

        if previous_plan != plan:

            raise RuntimeError(
                "Le plan du stage "
                "a changé."
            )

    else:

        save_json(
            plan_file,
            plan,
        )

    metadata = build_metadata(
        strategy,
        stage,
        seed,
        source_adapter,
        plan,
        probe_ids,
    )

    if metadata_file.is_file():

        if (
            load_json(
                metadata_file
            )
            != metadata
        ):

            raise RuntimeError(
                "Métadonnées incompatibles."
            )

    else:

        save_json(
            metadata_file,
            metadata,
        )

    dataset = build_dataset(
        plan,
        rows_by_id,
        keys_by_id,
        tokenizer,
        evaluator,
    )

    config = build_grpo_config(
        stage_dir,
        seed,
    )

    model = load_model(
        source_adapter,
        trainable=True,
    )

    holder = {}

    reward_function = (
        make_reward_function(
            evaluator,
            mbpp_run_case,
            events_file,
            holder,
        )
    )

    trainer_parameters = set(
        inspect.signature(
            GRPOTrainer.__init__
        ).parameters
    )

    if (
        "processing_class"
        in trainer_parameters
    ):

        tokenizer_argument = {
            "processing_class":
                tokenizer
        }

    elif (
        "tokenizer"
        in trainer_parameters
    ):

        tokenizer_argument = {
            "tokenizer":
                tokenizer
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

    holder[
        "trainer"
    ] = trainer

    checkpoint = (
        get_last_checkpoint(
            str(
                stage_dir
            )
        )
    )

    if checkpoint:

        checkpoint_step = int(
            Path(
                checkpoint
            ).name.split(
                "-"
            )[-1]
        )

        events = (
            load_jsonl_if_exists(
                events_file
            )
        )

        valid_events = [
            event
            for event in events
            if (
                event.get(
                    "step_before_update"
                )
                is not None
                and event[
                    "step_before_update"
                ]
                < checkpoint_step
            )
        ]

        if (
            len(
                valid_events
            )
            != len(
                events
            )
        ):

            events_file.unlink(
                missing_ok=True
            )

            for event in valid_events:

                append_jsonl(
                    events_file,
                    event,
                )

        print(
            "\nReprise : "
            f"{checkpoint}"
        )

    else:

        if events_file.is_file():

            raise RuntimeError(
                "Journal présent sans "
                "checkpoint exploitable."
            )

        print(
            "\nNouvel entraînement."
        )

    train_result = trainer.train(
        resume_from_checkpoint=(
            checkpoint
        )
    )

    trainer.save_model(
        str(
            final_adapter
        )
    )

    tokenizer.save_pretrained(
        str(
            final_adapter
        )
    )

    trainer.save_state()

    save_json(
        training_log_file,
        trainer.state.log_history,
    )

    events = (
        load_jsonl_if_exists(
            events_file
        )
    )

    all_rewards = [
        reward
        for event in events
        for reward in event[
            "rewards"
        ]
    ]

    correct = sum(
        event["correct"]
        for event in events
    )

    groups_with_signal = sum(
        len(
            set(
                event["rewards"]
            )
        ) > 1
        for event in events
    )

    reward_report = {
        "groups":
            len(events),

        "generations":
            len(
                all_rewards
            ),

        "correct":
            correct,

        "groups_with_signal":
            groups_with_signal,

        "reward_distribution":
            dict(
                Counter(
                    str(
                        reward
                    )
                    for reward
                    in all_rewards
                )
            ),

        "events":
            events,
    }

    save_json(
        reward_report_file,
        reward_report,
    )

    completion = {
        "experiment":
            "continual_rl_v2",

        "strategy":
            strategy,

        "stage_index":
            stage["index"],

        "stage_name":
            stage["name"],

        "global_step":
            trainer.state.global_step,

        "training_loss":
            train_result.training_loss,

        "groups":
            len(events),

        "generations":
            len(
                all_rewards
            ),

        "correct_generations":
            correct,

        "groups_with_signal":
            groups_with_signal,

        "final_adapter":
            str(
                final_adapter
            ),

        "validation_evaluated":
            False,

        "test_evaluated":
            False,
    }

    save_json(
        completed_file,
        completion,
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        f"STAGE {stage['index']} TERMINÉ "
        f"- {strategy}"
    )

    print(
        "=" * 65
    )

    print(
        "Domaine : "
        f"{DOMAIN_LABELS[stage['domain']]}"
    )

    print(
        "Étapes : "
        f"{trainer.state.global_step}"
    )

    print(
        "Générations : "
        f"{len(all_rewards)}"
    )

    print(
        "Correctes : "
        f"{correct}"
    )

    print(
        "Groupes avec signal : "
        f"{groups_with_signal}/"
        f"{len(events)}"
    )

    print(
        "Adaptateur : "
        f"{final_adapter}"
    )

    release_model(
        model
    )

    return final_adapter


# ============================================================
# 24. PIPELINE D'UNE STRATÉGIE
# ============================================================

def run_strategy(
    strategy,
    seed,
    rows_by_id,
    keys_by_id,
    training_pools,
    probe_ids,
    tokenizer,
    evaluator,
    mbpp_run_case,
):

    root = (
        OUTPUT_ROOT
        / f"seed_{seed}"
        / strategy
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_file = (
        root
        / "replay_memory.json"
    )

    memory = (
        load_or_create_memory(
            memory_file,
            training_pools,
        )
    )

    adaptive_file = (
        root
        / "adaptive_state.json"
    )

    if (
        strategy
        == "adaptive_replay"
    ):

        adaptive_state = (
            load_adaptive_state(
                adaptive_file
            )
        )

        initialize_adaptive_reference(
            adaptive_state,
            adaptive_file,
            probe_ids,
            rows_by_id,
            keys_by_id,
            evaluator,
            tokenizer,
        )

    else:

        adaptive_state = None

    current_adapter = (
        BASE_ADAPTER
    )

    summaries = []

    for stage in STAGES:

        index = stage[
            "index"
        ]

        stage_dir = (
            root
            / (
                f"stage_{index}_"
                f"{stage['name']}"
            )
        )

        plan_file = (
            stage_dir
            / "stage_plan.json"
        )

        if plan_file.is_file():

            plan = load_json(
                plan_file
            )

        else:

            adaptive_info = None

            if (
                strategy
                == "adaptive_replay"
            ):

                old_domains = list(
                    DOMAIN_ORDER[
                        :index
                    ]
                )

                adaptive_info = (
                    adaptive_allocation(
                        state=(
                            adaptive_state
                        ),
                        state_file=(
                            adaptive_file
                        ),
                        current_adapter=(
                            current_adapter
                        ),
                        old_domains=(
                            old_domains
                        ),
                        stage_index=(
                            index
                        ),
                        probe_ids=(
                            probe_ids
                        ),
                        rows_by_id=(
                            rows_by_id
                        ),
                        keys_by_id=(
                            keys_by_id
                        ),
                        evaluator=(
                            evaluator
                        ),
                        tokenizer=(
                            tokenizer
                        ),
                    )
                )

            plan = build_stage_plan(
                strategy=(
                    strategy
                ),
                stage=stage,
                training_pools=(
                    training_pools
                ),
                memory=memory,
                seed=seed,
                adaptive_info=(
                    adaptive_info
                ),
            )

            save_json(
                plan_file,
                plan,
            )

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"{strategy.upper()} "
            f"- STAGE {index}/3"
        )

        print(
            DOMAIN_LABELS[
                stage["domain"]
            ]
        )

        print(
            "#" * 70
        )

        print(
            "Nouveaux groupes : "
            f"{plan['current_groups']}"
        )

        print(
            "Replay : "
            f"{plan['replay_groups']}"
        )

        if plan[
            "replay_allocation"
        ]:

            print(
                "Allocation : "
                f"{plan['replay_allocation']}"
            )

        current_adapter = train_stage(
            strategy=strategy,
            stage=stage,
            seed=seed,
            source_adapter=(
                current_adapter
            ),
            stage_dir=(
                stage_dir
            ),
            plan=plan,
            rows_by_id=(
                rows_by_id
            ),
            keys_by_id=(
                keys_by_id
            ),
            tokenizer=(
                tokenizer
            ),
            evaluator=(
                evaluator
            ),
            mbpp_run_case=(
                mbpp_run_case
            ),
            probe_ids=(
                probe_ids
            ),
        )

        update_memory(
            memory=memory,
            domain=(
                stage["domain"]
            ),
            plan=plan,
            memory_file=(
                memory_file
            ),
        )

        summary = load_json(
            stage_dir
            / "completed.json"
        )

        summary[
            "replay_allocation"
        ] = plan[
            "replay_allocation"
        ]

        summaries.append(
            summary
        )

    strategy_summary = {
        "experiment":
            "continual_rl_v2",

        "strategy":
            strategy,

        "seed":
            seed,

        "stage0_adapter":
            str(
                BASE_ADAPTER
            ),

        "stages":
            summaries,

        "final_adapter":
            str(
                current_adapter
            ),

        "validation_evaluated":
            False,

        "test_evaluated":
            False,
    }

    save_json(
        root
        / "strategy_summary.json",
        strategy_summary,
    )

    return strategy_summary


# ============================================================
# 25. BILAN GLOBAL
# ============================================================

def print_summary(
    summaries,
    seed,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CONTINUAL RL V2 - "
        "ENTRAÎNEMENT TERMINÉ"
    )

    print(
        "=" * 72
    )

    print(
        f"Seed : {seed}"
    )

    print(
        "Checkpoint commun : "
        "SFT stage0"
    )

    print(
        "Récompense : "
        "correctness-only"
    )

    print(
        "Budget par stage : "
        f"{STAGE_STEPS} groupes × "
        f"{NUM_GENERATIONS} générations"
    )

    print(
        "Replay Fixed/Adaptive : "
        f"{REPLAY_GROUPS}/"
        f"{STAGE_STEPS} = "
        f"{100 * REPLAY_GROUPS / STAGE_STEPS:.0f} %"
    )

    print(
        "Validation utilisée "
        "pour entraîner : NON"
    )

    print(
        "Test indépendant utilisé : NON"
    )

    for strategy in summaries:

        print(
            "\n"
            + "-" * 72
        )

        print(
            strategy[
                "strategy"
            ]
        )

        for stage in strategy[
            "stages"
        ]:

            print(
                f"Stage "
                f"{stage['stage_index']} "
                f"{stage['stage_name']} : "
                f"{stage['correct_generations']}/"
                f"{stage['generations']} "
                "générations correctes | "
                f"signal="
                f"{stage['groups_with_signal']}/"
                f"{stage['groups']} | "
                "replay="
                f"{stage['replay_allocation']}"
            )

        print(
            "Final : "
            f"{strategy['final_adapter']}"
        )

    print(
        "\nProchaine étape : "
        "évaluation rétrospective "
        "des checkpoints sur la validation."
    )


# ============================================================
# 26. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=(
            "all",
            *STRATEGIES,
        ),
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            SEED_DEFAULT
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 72
    )

    print(
        "CONTINUAL & DATA-EFFICIENT RL "
        "FOR LLM REASONING"
    )

    print(
        "=" * 72
    )

    print(
        "Stage 0 : GSM8K / SFT existant"
    )

    print(
        "Stage 1 : Algèbre avancée"
    )

    print(
        "Stage 2 : Maths compétition"
    )

    print(
        "Stage 3 : MBPP programmation"
    )

    print(
        "Récompense : justesse uniquement"
    )

    print(
        "Validation/test pendant "
        "l'entraînement : NON"
    )

    check_environment()

    set_seed(
        args.seed
    )

    evaluator = (
        load_math_evaluator()
    )

    mbpp_run_case = (
        load_mbpp_runner()
    )

    (
        rows,
        keys_by_id,
        by_domain,
    ) = load_benchmark()

    rows_by_id = {
        row["id"]: row
        for row in rows
    }

    probe_ids = build_probe_ids(
        by_domain,
        args.seed,
    )

    training_pools = (
        build_training_pools(
            by_domain,
            probe_ids,
            args.seed,
        )
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

    tokenizer.padding_side = "left"

    if (
        args.strategy
        == "all"
    ):

        strategies = list(
            STRATEGIES
        )

    else:

        strategies = [
            args.strategy
        ]

    summaries = []

    for strategy in strategies:

        summary = run_strategy(
            strategy=(
                strategy
            ),
            seed=(
                args.seed
            ),
            rows_by_id=(
                rows_by_id
            ),
            keys_by_id=(
                keys_by_id
            ),
            training_pools=(
                training_pools
            ),
            probe_ids=(
                probe_ids
            ),
            tokenizer=(
                tokenizer
            ),
            evaluator=(
                evaluator
            ),
            mbpp_run_case=(
                mbpp_run_case
            ),
        )

        summaries.append(
            summary
        )

    print_summary(
        summaries,
        args.seed,
    )


if __name__ == "__main__":

    main()
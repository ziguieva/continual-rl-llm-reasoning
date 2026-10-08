
import argparse
import ast
import hashlib
import importlib.util
import json
import re

from collections import Counter
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BENCHMARK = ROOT / "data/benchmark_v2_v1"

SANDBOX_SCRIPT = (
    ROOT / "experiments/33_audit_mbpp_sandbox.py"
)

OUTPUT_DIR = (
    ROOT / "results/benchmark_v2_mbpp_qwen_baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATIONS_FILE = (
    OUTPUT_DIR / "generations.jsonl"
)

GRADES_FILE = (
    OUTPUT_DIR / "grades.jsonl"
)

METADATA_FILE = (
    OUTPUT_DIR / "metadata.json"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "summary.json"
)

MAX_NEW_TOKENS = 512

SEED = 42

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SYSTEM_PROMPT = (
    "Tu es un ingénieur logiciel Python. "
    "Résous le problème de programmation donné. "
    "Écris uniquement le code Python complet, "
    "avec les fonctions nécessaires. "
    "Ne fournis ni explication, ni exemple "
    "d'utilisation, ni bloc Markdown. "
    "Le code doit pouvoir être exécuté "
    "directement et réussir les tests."
)

torch.manual_seed(SEED)

# ============================================================
# 2. UTILITAIRES
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(path):

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


def append_jsonl(path, record):

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def save_json(path, record):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            indent=2,
            ensure_ascii=False,
        )


def index_records(records):

    indexed = {}

    for record in records:

        example_id = record["id"]

        if example_id in indexed:

            raise ValueError(
                f"Identifiant dupliqué : {example_id}"
            )

        indexed[example_id] = record

    return indexed

# ============================================================
# 3. IMPORTER LE SANDBOX DE L'ÉTAPE 33
# ============================================================

def load_sandbox():

    if not SANDBOX_SCRIPT.is_file():

        raise FileNotFoundError(
            SANDBOX_SCRIPT
        )

    spec = importlib.util.spec_from_file_location(
        "mbpp_sandbox",
        SANDBOX_SCRIPT,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Import du sandbox impossible."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module

# ============================================================
# 4. EXTRACTION DU CODE GÉNÉRÉ
# ============================================================

def extract_code(response):

    raw = response.strip()

    if not raw:
        return ""

    # Cas d'une réponse avec un bloc Markdown.

    fenced = re.findall(
        r"```(?:python|py)?[ \t]*\n"
        r"(.*?)```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:

        return fenced[0].strip()

    # Cas où Qwen retourne directement
    # un programme Python valide.

    try:

        ast.parse(raw)

        return raw

    except SyntaxError:
        pass

    # Cas où le programme est précédé
    # d'une courte explication.

    lines = raw.splitlines()

    pattern = re.compile(
        r"^\s*(?:"
        r"def\s+|"
        r"async\s+def\s+|"
        r"class\s+|"
        r"import\s+|"
        r"from\s+|"
        r"@"
        r")"
    )

    for index, line in enumerate(lines):

        if not pattern.match(line):
            continue

        candidate = "\n".join(
            lines[index:]
        ).strip()

        try:

            ast.parse(candidate)

            return candidate

        except SyntaxError:
            continue

    # Une sortie mal formée reste
    # enregistrée pour l'audit.

    return raw

# ============================================================
# 5. CHARGEMENT DES QUESTIONS MBPP
# ============================================================

def load_cases(sandbox):

    reference_cases = (
        sandbox.load_mbpp_validation()
    )

    questions = load_jsonl(
        BENCHMARK / "validation.jsonl"
    )

    questions_by_id = index_records(
        questions
    )

    cases = []

    for reference in reference_cases:

        example_id = reference["id"]

        question = questions_by_id[
            example_id
        ]

        if question["domain"] != "programmation":

            raise ValueError(
                f"Domaine incorrect : {example_id}"
            )

        cases.append({
            "id": example_id,
            "question": question["question"],
            "setup": reference["setup"],
            "tests": reference["tests"],
        })

    if len(cases) != 90:

        raise ValueError(
            f"90 questions attendues, "
            f"{len(cases)} trouvées."
        )

    return cases

# ============================================================
# 6. CHARGEMENT DU MODÈLE
# ============================================================

def load_model():

    print("\nChargement de Qwen...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to(DEVICE)
    )

    model.eval()

    return model, tokenizer

# ============================================================
# 7. GÉNÉRATION D'UNE SOLUTION
# ============================================================

def generate_solution(
    model,
    tokenizer,
    question,
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
            ),
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated = (
        output[0][input_length:]
    )

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    token_count = len(
        generated
    )

    last_token = (
        int(generated[-1])
        if token_count
        else None
    )

    stop_tokens = {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }

    truncated = (
        token_count >= MAX_NEW_TOKENS
        and last_token not in stop_tokens
    )

    return {
        "response": response,
        "code": extract_code(response),
        "tokens": token_count,
        "truncated": truncated,
    }

# ============================================================
# 8. ÉVALUATION D'UNE SOLUTION DANS DOCKER
# ============================================================

def grade_solution(
    sandbox,
    case,
    generation,
):

    code = generation["code"]

    total_tests = len(
        case["tests"]
    )

    if not code.strip():

        return {
            "status": "empty_code",
            "tests_passed": 0,
            "tests_total": total_tests,
            "error": "Aucun code généré.",
        }

    # Vérification syntaxique uniquement.
    # Aucun code n'est exécuté sur le Mac.

    try:

        ast.parse(code)

    except SyntaxError as error:

        return {
            "status": "syntax_error",
            "tests_passed": 0,
            "tests_total": total_tests,
            "error": str(error),
        }

    # Le conteneur reçoit le programme
    # et les tests seulement au moment
    # de la correction.

    docker_case = {
        "id": case["id"],
        "setup": case["setup"],
        "solution": code,
        "tests": case["tests"],
    }

    return sandbox.run_case(
        docker_case
    )

# ============================================================
# 9. EMPREINTE DU PROTOCOLE
# ============================================================

def build_metadata(
    sandbox,
    docker_info,
):

    protocol = {
        "model": MODEL_NAME,
        "model_type": "base",
        "partition": "validation",
        "domain": "programmation",
        "seed": SEED,
        "device": DEVICE,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "questions_hash": sha256_file(
            BENCHMARK / "validation.jsonl"
        ),
        "grading_hash": sha256_file(
            BENCHMARK
            / "grading"
            / "validation_keys.jsonl"
        ),
        "sandbox_hash": sha256_file(
            SANDBOX_SCRIPT
        ),
        "docker_image": (
            docker_info["image_id"]
        ),
        "docker_timeout_seconds": (
            sandbox.TIMEOUT_SECONDS
        ),
    }

    return protocol

# ============================================================
# 10. ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Nombre de problèmes à évaluer.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Évaluer les 90 problèmes.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK V2 - QWEN MBPP BASELINE")
    print("=" * 65)

    print(f"Modèle : {MODEL_NAME}")
    print(f"Device : {DEVICE}")

    if DEVICE != "mps":

        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    sandbox = load_sandbox()

    docker_info = (
        sandbox.check_docker()
    )

    cases = load_cases(
        sandbox
    )

    if args.all:

        selected = cases

    else:

        if not 1 <= args.limit <= 90:

            raise ValueError(
                "--limit doit être entre 1 et 90."
            )

        selected = cases[
            :args.limit
        ]

    print(
        f"Problèmes sélectionnés : "
        f"{len(selected)}"
    )

    current_metadata = build_metadata(
        sandbox,
        docker_info,
    )

    # --------------------------------------------------------
    # Éviter de mélanger des évaluations
    # réalisées avec deux protocoles différents.
    # --------------------------------------------------------

    if METADATA_FILE.is_file():

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            saved_metadata = json.load(
                file
            )

        if saved_metadata != current_metadata:

            raise RuntimeError(
                "Le protocole a changé. "
                "Ne mélange pas les anciens "
                "résultats avec les nouveaux."
            )

    else:

        if (
            GENERATIONS_FILE.exists()
            or GRADES_FILE.exists()
        ):

            raise RuntimeError(
                "Des résultats existent sans "
                "fichier de métadonnées."
            )

        save_json(
            METADATA_FILE,
            current_metadata,
        )

    generations = index_records(
        load_jsonl(
            GENERATIONS_FILE
        )
    )

    grades = index_records(
        load_jsonl(
            GRADES_FILE
        )
    )

    # --------------------------------------------------------
    # Charger Qwen uniquement si au moins
    # une génération manque.
    # --------------------------------------------------------

    missing_generations = [
        case
        for case in selected
        if case["id"] not in generations
    ]

    if missing_generations:

        model, tokenizer = (
            load_model()
        )

    else:

        model = None
        tokenizer = None

    # ========================================================
    # 11. BOUCLE D'ÉVALUATION
    # ========================================================

    for index, case in enumerate(
        selected,
        start=1,
    ):

        example_id = case["id"]

        if example_id not in generations:

            generation = generate_solution(
                model,
                tokenizer,
                case["question"],
            )

            generation_record = {
                "id": example_id,
                **generation,
            }

            append_jsonl(
                GENERATIONS_FILE,
                generation_record,
            )

            generations[
                example_id
            ] = generation_record

        generation = generations[
            example_id
        ]

        if example_id not in grades:

            grade = grade_solution(
                sandbox,
                case,
                generation,
            )

            grade_record = {
                "id": example_id,
                **grade,
            }

            grades[
                example_id
            ] = grade_record

            # On ne conserve pas en cache
            # les erreurs temporaires de Docker.

            temporary_errors = {
                "container_error",
                "timeout",
                "invalid_runner_output",
                "invalid_runner_json",
            }

            if grade["status"] not in temporary_errors:

                append_jsonl(
                    GRADES_FILE,
                    grade_record,
                )

        result = grades[
            example_id
        ]

        print(
            f"[{index}/{len(selected)}] "
            f"{example_id} : "
            f"{result['status']} "
            f"({result['tests_passed']}/"
            f"{result['tests_total']})"
        )

    # ========================================================
    # 12. BILAN
    # ========================================================

    evaluated = [
        grades[case["id"]]
        for case in selected
    ]

    statuses = Counter(
        record["status"]
        for record in evaluated
    )

    passed = sum(
        record["status"] == "passed"
        and record["tests_passed"]
        == record["tests_total"]
        for record in evaluated
    )

    total = len(
        evaluated
    )

    total_tests = sum(
        record["tests_total"]
        for record in evaluated
    )

    passed_tests = sum(
        record["tests_passed"]
        for record in evaluated
    )

    pass_rate = (
        100 * passed / total
        if total
        else 0.0
    )

    summary = {
        "model": MODEL_NAME,
        "partition": "validation",
        "evaluated": total,
        "passed_exercises": passed,
        "pass_at_1": pass_rate,
        "tests_passed": passed_tests,
        "tests_total": total_tests,
        "status_counts": dict(statuses),
        "generated_file": str(
            GENERATIONS_FILE
        ),
        "grades_file": str(
            GRADES_FILE
        ),
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )

    print("\n" + "=" * 65)
    print("BILAN QWEN - MBPP VALIDATION")
    print("=" * 65)

    print(
        f"Problèmes évalués : {total}"
    )

    print(
        f"Exercices réussis : "
        f"{passed}/{total}"
    )

    print(
        f"Pass@1 : {pass_rate:.2f}%"
    )

    print(
        f"Tests réussis : "
        f"{passed_tests}/{total_tests}"
    )

    for status, count in sorted(
        statuses.items()
    ):

        print(
            f"{status}: {count}"
        )

    print(
        f"\nRésumé : {SUMMARY_FILE}"
    )

    if total < 90:

        print(
            "Évaluation partielle : "
            "90 exercices sont prévus."
        )

    else:

        print(
            "Évaluation complète "
            "de MBPP validation terminée."
        )

# ============================================================
# 13. EXÉCUTION
# ============================================================

if __name__ == "__main__":

    main()

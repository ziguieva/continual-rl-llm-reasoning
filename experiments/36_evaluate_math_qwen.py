
import argparse
import hashlib
import json
import re

from collections import Counter
from pathlib import Path

import torch

from math_verify import parse, verify

from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)

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

OUTPUT_DIR = (
    ROOT
    / "results"
    / "benchmark_v2_math_qwen_baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"
GRADES_FILE = OUTPUT_DIR / "grades.jsonl"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
)

EXPECTED_COUNTS = {
    "maths_appliques": 150,
    "algebre_avancee": 100,
    "maths_competition": 100,
}

MAX_NEW_TOKENS = 768

SEED = 42

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SYSTEM_PROMPT = (
    "You are solving a mathematical reasoning problem. "
    "Work through the problem carefully and show "
    "the essential reasoning steps. "
    "End your response with exactly one final line "
    "in this format: "
    "REPONSE: \\boxed{your final mathematical answer}. "
    "Put only the final answer inside the box. "
    "Do not write anything after the final line."
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
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    ids = [
        row["id"]
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Identifiants dupliqués : {path}"
        )

    return rows


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


def save_json(path, data):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def index_by_id(rows):

    return {
        row["id"]: row
        for row in rows
    }

# ============================================================
# 3. CHARGEMENT DE LA VALIDATION
# ============================================================

def load_cases():

    questions = load_jsonl(
        BENCHMARK / "validation.jsonl"
    )

    corrections = index_by_id(
        load_jsonl(
            BENCHMARK
            / "grading"
            / "validation_keys.jsonl"
        )
    )

    groups = {
        domain: []
        for domain in DOMAINS
    }

    for question in questions:

        domain = question["domain"]

        if domain not in DOMAINS:
            continue

        example_id = question["id"]

        if example_id not in corrections:

            raise ValueError(
                f"Correction absente : {example_id}"
            )

        correction = corrections[
            example_id
        ]

        if correction["domain"] != domain:

            raise ValueError(
                f"Domaine incohérent : {example_id}"
            )

        expected = str(
            correction["expected"]
        ).strip()

        if not expected:

            raise ValueError(
                f"Réponse attendue vide : {example_id}"
            )

        groups[domain].append({
            "id": example_id,
            "domain": domain,
            "question": question["question"],
            "expected": expected,
        })

    for domain, expected_count in EXPECTED_COUNTS.items():

        actual_count = len(
            groups[domain]
        )

        if actual_count != expected_count:

            raise ValueError(
                f"{domain} : "
                f"{actual_count} questions, "
                f"{expected_count} attendues."
            )

    # Entrelacer les domaines :
    # avec --limit 3, un problème
    # de chaque domaine est évalué.

    cases = []

    max_length = max(
        len(group)
        for group in groups.values()
    )

    for index in range(max_length):

        for domain in DOMAINS:

            group = groups[domain]

            if index < len(group):

                cases.append(
                    group[index]
                )

    if len(cases) != 350:

        raise ValueError(
            "350 problèmes mathématiques attendus."
        )

    return cases

# ============================================================
# 4. VÉRIFICATEUR MATHÉMATIQUE
# ============================================================

EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        boxed_match_priority=0,
    ),
    ExprExtractionConfig(),
]


def parse_expression(text):

    return parse(
        text,
        extraction_config=EXTRACTION_CONFIG,
        fallback_mode="no_fallback",
    )


def parse_gold(expected):

    # La correction peut être un nombre,
    # une fraction, une expression ou du LaTeX.

    candidates = [
        expected,
        "\\boxed{" + expected + "}",
    ]

    for candidate in candidates:

        parsed = parse_expression(
            candidate
        )

        if parsed:

            return parsed

    return []


def last_boxed(response):

    # Extraction du dernier \boxed{...},
    # y compris lorsque des accolades
    # sont imbriquées.

    matches = list(
        re.finditer(
            r"\\boxed\s*\{",
            response,
        )
    )

    for match in reversed(matches):

        opening = response.find(
            "{",
            match.start(),
        )

        depth = 0

        for index in range(
            opening,
            len(response),
        ):

            char = response[index]

            if char == "{":
                depth += 1

            elif char == "}":

                depth -= 1

                if depth == 0:

                    return response[
                        match.start():index + 1
                    ]

    return None


def extract_final(response):

    # Priorité au dernier résultat encadré.

    boxed = last_boxed(
        response
    )

    if boxed is not None:

        return {
            "answer": boxed,
            "extraction": "boxed",
        }

    # Alternative : ligne finale explicite.

    pattern = (
        r"(?im)^\s*"
        r"(?:REPONSE|RÉPONSE|FINAL_ANSWER|"
        r"Final answer)\s*:\s*(.+?)\s*$"
    )

    matches = re.findall(
        pattern,
        response,
    )

    if matches:

        return {
            "answer": matches[-1].strip(),
            "extraction": "final_line",
        }

    return {
        "answer": "",
        "extraction": "no_final_answer",
    }


def grade_math(response, gold_parsed):

    final = extract_final(
        response
    )

    answer = final["answer"]

    strict_format = bool(
        re.search(
            r"(?im)^\s*REPONSE\s*:\s*"
            r"\$?\s*\\boxed\s*\{",
            response,
        )
    )

    if not answer:

        return {
            "status": "no_final_answer",
            "correct": False,
            "format_ok": strict_format,
            "extraction": final["extraction"],
            "extracted_answer": "",
        }

    predicted = parse_expression(
        answer
    )

    if not predicted:

        return {
            "status": "unparsed_answer",
            "correct": False,
            "format_ok": strict_format,
            "extraction": final["extraction"],
            "extracted_answer": answer,
        }

    try:

        correct = bool(
            verify(
                gold_parsed,
                predicted,
            )
        )

    except Exception:

        return {
            "status": "verification_error",
            "correct": False,
            "format_ok": strict_format,
            "extraction": final["extraction"],
            "extracted_answer": answer,
        }

    return {
        "status": (
            "correct"
            if correct
            else "incorrect"
        ),
        "correct": correct,
        "format_ok": strict_format,
        "extraction": final["extraction"],
        "extracted_answer": answer,
    }

# ============================================================
# 5. TESTS DU VÉRIFICATEUR
# ============================================================

def run_verifier_tests():

    tests = [
        (
            "REPONSE: \\boxed{42}",
            "42",
            True,
        ),
        (
            "REPONSE: \\boxed{41}",
            "42",
            False,
        ),
        (
            "REPONSE: \\boxed{\\frac{3}{4}}",
            "\\frac{6}{8}",
            True,
        ),
        (
            "REPONSE: \\boxed{2938}",
            "2938",
            True,
        ),
        (
            "Calculation: 42",
            "42",
            False,
        ),
    ]

    passed = 0

    for response, expected, target in tests:

        gold = parse_gold(
            expected
        )

        if not gold:

            raise RuntimeError(
                f"Correction non interprétable : "
                f"{expected}"
            )

        result = grade_math(
            response,
            gold,
        )

        if result["correct"] != target:

            raise RuntimeError(
                "Échec d'un test du vérificateur : "
                f"{response}"
            )

        passed += 1

    return passed


def audit_gold(cases):

    parsed_gold = {}

    failures = []

    for case in cases:

        example_id = case["id"]

        parsed = parse_gold(
            case["expected"]
        )

        if parsed:

            parsed_gold[
                example_id
            ] = parsed

        else:

            failures.append(
                example_id
            )

    if failures:

        raise RuntimeError(
            "Certaines corrections sont "
            "non interprétables : "
            + ", ".join(
                failures[:10]
            )
        )

    return parsed_gold

# ============================================================
# 6. CHARGEMENT DE QWEN
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
# 7. GÉNÉRATION
# ============================================================

def generate_response(
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

    generated = output[0][
        input_length:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    token_count = len(
        generated
    )

    last_token = (
        int(generated[-1].item())
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
        "tokens": token_count,
        "truncated": truncated,
    }

# ============================================================
# 8. MÉTADONNÉES DU PROTOCOLE
# ============================================================

def build_metadata():

    return {
        "model": MODEL_NAME,
        "partition": "validation",
        "domains": list(DOMAINS),
        "seed": SEED,
        "device": DEVICE,
        "dtype": "float32",
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "system_prompt": SYSTEM_PROMPT,
        "questions_hash": sha256_file(
            BENCHMARK / "validation.jsonl"
        ),
        "grading_hash": sha256_file(
            BENCHMARK
            / "grading"
            / "validation_keys.jsonl"
        ),
        "evaluator_hash": sha256_file(
            Path(__file__).resolve()
        ),
    }

# ============================================================
# 9. RÉSUMÉ DES RÉSULTATS
# ============================================================

def print_summary(cases, generations, grades):

    domain_stats = {
        domain: {
            "total": 0,
            "correct": 0,
            "format_ok": 0,
            "truncated": 0,
        }
        for domain in DOMAINS
    }

    statuses = Counter()

    for case in cases:

        example_id = case["id"]
        domain = case["domain"]

        grade = grades[
            example_id
        ]

        generation = generations[
            example_id
        ]

        stats = domain_stats[
            domain
        ]

        stats["total"] += 1

        stats["correct"] += int(
            grade["correct"]
        )

        stats["format_ok"] += int(
            grade["format_ok"]
        )

        stats["truncated"] += int(
            generation["truncated"]
        )

        statuses[
            grade["status"]
        ] += 1

    total = len(cases)

    correct = sum(
        stats["correct"]
        for stats in domain_stats.values()
    )

    format_ok = sum(
        stats["format_ok"]
        for stats in domain_stats.values()
    )

    truncated = sum(
        stats["truncated"]
        for stats in domain_stats.values()
    )

    accuracy = (
        100 * correct / total
        if total
        else 0.0
    )

    summary = {
        "model": MODEL_NAME,
        "partition": "validation",
        "evaluated": total,
        "correct": correct,
        "accuracy_at_1_greedy": accuracy,
        "format_ok": format_ok,
        "truncated": truncated,
        "domains": domain_stats,
        "status_counts": dict(statuses),
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )

    print("\n" + "=" * 65)
    print("BILAN QWEN - MATHÉMATIQUES V2")
    print("=" * 65)

    print(
        f"Problèmes évalués : {total}"
    )

    print(
        f"Réponses correctes : "
        f"{correct}/{total}"
    )

    print(
        f"Exactitude@1 (greedy) : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Format respecté : "
        f"{format_ok}/{total}"
    )

    print(
        f"Générations tronquées : "
        f"{truncated}"
    )

    print("\nPAR DOMAINE")

    for domain, stats in domain_stats.items():

        domain_total = stats[
            "total"
        ]

        if domain_total == 0:
            continue

        domain_accuracy = (
            100
            * stats["correct"]
            / domain_total
        )

        print(
            f"{domain}: "
            f"{stats['correct']}"
            f"/{domain_total} "
            f"({domain_accuracy:.2f}%)"
        )

    print("\nSTATUTS")

    for status, count in sorted(
        statuses.items()
    ):

        print(
            f"{status}: {count}"
        )

    print(
        f"\nRésumé : {SUMMARY_FILE}"
    )

    if total < 350:

        print(
            "Évaluation partielle : "
            "350 problèmes sont prévus."
        )

    else:

        print(
            "Évaluation complète "
            "de la validation mathématique."
        )

# ============================================================
# 10. PROGRAMME PRINCIPAL
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
        help="Évaluer les 350 problèmes.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK V2 - QWEN MATH BASELINE")
    print("=" * 65)

    print(
        f"Modèle : {MODEL_NAME}"
    )

    print(
        f"Device : {DEVICE}"
    )

    if DEVICE != "mps":

        raise RuntimeError(
            "GPU Apple MPS indisponible."
        )

    cases = load_cases()

    # Vérifier toutes les corrections
    # avant de charger Qwen.

    tests_passed = run_verifier_tests()

    print(
        f"Tests du vérificateur : "
        f"{tests_passed}/{tests_passed} OK"
    )

    parsed_gold = audit_gold(
        cases
    )

    print(
        "Corrections de validation "
        f"interprétables : "
        f"{len(parsed_gold)}/350"
    )

    if args.all:

        selected = cases

    else:

        if not 1 <= args.limit <= 350:

            raise ValueError(
                "--limit doit être "
                "entre 1 et 350."
            )

        selected = cases[
            :args.limit
        ]

    print(
        f"Problèmes sélectionnés : "
        f"{len(selected)}"
    )

    metadata = build_metadata()

    # Ne jamais mélanger deux protocoles.

    if METADATA_FILE.is_file():

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            previous_metadata = (
                json.load(file)
            )

        if previous_metadata != metadata:

            raise RuntimeError(
                "Le protocole a changé. "
                "Conserve les anciens résultats "
                "et utilise un nouveau dossier "
                "pour une autre expérience."
            )

    else:

        if (
            GENERATIONS_FILE.exists()
            or GRADES_FILE.exists()
        ):

            raise RuntimeError(
                "Résultats existants "
                "sans métadonnées."
            )

        save_json(
            METADATA_FILE,
            metadata,
        )

    generations = index_by_id(
        load_jsonl(GENERATIONS_FILE)
        if GENERATIONS_FILE.exists()
        else []
    )

    grades = index_by_id(
        load_jsonl(GRADES_FILE)
        if GRADES_FILE.exists()
        else []
    )

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
    # 11. ÉVALUATION
    # ========================================================

    for index, case in enumerate(
        selected,
        start=1,
    ):

        example_id = case["id"]

        if example_id not in generations:

            generation = generate_response(
                model,
                tokenizer,
                case["question"],
            )

            record = {
                "id": example_id,
                **generation,
            }

            append_jsonl(
                GENERATIONS_FILE,
                record,
            )

            generations[
                example_id
            ] = record

        generation = generations[
            example_id
        ]

        if example_id not in grades:

            grade = grade_math(
                generation["response"],
                parsed_gold[example_id],
            )

            record = {
                "id": example_id,
                "domain": case["domain"],
                **grade,
            }

            append_jsonl(
                GRADES_FILE,
                record,
            )

            grades[
                example_id
            ] = record

        grade = grades[
            example_id
        ]

        print(
            f"[{index}/{len(selected)}] "
            f"{case['domain']} : "
            f"{grade['status']} "
            f"({generation['tokens']} tokens)"
        )

    # ========================================================
    # 12. BILAN
    # ========================================================

    print_summary(
        selected,
        generations,
        grades,
    )


if __name__ == "__main__":

    main()

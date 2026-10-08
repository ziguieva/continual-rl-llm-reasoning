
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid

from collections import Counter
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK = ROOT / "data/benchmark_v2_v1"

RESULTS = ROOT / "results"

SCRATCH = RESULTS / "_mbpp_docker_scratch"

REPORT = (
    RESULTS
    / "benchmark_v2_mbpp_sandbox_audit.json"
)

IMAGE = "python:3.11-slim"

TIMEOUT_SECONDS = 20

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

SCRATCH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. PROGRAMME EXÉCUTÉ DANS DOCKER
# ============================================================

RUNNER_SOURCE = r'''
import contextlib
import io
import json
import sys

with open(
    "/work/case.json",
    "r",
    encoding="utf-8",
) as file:
    case = json.load(file)

namespace = {
    "__name__": "__main__",
}

result = {
    "status": "passed",
    "tests_passed": 0,
    "tests_total": len(case["tests"]),
    "error": "",
}

captured_stdout = io.StringIO()
captured_stderr = io.StringIO()

try:

    with (
        contextlib.redirect_stdout(captured_stdout),
        contextlib.redirect_stderr(captured_stderr),
    ):

        setup = case["setup"]

        if setup.strip():

            exec(
                compile(
                    setup,
                    "<test_setup>",
                    "exec",
                ),
                namespace,
            )

        exec(
            compile(
                case["solution"],
                "<reference_solution>",
                "exec",
            ),
            namespace,
        )

        for index, test in enumerate(
            case["tests"]
        ):

            exec(
                compile(
                    test,
                    f"<test_{index}>",
                    "exec",
                ),
                namespace,
            )

            result["tests_passed"] += 1

except ModuleNotFoundError as error:

    result["status"] = "missing_dependency"
    result["error"] = str(error)

except AssertionError as error:

    result["status"] = "failed_assertion"
    result["error"] = str(error)

except SyntaxError as error:

    result["status"] = "syntax_error"
    result["error"] = str(error)

except BaseException as error:

    result["status"] = "execution_error"

    result["error"] = (
        f"{type(error).__name__}: {error}"
    )

sys.__stdout__.write(
    "MBPP_AUDIT_RESULT="
    + json.dumps(result)
    + "\n"
)
'''


# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def load_jsonl(path):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_report(data):

    with REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 4. VÉRIFICATION DE DOCKER
# ============================================================

def check_docker():

    try:

        info = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "{{.ServerVersion}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as error:

        raise RuntimeError(
            "Docker est introuvable "
            "ou ne répond pas."
        ) from error

    if info.returncode != 0:

        raise RuntimeError(
            "Docker Desktop doit être démarré.\n"
            + info.stderr
        )

    image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            IMAGE,
            "--format",
            "{{.Id}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if image.returncode != 0:

        raise RuntimeError(
            "Image Python absente. "
            "Exécute : docker pull "
            + IMAGE
        )

    return {
        "docker_server": info.stdout.strip(),
        "image": IMAGE,
        "image_id": image.stdout.strip(),
    }


# ============================================================
# 5. CHARGEMENT DES EXERCICES MBPP
# ============================================================

def load_mbpp_validation():

    questions = load_jsonl(
        BENCHMARK / "validation.jsonl"
    )

    corrections = load_jsonl(
        BENCHMARK
        / "grading"
        / "validation_keys.jsonl"
    )

    keys = {
        row["id"]: row
        for row in corrections
    }

    if len(keys) != len(corrections):

        raise ValueError(
            "Corrections dupliquées."
        )

    cases = []

    for question in questions:

        if question["domain"] != "programmation":
            continue

        example_id = question["id"]

        if example_id not in keys:

            raise ValueError(
                f"Correction absente : {example_id}"
            )

        key = keys[example_id]

        tests = key.get(
            "tests"
        ) or {}

        all_tests = (
            (tests.get("public") or [])
            + (tests.get("hidden") or [])
            + (tests.get("challenge") or [])
        )

        if not all_tests:

            raise ValueError(
                f"Aucun test : {example_id}"
            )

        solution = (
            key.get("reference_solution")
            or ""
        )

        if not solution.strip():

            raise ValueError(
                f"Solution absente : {example_id}"
            )

        cases.append({
            "id": example_id,
            "setup": tests.get("setup") or "",
            "solution": solution,
            "tests": all_tests,
        })

    if len(cases) != 90:

        raise ValueError(
            f"90 exercices MBPP attendus, "
            f"{len(cases)} trouvés."
        )

    return cases


# ============================================================
# 6. EXÉCUTION ISOLÉE DANS DOCKER
# ============================================================

def run_case(case):

    with tempfile.TemporaryDirectory(
        prefix="mbpp_",
        dir=SCRATCH,
    ) as directory:

        workdir = Path(
            directory
        )

        os.chmod(
            workdir,
            0o755,
        )

        case_file = (
            workdir / "case.json"
        )

        runner_file = (
            workdir / "runner.py"
        )

        case_file.write_text(
            json.dumps(
                case,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        runner_file.write_text(
            RUNNER_SOURCE,
            encoding="utf-8",
        )

        os.chmod(
            case_file,
            0o644,
        )

        os.chmod(
            runner_file,
            0o644,
        )

        container_name = (
            "mbpp-v2-"
            + uuid.uuid4().hex[:12]
        )

        command = [
            "docker",
            "run",

            "--rm",

            "--name",
            container_name,

            # Réseau désactivé.
            "--network",
            "none",

            # Utiliser uniquement
            # l'image déjà téléchargée.
            "--pull",
            "never",

            # Système en lecture seule.
            "--read-only",

            # Supprimer les capacités
            # Linux supplémentaires.
            "--cap-drop",
            "ALL",

            "--security-opt",
            "no-new-privileges=true",

            # Utilisateur non privilégié.
            "--user",
            "65534:65534",

            # Limites de ressources.
            "--memory",
            "512m",

            "--cpus",
            "1",

            "--pids-limit",
            "64",

            # Espace temporaire limité.
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",

            # Monter seulement les fichiers
            # nécessaires, en lecture seule.
            "--mount",
            (
                "type=bind,"
                f"src={workdir.resolve()},"
                "dst=/work,"
                "readonly"
            ),

            "--workdir",
            "/work",

            IMAGE,

            "python",
            "-I",
            "-B",
            "/work/runner.py",
        ]

        try:

            execution = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )

        except subprocess.TimeoutExpired:

            # Arrêter le conteneur
            # en cas de dépassement du délai.
            try:

                subprocess.run(
                    [
                        "docker",
                        "rm",
                        "-f",
                        container_name,
                    ],
                    capture_output=True,
                    timeout=10,
                )

            except subprocess.TimeoutExpired:
                pass

            return {
                "status": "timeout",
                "tests_passed": 0,
                "tests_total": len(case["tests"]),
                "error": (
                    f"Délai de "
                    f"{TIMEOUT_SECONDS}s dépassé."
                ),
            }

        if execution.returncode != 0:

            return {
                "status": "container_error",
                "tests_passed": 0,
                "tests_total": len(case["tests"]),
                "error": execution.stderr[-1500:],
            }

        marker = (
            "MBPP_AUDIT_RESULT="
        )

        result_lines = [
            line[len(marker):]
            for line in execution.stdout.splitlines()
            if line.startswith(marker)
        ]

        if len(result_lines) != 1:

            return {
                "status": "invalid_runner_output",
                "tests_passed": 0,
                "tests_total": len(case["tests"]),
                "error": execution.stdout[-1500:],
            }

        try:

            return json.loads(
                result_lines[0]
            )

        except json.JSONDecodeError:

            return {
                "status": "invalid_runner_json",
                "tests_passed": 0,
                "tests_total": len(case["tests"]),
                "error": result_lines[0][-1500:],
            }


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help=(
            "Nombre d'exercices à vérifier. "
            "Par défaut : 3."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Vérifier les 90 exercices.",
    )

    args = parser.parse_args()

    print("=" * 65)

    print(
        "BENCHMARK V2 - AUDIT MBPP DANS DOCKER"
    )

    print("=" * 65)

    docker_info = check_docker()

    print(
        f"Docker : "
        f"{docker_info['docker_server']}"
    )

    print(
        f"Image : "
        f"{docker_info['image']}"
    )

    cases = load_mbpp_validation()

    if args.all:

        selected = cases

    else:

        if args.limit < 1:

            raise ValueError(
                "--limit doit être positif."
            )

        selected = cases[
            :args.limit
        ]

    protocol_hash = sha256_text(
        RUNNER_SOURCE
        + docker_info["image_id"]
        + str(TIMEOUT_SECONDS)
    )

    report = {
        "benchmark": "V2",
        "partition": "validation",
        "domain": "programmation",
        "dataset": "MBPP",
        "docker": docker_info,
        "protocol_hash": protocol_hash,
        "reference_solutions_only": True,
        "model_inference_performed": False,
        "cases": {},
    }

    # Réutiliser les résultats
    # déjà calculés.
    if REPORT.is_file():

        with REPORT.open(
            "r",
            encoding="utf-8",
        ) as file:

            previous = json.load(
                file
            )

        if (
            previous.get("protocol_hash")
            == protocol_hash
        ):

            report["cases"] = (
                previous.get("cases")
                or {}
            )

    for index, case in enumerate(
        selected,
        start=1,
    ):

        example_id = case[
            "id"
        ]

        case_hash = sha256_text(
            json.dumps(
                case,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        previous = report[
            "cases"
        ].get(
            example_id
        )

        if (
            previous is not None
            and previous.get("case_hash")
            == case_hash
        ):

            result = previous[
                "result"
            ]

        else:

            result = run_case(
                case
            )

            report["cases"][
                example_id
            ] = {
                "case_hash": case_hash,
                "result": result,
            }

            # Sauvegarder après
            # chaque exercice.
            save_report(
                report
            )

        print(
            f"[{index}/{len(selected)}] "
            f"{example_id} : "
            f"{result['status']} "
            f"({result['tests_passed']}/"
            f"{result['tests_total']} tests)"
        )

    selected_results = [
        report["cases"][
            case["id"]
        ]["result"]
        for case in selected
    ]

    stats = Counter(
        item["status"]
        for item in selected_results
    )

    total_tests = sum(
        item["tests_total"]
        for item in selected_results
    )

    passed_tests = sum(
        item["tests_passed"]
        for item in selected_results
    )

    report["last_run"] = {
        "selected_cases": len(selected),
        "status_counts": dict(stats),
        "tests_passed": passed_tests,
        "tests_total": total_tests,
    }

    save_report(
        report
    )

    print("\n" + "=" * 65)

    print(
        "BILAN DE L'AUDIT MBPP"
    )

    print("=" * 65)

    print(
        f"Exercices : {len(selected)}"
    )

    for status, count in sorted(
        stats.items()
    ):

        print(
            f"{status}: {count}"
        )

    print(
        f"Tests réussis : "
        f"{passed_tests}/{total_tests}"
    )

    print(
        f"Rapport : {REPORT}"
    )

    if len(selected) < 90:

        print(
            "\nAudit partiel : "
            "les 90 exercices ne sont "
            "pas encore vérifiés."
        )

    elif stats.get("passed", 0) == 90:

        print(
            "\nLes 90 solutions de référence "
            "ont réussi les tests."
        )

    else:

        print(
            "\nCertains exercices nécessitent "
            "une analyse avant "
            "l'évaluation des modèles."
        )


if __name__ == "__main__":
    main()

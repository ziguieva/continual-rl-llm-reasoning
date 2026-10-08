
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "data/benchmark_v2_v1"

SEED = 42

# On conserve les partitions officielles :
# jamais de données de test officielles
# dans notre entraînement.

SOURCES = {
    "gsm8k": "openai/gsm8k",
    "math": "EleutherAI/hendrycks_math",
    "mbpp": "google-research-datasets/mbpp",
    "bbh": "Joschka/big_bench_hard",
    "humaneval": "evalplus/humanevalplus",
}

public = {
    "train": [],
    "validation": [],
    "test": [],
}

grading = {
    "train": [],
    "validation": [],
    "test": [],
}

source_metadata = []

seen_questions = {}

print("=" * 65)
print("BENCHMARK V2 - PRÉPARATION")
print("=" * 65)

# ============================================================
# 2. OUTILS
# ============================================================

def write_jsonl(path, records):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def question_hash(question):

    normalized = " ".join(
        question.split()
    ).casefold()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def stable_rng(identifier):

    digest = hashlib.sha256(
        identifier.encode("utf-8")
    ).digest()

    offset = int.from_bytes(
        digest[:8],
        "big",
    )

    return random.Random(
        SEED + offset
    )


def load_source(
    source,
    config,
    split,
):

    print(
        f"Chargement : {source} "
        f"/ {config} / {split}"
    )

    dataset = load_dataset(
        source,
        config,
        split=split,
    )

    source_metadata.append({
        "source": source,
        "config": config,
        "official_split": split,
        "fingerprint": dataset._fingerprint,
        "total_rows": len(dataset),
    })

    return dataset


def shuffled_indices(
    dataset,
    identifier,
    predicate=None,
):

    indices = []

    for index, row in enumerate(dataset):

        if (
            predicate is None
            or predicate(row)
        ):

            indices.append(index)

    rng = stable_rng(
        identifier
    )

    rng.shuffle(
        indices
    )

    return indices


# ============================================================
# 3. EXTRACTION DES RÉPONSES
# ============================================================

def gsm8k_answer(answer):

    match = re.search(
        r"####\s*(.+?)\s*$",
        answer,
        flags=re.DOTALL,
    )

    if not match:

        raise ValueError(
            "Réponse GSM8K non reconnue."
        )

    return match.group(1).strip()


def extract_last_boxed(solution):

    # Extraction d'une expression LaTeX
    # \boxed{...}, y compris avec des
    # accolades imbriquées.

    marker = r"\boxed{"

    start = solution.rfind(
        marker
    )

    if start == -1:

        return None

    start += len(
        marker
    )

    depth = 1

    position = start

    while position < len(solution):

        character = solution[
            position
        ]

        if character == "{":

            depth += 1

        elif character == "}":

            depth -= 1

            if depth == 0:

                return solution[
                    start:position
                ].strip()

        position += 1

    return None


def math_level_at_least_three(row):

    label = str(
        row.get("level", "")
    )

    match = re.search(
        r"\d+",
        label,
    )

    if not match:

        return False

    level = int(
        match.group()
    )

    return 3 <= level <= 5


# ============================================================
# 4. AJOUT D'UN EXERCICE
# ============================================================

def add_example(
    partition,
    domain,
    subtype,
    source,
    source_config,
    official_split,
    original_index,
    question,
    expected,
    reference_solution="",
    tests=None,
    metadata=None,
):

    if partition not in public:

        raise ValueError(
            f"Partition inconnue : {partition}"
        )

    if not question.strip():

        raise ValueError(
            "Question vide."
        )

    fingerprint = question_hash(
        question
    )

    if fingerprint in seen_questions:

        previous = seen_questions[
            fingerprint
        ]

        raise ValueError(
            "Fuite ou doublon détecté : "
            f"{previous} / "
            f"{source_config}:{original_index}"
        )

    example_id = (
        f"v2_{source_config}_"
        f"{official_split}_"
        f"{original_index:05d}"
    )

    if any(
        example_id == item["id"]
        for partition_rows in public.values()
        for item in partition_rows
    ):

        raise ValueError(
            f"ID dupliqué : {example_id}"
        )

    seen_questions[
        fingerprint
    ] = example_id

    public_record = {
        "id": example_id,
        "domain": domain,
        "subtype": subtype,
        "source": source,
        "source_config": source_config,
        "official_split": official_split,
        "partition": partition,
        "question": question,
        "question_hash": fingerprint,
        "metadata": metadata or {},
    }

    grading_record = {
        "id": example_id,
        "domain": domain,
        "expected": expected,
        "reference_solution": reference_solution,
        "tests": tests or {},
    }

    # Les solutions d'entraînement sont
    # nécessaires au SFT. Elles ne sont
    # jamais publiées pour validation/test.

    if partition == "train":

        public_record["target"] = (
            reference_solution
            if reference_solution
            else expected
        )

    public[
        partition
    ].append(
        public_record
    )

    grading[
        partition
    ].append(
        grading_record
    )


# ============================================================
# 5. GSM8K - PROBLÈMES MULTI-ÉTAPES
# ============================================================

print("\n[1/6] GSM8K")

gsm_train = load_source(
    SOURCES["gsm8k"],
    "main",
    "train",
)

gsm_test = load_source(
    SOURCES["gsm8k"],
    "main",
    "test",
)

train_indices = shuffled_indices(
    gsm_train,
    "gsm8k_train",
)

test_indices = shuffled_indices(
    gsm_test,
    "gsm8k_test",
)

# 600 train + 150 validation,
# tous issus du train officiel.

for partition, indices in [
    ("train", train_indices[:600]),
    (
        "validation",
        train_indices[600:750],
    ),
    ("test", test_indices[:150]),
]:

    dataset = (
        gsm_test
        if partition == "test"
        else gsm_train
    )

    original_split = (
        "test"
        if partition == "test"
        else "train"
    )

    for index in indices:

        row = dataset[
            index
        ]

        add_example(
            partition=partition,
            domain="maths_appliques",
            subtype="gsm8k",
            source=SOURCES["gsm8k"],
            source_config="gsm8k_main",
            official_split=original_split,
            original_index=index,
            question=row["question"],
            expected=gsm8k_answer(
                row["answer"]
            ),
            reference_solution=row["answer"],
        )


# ============================================================
# 6. MATH - MATHÉMATIQUES AVANCÉES
# ============================================================

print("\n[2/6] MATH - mathématiques")

def prepare_math_config(
    config,
    domain,
):

    source = SOURCES[
        "math"
    ]

    train_data = load_source(
        source,
        config,
        "train",
    )

    test_data = load_source(
        source,
        config,
        "test",
    )

    train_indices = shuffled_indices(
        train_data,
        f"math_{config}_train",
        math_level_at_least_three,
    )

    test_indices = shuffled_indices(
        test_data,
        f"math_{config}_test",
        math_level_at_least_three,
    )

    # Par catégorie :
    # 150 train + 50 validation
    # + 50 test officiel.

    if len(train_indices) < 200:

        raise ValueError(
            f"Pas assez de train : {config}"
        )

    if len(test_indices) < 50:

        raise ValueError(
            f"Pas assez de test : {config}"
        )

    groups = [
        (
            "train",
            "train",
            train_data,
            train_indices[:150],
        ),
        (
            "validation",
            "train",
            train_data,
            train_indices[150:200],
        ),
        (
            "test",
            "test",
            test_data,
            test_indices[:50],
        ),
    ]

    for (
        partition,
        official_split,
        dataset,
        indices,
    ) in groups:

        for index in indices:

            row = dataset[
                index
            ]

            solution = row[
                "solution"
            ]

            expected = extract_last_boxed(
                solution
            )

            add_example(
                partition=partition,
                domain=domain,
                subtype=config,
                source=source,
                source_config=(
                    f"math_{config}"
                ),
                official_split=official_split,
                original_index=index,
                question=row["problem"],
                expected=expected,
                reference_solution=solution,
                metadata={
                    "level": row["level"],
                    "type": row["type"],
                    "automatic_answer_extraction": (
                        expected is not None
                    ),
                },
            )


for config in (
    "number_theory",
    "counting_and_probability",
):

    prepare_math_config(
        config,
        "maths_competition",
    )


# ============================================================
# 7. MATH - ALGÈBRE AVANCÉE
# ============================================================

print("\n[3/6] MATH - algèbre")

for config in (
    "algebra",
    "intermediate_algebra",
):

    prepare_math_config(
        config,
        "algebre_avancee",
    )


# ============================================================
# 8. MBPP - PROGRAMMATION
# ============================================================

print("\n[4/6] MBPP")

mbpp_source = SOURCES[
    "mbpp"
]

mbpp_train = load_source(
    mbpp_source,
    "full",
    "train",
)

mbpp_validation = load_source(
    mbpp_source,
    "full",
    "validation",
)

mbpp_test = load_source(
    mbpp_source,
    "full",
    "test",
)

mbpp_groups = [
    (
        "train",
        "train",
        mbpp_train,
        list(range(len(mbpp_train))),
    ),
    (
        "validation",
        "validation",
        mbpp_validation,
        list(range(len(mbpp_validation))),
    ),
    (
        "test",
        "test",
        mbpp_test,
        shuffled_indices(
            mbpp_test,
            "mbpp_test",
        )[:100],
    ),
]

for (
    partition,
    official_split,
    dataset,
    indices,
) in mbpp_groups:

    for index in indices:

        row = dataset[
            index
        ]

        # Un test public sert d'exemple
        # dans l'énoncé. Les autres tests
        # restent dans les fichiers de
        # vérification.

        tests = row.get(
            "test_list"
        ) or []

        challenge = row.get(
            "challenge_test_list"
        ) or []

        question = row["text"]

        if tests:

            question += (
                "\n\nExemple de test :\n"
                + str(tests[0])
            )

        add_example(
            partition=partition,
            domain="programmation",
            subtype="mbpp",
            source=mbpp_source,
            source_config="mbpp_full",
            official_split=official_split,
            original_index=index,
            question=question,
            expected=None,
            reference_solution=(
                row["code"]
            ),
            tests={
                "public": tests[:1],
                "hidden": tests[1:],
                "challenge": challenge,
                "setup": row.get(
                    "test_setup_code"
                ) or "",
            },
            metadata={
                "task_id": row["task_id"],
            },
        )


# ============================================================
# 9. BIG-BENCH HARD - LOGIQUE
# ============================================================

print("\n[5/6] BIG-Bench Hard")

bbh_config = (
    "logical_deduction_five_objects"
)

bbh = load_source(
    SOURCES["bbh"],
    bbh_config,
    bbh_config,
)

bbh_indices = shuffled_indices(
    bbh,
    "bbh_logical_test",
)[:100]

for index in bbh_indices:

    row = bbh[
        index
    ]

    question = row[
        "question"
    ]

    choices = row.get(
        "choices"
    ) or {}

    labels = choices.get(
        "label"
    ) or []

    texts = choices.get(
        "text"
    ) or []

    if labels and texts:

        options = "\n".join(
            f"({label}) {text}"
            for label, text in zip(
                labels,
                texts,
            )
        )

        question += (
            "\n\nChoix possibles :\n"
            + options
        )

    add_example(
        partition="test",
        domain="raisonnement_logique",
        subtype=bbh_config,
        source=SOURCES["bbh"],
        source_config=bbh_config,
        official_split=bbh_config,
        original_index=index,
        question=question,
        expected=row["target"],
        metadata={
            "transfer_only": True,
        },
    )


# ============================================================
# 10. HUMANEVAL+ - PROGRAMMATION AVANCÉE
# ============================================================

print("\n[6/6] HumanEval+")

humaneval = load_source(
    SOURCES["humaneval"],
    "default",
    "test",
)

for index, row in enumerate(
    humaneval
):

    add_example(
        partition="test",
        domain="programmation_avancee",
        subtype="humaneval_plus",
        source=SOURCES["humaneval"],
        source_config="humaneval_plus",
        official_split="test",
        original_index=index,
        question=row["prompt"],
        expected=None,
        reference_solution=(
            row.get(
                "canonical_solution"
            ) or ""
        ),
        tests={
            "test": row.get(
                "test"
            ) or "",
            "entry_point": row.get(
                "entry_point"
            ) or "",
        },
        metadata={
            "task_id": row[
                "task_id"
            ],
            "transfer_only": True,
        },
    )


# ============================================================
# 11. CONTRÔLE FINAL
# ============================================================

print("\nVérification des partitions...")

train_ids = {
    row["id"]
    for row in public["train"]
}

validation_ids = {
    row["id"]
    for row in public["validation"]
}

test_ids = {
    row["id"]
    for row in public["test"]
}

if train_ids & validation_ids:

    raise ValueError(
        "Fuite train/validation."
    )

if train_ids & test_ids:

    raise ValueError(
        "Fuite train/test."
    )

if validation_ids & test_ids:

    raise ValueError(
        "Fuite validation/test."
    )

expected_counts = {
    "train": 1574,
    "validation": 440,
    "test": 714,
}

for partition, count in expected_counts.items():

    actual = len(
        public[partition]
    )

    if actual != count:

        raise ValueError(
            f"{partition} : "
            f"{actual} au lieu de {count}."
        )

    public_ids = {
        row["id"]
        for row in public[partition]
    }

    grading_ids = {
        row["id"]
        for row in grading[partition]
    }

    if public_ids != grading_ids:

        raise ValueError(
            f"Clés incohérentes : {partition}"
        )

# Les références ne sont pas incluses
# dans les fichiers publics de
# validation et de test.

for partition in (
    "validation",
    "test",
):

    if any(
        "target" in row
        for row in public[partition]
    ):

        raise ValueError(
            "Fuite de réponses."
        )

# ============================================================
# 12. SAUVEGARDE
# ============================================================

# Refuser l'écrasement du benchmark
# évite de modifier accidentellement
# un jeu de test déjà figé.

if OUTPUT.exists() and any(
    OUTPUT.iterdir()
):

    raise FileExistsError(
        "Le Benchmark V2 existe déjà. "
        "Aucun fichier n'a été écrasé."
    )

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

grading_dir = (
    OUTPUT / "grading"
)

grading_dir.mkdir(
    parents=True,
    exist_ok=True,
)

file_hashes = {}

for partition in (
    "train",
    "validation",
    "test",
):

    public_file = (
        OUTPUT
        / f"{partition}.jsonl"
    )

    grading_file = (
        grading_dir
        / f"{partition}_keys.jsonl"
    )

    write_jsonl(
        public_file,
        public[partition],
    )

    write_jsonl(
        grading_file,
        grading[partition],
    )

    file_hashes[
        str(public_file.relative_to(OUTPUT))
    ] = sha256_file(
        public_file
    )

    file_hashes[
        str(grading_file.relative_to(OUTPUT))
    ] = sha256_file(
        grading_file
    )

# ============================================================
# 13. MANIFESTE SCIENTIFIQUE
# ============================================================

counts = {}

for partition, records in public.items():

    counts[partition] = dict(
        Counter(
            row["domain"]
            for row in records
        )
    )

missing_math_answers = sum(
    row["expected"] is None
    and row["domain"] in {
        "maths_competition",
        "algebre_avancee",
    }
    for partition in grading
    for row in grading[partition]
)

manifest = {
    "name": "Benchmark V2",
    "version": "v1",
    "seed": SEED,
    "counts": counts,
    "expected_counts": expected_counts,
    "sources": source_metadata,
    "file_hashes": file_hashes,
    "missing_math_answer_extractions": (
        missing_math_answers
    ),
    "protocol": {
        "official_test_only_for_test": True,
        "train_and_validation_disjoint": True,
        "answers_separated": True,
        "no_model_evaluation_performed": True,
        "no_untrusted_code_executed": True,
        "external_benchmarks_may_be_in_pretraining": True,
        "test_selection_must_be_frozen": True,
    },
}

manifest_file = (
    OUTPUT / "manifest.json"
)

with manifest_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )

# ============================================================
# 14. BILAN
# ============================================================

print("\n" + "=" * 65)
print("BENCHMARK V2 CONSTRUIT")
print("=" * 65)

for partition in (
    "train",
    "validation",
    "test",
):

    print(
        f"\n{partition.upper()} : "
        f"{len(public[partition])}"
    )

    for domain, count in sorted(
        counts[partition].items()
    ):

        print(
            f"  {domain}: {count}"
        )

print(
    "\nRéponses MATH sans extraction "
    f"automatique : {missing_math_answers}"
)

print(
    f"\nDossier : {OUTPUT}"
)

print(
    f"Manifeste : {manifest_file}"
)

print(
    "\nAucun modèle n'a été évalué "
    "sur le nouveau test."
)

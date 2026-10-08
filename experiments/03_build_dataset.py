
import json
import random
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
rng = random.Random(SEED)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": 120,
    "validation": 30,
    "test": 50
}

DOMAINS = ["maths", "algebre", "code"]

# ============================================================
# 2. GÉNÉRATION DES QUESTIONS DE MATHÉMATIQUES
# ============================================================

def generate_maths():

    operation = rng.randint(0, 4)

    if operation == 0:
        a = rng.randint(10, 250)
        b = rng.randint(10, 250)

        question = f"Calcule : {a} + {b}"
        answer = a + b

    elif operation == 1:
        a = rng.randint(50, 300)
        b = rng.randint(1, a)

        question = f"Calcule : {a} - {b}"
        answer = a - b

    elif operation == 2:
        a = rng.randint(2, 50)
        b = rng.randint(2, 20)

        question = f"Calcule : {a} * {b}"
        answer = a * b

    elif operation == 3:
        b = rng.randint(2, 20)
        answer = rng.randint(2, 50)

        a = b * answer

        question = f"Calcule : {a} / {b}"

    else:
        a = rng.randint(20, 100)
        b = rng.randint(1, a - 1)
        c = rng.randint(2, 10)

        question = f"Calcule : ({a} - {b}) * {c}"
        answer = (a - b) * c

    return question, answer

# ============================================================
# 3. GÉNÉRATION DES QUESTIONS D'ALGÈBRE
# ============================================================

def generate_algebra():

    operation = rng.randint(0, 3)

    a = rng.randint(2, 15)
    b = rng.randint(1, 20)
    x = rng.randint(1, 30)

    if operation == 0:

        c = a * x + b

        question = (
            f"Résous l'équation : "
            f"{a}x + {b} = {c}"
        )

    elif operation == 1:

        c = a * x - b

        question = (
            f"Résous l'équation : "
            f"{a}x - {b} = {c}"
        )

    elif operation == 2:

        c = a * (x + b)

        question = (
            f"Résous l'équation : "
            f"{a}(x + {b}) = {c}"
        )

    else:

        x = a * rng.randint(1, 20)
        c = x // a + b

        question = (
            f"Résous l'équation : "
            f"x/{a} + {b} = {c}"
        )

    return question, x

# ============================================================
# 4. GÉNÉRATION DES QUESTIONS DE PROGRAMMATION
# ============================================================

def generate_code():

    operation = rng.randint(0, 4)

    if operation == 0:

        a = rng.randint(2, 9)
        b = rng.randint(2, 4)
        c = rng.randint(1, 20)

        expression = f"print({a}**{b} + {c})"
        answer = a**b + c

    elif operation == 1:

        a = rng.randint(10, 200)
        b = rng.randint(2, 20)

        expression = f"print({a} // {b})"
        answer = a // b

    elif operation == 2:

        values = [
            rng.randint(1, 100)
            for _ in range(3)
        ]

        expression = f"print(sum({values}))"
        answer = sum(values)

    elif operation == 3:

        values = [
            rng.randint(1, 100)
            for _ in range(rng.randint(2, 10))
        ]

        expression = f"print(len({values}))"
        answer = len(values)

    else:

        a = rng.randint(10, 200)
        b = rng.randint(2, 20)
        c = rng.randint(1, 10)

        expression = f"print({a} % {b} + {c})"
        answer = a % b + c

    question = (
        "Quel est le résultat de ce programme "
        f"Python ?\n{expression}"
    )

    return question, answer

# ============================================================
# 5. CRÉATION DES DATASETS
# ============================================================

GENERATORS = {
    "maths": generate_maths,
    "algebre": generate_algebra,
    "code": generate_code
}

datasets = {
    split: []
    for split in SPLITS
}

# Empêche qu'une question identique
# apparaisse dans plusieurs jeux de données.
seen_questions = set()

for domain in DOMAINS:

    generator = GENERATORS[domain]

    total = sum(SPLITS.values())

    examples = []
    attempts = 0

    while len(examples) < total:

        attempts += 1

        if attempts > 100000:
            raise RuntimeError(
                f"Génération impossible : {domain}"
            )

        question, answer = generator()

        if question in seen_questions:
            continue

        seen_questions.add(question)

        examples.append({
            "domain": domain,
            "question": question,
            "answer": answer
        })

    rng.shuffle(examples)

    start = 0

    for split, count in SPLITS.items():

        subset = examples[start:start + count]

        for index, example in enumerate(
            subset,
            start=1
        ):

            record = {
                "id": (
                    f"{domain}_{split}_"
                    f"{index:04d}"
                ),
                **example
            }

            datasets[split].append(record)

        start += count

# ============================================================
# 6. SAUVEGARDE JSONL
# ============================================================

print("=" * 60)
print("CONTINUAL RL - DATASET GENERATION")
print("=" * 60)

manifest = {
    "seed": SEED,
    "domains": DOMAINS,
    "splits": {}
}

for split, examples in datasets.items():

    rng.shuffle(examples)

    output_path = (
        OUTPUT_DIR / f"{split}.jsonl"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

    manifest["splits"][split] = {
        "total": len(examples),
        "by_domain": {
            domain: sum(
                row["domain"] == domain
                for row in examples
            )
            for domain in DOMAINS
        }
    }

    print(
        f"{split:12s} : "
        f"{len(examples)} questions"
    )

# ============================================================
# 7. SAUVEGARDE DES INFORMATIONS
# ============================================================

manifest_path = (
    OUTPUT_DIR / "manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 8. VÉRIFICATION FINALE
# ============================================================

all_records = [
    record
    for examples in datasets.values()
    for record in examples
]

questions = [
    record["question"]
    for record in all_records
]

assert len(questions) == len(set(questions))
assert len(all_records) == 600

print("-" * 60)

print(f"Total : {len(all_records)} questions")
print("Doublons : 0")
print("Statut : DATASET VALIDÉ")
print(f"Dossier : {OUTPUT_DIR}")

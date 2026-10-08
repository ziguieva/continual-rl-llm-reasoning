
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "mps"
    else torch.float32
)

MAX_NEW_TOKENS = 256

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. DATASET DE TEST
# ============================================================

TESTS = [

    # Mathématiques
    ("maths", "28 + 17", 45),
    ("maths", "9 * 8", 72),
    ("maths", "144 / 12", 12),
    ("maths", "(35 - 11) * 2", 48),

    # Algèbre
    ("algebre", "3x + 5 = 20. Trouve x.", 5),
    ("algebre", "2x - 7 = 11. Trouve x.", 9),
    ("algebre", "5(x + 2) = 35. Trouve x.", 5),
    ("algebre", "x/4 + 6 = 10. Trouve x.", 16),

    # Programmation
    (
        "code",
        "Quel est le résultat Python : print(2**4 + 1)",
        17
    ),
    (
        "code",
        "Quel est le résultat Python : print(10 // 3)",
        3
    ),
    (
        "code",
        "Quel est le résultat Python : print(sum([2, 5, 8]))",
        15
    ),
    (
        "code",
        "Quel est le résultat Python : print(len([4, 7, 9]))",
        3
    ),
]

# ============================================================
# 3. CHARGEMENT DU MODÈLE
# ============================================================

print("=" * 60)
print("CONTINUAL RL - BASELINE V3")
print("=" * 60)

print(f"Appareil : {DEVICE}")
print(f"Modèle : {MODEL_NAME}")

print("\nChargement du modèle...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE
).to(DEVICE)

model.eval()

print("Modèle chargé avec succès.")

# ============================================================
# 4. GÉNÉRATION DES RÉPONSES
# ============================================================

def generate_response(question):

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant spécialisé en "
                "mathématiques, algèbre et programmation Python. "
                "Résous chaque problème avec précision. "
                "Termine obligatoirement par une ligne "
                "au format REPONSE: <entier>."
            )
        },
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Tu peux détailler ton raisonnement. "
                "Termine par REPONSE: <entier>."
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    token_count = len(generated_tokens)

    # Détection d'une génération interrompue
    # parce que la limite de tokens a été atteinte.
    eos_reached = (
        generated_tokens[-1].item()
        == tokenizer.eos_token_id
        if token_count > 0
        else False
    )

    truncated = (
        token_count >= MAX_NEW_TOKENS
        and not eos_reached
    )

    return response, token_count, truncated

# ============================================================
# 5. EXTRACTION DE LA RÉPONSE
# ============================================================

def extract_answer(response):

    # Méthode 1 :
    # REPONSE: 15
    # RÉPONSE: 15
    # RESPONSE: 15

    pattern = (
        r"(?:R[ÉE]PONSE|RESPONSE)\s*:\s*"
        r"(?:x\s*=\s*)?(-?\d+)(?:\.0+)?"
    )

    matches = re.findall(
        pattern,
        response,
        flags=re.IGNORECASE
    )

    if matches:
        return int(matches[-1]), "explicit"

    # Méthode 2 : réponse numérique seule.
    # Exemples :
    # 17
    # x = 5

    match = re.fullmatch(
        r"\s*(?:x\s*=\s*)?"
        r"(-?\d+)(?:\.0+)?\s*[.!]?\s*",
        response
    )

    if match:
        return int(match.group(1)), "numeric"

    # Méthode 3 : réponse dans la dernière phrase.
    # Exemple :
    # "Le résultat est 17."
    # "The final answer is 48."
    # "La fonction affiche `15`."

    lines = response.strip().splitlines()

    if not lines:
        return None, "not_found"

    last_line = lines[-1].strip()

    pattern = (
        r"(?:est|is|affiche|vaut)\s+"
        r"[`*]*(-?\d+)(?:\.0+)?[`*]*"
        r"[.!]?\s*$"
    )

    match = re.search(
        pattern,
        last_line,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1)), "fallback"

    # Aucune réponse finale identifiable.
    return None, "not_found"

# ============================================================
# 6. INITIALISATION DES RÉSULTATS
# ============================================================

results = []

scores = {
    domain: {
        "correct": 0,
        "total": 0,
        "format_errors": 0,
        "truncated": 0
    }
    for domain in ["maths", "algebre", "code"]
}

print("\nDébut de l'évaluation...\n")

# ============================================================
# 7. ÉVALUATION DU MODÈLE
# ============================================================

for index, (domain, question, expected) in enumerate(
    TESTS,
    start=1
):

    print("-" * 60)

    print(f"Test : {index}/{len(TESTS)}")
    print(f"Domaine : {domain}")
    print(f"Question : {question}")

    response, token_count, truncated = (
        generate_response(question)
    )

    predicted, extraction_method = (
        extract_answer(response)
    )

    correct = (
        predicted is not None
        and predicted == expected
    )

    format_error = predicted is None

    # Mise à jour des métriques
    scores[domain]["total"] += 1

    scores[domain]["correct"] += int(
        correct
    )

    scores[domain]["format_errors"] += int(
        format_error
    )

    scores[domain]["truncated"] += int(
        truncated
    )

    # Sauvegarde du résultat individuel
    result = {
        "domain": domain,
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "extraction_method": extraction_method,
        "format_error": format_error,
        "token_count": token_count,
        "truncated": truncated,
        "full_response": response
    }

    results.append(result)

    # Affichage
    print(f"Réponse attendue : {expected}")
    print(f"Réponse extraite : {predicted}")
    print(f"Correct : {correct}")
    print(f"Extraction : {extraction_method}")
    print(f"Tokens générés : {token_count}")
    print(f"Tronqué : {truncated}")

    if format_error:

        print(
            "Attention : réponse finale non reconnue."
        )

        print(f"Sortie complète : {response}")

    if truncated:

        print(
            "Attention : limite de tokens atteinte."
        )

# ============================================================
# 8. SAUVEGARDE DES RÉSULTATS CSV
# ============================================================

csv_path = RESULTS_DIR / "baseline_v3.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)

# ============================================================
# 9. CALCUL DES MÉTRIQUES
# ============================================================

summary = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "domains": {},
    "global": {}
}

total_correct = 0
total_questions = 0
total_format_errors = 0
total_truncated = 0

print("\n" + "=" * 60)

print("RÉSULTATS BASELINE V3")

print("=" * 60)

for domain, score in scores.items():

    correct = score["correct"]

    total = score["total"]

    accuracy = (
        100 * correct / total
        if total > 0
        else 0.0
    )

    summary["domains"][domain] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "format_errors": score["format_errors"],
        "truncated": score["truncated"]
    }

    total_correct += correct

    total_questions += total

    total_format_errors += score[
        "format_errors"
    ]

    total_truncated += score[
        "truncated"
    ]

    print(
        f"{domain.upper():10s} : "
        f"{accuracy:.1f}% "
        f"({correct}/{total})"
    )

# ============================================================
# 10. SCORE GLOBAL
# ============================================================

global_accuracy = (
    100 * total_correct / total_questions
    if total_questions > 0
    else 0.0
)

summary["global"] = {
    "correct": total_correct,
    "total": total_questions,
    "accuracy": global_accuracy,
    "format_errors": total_format_errors,
    "truncated": total_truncated
}

print("-" * 60)

print(
    f"SCORE GLOBAL : {global_accuracy:.1f}%"
)

print(
    f"Erreurs de format : {total_format_errors}"
)

print(
    f"Générations tronquées : {total_truncated}"
)

# ============================================================
# 11. SAUVEGARDE JSON
# ============================================================

json_path = (
    RESULTS_DIR / "baseline_v3_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 12. FIN
# ============================================================

print("\nÉvaluation terminée.")

print(f"CSV : {csv_path}")

print(f"JSON : {json_path}")

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    ROOT
    / "results"
    / "continual_rl_v2"
    / "seed_42"
    / "adaptive_replay"
    / "stage3_programmation"
)

RECORDS_FILE = (
    RESULT_DIR
    / "records.jsonl"
)

BACKUP_FILE = (
    RESULT_DIR
    / "records_before_mbpp_container_repair.jsonl"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "summary.json"
)

COMPLETED_FILE = (
    RESULT_DIR
    / "completed.json"
)


def load_jsonl(path):

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


def write_jsonl(
    path,
    rows,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    print(
        "=" * 72
    )

    print(
        "RÉPARATION ÉVALUATION "
        "ADAPTIVE REPLAY / MBPP"
    )

    print(
        "=" * 72
    )

    if not RECORDS_FILE.is_file():

        raise FileNotFoundError(
            RECORDS_FILE
        )

    rows = load_jsonl(
        RECORDS_FILE
    )

    print(
        f"Résultats actuels : "
        f"{len(rows)}"
    )

    bad_mbpp = [
        row
        for row in rows
        if (
            row.get(
                "domain"
            )
            == "programmation"
            and row.get(
                "status"
            )
            == "container_error"
        )
    ]

    good_rows = [
        row
        for row in rows
        if not (
            row.get(
                "domain"
            )
            == "programmation"
            and row.get(
                "status"
            )
            == "container_error"
        )
    ]

    programming_rows = [
        row
        for row in rows
        if row.get(
            "domain"
        )
        == "programmation"
    ]

    print(
        "MBPP présents : "
        f"{len(programming_rows)}"
    )

    print(
        "MBPP container_error : "
        f"{len(bad_mbpp)}"
    )

    print(
        "Résultats conservés : "
        f"{len(good_rows)}"
    )

    if len(
        bad_mbpp
    ) != 90:

        raise RuntimeError(
            "Sécurité : exactement "
            "90 container_error MBPP "
            "étaient attendus."
        )

    math_rows = [
        row
        for row in good_rows
        if row.get(
            "domain"
        )
        != "programmation"
    ]

    if len(
        math_rows
    ) != 350:

        raise RuntimeError(
            "Sécurité : 350 résultats "
            "mathématiques étaient attendus, "
            f"{len(math_rows)} trouvés."
        )

    if BACKUP_FILE.exists():

        print(
            "\nBackup déjà présent :"
        )

        print(
            BACKUP_FILE
        )

    else:

        shutil.copy2(
            RECORDS_FILE,
            BACKUP_FILE,
        )

        print(
            "\nBackup créé :"
        )

        print(
            BACKUP_FILE
        )

    write_jsonl(
        RECORDS_FILE,
        good_rows,
    )

    # Le stage ne doit plus être marqué
    # comme complètement évalué.
    for path in (
        SUMMARY_FILE,
        COMPLETED_FILE,
    ):

        if path.exists():

            path.unlink()

            print(
                "Supprimé : "
                f"{path.name}"
            )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RÉPARATION TERMINÉE"
    )

    print(
        "=" * 72
    )

    print(
        "350 résultats mathématiques "
        "ont été conservés."
    )

    print(
        "90 résultats MBPP erronés "
        "ont été retirés."
    )

    print(
        "\nRelance maintenant :"
    )

    print(
        "python experiments/"
        "54_evaluate_continual.py "
        "--strategy adaptive_replay "
        "--seed 42"
    )


if __name__ == "__main__":

    main()
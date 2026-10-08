import csv
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib est requis.\n"
        "Installe-le avec : pip install matplotlib"
    )


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SEED = 42

RESULT_ROOT = (
    ROOT
    / "results"
    / "continual_rl_v2"
    / f"seed_{SEED}"
)

OUTPUT_DIR = (
    RESULT_ROOT
    / "final_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STRATEGIES = (
    "no_replay",
    "fixed_replay",
    "adaptive_replay",
)

DISPLAY_NAMES = {
    "no_replay": "No Replay",
    "fixed_replay": "Fixed Replay",
    "adaptive_replay": "Adaptive Replay",
}

DOMAINS = (
    "maths_appliques",
    "algebre_avancee",
    "maths_competition",
    "programmation",
)

DOMAIN_NAMES = {
    "maths_appliques": "GSM8K",
    "algebre_avancee": "Algèbre",
    "maths_competition": "Compétition",
    "programmation": "MBPP",
}

DOMAIN_STAGE = {
    "maths_appliques": 0,
    "algebre_avancee": 1,
    "maths_competition": 2,
    "programmation": 3,
}

STAGE_PATHS = {
    0: (
        RESULT_ROOT
        / "common"
        / "stage0_gsm8k"
        / "summary.json"
    ),
    1: "stage1_algebre_avancee",
    2: "stage2_maths_competition",
    3: "stage3_programmation",
}


# ============================================================
# 2. JSON
# ============================================================

def load_json(path):

    if not path.is_file():

        raise FileNotFoundError(
            f"Fichier manquant : {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(path, data):

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 3. CHARGER LES RÉSUMÉS
# ============================================================

def load_summaries():

    summaries = {
        "stage0": load_json(
            STAGE_PATHS[0]
        )
    }

    for strategy in STRATEGIES:

        summaries[
            strategy
        ] = {}

        for stage in (
            1,
            2,
            3,
        ):

            path = (
                RESULT_ROOT
                / strategy
                / STAGE_PATHS[
                    stage
                ]
                / "summary.json"
            )

            summaries[
                strategy
            ][stage] = load_json(
                path
            )

    return summaries


# ============================================================
# 4. MATRICE A[t,d]
# ============================================================

def build_matrix(
    strategy,
    summaries,
):

    matrix = {}

    matrix["0"] = {
        "maths_appliques":
            summaries[
                "stage0"
            ][
                "domains"
            ][
                "maths_appliques"
            ][
                "accuracy"
            ]
    }

    for stage in (
        1,
        2,
        3,
    ):

        summary = (
            summaries[
                strategy
            ][stage]
        )

        matrix[
            str(stage)
        ] = {}

        for domain in DOMAINS:

            if (
                DOMAIN_STAGE[
                    domain
                ]
                <= stage
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
# 5. MÉTRIQUES CONTINUAL
# ============================================================

def compute_metrics(
    matrix,
):

    final_stage = 3

    acquisition = {}
    final_accuracy = {}
    forgetting = {}
    retention = {}
    backward_transfer = {}

    for domain in DOMAINS:

        first_stage = (
            DOMAIN_STAGE[
                domain
            ]
        )

        first_accuracy = (
            matrix[
                str(first_stage)
            ][domain]
        )

        final = (
            matrix[
                str(final_stage)
            ][domain]
        )

        history = []

        for stage in range(
            first_stage,
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

        best = max(
            history
        )

        acquisition[
            domain
        ] = first_accuracy

        final_accuracy[
            domain
        ] = final

        forgetting[
            domain
        ] = max(
            0.0,
            best - final,
        )

        backward_transfer[
            domain
        ] = (
            final
            - first_accuracy
        )

        if first_accuracy > 0:

            retention[
                domain
            ] = (
                final
                / first_accuracy
            )

        else:

            retention[
                domain
            ] = None

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

    average_forgetting = (
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

    average_bwt = (
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

        "forgetting":
            forgetting,

        "retention":
            retention,

        "backward_transfer":
            backward_transfer,

        "average_accuracy_by_stage":
            average_accuracy_by_stage,

        "final_average_accuracy":
            final_average_accuracy,

        "average_forgetting_old_domains":
            average_forgetting,

        "average_backward_transfer_old_domains":
            average_bwt,
    }


# ============================================================
# 6. TABLEAU FINAL CSV
# ============================================================

def write_final_table(
    metrics,
):

    path = (
        OUTPUT_DIR
        / "final_results.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "strategy",
            "gsm8k_accuracy_pct",
            "algebra_accuracy_pct",
            "competition_accuracy_pct",
            "mbpp_accuracy_pct",
            "final_average_accuracy_pct",
            "average_forgetting_pct_points",
            "average_backward_transfer_pct_points",
        ])

        for strategy in STRATEGIES:

            m = metrics[
                strategy
            ]

            writer.writerow([
                strategy,

                100 * m[
                    "final_accuracy"
                ][
                    "maths_appliques"
                ],

                100 * m[
                    "final_accuracy"
                ][
                    "algebre_avancee"
                ],

                100 * m[
                    "final_accuracy"
                ][
                    "maths_competition"
                ],

                100 * m[
                    "final_accuracy"
                ][
                    "programmation"
                ],

                100 * m[
                    "final_average_accuracy"
                ],

                100 * m[
                    "average_forgetting_old_domains"
                ],

                100 * m[
                    "average_backward_transfer_old_domains"
                ],
            ])

    return path


# ============================================================
# 7. MATRICE COMPLÈTE CSV
# ============================================================

def write_matrix_csv(
    matrices,
):

    path = (
        OUTPUT_DIR
        / "continual_accuracy_matrix.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "strategy",
            "stage",
            "gsm8k",
            "algebra",
            "competition",
            "mbpp",
            "average_seen",
        ])

        for strategy in STRATEGIES:

            matrix = matrices[
                strategy
            ]

            for stage in range(
                4
            ):

                stage_values = (
                    matrix[
                        str(stage)
                    ]
                )

                seen = list(
                    stage_values.values()
                )

                writer.writerow([
                    strategy,
                    stage,

                    (
                        100
                        * stage_values[
                            "maths_appliques"
                        ]
                        if (
                            "maths_appliques"
                            in stage_values
                        )
                        else ""
                    ),

                    (
                        100
                        * stage_values[
                            "algebre_avancee"
                        ]
                        if (
                            "algebre_avancee"
                            in stage_values
                        )
                        else ""
                    ),

                    (
                        100
                        * stage_values[
                            "maths_competition"
                        ]
                        if (
                            "maths_competition"
                            in stage_values
                        )
                        else ""
                    ),

                    (
                        100
                        * stage_values[
                            "programmation"
                        ]
                        if (
                            "programmation"
                            in stage_values
                        )
                        else ""
                    ),

                    (
                        100
                        * sum(seen)
                        / len(seen)
                    ),
                ])

    return path


# ============================================================
# 8. FIGURE 1 — ACCURACY MOYENNE FINALE
# ============================================================

def plot_final_average_accuracy(
    metrics,
):

    labels = [
        DISPLAY_NAMES[
            strategy
        ]
        for strategy
        in STRATEGIES
    ]

    values = [
        100
        * metrics[
            strategy
        ][
            "final_average_accuracy"
        ]
        for strategy
        in STRATEGIES
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Average accuracy (%)"
    )

    ax.set_title(
        "Final Average Accuracy after Stage 3"
    )

    ax.set_ylim(
        0,
        max(values) + 6,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.4,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "01_final_average_accuracy.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# 9. FIGURE 2 — PERFORMANCE PAR DOMAINE
# ============================================================

def plot_final_domain_accuracy(
    metrics,
):

    x = list(
        range(
            len(DOMAINS)
        )
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    for index, strategy in enumerate(
        STRATEGIES
    ):

        values = [
            100
            * metrics[
                strategy
            ][
                "final_accuracy"
            ][domain]
            for domain
            in DOMAINS
        ]

        positions = [
            value
            + (
                index - 1
            ) * width
            for value
            in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=(
                DISPLAY_NAMES[
                    strategy
                ]
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels([
        DOMAIN_NAMES[
            domain
        ]
        for domain
        in DOMAINS
    ])

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Final Accuracy by Domain"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "02_final_domain_accuracy.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# 10. FIGURE 3 — FORGETTING
# ============================================================

def plot_forgetting(
    metrics,
):

    labels = [
        DISPLAY_NAMES[
            strategy
        ]
        for strategy
        in STRATEGIES
    ]

    values = [
        100
        * metrics[
            strategy
        ][
            "average_forgetting_old_domains"
        ]
        for strategy
        in STRATEGIES
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Average forgetting (percentage points)"
    )

    ax.set_title(
        "Average Forgetting on Previous Domains"
    )

    ax.set_ylim(
        0,
        max(values) + 1.5,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "03_average_forgetting.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# 11. FIGURE 4 — BACKWARD TRANSFER
# ============================================================

def plot_backward_transfer(
    metrics,
):

    labels = [
        DISPLAY_NAMES[
            strategy
        ]
        for strategy
        in STRATEGIES
    ]

    values = [
        100
        * metrics[
            strategy
        ][
            "average_backward_transfer_old_domains"
        ]
        for strategy
        in STRATEGIES
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "Backward transfer (percentage points)"
    )

    ax.set_title(
        "Backward Transfer on Previous Domains"
    )

    for bar, value in zip(
        bars,
        values,
    ):

        offset = (
            0.1
            if value >= 0
            else -0.3
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va=(
                "bottom"
                if value >= 0
                else "top"
            ),
        )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "04_backward_transfer.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# 12. FIGURES DE TRAJECTOIRE
# ============================================================

def plot_strategy_trajectory(
    strategy,
    matrix,
):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for domain in DOMAINS:

        stages = []

        values = []

        for stage in range(
            DOMAIN_STAGE[
                domain
            ],
            4,
        ):

            if (
                domain
                in matrix[
                    str(stage)
                ]
            ):

                stages.append(
                    stage
                )

                values.append(
                    100
                    * matrix[
                        str(stage)
                    ][domain]
                )

        ax.plot(
            stages,
            values,
            marker="o",
            label=(
                DOMAIN_NAMES[
                    domain
                ]
            ),
        )

    ax.set_xticks(
        [0, 1, 2, 3]
    )

    ax.set_xticklabels([
        "Stage 0\nGSM8K",
        "Stage 1\nAlgèbre",
        "Stage 2\nCompétition",
        "Stage 3\nMBPP",
    ])

    ax.set_ylabel(
        "Validation accuracy (%)"
    )

    ax.set_title(
        f"Continual Accuracy Trajectory — "
        f"{DISPLAY_NAMES[strategy]}"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / (
            "trajectory_"
            f"{strategy}.png"
        )
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# 13. RAPPORT MARKDOWN FINAL
# ============================================================

def write_report(
    matrices,
    metrics,
):

    path = (
        OUTPUT_DIR
        / "FINAL_RESULTS.md"
    )

    lines = []

    lines.append(
        "# Continual & Data-Efficient RL "
        "for LLM Reasoning"
    )

    lines.append("")

    lines.append(
        "## Final validation results — Seed 42"
    )

    lines.append("")

    lines.append(
        "| Strategy | GSM8K | Algebra | "
        "Competition | MBPP | Final avg. | "
        "Forgetting | BWT |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for strategy in STRATEGIES:

        m = metrics[
            strategy
        ]

        final = (
            m[
                "final_accuracy"
            ]
        )

        lines.append(
            f"| {DISPLAY_NAMES[strategy]} "
            f"| {100 * final['maths_appliques']:.2f}% "
            f"| {100 * final['algebre_avancee']:.2f}% "
            f"| {100 * final['maths_competition']:.2f}% "
            f"| {100 * final['programmation']:.2f}% "
            f"| {100 * m['final_average_accuracy']:.2f}% "
            f"| {100 * m['average_forgetting_old_domains']:.2f} pts "
            f"| {100 * m['average_backward_transfer_old_domains']:+.2f} pts |"
        )

    lines.append("")
    lines.append(
        "## Main observations"
    )
    lines.append("")

    no_replay = metrics[
        "no_replay"
    ]

    fixed = metrics[
        "fixed_replay"
    ]

    adaptive = metrics[
        "adaptive_replay"
    ]

    fixed_gain = (
        100
        * (
            fixed[
                "final_average_accuracy"
            ]
            - no_replay[
                "final_average_accuracy"
            ]
        )
    )

    adaptive_gain = (
        100
        * (
            adaptive[
                "final_average_accuracy"
            ]
            - no_replay[
                "final_average_accuracy"
            ]
        )
    )

    lines.append(
        f"- Fixed Replay obtains the highest "
        f"final average accuracy: "
        f"**{100 * fixed['final_average_accuracy']:.2f}%**."
    )

    lines.append(
        f"- Fixed Replay improves final average "
        f"accuracy by **{fixed_gain:+.2f} points** "
        f"relative to No Replay."
    )

    lines.append(
        f"- Adaptive Replay improves final average "
        f"accuracy by **{adaptive_gain:+.2f} points** "
        f"relative to No Replay."
    )

    lines.append(
        f"- Average forgetting falls from "
        f"**{100 * no_replay['average_forgetting_old_domains']:.2f}** "
        f"points without replay to "
        f"**{100 * fixed['average_forgetting_old_domains']:.2f}** "
        f"with Fixed Replay."
    )

    lines.append(
        f"- Fixed Replay produces positive average "
        f"backward transfer "
        f"(**{100 * fixed['average_backward_transfer_old_domains']:+.2f} points**)."
    )

    lines.append(
        f"- Adaptive Replay reaches "
        f"**{100 * adaptive['final_average_accuracy']:.2f}%** "
        f"final average accuracy, between Fixed Replay "
        f"and No Replay."
    )

    lines.append("")
    lines.append(
        "## Interpretation"
    )
    lines.append("")

    lines.append(
        "For seed 42, replay improves retention across "
        "the sequential reasoning curriculum. "
        "Fixed Replay provides the strongest overall "
        "trade-off between final performance and retention."
    )

    lines.append("")

    lines.append(
        "The current Adaptive Replay controller does not "
        "yet demonstrate a clear advantage over Fixed Replay. "
        "During this experiment its replay allocations were "
        "effectively identical to the fixed allocation, so "
        "differences between both strategies cannot be "
        "attributed confidently to adaptive allocation."
    )

    lines.append("")

    lines.append(
        "Therefore the main supported conclusion is that "
        "**replay mitigates forgetting in this continual "
        "reasoning setting**, while the benefit of adaptive "
        "replay remains to be demonstrated."
    )

    lines.append("")
    lines.append(
        "## Limitations"
    )
    lines.append("")

    lines.append(
        "- Results reported here correspond to one seed "
        "(seed 42)."
    )

    lines.append(
        "- The Qwen2.5-0.5B model shows limited absolute "
        "performance on advanced mathematics."
    )

    lines.append(
        "- The adaptive replay controller did not create "
        "meaningfully different replay allocations."
    )

    lines.append(
        "- Validation was used only retrospectively; "
        "the independent 714-example test set remains closed."
    )

    lines.append("")
    lines.append(
        "## Independent test status"
    )
    lines.append("")

    lines.append(
        "**Not evaluated.** The 714-example independent "
        "test set remains untouched."
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return path


# ============================================================
# 14. TERMINAL SUMMARY
# ============================================================

def print_summary(
    metrics,
):

    print(
        "\n"
        + "=" * 82
    )

    print(
        "FINAL CONTINUAL RL ANALYSIS — SEED 42"
    )

    print(
        "=" * 82
    )

    print(
        f"{'Strategy':<20}"
        f"{'GSM8K':>9}"
        f"{'Algebra':>9}"
        f"{'Comp':>9}"
        f"{'MBPP':>9}"
        f"{'Average':>10}"
        f"{'Forget':>9}"
        f"{'BWT':>9}"
    )

    print(
        "-" * 82
    )

    for strategy in STRATEGIES:

        m = metrics[
            strategy
        ]

        final = (
            m[
                "final_accuracy"
            ]
        )

        print(
            f"{DISPLAY_NAMES[strategy]:<20}"
            f"{100 * final['maths_appliques']:>8.2f}%"
            f"{100 * final['algebre_avancee']:>8.2f}%"
            f"{100 * final['maths_competition']:>8.2f}%"
            f"{100 * final['programmation']:>8.2f}%"
            f"{100 * m['final_average_accuracy']:>9.2f}%"
            f"{100 * m['average_forgetting_old_domains']:>8.2f}"
            f"{100 * m['average_backward_transfer_old_domains']:>+8.2f}"
        )

    print(
        "\nTest indépendant : NON UTILISÉ"
    )

    print(
        "\nRésultats sauvegardés dans :"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "CONTINUAL RL V2 — FINAL ANALYSIS"
    )

    print(
        "=" * 72
    )

    print(
        "Aucun entraînement."
    )

    print(
        "Aucun modèle chargé."
    )

    print(
        "Analyse des résultats existants uniquement."
    )

    summaries = (
        load_summaries()
    )

    matrices = {}

    metrics = {}

    for strategy in STRATEGIES:

        matrices[
            strategy
        ] = build_matrix(
            strategy,
            summaries,
        )

        metrics[
            strategy
        ] = compute_metrics(
            matrices[
                strategy
            ]
        )

    save_json(
        OUTPUT_DIR
        / "accuracy_matrices.json",
        matrices,
    )

    save_json(
        OUTPUT_DIR
        / "final_metrics.json",
        metrics,
    )

    final_csv = (
        write_final_table(
            metrics
        )
    )

    matrix_csv = (
        write_matrix_csv(
            matrices
        )
    )

    figures = []

    figures.append(
        plot_final_average_accuracy(
            metrics
        )
    )

    figures.append(
        plot_final_domain_accuracy(
            metrics
        )
    )

    figures.append(
        plot_forgetting(
            metrics
        )
    )

    figures.append(
        plot_backward_transfer(
            metrics
        )
    )

    for strategy in STRATEGIES:

        figures.append(
            plot_strategy_trajectory(
                strategy,
                matrices[
                    strategy
                ],
            )
        )

    report = write_report(
        matrices,
        metrics,
    )

    print_summary(
        metrics
    )

    print(
        "\nFichiers produits :"
    )

    print(
        final_csv
    )

    print(
        matrix_csv
    )

    print(
        report
    )

    for figure in figures:

        print(
            figure
        )


if __name__ == "__main__":

    main()
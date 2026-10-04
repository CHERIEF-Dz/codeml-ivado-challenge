"""Fairlearn mitigation using demographic parity and the fixed grant budget."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    selection_rate,
)
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
SUBMISSION_PATH = HERE.parent / "predictions.csv"
DATA_DIR = HERE / "data"
REMOTE_REGIONS = [
    "Bas-Saint-Laurent",
    "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine",
]
BUDGET_MIN, BUDGET_MAX = 0.36, 0.44
EPS_BOUNDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.27]
CATEGORICAL_COLUMNS = [
    "programme_etudes",
    "region_administrative",
    "code_postal_3",
]


def groupe_region(frame):
    return np.where(
        frame["region_administrative"].isin(REMOTE_REGIONS),
        "Eloignee",
        "Centre",
    )


def encoder(frame, columns):
    features = frame.drop(
        columns=["id_candidat", "decision_octroi", "groupe", "stratification"],
        errors="ignore",
    )
    return pd.get_dummies(features, columns=CATEGORICAL_COLUMNS).reindex(
        columns=columns, fill_value=0
    )


def nouveau_modele(bound):
    estimator = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    return ExponentiatedGradient(
        estimator=estimator,
        constraints=DemographicParity(difference_bound=bound),
        eps=0.01,
    )


def taux_par_groupe(y_pred, sensitive_features):
    rates = {}
    for group in ("Centre", "Eloignee"):
        mask = np.asarray(sensitive_features) == group
        rates[group] = float(np.mean(np.asarray(y_pred)[mask]))
    return rates


def main():
    historical = pd.read_csv(DATA_DIR / "donnees_demandes.csv")
    candidates = pd.read_csv(DATA_DIR / "candidats_evaluation.csv")
    historical["groupe"] = groupe_region(historical)
    historical["stratification"] = (
        historical["groupe"].astype(str)
        + "_"
        + historical["decision_octroi"].astype(str)
    )

    train_frame, remaining = train_test_split(
        historical,
        test_size=0.4,
        random_state=42,
        stratify=historical["stratification"],
    )
    validation_frame, test_frame = train_test_split(
        remaining,
        test_size=0.5,
        random_state=42,
        stratify=remaining["stratification"],
    )

    X_train = pd.get_dummies(
        train_frame.drop(
            columns=[
                "id_candidat",
                "decision_octroi",
                "groupe",
                "stratification",
            ]
        ),
        columns=CATEGORICAL_COLUMNS,
    )
    X_validation = encoder(validation_frame, X_train.columns)
    X_test = encoder(test_frame, X_train.columns)
    y_train = train_frame["decision_octroi"]
    y_validation = validation_frame["decision_octroi"]
    y_test = test_frame["decision_octroi"]
    sensitive_train = train_frame["groupe"]
    sensitive_validation = validation_frame["groupe"]
    sensitive_test = test_frame["groupe"]

    results = []
    validation_models = {}
    print("Balayage de DemographicParity sur l'ensemble de validation...")
    for bound in EPS_BOUNDS:
        mitigator = nouveau_modele(bound)
        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_train,
        )
        predictions = mitigator.predict(X_validation)
        group_rates = taux_par_groupe(predictions, sensitive_validation)
        result = {
            "difference_bound": bound,
            "accuracy_historical_label": accuracy_score(
                y_validation, predictions
            ),
            "demographic_parity_difference": demographic_parity_difference(
                y_validation,
                predictions,
                sensitive_features=sensitive_validation,
            ),
            "grant_rate": float(np.mean(predictions)),
            "centre_selection_rate": group_rates["Centre"],
            "eloignee_selection_rate": group_rates["Eloignee"],
        }
        results.append(result)
        validation_models[bound] = mitigator
        print(
            f"bound={bound:>4.2f} | DP gap={result['demographic_parity_difference']:.4f} "
            f"| accuracy={result['accuracy_historical_label']:.4f} "
            f"| octroi={result['grant_rate']:.2%}"
        )

    results_frame = pd.DataFrame(results)
    results_frame["budget_respected"] = results_frame["grant_rate"].between(
        BUDGET_MIN, BUDGET_MAX
    )
    admissible = results_frame[results_frame["budget_respected"]]
    if admissible.empty:
        raise RuntimeError(
            "Aucun réglage ne respecte l'enveloppe de 36–44 % sur validation; "
            "aucune soumission n'a été produite."
        )

    selected = admissible.sort_values(
        ["demographic_parity_difference", "accuracy_historical_label"],
        ascending=[True, False],
    ).iloc[0]
    selected_bound = float(selected["difference_bound"])
    results_frame.to_csv(HERE / "pareto_results_demographic_parity.csv", index=False)

    # The untouched test set is used once to report the selected validation setting.
    test_model = validation_models[selected_bound]
    test_predictions = test_model.predict(X_test)
    test_group_rates = taux_par_groupe(test_predictions, sensitive_test)
    test_dp_gap = demographic_parity_difference(
        y_test, test_predictions, sensitive_features=sensitive_test
    )
    print("\n--- Réglage retenu sur validation ---")
    print(f"DemographicParity difference_bound : {selected_bound:.2f}")
    print(
        "Écart de parité démographique (validation) : "
        f"{selected['demographic_parity_difference']:.4f}"
    )
    print(f"Accuracy vs décision historique (validation) : {selected['accuracy_historical_label']:.4f}")
    print(f"Taux d'octroi validation : {selected['grant_rate']:.2%}")
    print("\n--- Évaluation une fois sur le test ---")
    print(f"Écart DP vs étiquette historique : {test_dp_gap:.4f}")
    print(f"Accuracy vs étiquette historique : {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Taux global : {np.mean(test_predictions):.2%}")
    print(
        "Taux par groupe : "
        f"Centre {test_group_rates['Centre']:.2%}, "
        f"Éloignée {test_group_rates['Eloignee']:.2%}"
    )

    # Refit the chosen constraint on train + validation; keep test out of training.
    development = pd.concat([train_frame, validation_frame]).sort_index()
    X_development = pd.get_dummies(
        development.drop(
            columns=[
                "id_candidat",
                "decision_octroi",
                "groupe",
                "stratification",
            ]
        ),
        columns=CATEGORICAL_COLUMNS,
    )
    X_candidates = encoder(candidates, X_development.columns)
    final_model = nouveau_modele(selected_bound)
    final_model.fit(
        X_development,
        development["decision_octroi"],
        sensitive_features=development["groupe"],
    )
    candidate_predictions = final_model.predict(X_candidates).astype(int)
    candidate_groups = groupe_region(candidates)
    candidate_group_rates = taux_par_groupe(
        candidate_predictions, candidate_groups
    )
    candidate_rate = float(np.mean(candidate_predictions))
    if not BUDGET_MIN <= candidate_rate <= BUDGET_MAX:
        raise RuntimeError(
            f"Le taux d'octroi sur évaluation ({candidate_rate:.2%}) est hors "
            "budget 36–44 %; aucune soumission n'a été écrite."
        )

    submission = pd.DataFrame(
        {
            "id_candidat": candidates["id_candidat"],
            "decision_octroi": candidate_predictions,
        }
    )
    if len(submission) != 4000 or submission["id_candidat"].duplicated().any():
        raise ValueError("La soumission doit contenir 4 000 identifiants uniques.")
    if not submission["decision_octroi"].isin([0, 1]).all():
        raise ValueError("Les décisions de la soumission doivent être binaires.")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("\n--- Soumission candidats d'évaluation ---")
    print(f"Lignes : {len(submission)}")
    print(f"Taux d'octroi : {candidate_rate:.2%} (budget respecté)")
    print(
        "Taux de sélection observés : "
        f"Centre {candidate_group_rates['Centre']:.2%}, "
        f"Éloignée {candidate_group_rates['Eloignee']:.2%}"
    )
    print(
        "Écart DP observé sur l'évaluation : "
        f"{abs(candidate_group_rates['Centre'] - candidate_group_rates['Eloignee']):.4f}"
    )
    print(f"Soumission écrite : {SUBMISSION_PATH}")

    # Plot only the validation results. Fairness is the primary selection axis;
    # historical-label accuracy is reported as a caveated utility proxy.
    budget_results = results_frame[results_frame["budget_respected"]].copy()
    frontier_indices = []
    for index, point in budget_results.iterrows():
        dominated = (
            (budget_results["demographic_parity_difference"]
             <= point["demographic_parity_difference"])
            & (budget_results["accuracy_historical_label"]
               >= point["accuracy_historical_label"])
            & (
                (budget_results["demographic_parity_difference"]
                 < point["demographic_parity_difference"])
                | (budget_results["accuracy_historical_label"]
                   > point["accuracy_historical_label"])
            )
        )
        if not dominated.any():
            frontier_indices.append(index)
    frontier = budget_results.loc[frontier_indices].sort_values(
        "demographic_parity_difference"
    )

    colors = results_frame["difference_bound"]
    fig, axis = plt.subplots(figsize=(9, 6))
    points = axis.scatter(
        results_frame["demographic_parity_difference"],
        results_frame["accuracy_historical_label"],
        c=colors,
        cmap="viridis",
        s=75,
        marker="o",
        edgecolors=np.where(results_frame["budget_respected"], "black", "none"),
        linewidths=0.8,
    )
    axis.plot(
        frontier["demographic_parity_difference"],
        frontier["accuracy_historical_label"],
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label="Frontière non dominée (budget respecté)",
    )
    axis.scatter(
        [selected["demographic_parity_difference"]],
        [selected["accuracy_historical_label"]],
        marker="*",
        s=260,
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
        label=f"Réglage retenu (bound={selected_bound:.2f})",
        zorder=5,
    )
    axis.set_xlabel("Écart de parité démographique (plus bas = taux plus proches)")
    axis.set_ylabel("Exactitude vs décisions historiques (utilité indicative)")
    axis.set_title("Front de compromis — parité démographique (validation)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    fig.colorbar(points, ax=axis, label="difference_bound")
    axis.text(
        0.01,
        0.01,
        "Contour noir : taux d'octroi validation dans [36 %, 44 %].\n"
        "L'exactitude vise l'étiquette historique, pas l'étalon caché.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(HERE / "pareto_final.png", dpi=160)
    plt.close(fig)
    print(f"Front de Pareto écrit : {HERE / 'pareto_final.png'}")


if __name__ == "__main__":
    main()

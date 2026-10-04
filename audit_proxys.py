"""Measure how well non-sensitive application fields predict the baseline region group."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/donnees_demandes.csv")
OUTPUT_PATH = Path("proxy_importance.png")
REMOTE_REGIONS = {
    "Bas-Saint-Laurent",
    "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine",
}


def main():
    applications = pd.read_csv(DATA_PATH)
    y = applications["region_administrative"].isin(REMOTE_REGIONS).astype(int)
    excluded = {
        "id_candidat",
        "decision_octroi",
        "region_administrative",
        "groupe",
        "strat",
    }
    X = applications.drop(columns=list(excluded.intersection(applications.columns)))

    categorical = [
        name for name in ("programme_etudes", "code_postal_3") if name in X.columns
    ]
    numeric = [name for name in X.columns if name not in categorical]
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Prédiction du groupe régional à partir des autres variables")
    print(f"Accuracy                 : {accuracy_score(y_test, predictions):.3f}")
    print(f"Balanced accuracy        : {balanced_accuracy_score(y_test, predictions):.3f}")
    print(f"Taux de base (Éloignée)  : {y_test.mean():.3f}")
    print("\nCorrélations linéaires des variables numériques avec le groupe Éloignée")
    numeric_with_group = X[numeric].copy()
    numeric_with_group["est_eloignee"] = y
    correlations = numeric_with_group.corr(numeric_only=True)["est_eloignee"].drop(
        "est_eloignee"
    )
    print(correlations.reindex(correlations.abs().sort_values(ascending=False).index).round(3))

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    ranking = pd.DataFrame(
        {
            "variable": X.columns,
            "importance_moyenne": importance.importances_mean,
            "ecart_type": importance.importances_std,
        }
    ).sort_values("importance_moyenne", ascending=False)
    print("\nImportance par permutation sur l'ensemble de test (balanced accuracy)")
    print(ranking.to_string(index=False, formatters={
        "importance_moyenne": "{:.4f}".format,
        "ecart_type": "{:.4f}".format,
    }))

    ordered = ranking.sort_values("importance_moyenne")
    colors = np.where(ordered["importance_moyenne"] > 0, "#2878B5", "#999999")
    plt.figure(figsize=(9, 5))
    plt.barh(
        ordered["variable"],
        ordered["importance_moyenne"],
        xerr=ordered["ecart_type"],
        color=colors,
        capsize=3,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Baisse de balanced accuracy après permutation")
    plt.title("Pouvoir proxy des variables pour prédire le groupe régional")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    print(f"\nGraphique enregistré : {OUTPUT_PATH}")
    print(
        "\nInterprétation : une importance positive indique que la variable aide "
        "à prédire le groupe dans ce modèle; ce n'est pas une preuve de causalité "
        "ni une mesure de discrimination à elle seule."
    )


if __name__ == "__main__":
    main()

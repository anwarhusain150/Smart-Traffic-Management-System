"""
train_models.py
---------------
Re-trains the FINAL (best) regression and classification models on the training
period and saves them to models/. The best hyperparameters were chosen in
Notebooks 03 and 04 (GridSearchCV) and saved as JSON files in models/.

Run from the project folder:
    python src/train_models.py
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append(str(Path(__file__).resolve().parent))
import data_preprocessing as dp  # noqa: E402
import model_utils as mu  # noqa: E402

MODELS_DIR = dp.ROOT / "models"

REGRESSORS = {"Random Forest": RandomForestRegressor, "Decision Tree": DecisionTreeRegressor}
CLASSIFIERS = {"Random Forest": RandomForestClassifier, "Decision Tree": DecisionTreeClassifier,
               "Logistic Regression": LogisticRegression, "KNN": KNeighborsClassifier, "SVM (RBF)": SVC}


def _build(kind: str):
    """Read best model name + params from JSON and build an untrained pipeline."""
    info = json.loads((MODELS_DIR / f"best_params_{kind}.json").read_text())
    name = info["best_model"]
    params = {k: v for k, v in info["params"][name].items() if "__" not in k}
    registry = REGRESSORS if kind == "regression" else CLASSIFIERS
    est = registry[name](**params)
    if "Forest" in name:
        est.set_params(n_estimators=150, n_jobs=-1)
    if hasattr(est, "random_state"):
        est.set_params(random_state=mu.RANDOM_STATE)
    if isinstance(est, SVC):
        est.set_params(probability=True)
    return name, Pipeline([("prep", mu.build_preprocessor()), ("model", est)])


def train_and_save():
    df = dp.load_clean_featured()
    for kind, target, fname in [("regression", "traffic_volume", "regressor.joblib"),
                                ("classification", "congestion_level", "classifier.joblib")]:
        name, pipe = _build(kind)
        X_train, _, y_train, _, _, _ = mu.time_split(df, target)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODELS_DIR / fname)
        print(f"Trained {name} for {kind} -> models/{fname}")


if __name__ == "__main__":
    train_and_save()

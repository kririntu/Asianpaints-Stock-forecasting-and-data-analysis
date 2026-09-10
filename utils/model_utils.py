import joblib
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


def train_models(X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    models = {

        "Logistic Regression": Pipeline([

            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                max_iter=5000
            ))

        ]),

        "Random Forest": RandomForestClassifier(

            n_estimators=300,

            max_depth=5,

            min_samples_leaf=10,

            random_state=42

        ),

        "XGBoost": XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42,

            eval_metric="logloss"

        )

    }

    best_model = None
    best_score = 0

    for model_name, model in models.items():

        print("\n" + "=" * 60)

        print(model_name)

        print("=" * 60)

        scores = []

        for fold, (train_idx, test_idx) in enumerate(

                tscv.split(X),

                start=1):

            X_train = X.iloc[train_idx]

            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]

            y_test = y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(

                y_test,

                y_pred

            )

            scores.append(score)

            print(

                f"Fold {fold}: {score:.4f}"

            )

        mean_score = np.mean(scores)

        print()

        print(f"Mean Accuracy : {mean_score:.4f}")

        print(f"Std Accuracy  : {np.std(scores):.4f}")

        print(f"Best Fold     : {np.max(scores):.4f}")

        print(f"Worst Fold    : {np.min(scores):.4f}")

        if mean_score > best_score:

            best_score = mean_score

            best_model = model

    print()

    print("=" * 60)

    print("Training Best Model On Entire Dataset")

    print("=" * 60)

    best_model.fit(X, y)

    joblib.dump(best_model, "model.pkl")

    joblib.dump(list(X.columns), "feature_columns.pkl")

    print()

    print("Best Model Saved Successfully")

    print(f"Best CV Accuracy : {best_score:.4f}")

    return best_model
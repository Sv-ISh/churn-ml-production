import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/churn_model.pkl")
RANDOM_STATE = 42


def load_data(path: str = "data/raw/churn.csv") -> pd.DataFrame:
    logger.info("Загрузка данных из %s", path)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info("ROC-AUC на тесте: %.4f", auc)

    return model


def save_model(model: RandomForestClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Модель сохранена в %s", path)


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    model = train_model(X, y)
    save_model(model)

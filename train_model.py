import pandas as pd
import re
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def preprocess_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    root = Path(__file__).resolve().parent
    data_file = root / "emails.csv"
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    df = pd.read_csv(data_file)
    # Keep first two meaningful columns (CSV may have many empty trailing columns)
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        second_col = df.columns[1]
        df = df[[first_col, second_col]].copy()
        # normalize column names to text/spam
        df.columns = ["text", "spam"]

    df = df.dropna()
    df = df[df["spam"].apply(lambda x: str(x).strip().isdigit())]
    df["spam"] = df["spam"].astype(int)

    df["text"] = df["text"].apply(preprocess_text)

    X = df["text"].tolist()
    y = df["spam"].values

    # simple pipeline: CountVectorizer + MultinomialNB
    cv = CountVectorizer(stop_words="english", max_features=10000)
    X_vec = cv.fit_transform(X)

    clf = MultinomialNB()
    clf.fit(X_vec, y)

    # save
    with open(model_dir / "cv.pkl", "wb") as f:
        pickle.dump(cv, f)
    with open(model_dir / "clf.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Training complete — model files written to:", model_dir)


if __name__ == "__main__":
    main()

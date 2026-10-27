"""Benchmark supervised classifiers and clustering methods on a folder of UCI CSV files.

Each CSV must use ';' as separator and contain a column named `target`
(for supervised runs). Results are written to results/.

Usage:
    python src/tabular_benchmark.py --data data/tabular
    python src/tabular_benchmark.py --data data/tabular --unsupervised
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42
TARGET_NAMES = ["target", "label", "class", "y"]


# ---------------------------------------------------------------- preprocessing
def read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin1")
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_")
                  .str.replace("(", "", regex=False).str.replace(")", "", regex=False))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop mostly-empty columns, impute the rest, remove duplicates."""
    df = df.loc[:, df.isna().mean() < 0.5].copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df.drop_duplicates()


def to_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median())


# ---------------------------------------------------------------- supervised
def build_models():
    """(model, needs_scaling) for each classifier."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=SEED), True),
        "SVM (RBF)": (SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED), False),
    }


def run_supervised(path: str, fig_dir: str) -> list:
    name = os.path.basename(path)
    df = clean(read_csv(path))
    if "target" not in df.columns:
        print(f"  skipped {name}: no 'target' column")
        return []

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["target"])
    X = to_numeric_features(df.drop(columns=["target"]))

    # 70 / 15 / 15 stratified split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    scaler = StandardScaler().fit(X_tr)
    scaled = (scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te))
    raw = (X_tr, X_val, X_te)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    labels = [str(c) for c in encoder.classes_]

    rows = []
    for model_name, (model, needs_scaling) in build_models().items():
        tr, va, te = scaled if needs_scaling else raw
        cv_scores = cross_val_score(model, tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(tr, y_tr)
        pred_te = model.predict(te)
        rows.append({
            "dataset": name,
            "model": model_name,
            "cv_acc_mean": cv_scores.mean(),
            "cv_acc_std": cv_scores.std(),
            "val_acc": accuracy_score(y_val, model.predict(va)),
            "test_acc": accuracy_score(y_te, pred_te),
            "test_f1_macro": f1_score(y_te, pred_te, average="macro"),
        })

        disp = ConfusionMatrixDisplay.from_predictions(y_te, pred_te, display_labels=labels,
                                                       values_format="d", cmap="Blues")
        disp.ax_.set_title(f"{model_name} | {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{os.path.splitext(name)[0]}_{model_name.split()[0].lower()}_cm.png"), dpi=110)
        plt.close("all")

    for r in rows:
        print(f"  {r['model']:20s} CV {r['cv_acc_mean']:.3f}±{r['cv_acc_std']:.3f}  test {r['test_acc']:.3f}")
    return rows


# ---------------------------------------------------------------- unsupervised
def run_unsupervised(path: str, fig_dir: str, n_clusters: int = 3) -> None:
    name = os.path.basename(path)
    df = read_csv(path)
    target = next((c for c in df.columns if c in TARGET_NAMES), None)
    if target:
        df = df.drop(columns=[target])
    X = StandardScaler().fit_transform(to_numeric_features(clean(df)))

    clusterings = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(X),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5).fit_predict(X),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X),
    }
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (method, labels) in zip(axes, clusterings.items()):
        for lab in np.unique(labels):
            m = labels == lab
            ax.scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.6, label="noise" if lab == -1 else f"cluster {lab}")
        ax.set(title=method, xlabel="PC 1", ylabel="PC 2")
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{os.path.splitext(name)[0]}_clusters.png"), dpi=110)
    plt.close(fig)
    print(f"  clustered {name}")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tabular", help="folder of ';'-separated CSV files")
    p.add_argument("--out", default="results")
    p.add_argument("--unsupervised", action="store_true", help="run clustering instead of classification")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")), key=lambda f: os.path.basename(f).lower())
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Found {len(files)} CSV files")

    if args.unsupervised:
        for f in files:
            run_unsupervised(f, fig_dir)
        return

    rows = []
    for f in files:
        print(f"\n{os.path.basename(f)}")
        rows += run_supervised(f, fig_dir)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out, "supervised_results.csv"), index=False)
    if not results.empty:
        print("\nMean test accuracy per model:")
        print(results.groupby("model")["test_acc"].mean().sort_values(ascending=False).round(3))


if __name__ == "__main__":
    main()

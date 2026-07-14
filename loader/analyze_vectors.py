"""Анализ сгенерированных векторов -> обоснование сжатия (слайд 7 + пункт рубрики).

Что делает:
  1. PCA по каждому каналу (image / text) -> кумулятивная explained variance
     -> сколько измерений держат 90/95/99% дисперсии (эффективная размерность).
     Это обоснование Matryoshka-обрезки 512 -> 256/128.
  2. Средний косинус между image- и text-вектором одного мема -> насколько каналы
     независимы (если близко к 1 — дублируют друг друга, fusion бесполезен).
  3. Сохраняет график eval/pca_variance.png для слайда.

Запуск (у кого есть parquet):
    pip install pandas pyarrow numpy scikit-learn matplotlib
    python loader/analyze_vectors.py path/to/embeddings_v1.parquet
"""
import os
import sys
import numpy as np
import pandas as pd


def find_vector_columns(df):
    """Колонки, где ячейка = список/массив чисел длины > 1 (эмбеддинги)."""
    cols = []
    for c in df.columns:
        v = df[c].dropna().iloc[0] if df[c].notna().any() else None
        if isinstance(v, (list, np.ndarray)) and len(np.asarray(v).ravel()) > 1:
            cols.append(c)
    return cols


def to_matrix(series):
    return np.vstack([np.asarray(x, dtype=np.float32).ravel() for x in series])


def pca_report(name, X):
    from sklearn.decomposition import PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    p = PCA(n_components=min(X.shape[1], X.shape[0]))
    p.fit(Xc)
    cum = np.cumsum(p.explained_variance_ratio_)
    dims = {t: int(np.searchsorted(cum, t) + 1) for t in (0.90, 0.95, 0.99)}
    print(f"\n[{name}] {X.shape[0]} векторов, dim={X.shape[1]}")
    print(f"  измерений для 90%: {dims[0.90]} | 95%: {dims[0.95]} | 99%: {dims[0.99]}")
    print(f"  дисперсии в первых 128 dim: {cum[min(127, len(cum)-1)]*100:.1f}%")
    return cum, dims


def main():
    if len(sys.argv) < 2:
        print("usage: python loader/analyze_vectors.py embeddings_v1.parquet")
        sys.exit(1)
    df = pd.read_parquet(sys.argv[1])
    print("колонки в parquet:", list(df.columns))
    vec_cols = find_vector_columns(df)
    print("нашёл векторные колонки:", vec_cols)
    if not vec_cols:
        print("Не нашёл эмбеддинги — укажи колонки вручную в коде.")
        sys.exit(1)

    curves = {}
    mats = {}
    for c in vec_cols:
        X = to_matrix(df[c])
        mats[c] = X
        curves[c], _ = pca_report(c, X)

    # независимость каналов: средний косинус image vs text одного мема
    img_c = next((c for c in vec_cols if "image" in c.lower() or "img" in c.lower()), None)
    txt_c = next((c for c in vec_cols if "text" in c.lower() or "ocr" in c.lower()), None)
    if img_c and txt_c and mats[img_c].shape == mats[txt_c].shape:
        A = mats[img_c] / (np.linalg.norm(mats[img_c], axis=1, keepdims=True) + 1e-9)
        B = mats[txt_c] / (np.linalg.norm(mats[txt_c], axis=1, keepdims=True) + 1e-9)
        cos = (A * B).sum(axis=1)
        print(f"\n[независимость каналов] средний косинус image<->text: {cos.mean():.3f} "
              f"(ниже = каналы разнее = fusion осмысленнее)")

    # график
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
        os.makedirs(out, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for c, cum in curves.items():
            plt.plot(range(1, len(cum) + 1), cum, label=c)
        for t in (0.90, 0.95):
            plt.axhline(t, ls="--", lw=0.7, color="gray")
        plt.xlabel("PCA components"); plt.ylabel("cumulative explained variance")
        plt.title("Vector analysis: effective dimensionality")
        plt.legend(); plt.tight_layout()
        png = os.path.join(out, "pca_variance.png")
        plt.savefig(png, dpi=130)
        print("\nграфик сохранён:", png)
    except ImportError:
        print("matplotlib не установлен — график пропущен, цифры выше.")


if __name__ == "__main__":
    main()

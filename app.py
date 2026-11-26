import csv
import io
import json
import re
import string
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_ROWS = 6500
MAX_COLUMNS = 40
MAX_CELL_CHARS = 8000


def read_csv(uploaded):
    content = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Ukuran CSV melebihi 2 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Simpan CSV dengan encoding UTF-8.") from None

    empty = "CSV harus punya header dan setidaknya satu baris data."
    try:
        if "\x00" in text:
            raise csv.Error
        records = (row for row in csv.reader(io.StringIO(text, newline=""), strict=True) if row)
        header = [name.strip() for name in next(records, [])]
        if not header:
            raise ValueError(empty)
        if len(header) > MAX_COLUMNS:
            raise ValueError("CSV dibatasi 40 kolom.")
        if not all(header) or len({name.casefold() for name in header}) != len(header):
            raise ValueError("Gunakan nama kolom yang tidak kosong dan tidak berulang.")
        if any(len(name) > MAX_CELL_CHARS for name in header):
            raise ValueError("Maksimal 8.000 karakter per sel atau nama kolom.")
        rows = []
        for row in records:
            if len(rows) >= MAX_ROWS:
                raise ValueError("CSV dibatasi 6.500 baris data.")
            if len(row) != len(header):
                raise ValueError("Setiap baris harus punya jumlah kolom yang sama dengan header. Pakai pemisah koma.")
            if any(len(value) > MAX_CELL_CHARS for value in row):
                raise ValueError("Maksimal 8.000 karakter per sel atau nama kolom.")
            rows.append([value if value else None for value in row])
    except csv.Error:
        raise ValueError("Berkas tidak bisa dibaca: format CSV tidak valid. Periksa kutip dan pemisah koma.") from None
    if not rows:
        raise ValueError(empty)
    return pd.DataFrame(rows, columns=header)


def show_exploration():
    st.header("Eksplorasi CSV")
    st.caption("Hasil berasal dari file yang diunggah dan memakai preprocessing sederhana, bukan metode atau hasil skripsi.")
    uploaded = st.file_uploader("Unggah CSV ulasan", type=["csv"], help="UTF-8, pemisah koma; maksimal 2 MB, 6.500 baris, 40 kolom, dan 8.000 karakter per sel.")
    if uploaded is None:
        st.info("Unggah CSV dengan kolom teks seperti ulasan, review, atau teks. Jangan unggah data pribadi.")
        return

    try:
        df = read_csv(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.write("### 1. Data Mentah")
    st.dataframe(df.head(), hide_index=True)

    columns = df.columns.tolist()
    preferred = ("ulasan", "review", "teks", "text", "content", "clean_text")
    column_by_name = {name.strip().casefold(): name for name in columns}
    default_column = next((column_by_name[name] for name in preferred if name in column_by_name), columns[0])
    kolom = st.selectbox("Pilih kolom teks ulasan", columns, index=columns.index(default_column))
    teks_asli = df[kolom].fillna("").astype(str)
    missing = teks_asli.str.strip().str.casefold().isin({"nan", "null", "none", "<na>", "na", "n/a"})
    teks_asli = teks_asli.mask(missing, "")

    # --- PREPROCESSING ---
    st.write("### 2. Preprocessing")
    stopwords = set("yang dan di ke dari pada untuk dengan ini itu ada adalah atau juga tidak bisa akan sangat sudah oleh karena sebagai dalam pada saat".split())

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = text.split()  # TOKENIZING
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]  # STOPWORD REMOVAL
        return " ".join(tokens)

    vectorizer = TfidfVectorizer(max_features=1000)
    clean = teks_asli.map(preprocess)
    has_tokens = clean.map(vectorizer.build_tokenizer()).astype(bool)
    if not has_tokens.any():
        st.error("Pilih kolom yang berisi teks ulasan; tidak ada kata yang dapat dipakai setelah pembersihan.")
        return

    X = vectorizer.fit_transform(clean[has_tokens]).tocsr()
    nonzero = X.getnnz(axis=1) > 0
    X = X[nonzero]
    X.sort_indices()
    clean = clean[has_tokens].iloc[nonzero]
    result = df.loc[clean.index].copy()
    preview = pd.DataFrame({"Ulasan": teks_asli.loc[clean.index], "Teks bersih": clean, "Token": clean.str.split()})
    st.caption(f"{len(df) - len(result)} baris tanpa kata yang dapat dipakai tidak diikutkan.")
    st.dataframe(preview.head(), hide_index=True)

    st.write("### 3. TF-IDF + K-Means")
    # Count distinct sparse vectors, ignoring float noise after normalization.
    unique_vectors = set()
    for a, b in zip(X.indptr, X.indptr[1:]):
        unique_vectors.add((X.indices[a:b].tobytes(), X.data[a:b].round(12).tobytes()))
        if len(unique_vectors) == 10:
            break
    max_k = len(unique_vectors)
    if max_k < 2:
        st.error("Clustering butuh minimal 2 vektor TF-IDF berbeda. Tambah variasi ulasan atau pilih kolom lain.")
        return
    if max_k == 2:
        k = 2
        st.caption("K = 2 karena hanya ada 2 vektor TF-IDF berbeda.")
    else:
        k = st.slider("Pilih jumlah klaster (K)", 2, max_k, min(6, max_k))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    counts = pd.Series(labels).value_counts().sort_index()
    if len(counts) != k:
        st.error("Jumlah klaster yang terbentuk kurang dari K. Turunkan K atau tambah variasi ulasan.")
        return
    st.success(f"Clustering selesai! Total data: {len(result)}, K = {k}")

    st.write("### 4. Evaluasi Klaster")
    if k == len(result):
        st.info("Metrik tidak dihitung jika setiap ulasan berada di klaster tersendiri. Turunkan K untuk menghitung metrik.")
    else:
        with config_context(working_memory=32):
            sil = silhouette_score(X, labels)
        dense = X.toarray()
        dbi = davies_bouldin_score(dense, labels)
        ch = calinski_harabasz_score(dense, labels)
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil:.4f}")
        c2.metric("Davies-Bouldin", f"{dbi:.4f}")
        c3.metric("Calinski-Harabasz", f"{ch:.1f}")
        st.caption("Silhouette mendekati 1 berarti pemisahan klaster lebih baik; Davies-Bouldin lebih baik saat lebih kecil.")

    st.write("### 5. Distribusi & Isi Tiap Klaster")
    distribution = pd.DataFrame({"Klaster": counts.index.astype(str), "Jumlah ulasan": counts.values})
    st.bar_chart(distribution, x="Klaster", y="Jumlah ulasan", height=220)
    st.dataframe(distribution, hide_index=True)
    terms = vectorizer.get_feature_names_out()
    for cluster, count in counts.items():
        with st.expander(f"Klaster {cluster} · {count} ulasan"):
            center = kmeans.cluster_centers_[cluster]
            top = [terms[i] for i in center.argsort()[-10:][::-1] if center[i] > 0]
            st.write("Kata kunci: " + ", ".join(top))
            st.dataframe(preview.loc[labels == cluster].head(), hide_index=True)

    for name, values in (("clean_text", clean), ("tokens", preview["Token"]), ("cluster_id", labels)):
        # Keep uploaded columns intact even when re-clustering a previous export.
        while name in result:
            name = f"hasil_{name}"
        result[name] = values
    st.download_button("Unduh Hasil Clustering (CSV)", result.to_csv(index=False), "hasil_clustering.csv", "text/csv", on_click="ignore")


def show_thesis():
    try:
        summary = json.loads(Path(__file__).with_name("hasil_skripsi.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.error("Ringkasan hasil skripsi tidak tersedia. Pastikan hasil_skripsi.json ada di samping app.py.")
        return

    st.header("Hasil Skripsi")
    st.caption("Ringkasan agregat dari hasil analisis skripsi. K = 6 adalah skenario visual sementara, bukan K optimal atau keputusan final.")
    raw, clean, k = (summary[key] for key in ("raw", "clean", "selected_k"))
    evaluation = pd.DataFrame(summary["comparison"]).rename(columns={"k": "K", "inertia": "WCSS", "silhouette": "Silhouette"})
    sil = evaluation.loc[evaluation["K"] == k, "Silhouette"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Ulasan mentah", f"{raw:,}".replace(",", "."))
    b.metric("Setelah preprocessing", f"{clean:,}".replace(",", "."))
    c.metric("K skenario visual", k)
    d.metric("Silhouette (Euclidean)", f"{sil:.6f}".replace(".", ","))
    st.caption("Silhouette mendekati nol pada seluruh K yang diuji: pemisahan klaster lemah. Tema adalah interpretasi kelompok, bukan label setiap ulasan.")

    distribution = pd.DataFrame(summary["clusters"]).rename(columns={"id": "Klaster", "theme": "Tema", "count": "Jumlah ulasan"})
    distribution["Persentase (%)"] = distribution["Jumlah ulasan"] / clean * 100
    st.subheader(f"Distribusi klaster · K = {k}")
    st.bar_chart(distribution.assign(Klaster=distribution["Klaster"].astype(str)), x="Klaster", y="Jumlah ulasan", height=180)
    st.dataframe(distribution, hide_index=True, height=247, column_order=["Klaster", "Tema", "Jumlah ulasan", "Persentase (%)"],
                 column_config={"Persentase (%)": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Perbandingan K = 2–6")
    st.dataframe(evaluation, hide_index=True, height=212, column_config={
        "WCSS": st.column_config.NumberColumn(format="%.2f"),
        "Silhouette": st.column_config.NumberColumn(format="%.6f"),
    })
    with st.expander(f"Rincian preprocessing · {raw - clean} baris tidak dipakai"):
        labels = {
            "empty_after_stem": "Kosong setelah stemming",
            "duplicate_clean_text": "Duplikat setelah pembersihan",
            "fewer_than_2_tokens": "Kurang dari dua token",
        }
        steps = pd.DataFrame(summary["dropped"]).rename(columns={"stage": "Tahap", "dropped": "Baris dikeluarkan"})
        steps["Tahap"] = steps["Tahap"].map(labels)
        st.dataframe(steps, hide_index=True)


st.set_page_config(page_title="K-Means Ulasan Tokopedia", layout="wide")
st.title("Clustering Ulasan Tokopedia")
view = st.radio("Tampilan", ("Hasil Skripsi", "Eksplorasi CSV"), horizontal=True, label_visibility="collapsed")
if view == "Hasil Skripsi":
    show_thesis()
else:
    show_exploration()

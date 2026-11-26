import streamlit as st
import pandas as pd
import re, string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="KMeans Ulasan Tokopedia", layout="wide")
st.title("📊 Clustering Ulasan Tokopedia - KMeans")

uploaded = st.file_uploader("Upload CSV Ulasan", type=["csv"])
if not uploaded:
    st.info("Upload dulu CSV mentah yang isinya kolom ulasan / review / teks.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("### 1. Data Mentah")
st.dataframe(df.head(), use_container_width=True)

# pilih kolom teks
kolom = st.selectbox("Pilih kolom teks ulasan", df.columns.tolist())
teks_asli = df[kolom].astype(str).fillna("")

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

df["clean_text"] = teks_asli.apply(preprocess)
df["tokens"] = df["clean_text"].apply(lambda x: x.split())

st.write("**Hasil Tokenizing (5 baris pertama):**")
st.dataframe(df[[kolom, "clean_text", "tokens"]].head(), use_container_width=True)

# --- TF-IDF + KMEANS ---
st.write("### 3. TF-IDF + KMeans")
k = st.slider("Pilih Jumlah Cluster (K)", 2, 10, 6)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["clean_text"])

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X)

st.success(f"Clustering selesai! Total data: {len(df)}, K = {k}")

# --- EVALUASI ---
st.write("### 4. Evaluasi Cluster")
sil = silhouette_score(X, df["cluster_id"])
dbi = davies_bouldin_score(X.toarray(), df["cluster_id"])
ch = calinski_harabasz_score(X.toarray(), df["cluster_id"])

c1, c2, c3 = st.columns(3)
c1.metric("Silhouette Score", f"{sil:.4f}")
c2.metric("Davies-Bouldin (makin kecil makin bagus)", f"{dbi:.4f}")
c3.metric("Calinski-Harabasz", f"{ch:.1f}")
st.caption("Silhouette: -1 sampai 1, makin dekat 1 makin bagus pemisahan clusternya.")

# --- DISTRIBUSI ---
st.write("### 5. Distribusi & Isi Tiap Cluster")
counts = df["cluster_id"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(counts.index.astype(str), counts.values)
ax.set_xlabel("Cluster")
ax.set_ylabel("Jumlah Ulasan")
st.pyplot(fig)
st.dataframe(counts.rename("Jumlah"), use_container_width=True)

# Top words + contoh per cluster
terms = vectorizer.get_feature_names_out()
for cl in sorted(df["cluster_id"].unique()):
    st.write(f"**Cluster {cl} - {len(df[df['cluster_id']==cl])} ulasan**")
    idx = kmeans.cluster_centers_[cl].argsort()[-10:][::-1]
    top = ", ".join([terms[i] for i in idx])
    st.write(f"Kata kunci: `{top}`")
    st.dataframe(df[df["cluster_id"]==cl].head(5)[[kolom, "clean_text", "cluster_id"]], use_container_width=True)

st.download_button("Download Hasil Clustering (CSV)", df.to_csv(index=False), "hasil_clustering.csv", "text/csv")

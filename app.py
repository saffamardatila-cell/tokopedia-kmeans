import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Ulasan IndiHome", layout="wide")
st.title("📊 Analisis Ulasan IndiHome - K-Means Clustering")
st.write("Mengelompokkan 5.955 ulasan menjadi 6 cluster_id berdasarkan kemiripan kata.")

@st.cache_data
def load_data():
    return pd.read_csv("dataset_klaster_k6.csv")

df = load_data()
st.sidebar.header("Filter")
kelompok_list = sorted(df['cluster_id'].unique().tolist())
pilihan = st.sidebar.multiselect("Pilih Kelompok:", kelompok_list, default=kelompok_list)
df_filter = df[df['cluster_id'].isin(pilihan)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Ulasan", len(df))
col2.metric("Total Kelompok", df['cluster_id'].nunique())
col3.metric("Yang Ditampilkan", len(df_filter))

st.subheader("Jumlah Ulasan per Kelompok")
jumlah = df['cluster_id'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(jumlah.index.astype(str), jumlah.values, color="#FF4B4B")
ax.set_xlabel("Kelompok")
ax.set_ylabel("Jumlah Ulasan")
for i, v in enumerate(jumlah.values):
    ax.text(i, v + 50, str(v), ha='center')
st.pyplot(fig)

st.subheader("Contoh Ulasan per Kelompok")
kolom_teks = [c for c in df.columns if 'ulasan' in c.lower() or 'teks' in c.lower() or 'review' in c.lower() or 'komentar' in c.lower()]
for k in kelompok_list:
    st.write(f"**Kelompok {k}** - {len(df[df['cluster_id']==k])} ulasan")
    st.dataframe(df[df['cluster_id']==k].head(5), use_container_width=True)

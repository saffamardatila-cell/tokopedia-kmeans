# Clustering Ulasan Tokopedia

**Hasil Skripsi** langsung menampilkan ringkasan agregat: 6.316 ulasan mentah, 5.955 setelah preprocessing, distribusi K = 6, dan perbandingan K = 2–6. Angka di `hasil_skripsi.json` berasal dari keluaran evaluasi, profil klaster, dan catatan preprocessing analisis skripsi, bukan dari `dataset_klaster_k6.csv` atau perhitungan ulang aplikasi. K = 6 hanya skenario visual sementara; Silhouette mendekati nol, jadi pemisahannya lemah. Mode ini tidak memerlukan dataset atau unggahan.

**Eksplorasi CSV** menghitung ulang TF-IDF dan K-Means dari CSV yang kamu unggah. Preprocessing-nya sederhana dan berbeda dari yang dipakai untuk skripsi. Gunakan CSV UTF-8 dengan header dan pemisah koma. Pilih kolom teks ulasan, bukan kolom angka/ID. Batasnya 2 MB, 6.500 baris, 40 kolom, dan 8.000 karakter per sel. Jangan unggah data pribadi, terutama jika aplikasinya di-host.

## Jalankan lokal

Dari direktori repo, gunakan Python 3.10–3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Di Streamlit Community Cloud, pilih Python 3.12 dan `app.py` sebagai entry point; tidak perlu secrets atau dataset. Simpan `hasil_skripsi.json` di samping `app.py`.

## Tes

```sh
python -m unittest discover -s tests -v
```

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def explore(csv_bytes):
    # AppTest 1.47 has no upload widget driver. Only replace browser transport.
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)):
        app = AppTest.from_file(str(APP)).run()
        if app.radio:
            app.radio[0].set_value("Eksplorasi CSV").run()
    return app


class AppTests(unittest.TestCase):
    def test_default_shows_actual_thesis_aggregates_without_upload(self):
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(app.radio and app.radio[0].value == "Hasil Skripsi")
        self.assertEqual([metric.value for metric in app.metric], ["6.316", "5.955", "6", "0,016276"])
        self.assertFalse(app.get("file_uploader"))
        tables = [element.value for element in app.dataframe]
        distribution = next(table for table in tables if "Tema" in table.columns)
        self.assertEqual(distribution["Jumlah ulasan"].tolist(), [324, 3647, 215, 669, 243, 857])
        evaluation = next(table for table in tables if "K" in table.columns)
        self.assertEqual(evaluation["K"].tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(evaluation.iloc[-1]["Silhouette"], 0.016275929555052648)
        self.assertTrue(any("sementara" in element.value for element in app.caption))

    def test_empty_vocabulary_shows_actionable_error(self):
        for sample in (b"rating\n4\n5\n", b"ulasan\n!!!\nyang dan\n", b"ulasan,rating\n,4\n,5\n"):
            with self.subTest(sample=sample):
                app = explore(sample)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertIn("Pilih kolom yang berisi teks ulasan", app.error[0].value)

    def test_fewer_vectors_than_default_k_still_clusters(self):
        app = explore(b"ulasan\npaket cepat\npaket tiba cepat\nvoucher diskon\npotongan voucher\n")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("K = 4", app.success[0].value)
        self.assertIn("Metrik tidak dihitung", app.info[0].value)
        self.assertTrue(app.get("download_button"))

    def test_duplicate_tfidf_vectors_limit_k_even_with_different_text(self):
        app = explore(b"ulasan\npaket cepat\ncepat paket\npaket cepat\nvoucher diskon\ndiskon voucher\nvoucher diskon\n")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("K = 2", app.success[0].value)
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(app.metric[0].value, "1.0000")

    def test_one_distinct_vector_cannot_cluster(self):
        app = explore(b"ulasan\npaket cepat\ncepat paket\npaket paket paket cepat cepat cepat\n")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("minimal 2 vektor", app.error[0].value)

    def test_null_unusable_and_tfidf_empty_rows_not_in_output(self):
        app = explore(b"ulasan,rating\n,5\nNaN,3\nNULL,4\n<NA>,5\n1234,4\n!!!a,4\npaket cepat,5\ncepat paket,4\nbarang tiba,3\nbarang datang,5\n")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("Total data: 4", app.success[0].value)
        self.assertEqual(app.dataframe[1].value["Ulasan"].tolist(), ["paket cepat", "cepat paket", "barang tiba", "barang datang"])

    def test_happy_path_exploration_shows_metrics_and_download(self):
        app = explore(b"rating,clean_text,cluster_id\n4,paket cepat,0\n5,cepat paket,0\n3,barang tiba,1\n4,barang datang,1\n4,voucher diskon,2\n5,potongan voucher,2\n3,murah belanja,3\n4,mudah belanja,3\n")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.selectbox[0].value, "clean_text")
        self.assertIn("K = 6", app.success[0].value)
        self.assertEqual(len(app.metric), 3)
        self.assertTrue(app.get("download_button"))

    def test_csv_errors_are_handled_without_disclosing_input(self):
        for sample, message in (
            (b"", "header dan setidaknya satu baris"),
            (b"ulasan\n", "header dan setidaknya satu baris"),
            (b"ulasan\n\xff", "UTF-8"),
            (b'ulasan\n"contoh belum tertutup\n', "format CSV tidak valid"),
            (b"ulasan,rating\npaket datang,4,lebih\n", "jumlah kolom"),
            (b"ulasan,rating\npaket datang\n", "jumlah kolom"),
            (b"ulasan,ulasan\npaket datang,4\n", "nama kolom"),
        ):
            with self.subTest(sample=sample):
                app = explore(sample)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertIn(message, app.error[0].value)

    def test_csv_resource_limits_reject_oversized_input(self):
        for sample, message in (
            (b"x" * (2 * 1024 * 1024 + 1), "2 MB"),
            (b"ulasan\n" + b"paket tiba\n" * 6501, "6.500 baris"),
            (b",".join(f"c{i}".encode() for i in range(41)) + b"\n", "40 kolom"),
            (b"ulasan\n" + b"x" * 8001 + b"\n", "8.000 karakter"),
        ):
            with self.subTest(message=message):
                app = explore(sample)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertIn(message, app.error[0].value)

    def test_utf8_bom_and_quoted_commas_work(self):
        app = explore('\ufeffUlasan,Skor\r\n"paket, cepat",5\r\n"barang, tiba",4\r\n'.encode())
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("K = 2", app.success[0].value)
        self.assertEqual(app.dataframe[0].value["Ulasan"].tolist(), ["paket, cepat", "barang, tiba"])

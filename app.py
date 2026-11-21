import json
import logging
import os
import traceback
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("health_score_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

st.set_page_config(
    page_title="Prediksi Health Score",
    page_icon="🩺",
    layout="wide"
)

MODEL_PKL_PATH = "best_health_predictor.pkl"
MODEL_JOBLIB_PATH = "health_score_model.joblib"
METADATA_PATH = "model_metadata.json"
DATA_PATH = "health.xlsx"


def _compute_metrics_from_data():
    if not os.path.exists(DATA_PATH):
        logger.warning("Dataset %s tidak ditemukan saat menghitung metrik; menggunakan metrik default.", DATA_PATH)
        return {"MAE": 5.7556, "RMSE": 9.7536, "R2": 0.3912}

    df = pd.read_excel(DATA_PATH, sheet_name="health")
    clean = df.dropna(subset=["health_score"]).copy()
    X = clean.drop(columns=["health_score"])
    y = clean["health_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "R2": round(float(r2_score(y_test, preds)), 4),
    }


@st.cache_resource
def load_or_train_model():
    metrics = None
    best_model_name = "Gradient Boosting (Tuned)"
    logger.info("Memulai proses pemuatan model/metadata.")

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "test_metrics" in meta:
                metrics = {
                    "MAE": round(float(meta["test_metrics"].get("MAE", 0)), 4),
                    "RMSE": round(float(meta["test_metrics"].get("RMSE", 0)), 4),
                    "R2": round(float(meta["test_metrics"].get("R2", 0)), 4),
                }
            best_model_name = meta.get("best_model", best_model_name)
            logger.info("Metadata berhasil dimuat dari %s (model=%s).", METADATA_PATH, best_model_name)
        except Exception as exc:
            logger.error("Gagal membaca %s: %s\n%s", METADATA_PATH, exc, traceback.format_exc())
            metrics = None

    if os.path.exists(MODEL_PKL_PATH):
        try:
            model = joblib.load(MODEL_PKL_PATH)
            if metrics is None:
                metrics = _compute_metrics_from_data()
            logger.info("Model berhasil dimuat dari %s.", MODEL_PKL_PATH)
            return model, metrics, best_model_name
        except Exception as exc:
            logger.error("Gagal memuat %s: %s\n%s", MODEL_PKL_PATH, exc, traceback.format_exc())
            st.error(f"Gagal memuat model '{MODEL_PKL_PATH}'. Detail error tercatat di {LOG_FILE}.")
            st.stop()

    if os.path.exists(MODEL_JOBLIB_PATH):
        try:
            artifact = joblib.load(MODEL_JOBLIB_PATH)
            logger.info("Model berhasil dimuat dari %s.", MODEL_JOBLIB_PATH)
            if isinstance(artifact, dict) and "pipeline" in artifact:
                art_metrics = artifact.get("metrics", {})
                norm_metrics = {
                    "MAE": art_metrics.get("mae", art_metrics.get("MAE", 5.7556)),
                    "RMSE": art_metrics.get("rmse", art_metrics.get("RMSE", 9.7536)),
                    "R2": art_metrics.get("r2", art_metrics.get("R2", 0.3912)),
                }
                return artifact["pipeline"], norm_metrics, best_model_name
            return artifact, metrics or _compute_metrics_from_data(), best_model_name
        except Exception as exc:
            logger.error("Gagal memuat %s: %s\n%s", MODEL_JOBLIB_PATH, exc, traceback.format_exc())
            st.error(f"Gagal memuat model '{MODEL_JOBLIB_PATH}'. Detail error tercatat di {LOG_FILE}.")
            st.stop()

    if not os.path.exists(DATA_PATH):
        msg = (
            "File 'best_health_predictor.pkl', 'health_score_model.joblib', atau 'health.xlsx' "
            "tidak ditemukan di direktori proyek."
        )
        logger.error(msg)
        st.error(f"{msg} Log tersimpan di `{LOG_FILE}`.")
        st.stop()

    try:
        logger.info("Melatih ulang model dari %s.", DATA_PATH)
        df = pd.read_excel(DATA_PATH, sheet_name="health")
        clean = df.dropna(subset=["health_score"]).copy()
        X = clean.drop(columns=["health_score"])
        y = clean["health_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.8,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        trained_metrics = {
            "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "R2": round(float(r2_score(y_test, preds)), 4),
        }

        joblib.dump(pipeline, MODEL_PKL_PATH)
        logger.info("Model baru berhasil dilatih dan disimpan ke %s. Metrik: %s", MODEL_PKL_PATH, trained_metrics)
        return pipeline, trained_metrics, "Gradient Boosting (Tuned)"
    except Exception as exc:
        logger.error("Gagal melatih model dari %s: %s\n%s", DATA_PATH, exc, traceback.format_exc())
        st.error(f"Terjadi kesalahan saat melatih model. Detail error tercatat di `{LOG_FILE}`.")
        st.stop()


try:
    pipeline, metrics, model_label = load_or_train_model()
except Exception as exc:
    logger.critical("Kesalahan fatal saat inisialisasi aplikasi: %s\n%s", exc, traceback.format_exc())
    st.error(f"Kesalahan fatal pada aplikasi. Detail error tercatat di `{LOG_FILE}`.")
    st.stop()

st.title("🩺 Aplikasi Prediksi Health Score")
st.write(
    "Masukkan data gaya hidup dan kondisi kesehatan untuk memprediksi **Health Score** "
    f"menggunakan model **{model_label}**."
)
st.caption("Catatan: Hasil prediksi bersifat dukungan analitik awal, bukan diagnosis atau keputusan medis.")

with st.expander("📊 Performa Model pada Data Testing", expanded=False):
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE", f"{metrics['MAE']}")
    col_m2.metric("RMSE", f"{metrics['RMSE']}")
    col_m3.metric("R² Score", f"{metrics['R2']}")

with st.expander("🧾 Log Aplikasi Terbaru", expanded=False):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-30:]
        st.code("".join(lines), language="text")
    else:
        st.info("Belum ada catatan log.")

st.divider()

with st.form("health_form"):
    st.subheader("📝 Input Data Pengguna")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (age - tahun)", min_value=18, max_value=100, value=35, step=1)
        bmi = st.number_input("Indeks Massa Tubuh (bmi)", min_value=12.0, max_value=60.0, value=24.5, step=0.1)
        exercise_hours = st.number_input("Jam Olahraga per Minggu (exercise_hours)", min_value=0.0, max_value=40.0, value=3.5, step=0.5)
        sleep_hours = st.number_input("Jam Tidur per Hari (sleep_hours)", min_value=1.0, max_value=24.0, value=7.0, step=0.5)

    with col2:
        smoking_years = st.number_input("Lama Merokok (smoking_years - tahun)", min_value=0, max_value=80, value=0, step=1)
        alcohol_units = st.number_input("Konsumsi Alkohol per Minggu (alcohol_units)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
        diet_quality = st.slider("Kualitas Pola Makan (diet_quality: 1 = Buruk, 5 = Sangat Baik)", min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button("🔍 Prediksi Health Score", use_container_width=True)

if submitted:
    if smoking_years > (age - 10):
        logger.warning("Input tidak valid: smoking_years=%s melebihi age-10=%s", smoking_years, age - 10)
        st.error("⚠️ Input tidak valid: `smoking_years` tidak logis terhadap usia (`age`).")
    else:
        try:
            input_df = pd.DataFrame([{
                "age": float(age),
                "bmi": float(bmi),
                "exercise_hours": float(exercise_hours),
                "sleep_hours": float(sleep_hours),
                "smoking_years": float(smoking_years),
                "alcohol_units": float(alcohol_units),
                "diet_quality": float(diet_quality),
            }])

            prediction = float(pipeline.predict(input_df)[0])
            logger.info("Prediksi berhasil: score=%.2f | input=%s", prediction, input_df.iloc[0].to_dict())

            if prediction >= 95:
                kategori = "Sangat Baik"
                st.success(f"### Hasil Prediksi Health Score: **{prediction:.2f}** ({kategori})")
            elif prediction >= 85:
                kategori = "Baik"
                st.info(f"### Hasil Prediksi Health Score: **{prediction:.2f}** ({kategori})")
            elif prediction >= 75:
                kategori = "Cukup"
                st.warning(f"### Hasil Prediksi Health Score: **{prediction:.2f}** ({kategori})")
            else:
                kategori = "Perlu Perhatian"
                st.error(f"### Hasil Prediksi Health Score: **{prediction:.2f}** ({kategori})")

            st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            logger.error("Gagal melakukan prediksi: %s\n%s", exc, traceback.format_exc())
            st.error(f"Terjadi kesalahan saat memproses prediksi. Detail error tercatat di `{LOG_FILE}`.")
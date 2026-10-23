import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG & CSS (MUST BE FIRST)
# ============================================================

st.set_page_config(
    page_title="Evaluasi dan Validasi",
    page_icon="📋",
    layout="wide"
)

css = Path("assets/style.css")
if css.exists():
    with open(css, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

st.markdown(
    """
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3C/svg%3E" style="display:none" onload="
        try {
            if (!window._beranda_lock) {
                window._beranda_lock = true;
                const fix = () => {
                    const links = document.querySelectorAll('[data-testid=stSidebarNav] a, [data-testid=stSidebarNavLink]');
                    if (links.length > 0) {
                        const sp = links[0].querySelector('span');
                        if (sp && sp.textContent === 'app') { sp.textContent = 'Beranda'; }
                    }
                };
                fix();
                new MutationObserver(fix).observe(document.body, { childList: true, subtree: true, characterData: true });
            }
        } catch(e) {}
    " />
    """,
    unsafe_allow_html=True
)

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from utils.prediction import (
    load_model
)



# ============================================================
# FUNGSI HTML
# ============================================================

def tampilkan_html(html):
    html_bersih = "".join(
        baris.strip()
        for baris in html.splitlines()
    )

    st.markdown(
        html_bersih,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

tampilkan_html(
    """
    <div class="hero-box">
        <div class="hero-title">
            Evaluasi dan Validasi Model
        </div>
        <div class="hero-subtitle">
            Analisis performa Random Forest dan SVR menggunakan
             data training, testing, dan Time Series K-Fold Cross-Validation.
        </div>
    </div>
    """
)


# ============================================================
# CEK DATA
# ============================================================

if "df_feature" not in st.session_state:

    st.warning(
        "Input dan proses data terlebih dahulu."
    )

    st.stop()


df_feature = st.session_state["df_feature"]


# ============================================================
# LOAD MODEL UNTUK EVALUASI TRAINING DAN TESTING
# ============================================================

with st.spinner(
    "Sedang menghitung evaluasi model..."
):

    bundle_rf = load_model(
        "models/model_rf_beras.joblib"
    )

    bundle_svr = load_model(
        "models/model_svr_beras.joblib"
    )


    fitur = bundle_rf["fitur"]


    X = df_feature[fitur]

    y = df_feature["Harga"]


    # ========================================================
    # SPLIT 80:20
    # ========================================================

    split_index = int(
        len(df_feature) * 0.8
    )


    X_train = X.iloc[
        :split_index
    ].copy()


    X_test = X.iloc[
        split_index:
    ].copy()


    y_train = y.iloc[
        :split_index
    ].copy()


    y_test = y.iloc[
        split_index:
    ].copy()


    # ========================================================
    # PREDIKSI TRAINING DAN TESTING
    # ========================================================

    pred_rf_train = (
        bundle_rf["model"]
        .predict(X_train)
    )


    pred_rf_test = (
        bundle_rf["model"]
        .predict(X_test)
    )


    pred_svr_train = (
        bundle_svr["model"]
        .predict(X_train)
    )


    pred_svr_test = (
        bundle_svr["model"]
        .predict(X_test)
    )


# ============================================================
# FUNGSI EVALUASI
# ============================================================

def metric_model(
    y_true,
    y_pred
):

    return {

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "MAPE":
            (
                mean_absolute_percentage_error(
                    y_true,
                    y_pred
                )
                * 100
            ),

        "R2":
            r2_score(
                y_true,
                y_pred
            )
    }


# ============================================================
# HASIL EVALUASI TRAINING DAN TESTING
# ============================================================

rf_train = metric_model(
    y_train,
    pred_rf_train
)

rf_test = metric_model(
    y_test,
    pred_rf_test
)

svr_train = metric_model(
    y_train,
    pred_svr_train
)

svr_test = metric_model(
    y_test,
    pred_svr_test
)


# ============================================================
# GAP TRAINING DAN TESTING
# ============================================================

gap_rf = abs(
    rf_train["R2"]
    - rf_test["R2"]
)


gap_svr = abs(
    svr_train["R2"]
    - svr_test["R2"]
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    tampilkan_html(
        f"""
        <div class="metric-card yellow">
            <div class="metric-label">
                RF Testing R²
            </div>
            <div class="metric-value">
                {rf_test['R2']:.4f}
            </div>
            <div class="metric-note">
                MAE {rf_test['MAE']:.2f}
            </div>
        </div>
        """
    )


with c2:

    tampilkan_html(
        f"""
        <div class="metric-card blue">
            <div class="metric-label">
                SVR Testing R²
            </div>
            <div class="metric-value">
                {svr_test['R2']:.4f}
            </div>
            <div class="metric-note">
                MAE {svr_test['MAE']:.2f}
            </div>
        </div>
        """
    )


with c3:

    tampilkan_html(
        f"""
        <div class="metric-card red">
            <div class="metric-label">
                RF Train-Test Gap
            </div>
            <div class="metric-value">
                {gap_rf:.4f}
            </div>
            <div class="metric-note">
                Selisih R²
            </div>
        </div>
        """
    )


with c4:

    tampilkan_html(
        f"""
        <div class="metric-card green">
            <div class="metric-label">
                SVR Train-Test Gap
            </div>
            <div class="metric-value">
                {gap_svr:.4f}
            </div>
            <div class="metric-note">
                Selisih R²
            </div>
        </div>
        """
    )


st.write("")


# ============================================================
# TRAIN VS TEST
# ============================================================

left, right = st.columns(2)


with left:

    with st.container(
        border=True
    ):

        st.subheader(
            "R² Training vs Testing"
        )


        fig_r2 = go.Figure()


        fig_r2.add_trace(
            go.Bar(
                x=[
                    "Random Forest",
                    "SVR"
                ],

                y=[
                    rf_train["R2"],
                    svr_train["R2"]
                ],

                name="Training",

                marker_color="#F5B700"
            )
        )


        fig_r2.add_trace(
            go.Bar(
                x=[
                    "Random Forest",
                    "SVR"
                ],

                y=[
                    rf_test["R2"],
                    svr_test["R2"]
                ],

                name="Testing",

                marker_color="#1877F2"
            )
        )


        fig_r2.add_hline(
            y=0,
            line_color="#B5B5B5",
            line_width=1
        )


        fig_r2.update_layout(
            height=390,

            barmode="group",

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            font=dict(
                color="#151923"
            ),

            legend=dict(
                orientation="h",
                y=1.1
            )
        )


        fig_r2.update_yaxes(
            gridcolor="#EFE9DD",
            zeroline=False
        )


        st.plotly_chart(
            fig_r2,
            use_container_width=True
        )


with right:

    with st.container(
        border=True
    ):

        st.subheader(
            "MAE Training vs Testing"
        )


        fig_mae = go.Figure()


        fig_mae.add_trace(
            go.Bar(
                x=[
                    "Random Forest",
                    "SVR"
                ],

                y=[
                    rf_train["MAE"],
                    svr_train["MAE"]
                ],

                name="Training",

                marker_color="#EE4B5A"
            )
        )


        fig_mae.add_trace(
            go.Bar(
                x=[
                    "Random Forest",
                    "SVR"
                ],

                y=[
                    rf_test["MAE"],
                    svr_test["MAE"]
                ],

                name="Testing",

                marker_color="#1877F2"
            )
        )


        fig_mae.update_layout(
            height=390,

            barmode="group",

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            font=dict(
                color="#151923"
            ),

            legend=dict(
                orientation="h",
                y=1.1
            )
        )


        fig_mae.update_yaxes(
            gridcolor="#EFE9DD",
            zeroline=False
        )


        st.plotly_chart(
            fig_mae,
            use_container_width=True
        )


# ============================================================
# TIME SERIES K-FOLD CROSS-VALIDATION
# ============================================================

st.write("")


with st.container(
    border=True
):

    st.subheader(
        "Time Series K-Fold Cross-Validation"
    )


    st.caption(
        "K = 5 dengan TimeSeriesSplit. "
        "Parameter model disamakan dengan konfigurasi Colab."
    )


    tscv = TimeSeriesSplit(
        n_splits=5
    )


    hasil_cv = []


    with st.spinner(
        "Sedang menjalankan 5 fold validation..."
    ):

        for fold, (
            train_idx,
            val_idx
        ) in enumerate(
            tscv.split(X_train),
            start=1
        ):


            # =================================================
            # DATA SETIAP FOLD
            # =================================================

            X_tr = X_train.iloc[
                train_idx
            ]


            X_val = X_train.iloc[
                val_idx
            ]


            y_tr = y_train.iloc[
                train_idx
            ]


            y_val = y_train.iloc[
                val_idx
            ]


            # =================================================
            # RANDOM FOREST
            # SAMA DENGAN COLAB
            # =================================================

            model_rf_cv = (
                RandomForestRegressor(
                    max_depth=10,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    n_estimators=200,
                    random_state=42
                )
            )


            model_rf_cv.fit(
                X_tr,
                y_tr
            )


            pred_rf_cv = (
                model_rf_cv.predict(
                    X_val
                )
            )


            mae_rf = (
                mean_absolute_error(
                    y_val,
                    pred_rf_cv
                )
            )


            # =================================================
            # SVR
            # SAMA DENGAN COLAB
            # =================================================

            scaler_fold = (
                StandardScaler()
            )


            X_tr_scaled = (
                scaler_fold.fit_transform(
                    X_tr
                )
            )


            X_val_scaled = (
                scaler_fold.transform(
                    X_val
                )
            )


            model_svr_cv = SVR(
                kernel="rbf",
                C=1000,
                epsilon=0.1,
                gamma=0.001
            )


            model_svr_cv.fit(
                X_tr_scaled,
                y_tr
            )


            pred_svr_cv = (
                model_svr_cv.predict(
                    X_val_scaled
                )
            )


            mae_svr = (
                mean_absolute_error(
                    y_val,
                    pred_svr_cv
                )
            )


            # =================================================
            # SIMPAN HASIL
            # =================================================

            hasil_cv.append({

                "Fold":
                    fold,

                "Jumlah Training":
                    len(train_idx),

                "Jumlah Validation":
                    len(val_idx),

                "MAE RF":
                    mae_rf,

                "MAE SVR":
                    mae_svr
            })


    hasil_cv = pd.DataFrame(
        hasil_cv
    )


    # ========================================================
    # METRIC CROSS-VALIDATION
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "MAE RF Rata-rata",
        f"{hasil_cv['MAE RF'].mean():.2f}"
    )


    c2.metric(
        "Std RF",
        f"{hasil_cv['MAE RF'].std():.2f}"
    )


    c3.metric(
        "MAE SVR Rata-rata",
        f"{hasil_cv['MAE SVR'].mean():.2f}"
    )


    c4.metric(
        "Std SVR",
        f"{hasil_cv['MAE SVR'].std():.2f}"
    )


    # ========================================================
    # GRAFIK CROSS-VALIDATION
    # ========================================================

    fig_cv = go.Figure()


    fig_cv.add_trace(
        go.Scatter(
            x=hasil_cv["Fold"],
            y=hasil_cv["MAE RF"],

            mode="lines+markers",

            name="Random Forest",

            line=dict(
                color="#F5B700",
                width=3
            ),

            marker=dict(
                size=9
            )
        )
    )


    fig_cv.add_trace(
        go.Scatter(
            x=hasil_cv["Fold"],
            y=hasil_cv["MAE SVR"],

            mode="lines+markers",

            name="SVR",

            line=dict(
                color="#1877F2",
                width=3
            ),

            marker=dict(
                size=9
            )
        )
    )


    fig_cv.update_layout(
        height=410,

        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            color="#151923"
        ),

        xaxis_title="Fold",

        yaxis_title="MAE",

        legend=dict(
            orientation="h",
            y=1.1
        )
    )


    fig_cv.update_xaxes(
        dtick=1,
        showgrid=False
    )


    fig_cv.update_yaxes(
        gridcolor="#EFE9DD",
        zeroline=False
    )


    st.plotly_chart(
        fig_cv,
        use_container_width=True
    )


    # ========================================================
    # TABEL CROSS-VALIDATION
    # ========================================================

    st.dataframe(
        hasil_cv.round(2),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# RINGKASAN EVALUASI MODEL
# ============================================================

st.write("")


with st.container(
    border=True
):

    st.subheader(
        "Ringkasan Evaluasi Model"
    )


    ringkasan = pd.DataFrame({

        "Model": [
            "Random Forest",
            "Random Forest",
            "SVR",
            "SVR"
        ],

        "Data": [
            "Training",
            "Testing",
            "Training",
            "Testing"
        ],

        "MAE": [
            rf_train["MAE"],
            rf_test["MAE"],
            svr_train["MAE"],
            svr_test["MAE"]
        ],

        "RMSE": [
            rf_train["RMSE"],
            rf_test["RMSE"],
            svr_train["RMSE"],
            svr_test["RMSE"]
        ],

        "MAPE (%)": [
            rf_train["MAPE"],
            rf_test["MAPE"],
            svr_train["MAPE"],
            svr_test["MAPE"]
        ],

        "R²": [
            rf_train["R2"],
            rf_test["R2"],
            svr_train["R2"],
            svr_test["R2"]
        ]
    })


    st.dataframe(
        ringkasan.round(4),

        use_container_width=True,

        hide_index=True
    )
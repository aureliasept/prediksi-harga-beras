import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG & CSS (MUST BE FIRST)
# ============================================================

st.set_page_config(
    page_title="Grafik dan Prediksi",
    page_icon="📊",
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

from utils.prediction import (
    load_model,
    predict_models
)

from utils.metrics import (
    compare_models
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
        <div class="hero-title">Prediksi Harga Beras</div>
        <div class="hero-subtitle">
            Perbandingan performa Random Forest Regressor
            dan Support Vector Regression pada data testing.
        </div>
    </div>
    """
)


# ============================================================
# CEK DATA
# ============================================================

if "df_feature" not in st.session_state:

    st.warning(
        "Input dan proses data terlebih dahulu "
        "pada halaman Input Data."
    )

    st.stop()


df_feature = st.session_state["df_feature"]


# ============================================================
# LOAD MODEL DAN PREDIKSI
# ============================================================

with st.spinner(
    "Sedang memuat model dan menghitung prediksi..."
):

    bundle_rf = load_model(
        "models/model_rf_beras.joblib"
    )

    bundle_svr = load_model(
        "models/model_svr_beras.joblib"
    )


    # Data testing 20% terakhir
    split_index = int(
        len(df_feature) * 0.8
    )


    data_test = (
        df_feature
        .iloc[split_index:]
        .reset_index(drop=True)
    )


    hasil = predict_models(
        data_test,
        bundle_rf,
        bundle_svr
    )


    evaluasi = compare_models(
        hasil["Aktual"],
        hasil["Prediksi_RF"],
        hasil["Prediksi_SVR"]
    )


st.session_state[
    "hasil_prediksi_dashboard"
] = hasil


# ============================================================
# HASIL EVALUASI
# ============================================================

rf = (
    evaluasi[
        evaluasi["Model"] == "Random Forest"
    ]
    .iloc[0]
)


svr = (
    evaluasi[
        evaluasi["Model"] == "SVR"
    ]
    .iloc[0]
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    tampilkan_html(
        f"""
        <div class="metric-card red">
            <div class="metric-label">Harga Aktual Terakhir</div>
            <div class="metric-value">
                Rp {hasil['Aktual'].iloc[-1]:,.0f}
            </div>
            <div class="metric-note">
                {hasil['Tanggal'].iloc[-1].strftime('%d %b %Y')}
            </div>
        </div>
        """
    )


with c2:

    tampilkan_html(
        f"""
        <div class="metric-card yellow">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value">
                Rp {hasil['Prediksi_RF'].iloc[-1]:,.0f}
            </div>
            <div class="metric-note">
                R² {rf['R²']:.4f}
            </div>
        </div>
        """
    )


with c3:

    tampilkan_html(
        f"""
        <div class="metric-card blue">
            <div class="metric-label">SVR</div>
            <div class="metric-value">
                Rp {hasil['Prediksi_SVR'].iloc[-1]:,.0f}
            </div>
            <div class="metric-note">
                R² {svr['R²']:.4f}
            </div>
        </div>
        """
    )


with c4:

    if svr["MAE"] < rf["MAE"]:
        model_terbaik = "SVR"
    else:
        model_terbaik = "Random Forest"


    tampilkan_html(
        f"""
        <div class="metric-card green">
            <div class="metric-label">Model Terbaik</div>
            <div class="metric-value">
                {model_terbaik}
            </div>
            <div class="metric-note">
                Berdasarkan MAE testing
            </div>
        </div>
        """
    )


st.write("")


# ============================================================
# GRAFIK UTAMA DAN RESIDUAL
# ============================================================

left, right = st.columns(
    [1.65, 1]
)


with left:

    with st.container(
        border=True
    ):

        st.subheader(
            "Prediksi vs Data Aktual"
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=hasil["Tanggal"],
                y=hasil["Aktual"],
                mode="lines",
                name="Aktual",

                line=dict(
                    color="#EE4B5A",
                    width=3
                )
            )
        )


        fig.add_trace(
            go.Scatter(
                x=hasil["Tanggal"],
                y=hasil["Prediksi_RF"],
                mode="lines",
                name="Random Forest",

                line=dict(
                    color="#F5B700",
                    width=2.8
                )
            )
        )


        fig.add_trace(
            go.Scatter(
                x=hasil["Tanggal"],
                y=hasil["Prediksi_SVR"],
                mode="lines",
                name="SVR",

                line=dict(
                    color="#1877F2",
                    width=3
                )
            )
        )


        fig.update_layout(
            height=450,

            hovermode="x unified",

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            margin=dict(
                l=20,
                r=20,
                t=15,
                b=20
            ),

            font=dict(
                color="#151923"
            ),

            legend=dict(
                orientation="h",
                y=1.08
            ),

            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)"
        )


        fig.update_xaxes(
            showgrid=False,
            linecolor="#D7D7D7"
        )


        fig.update_yaxes(
            gridcolor="#EFE9DD",
            tickprefix="Rp ",
            zeroline=False
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


with right:

    with st.container(
        border=True
    ):

        st.subheader(
            "Residual SVR"
        )


        residual_svr = (
            hasil["Aktual"]
            - hasil["Prediksi_SVR"]
        )


        fig_res = go.Figure()


        fig_res.add_trace(
            go.Scatter(
                x=hasil["Prediksi_SVR"],
                y=residual_svr,

                mode="markers",

                marker=dict(
                    color="#1877F2",
                    size=7,
                    opacity=0.7
                ),

                name="Residual"
            )
        )


        fig_res.add_hline(
            y=0,
            line_dash="dash",
            line_color="#EE4B5A"
        )


        fig_res.update_layout(
            height=450,

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            margin=dict(
                l=20,
                r=20,
                t=15,
                b=20
            ),

            font=dict(
                color="#151923"
            ),

            xaxis_title="Prediksi SVR",
            yaxis_title="Residual",

            showlegend=False
        )


        fig_res.update_xaxes(
            gridcolor="#F1EEE7"
        )


        fig_res.update_yaxes(
            gridcolor="#F1EEE7"
        )


        st.plotly_chart(
            fig_res,
            use_container_width=True
        )


# ============================================================
# PERBANDINGAN MODEL
# ============================================================

st.write("")

left, right = st.columns(2)


with left:

    with st.container(
        border=True
    ):

        st.subheader(
            "Perbandingan Error"
        )


        fig_error = go.Figure()


        fig_error.add_trace(
            go.Bar(
                x=[
                    "MAE",
                    "RMSE",
                    "MAPE"
                ],

                y=[
                    rf["MAE"],
                    rf["RMSE"],
                    rf["MAPE (%)"]
                ],

                name="Random Forest",

                marker_color="#F5B700"
            )
        )


        fig_error.add_trace(
            go.Bar(
                x=[
                    "MAE",
                    "RMSE",
                    "MAPE"
                ],

                y=[
                    svr["MAE"],
                    svr["RMSE"],
                    svr["MAPE (%)"]
                ],

                name="SVR",

                marker_color="#1877F2"
            )
        )


        fig_error.update_layout(
            height=360,

            barmode="group",

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            margin=dict(
                l=20,
                r=20,
                t=10,
                b=20
            ),

            font=dict(
                color="#151923"
            ),

            legend=dict(
                orientation="h",
                y=1.1
            )
        )


        fig_error.update_yaxes(
            gridcolor="#EFE9DD",
            zeroline=False
        )


        st.plotly_chart(
            fig_error,
            use_container_width=True
        )


with right:

    with st.container(
        border=True
    ):

        st.subheader(
            "R² Model"
        )


        fig_r2 = go.Figure()


        fig_r2.add_trace(
            go.Bar(
                x=[
                    "Random Forest",
                    "SVR"
                ],

                y=[
                    rf["R²"],
                    svr["R²"]
                ],

                marker_color=[
                    "#F5B700",
                    "#1877F2"
                ],

                text=[
                    f"{rf['R²']:.4f}",
                    f"{svr['R²']:.4f}"
                ],

                textposition="outside"
            )
        )


        fig_r2.add_hline(
            y=0,
            line_color="#B5B5B5",
            line_width=1
        )


        fig_r2.update_layout(
            height=360,

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            margin=dict(
                l=20,
                r=20,
                t=10,
                b=20
            ),

            font=dict(
                color="#151923"
            ),

            showlegend=False
        )


        fig_r2.update_yaxes(
            gridcolor="#EFE9DD",
            zeroline=False
        )


        st.plotly_chart(
            fig_r2,
            use_container_width=True
        )


# ============================================================
# TABEL
# ============================================================

st.write("")


with st.container(
    border=True
):

    st.subheader(
        "Riwayat Prediksi Testing"
    )


    tabel = hasil.copy()


    tabel["Aktual"] = (
        tabel["Aktual"]
        .round(2)
    )


    tabel["Prediksi_RF"] = (
        tabel["Prediksi_RF"]
        .round(2)
    )


    tabel["Prediksi_SVR"] = (
        tabel["Prediksi_SVR"]
        .round(2)
    )


    st.dataframe(
        tabel.tail(30),

        use_container_width=True,

        hide_index=True,

        column_config={

            "Tanggal":
                st.column_config.DateColumn(
                    "Tanggal",
                    format="DD/MM/YYYY"
                ),

            "Aktual":
                st.column_config.NumberColumn(
                    "Aktual",
                    format="Rp %.2f"
                ),

            "Prediksi_RF":
                st.column_config.NumberColumn(
                    "Random Forest",
                    format="Rp %.2f"
                ),

            "Prediksi_SVR":
                st.column_config.NumberColumn(
                    "SVR",
                    format="Rp %.2f"
                )
        }
    )
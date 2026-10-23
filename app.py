import streamlit as st
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Prediksi Harga Beras",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stSidebarNav"] > div:first-child a,
    [data-testid="stSidebarNav"] > ul > li:first-child a,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > div:first-child a,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > li:first-child a {
        font-size: 0 !important;
    }
    [data-testid="stSidebarNav"] > div:first-child a *,
    [data-testid="stSidebarNav"] > ul > li:first-child a *,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > div:first-child a *,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > li:first-child a * {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
    }
    [data-testid="stSidebarNav"] > div:first-child a::after,
    [data-testid="stSidebarNav"] > ul > li:first-child a::after,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > div:first-child a::after,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] > li:first-child a::after {
        content: "Beranda" !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: inherit !important;
        visibility: visible !important;
        display: inline-block !important;
        white-space: nowrap !important;
    }
    </style>
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

css = Path("assets/style.css")

if css.exists():
    with open(css, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Prediksi Harga Beras Kualitas Medium I"
)

st.caption(
    "Perbandingan Random Forest Regressor "
    "dan Support Vector Regression"
)


# ============================================================
# INFORMASI UTAMA
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:
    st.markdown(
        """
        <div class="metric-card blue">
            <div class="metric-label">
                Periode
            </div>
            <div class="metric-value home-value">
                2020–2025
            </div>
            <div class="metric-note">
                Data historis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:
    st.markdown(
        """
        <div class="metric-card yellow">
            <div class="metric-label">
                Wilayah
            </div>
            <div class="metric-value home-value">
                Jawa Barat
            </div>
            <div class="metric-note">
                Wilayah penelitian
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:
    st.markdown(
        """
        <div class="metric-card green">
            <div class="metric-label">
                Model
            </div>
            <div class="metric-value home-value">
                2 Model
            </div>
            <div class="metric-note">
                RF dan SVR
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:
    st.markdown(
        """
        <div class="metric-card red">
            <div class="metric-label">
                Forecast
            </div>
            <div class="metric-value home-value">
                3 Bulan
            </div>
            <div class="metric-note">
                Januari–Maret 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# TENTANG DASHBOARD
# ============================================================

left, right = st.columns(
    [1.6, 1]
)


with left:

    with st.container(
        border=True
    ):

        st.subheader(
            "Tentang Dashboard"
        )

        st.write(
            """
            Dashboard ini digunakan untuk memantau hasil prediksi
            harga Beras Kualitas Medium I di Jawa Barat
             menggunakan Random Forest Regressor dan
            Support Vector Regression.
            """
        )

        st.write(
            """
            Data diproses menggunakan fitur lag dan rolling mean,
            kemudian kedua model dibandingkan berdasarkan
            performa prediksi, evaluasi, dan validasi.
            """
        )


with right:

    with st.container(
        border=True
    ):

        st.subheader(
            "Model"
        )

        st.write(
            "**Random Forest Regressor**"
        )

        st.caption(
            "Model ensemble berbasis decision tree."
        )

        st.write(
            "**Support Vector Regression**"
        )

        st.caption(
            "Model regresi berbasis support vector."
        )


st.write("")


# ============================================================
# MENU
# ============================================================

with st.container(
    border=True
):

    st.subheader(
        "Menu Dashboard"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.write(
            "**Input Data**"
        )

        st.caption(
            "Upload dan proses data harga."
        )


        st.write(
            "**Grafik dan Prediksi**"
        )

        st.caption(
            "Perbandingan aktual, Random Forest, dan SVR."
        )


    with c2:

        st.write(
            "**Evaluasi dan Validasi**"
        )

        st.caption(
            "MAE, RMSE, MAPE, R², dan Time Series K-Fold."
        )


        st.write(
            "**Prediksi 3 Bulan**"
        )

        st.caption(
            "Prediksi Harga Beras Kualitas Medium IJanuari sampai Maret 2026."
        )


st.divider()

st.caption(
    "Prediksi Harga Beras Kualitas Medium I — Jawa Barat"
)
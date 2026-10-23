import streamlit as st
import plotly.graph_objects as go

from pathlib import Path

from utils.prediction import (
    load_model,
    forecast_3_months
)


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
    <div style="text-align: center; margin-top: 0.2rem; margin-bottom: 1.8rem;">
        <h1 style="text-align: center; font-size: 2.15rem; font-weight: 700; color: #151923; margin-bottom: 6px; letter-spacing: -0.02em;">
            SISTEM PREDIKSI HARGA BERAS
        </h1>
        <p style="color: #667085; font-size: 1.02rem; margin: 0;">
            Silakan masukkan parameter untuk memulai peramalan
        </p>
    </div>
    """
)


# ============================================================
# PARAMETER CARD
# ============================================================

with st.container(border=True):

    c_param1, c_param2, c_param3 = st.columns(3)

    with c_param1:
        st.text_input(
            "Model Peramalan",
            value="SVR & Random Forest",
            disabled=True
        )

    with c_param2:
        st.text_input(
            "Komoditas",
            value="Beras Medium",
            disabled=True
        )

    with c_param3:
        periode_pilihan = st.selectbox(
            "Periode Prediksi (Hari)",
            ["7 Hari", "30 Hari", "60 Hari", "90 Hari"],
            index=3
        )

    st.write("")

    col_b1, col_b2, col_b3 = st.columns([1.3, 1, 1.3])

    with col_b2:
        btn_prediksi = st.button(
            "Mulai Prediksi",
            type="primary",
            use_container_width=True
        )


st.divider()


# ============================================================
# PROSES PREDIKSI
# ============================================================

periode_map = {
    "7 Hari": 7,
    "30 Hari": 30,
    "60 Hari": 60,
    "90 Hari": 90
}
n_days = periode_map.get(periode_pilihan, 90)

if btn_prediksi:
    if "df_ts" not in st.session_state:
        st.warning(
            "Data time series belum tersedia. Silakan upload dan proses data terlebih dahulu pada halaman Input Data."
        )
    else:
        with st.spinner(f"Sedang membuat prediksi {periode_pilihan}..."):
            bundle_rf = load_model(
                "models/model_rf_beras.joblib"
            )

            bundle_svr = load_model(
                "models/model_svr_beras.joblib"
            )

            (
                raw_harian,
                _
            ) = forecast_3_months(
                st.session_state["df_ts"],
                bundle_rf,
                bundle_svr
            )

            # Potong harian sesuai periode yang dipilih (7, 30, 60, atau 90 hari)
            hasil_harian = raw_harian.head(n_days).copy()

            hasil_bulanan = (
                hasil_harian
                .groupby("Bulan")[
                    [
                        "Prediksi_RF",
                        "Prediksi_SVR"
                    ]
                ]
                .mean()
                .reset_index()
            )

            hasil_bulanan["Bulan"] = (
                hasil_bulanan["Bulan"]
                .dt.to_timestamp()
            )

            st.session_state["hasil_harian_2026"] = hasil_harian
            st.session_state["hasil_bulanan_2026"] = hasil_bulanan
            st.session_state["forecast_3_bulan"] = hasil_bulanan
            st.session_state["periode_terpilih_2026"] = periode_pilihan
            st.session_state["sudah_prediksi_2026"] = True


sudah_prediksi = (
    st.session_state.get("sudah_prediksi_2026", False)
    and ("df_ts" in st.session_state)
    and ("hasil_bulanan_2026" in st.session_state)
)


if not sudah_prediksi:

    # ============================================================
    # EMPTY STATE PLACEHOLDER (SEBELUM PREDIKSI)
    # ============================================================

    tampilkan_html(
        f"""
        <div style="text-align: center; padding: 15px 20px 35px 20px;">
            <p style="color: #667085; font-size: 0.98rem; margin-bottom: 28px;">
                Hasil prediksi harian selama {periode_pilihan.lower()} ke depan akan muncul di sini setelah Anda mengklik tombol &quot;Mulai Prediksi&quot;.
            </p>
            <div style="display: flex; justify-content: center; margin-bottom: 22px;">
                <svg width="135" height="102" viewBox="0 0 135 102" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- Window Frame -->
                    <rect x="8" y="8" width="86" height="58" rx="6" stroke="#9aa5b5" stroke-width="1.8" stroke-dasharray="3 3"/>
                    <line x1="8" y1="22" x2="94" y2="22" stroke="#9aa5b5" stroke-width="1.4"/>
                    <circle cx="16" cy="15" r="2.2" fill="#9aa5b5"/>
                    <circle cx="23" cy="15" r="2.2" fill="#9aa5b5"/>
                    <circle cx="30" cy="15" r="2.2" fill="#9aa5b5"/>
                    
                    <!-- Line Chart -->
                    <path d="M19 50 L36 35 L50 43 L72 26" stroke="#1877f2" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M64 26 H72 V34" stroke="#1877f2" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                    
                    <!-- Calendar Card -->
                    <rect x="60" y="36" width="68" height="58" rx="8" fill="#ffffff" stroke="#9aa5b5" stroke-width="1.8"/>
                    <path d="M60 44 C60 40.5 62.9 37.6 66.4 37.6 H121.6 C125.1 37.6 128 40.5 128 44 V50 H60 V44 Z" fill="#fff9eb"/>
                    <line x1="60" y1="50" x2="128" y2="50" stroke="#9aa5b5" stroke-width="1.4"/>
                    <rect x="72" y="33" width="4.5" height="8.5" rx="2" fill="#f5b700"/>
                    <rect x="110" y="33" width="4.5" height="8.5" rx="2" fill="#f5b700"/>
                    
                    <!-- Clock -->
                    <circle cx="94" cy="70" r="14.5" fill="#fffdf8" stroke="#f5b700" stroke-width="1.8"/>
                    <path d="M94 61 V70 L100 73" stroke="#f5b700" stroke-width="1.8" stroke-linecap="round"/>
                </svg>
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #151923; margin-bottom: 5px;">
                Data Historis PIHPS 2020-2025
            </div>
            <div style="font-size: 0.92rem; color: #667085;">
                Hasil prediksi akan ditampilkan dalam bentuk grafik interaktif dan tabel metrik evaluasi.
            </div>
        </div>
        """
    )

    st.stop()


df_ts = st.session_state["df_ts"]
hasil_harian = st.session_state["hasil_harian_2026"]
hasil_bulanan = st.session_state["hasil_bulanan_2026"]
periode_aktif = st.session_state.get("periode_terpilih_2026", periode_pilihan)


# ============================================================
# VALIDASI
# ============================================================

if len(hasil_bulanan) == 0:
    st.error(
        "Hasil forecast tidak tersedia."
    )
    st.stop()


nama_bulan_indo = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

bulan_nama = [
    nama_bulan_indo.get(b.month, b.strftime("%B"))
    for b in hasil_bulanan["Bulan"]
]


# ============================================================
# KPI CARDS
# ============================================================

num_kpi = len(hasil_bulanan)
cols = st.columns(1 + num_kpi)


with cols[0]:

    tampilkan_html(
        f"""
        <div class="metric-card green">
            <div class="metric-label">
                Data Terakhir
            </div>
            <div class="metric-value">
                {df_ts['Tanggal'].max().strftime('%d %b %Y')}
            </div>
            <div class="metric-note">
                Dasar forecast
            </div>
        </div>
        """
    )


for i in range(num_kpi):

    rf_value = (
        hasil_bulanan[
            "Prediksi_RF"
        ]
        .iloc[i]
    )

    svr_value = (
        hasil_bulanan[
            "Prediksi_SVR"
        ]
        .iloc[i]
    )

    rata = (
        rf_value
        + svr_value
    ) / 2

    warna = [
        "blue",
        "yellow",
        "red"
    ][i % 3]

    with cols[i + 1]:

        tampilkan_html(
            f"""
            <div class="metric-card {warna}">
                <div class="metric-label">
                    {bulan_nama[i]} 2026
                </div>
                <div class="metric-value">
                    Rp {rata:,.0f}
                </div>
                <div class="metric-note">
                    Rata-rata RF & SVR
                </div>
            </div>
            """
        )


st.write("")


# ============================================================
# GRAFIK UTAMA
# ============================================================

with st.container(
    border=True
):

    st.subheader(
        f"Forecast Harian — {periode_aktif}"
    )


    histori = (
        df_ts
        .tail(90)
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=histori["Tanggal"],
            y=histori["Harga"],

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
            x=hasil_harian["Tanggal"],
            y=hasil_harian["Prediksi_RF"],

            mode="lines",

            name="Random Forest",

            line=dict(
                color="#F5B700",
                width=3
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=hasil_harian["Tanggal"],
            y=hasil_harian["Prediksi_SVR"],

            mode="lines",

            name="SVR",

            line=dict(
                color="#1877F2",
                width=3
            )
        )
    )


    fig.update_layout(
        height=500,

        hovermode="x unified",

        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

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
        showgrid=False
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


# ============================================================
# GRAFIK BULANAN
# ============================================================

st.write("")


left, right = st.columns(
    [1.3, 1]
)


with left:

    with st.container(
        border=True
    ):

        st.subheader(
            "Prediksi Rata-Rata Bulanan"
        )


        fig_month = go.Figure()


        fig_month.add_trace(
            go.Bar(
                x=bulan_nama,

                y=hasil_bulanan[
                    "Prediksi_RF"
                ],

                name="Random Forest",

                marker_color="#F5B700"
            )
        )


        fig_month.add_trace(
            go.Bar(
                x=bulan_nama,

                y=hasil_bulanan[
                    "Prediksi_SVR"
                ],

                name="SVR",

                marker_color="#1877F2"
            )
        )


        fig_month.update_layout(
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


        fig_month.update_yaxes(
            gridcolor="#EFE9DD",

            tickprefix="Rp ",

            zeroline=False
        )


        st.plotly_chart(
            fig_month,
            use_container_width=True
        )


with right:

    with st.container(
        border=True
    ):

        st.subheader(
            "Perbedaan Prediksi Model"
        )


        selisih = (
            hasil_bulanan[
                "Prediksi_RF"
            ]

            -

            hasil_bulanan[
                "Prediksi_SVR"
            ]
        ).abs()


        fig_diff = go.Figure()


        fig_diff.add_trace(
            go.Bar(
                x=bulan_nama,

                y=selisih,

                marker_color="#2CA66F",

                text=[
                    f"Rp {x:,.0f}"
                    for x in selisih
                ],

                textposition="outside"
            )
        )


        fig_diff.update_layout(
            height=390,

            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",

            font=dict(
                color="#151923"
            ),

            showlegend=False
        )


        fig_diff.update_yaxes(
            gridcolor="#EFE9DD",

            tickprefix="Rp ",

            zeroline=False
        )


        st.plotly_chart(
            fig_diff,
            use_container_width=True
        )


# ============================================================
# DETAIL PER BULAN
# ============================================================

st.write("")


with st.container(
    border=True
):

    st.subheader(
        "Detail Prediksi Bulanan"
    )


    list_column = st.columns(len(hasil_bulanan))


    for i, col in enumerate(
        list_column
    ):

        rf_value = (
            hasil_bulanan[
                "Prediksi_RF"
            ]
            .iloc[i]
        )


        svr_value = (
            hasil_bulanan[
                "Prediksi_SVR"
            ]
            .iloc[i]
        )


        with col:

            st.markdown(
                f"### {bulan_nama[i]} 2026"
            )


            st.metric(
                "Random Forest",
                f"Rp {rf_value:,.0f}"
            )


            st.metric(
                "SVR",
                f"Rp {svr_value:,.0f}"
            )


# ============================================================
# TABEL
# ============================================================

st.write("")


with st.container(
    border=True
):

    st.subheader(
        f"Tabel Prediksi Bulanan — {periode_aktif}"
    )


    tabel = hasil_bulanan.copy()


    tabel["Bulan"] = (
        bulan_nama
    )


    tabel["Prediksi_RF"] = (
        tabel[
            "Prediksi_RF"
        ]
        .round(2)
    )


    tabel["Prediksi_SVR"] = (
        tabel[
            "Prediksi_SVR"
        ]
        .round(2)
    )


    tabel = tabel[
        [
            "Bulan",
            "Prediksi_RF",
            "Prediksi_SVR"
        ]
    ]


    st.dataframe(
        tabel,

        use_container_width=True,

        hide_index=True,

        column_config={

            "Bulan":
                st.column_config.TextColumn(
                    "Bulan"
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


    st.download_button(
        label=
            f"Download Prediksi ({periode_aktif})",

        data=
            tabel
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            ),

        file_name=
            f"prediksi_beras_{periode_aktif.lower().replace(' ', '_')}.csv",

        mime=
            "text/csv"
    )


st.caption(
    "Prediksi bulanan diperoleh dari rata-rata "
    "hasil forecast harian pada masing-masing bulan."
)
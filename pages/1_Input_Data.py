import streamlit as st
from pathlib import Path

# ============================================================
# CONFIG & CSS (MUST BE FIRST)
# ============================================================

st.set_page_config(
    page_title="Input Data",
    page_icon="📂",
    layout="wide"
)

css = Path("assets/style.css")
if css.exists():
    with open(css, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

from utils.data_processing import (
    load_data,
    prepare_data,
    create_features
)



st.title("Input Data")
st.caption("Upload data harga Beras Kualitas Medium I")


uploaded_file = st.file_uploader(
    "Pilih file Excel",
    type=["xlsx", "xls"]
)


if uploaded_file is not None:

    try:
        df_raw = load_data(uploaded_file)

        st.subheader("Data Awal")

        st.dataframe(
            df_raw,
            use_container_width=True
        )

        if st.button(
            "Proses Data",
            type="primary"
        ):

            with st.spinner("Sedang memproses data..."):

                df_ts = prepare_data(df_raw)
                df_feature = create_features(df_ts)

                st.session_state["df_raw"] = df_raw
                st.session_state["df_ts"] = df_ts
                st.session_state["df_feature"] = df_feature

            st.success("Data berhasil diproses.")


            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Data Mentah",
                len(df_raw)
            )

            c2.metric(
                "Data Time Series",
                len(df_ts)
            )

            c3.metric(
                "Data Siap Model",
                len(df_feature)
            )

            c4.metric(
                "Missing Value",
                df_feature.isna().sum().sum()
            )


            st.divider()

            st.subheader("Ringkasan Data")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Tanggal Awal",
                df_feature["Tanggal"].min().strftime("%d %b %Y")
            )

            c2.metric(
                "Tanggal Akhir",
                df_feature["Tanggal"].max().strftime("%d %b %Y")
            )

            c3.metric(
                "Harga Terakhir",
                f"Rp {df_feature['Harga'].iloc[-1]:,.0f}"
            )


            st.divider()

            st.subheader("Data Time Series")

            st.dataframe(
                df_ts.tail(20),
                use_container_width=True,
                hide_index=True
            )


            st.divider()

            st.subheader("Hasil Feature Engineering")

            st.dataframe(
                df_feature.tail(20),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(
            f"Data tidak dapat diproses: {e}"
        )

else:
    st.info(
        "Upload file Excel untuk mulai memproses data."
    )
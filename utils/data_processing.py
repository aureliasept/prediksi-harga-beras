import numpy as np
import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

def load_data(source):
    return pd.read_excel(source)


# ============================================================
# PREPROCESSING DATA
# ============================================================

def prepare_data(
    df,
    komoditas="Beras Kualitas Medium I"
):

    # Mengambil data komoditas
    data = df[
        df["Komoditas (Rp)"] == komoditas
    ].copy()


    if data.empty:

        raise ValueError(
            f"Komoditas '{komoditas}' tidak ditemukan."
        )


    # Mengubah data menjadi format time series
    data = (
        data
        .iloc[:, 2:]
        .T
        .reset_index()
    )


    data.columns = [
        "Tanggal",
        "Harga"
    ]


    # ========================================================
    # FORMAT TANGGAL
    # ========================================================

    data["Tanggal"] = pd.to_datetime(
        data["Tanggal"]
        .astype(str)
        .str.strip(),

        format="%d/ %m/ %Y",

        errors="coerce"
    )


    # ========================================================
    # MEMBERSIHKAN HARGA
    # ========================================================

    data["Harga"] = (
        data["Harga"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.strip()
        .replace(
            "-",
            np.nan
        )
    )


    data["Harga"] = pd.to_numeric(
        data["Harga"],
        errors="coerce"
    )


    # ========================================================
    # MENGURUTKAN BERDASARKAN WAKTU
    # ========================================================

    data = (
        data
        .sort_values("Tanggal")
        .reset_index(drop=True)
    )


    # ========================================================
    # INTERPOLASI LINEAR
    # ========================================================

    data["Harga"] = (
        data["Harga"]
        .interpolate(
            method="linear",
            limit_area="inside"
        )
    )


    # ========================================================
    # MENGHAPUS DATA YANG MASIH KOSONG
    # ========================================================

    data = (
        data
        .dropna(
            subset=[
                "Tanggal",
                "Harga"
            ]
        )
        .reset_index(drop=True)
    )


    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    df = data.copy()


    # Fitur lag
    df["lag_1"] = (
        df["Harga"]
        .shift(1)
    )


    df["lag_7"] = (
        df["Harga"]
        .shift(7)
    )


    df["lag_30"] = (
        df["Harga"]
        .shift(30)
    )


    df["lag_60"] = (
        df["Harga"]
        .shift(60)
    )


    # ========================================================
    # ROLLING MEAN
    # ========================================================

    harga_sebelumnya = (
        df["Harga"]
        .shift(1)
    )


    df["rolling_mean_7"] = (
        harga_sebelumnya
        .rolling(
            window=7
        )
        .mean()
    )


    df["rolling_mean_30"] = (
        harga_sebelumnya
        .rolling(
            window=30
        )
        .mean()
    )


    # Menghapus data yang belum memiliki histori lengkap
    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )


    return df


# ============================================================
# DAFTAR FITUR
# ============================================================

def get_features():

    return [
        "lag_1",
        "lag_7",
        "lag_30",
        "lag_60",
        "rolling_mean_7",
        "rolling_mean_30"
    ]
import joblib
import numpy as np
import pandas as pd


def load_model(path):
    return joblib.load(path)


def predict_models(data, bundle_rf, bundle_svr):

    fitur = bundle_rf["fitur"]

    X = data[fitur]

    hasil = data[["Tanggal", "Harga"]].copy()

    hasil.columns = [
        "Tanggal",
        "Aktual"
    ]

    hasil["Prediksi_RF"] = (
        bundle_rf["model"]
        .predict(X)
    )

    hasil["Prediksi_SVR"] = (
        bundle_svr["model"]
        .predict(X)
    )

    return hasil


def forecast_model(
    model,
    histori,
    tanggal_mulai,
    tanggal_akhir,
    fitur
):

    history = list(
        histori.astype(float)
    )

    hasil = []

    tanggal_prediksi = pd.date_range(
        start=tanggal_mulai,
        end=tanggal_akhir,
        freq="B"
    )


    for tanggal in tanggal_prediksi:

        row = {

            "lag_1":
                history[-1],

            "lag_7":
                history[-7],

            "lag_30":
                history[-30],

            "lag_60":
                history[-60],

            "rolling_mean_7":
                np.mean(
                    history[-7:]
                ),

            "rolling_mean_30":
                np.mean(
                    history[-30:]
                )
        }


        X = pd.DataFrame(
            [row]
        )[fitur]


        prediksi = float(
            model.predict(X)[0]
        )


        hasil.append({

            "Tanggal":
                tanggal,

            "Prediksi":
                prediksi
        })


        history.append(
            prediksi
        )


    return pd.DataFrame(
        hasil
    )


def forecast_3_months(
    data,
    bundle_rf,
    bundle_svr
):

    histori = data["Harga"].copy()


    tanggal_mulai = pd.Timestamp(
        "2026-01-01"
    )

    tanggal_akhir = pd.Timestamp(
        "2026-03-31"
    )


    pred_rf = forecast_model(

        bundle_rf["model"],

        histori,

        tanggal_mulai,

        tanggal_akhir,

        bundle_rf["fitur"]
    )


    pred_svr = forecast_model(

        bundle_svr["model"],

        histori,

        tanggal_mulai,

        tanggal_akhir,

        bundle_svr["fitur"]
    )


    hasil_harian = pd.DataFrame({

        "Tanggal":
            pred_rf["Tanggal"],

        "Prediksi_RF":
            pred_rf["Prediksi"],

        "Prediksi_SVR":
            pred_svr["Prediksi"]
    })


    hasil_harian["Bulan"] = (

        hasil_harian["Tanggal"]
        .dt.to_period("M")
    )


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


    return (
        hasil_harian,
        hasil_bulanan
    )
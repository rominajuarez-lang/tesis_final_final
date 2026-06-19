@st.cache_data
def evaluar_todos(ventas, forecast_empresa, maestro):
    resumen = []
    detalles = {}
    pronosticos_2026 = []

    productos = sorted(
        set(ventas["product_id"].astype(str)) &
        set(forecast_empresa["product_id"].astype(str))
    )

    for producto in productos:
        r = evaluar_sku(
            producto,
            ventas,
            forecast_empresa,
            maestro
        )

        if r is None:
            continue

        detalles[producto] = r
        pronosticos_2026.append(r["forecast_2026"])

        resumen.append({
            "product_id": producto,
            "mejor_modelo": r["mejor_modelo"],
            "wMAPE empresa (%)": r["wmape_empresa"],
            "wMAPE propuesto (%)": r["wmape_propuesto"],
            "Bias empresa (%)": r["bias_empresa"],
            "Bias propuesto (%)": r["bias_propuesto"],
            "Error empresa S/": r["error_empresa"],
            "Error propuesto S/": r["error_propuesto"],
            "Ahorro potencial S/": r["ahorro"]
        })

    resumen = pd.DataFrame(resumen)

    if len(pronosticos_2026) > 0:
        df_forecast_2026 = pd.concat(
            pronosticos_2026,
            ignore_index=True
        )
    else:
        df_forecast_2026 = pd.DataFrame()

    return resumen, detalles, df_forecast_2026

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecast Intelligence 2025-2026",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Forecast Intelligence Framework")
st.caption("Comparación 2025: Forecast Empresa vs Forecast Propuesto | Pronóstico 2026 por SKU")

METODOS = ["Regresión lineal", "SES", "ARIMA"]


# =========================================================
# CARGA Y LIMPIEZA
# =========================================================
def normalizar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "sku": "product_id",
        "producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "cod_producto_v5": "product_id",
        "product_id": "product_id",

        "descripcion": "description",
        "descripción": "description",
        "description": "description",

        "fecha": "date",
        "mes": "date",
        "date": "date",

        "ventas": "demand_real",
        "venta": "demand_real",
        "und final": "demand_real",
        "demanda": "demand_real",
        "demand_real": "demand_real",

        "forecast_comercial": "forecast_company",
        "forecast empresa": "forecast_company",
        "forecast_company": "forecast_company",

        "costo_unitario": "unit_cost",
        "costo unitario": "unit_cost",
        "unit_cost": "unit_cost",
        "precio bp 2025 sin igv": "unit_cost",
    }

    return df.rename(columns={c: alias.get(c, c) for c in df.columns})


def leer_base(uploaded_file):
    maestro = pd.read_excel(uploaded_file, sheet_name="Maestro_SKU")
    ventas = pd.read_excel(uploaded_file, sheet_name="Ventas_Historicas")
    forecast = pd.read_excel(uploaded_file, sheet_name="Forecast_Comercial")

    maestro = normalizar_columnas(maestro)
    ventas = normalizar_columnas(ventas)
    forecast = normalizar_columnas(forecast)

    maestro["product_id"] = maestro["product_id"].astype(str)
    maestro["unit_cost"] = pd.to_numeric(maestro["unit_cost"], errors="coerce").fillna(0)

    if "description" not in maestro.columns:
        maestro["description"] = maestro["product_id"]

    ventas["date"] = pd.to_datetime(ventas["date"], errors="coerce")
    ventas["product_id"] = ventas["product_id"].astype(str)
    ventas["demand_real"] = pd.to_numeric(ventas["demand_real"], errors="coerce").fillna(0)

    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast["product_id"] = forecast["product_id"].astype(str)
    forecast["forecast_company"] = pd.to_numeric(forecast["forecast_company"], errors="coerce").fillna(0)

    ventas = ventas.dropna(subset=["date"])
    forecast = forecast.dropna(subset=["date"])

    ventas["date"] = ventas["date"].dt.to_period("M").dt.to_timestamp()
    forecast["date"] = forecast["date"].dt.to_period("M").dt.to_timestamp()

    ventas_mensual = (
        ventas.groupby(["product_id", "date"], as_index=False)["demand_real"]
        .sum()
        .sort_values(["product_id", "date"])
    )

    forecast_mensual = (
        forecast.groupby(["product_id", "date"], as_index=False)["forecast_company"]
        .sum()
        .sort_values(["product_id", "date"])
    )

    return maestro, ventas_mensual, forecast_mensual


# =========================================================
# MODELOS
# =========================================================
def pred_regresion(train, pasos):
    y = np.array(train, dtype=float)
    x = np.arange(len(y)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, y)

    x_future = np.arange(len(y), len(y) + pasos).reshape(-1, 1)
    return np.maximum(0, modelo.predict(x_future))


def pred_ses(train, pasos):
    y = np.array(train, dtype=float)

    if len(y) < 3:
        return np.repeat(np.mean(y), pasos)

    try:
        modelo = SimpleExpSmoothing(y, initialization_method="estimated")
        ajuste = modelo.fit(optimized=True)
        return np.maximum(0, np.asarray(ajuste.forecast(pasos)))
    except Exception:
        return np.repeat(np.mean(y), pasos)


def pred_arima(train, pasos):
    y = np.array(train, dtype=float)

    if len(y) < 8:
        return pred_ses(y, pasos)

    try:
        modelo = ARIMA(y, order=(1, 1, 1))
        ajuste = modelo.fit()
        return np.maximum(0, np.asarray(ajuste.forecast(pasos)))
    except Exception:
        return pred_ses(y, pasos)


def generar_prediccion(train, metodo, pasos):
    if metodo == "Regresión lineal":
        return pred_regresion(train, pasos)
    if metodo == "SES":
        return pred_ses(train, pasos)
    if metodo == "ARIMA":
        return pred_arima(train, pasos)
    return pred_ses(train, pasos)


# =========================================================
# MÉTRICAS
# =========================================================
def wmape(real, pred):
    real = np.array(real, dtype=float)
    pred = np.array(pred, dtype=float)
    return np.sum(np.abs(real - pred)) / max(np.sum(np.abs(real)), 1) * 100


def bias_pct(real, pred):
    real = np.array(real, dtype=float)
    pred = np.array(pred, dtype=float)
    return np.sum(pred - real) / max(np.sum(real), 1) * 100


def errores_valorizados(real, pred, costo):
    real = np.array(real, dtype=float)
    pred = np.array(pred, dtype=float)

    exceso = np.maximum(pred - real, 0)
    faltante = np.maximum(real - pred, 0)
    error_abs = np.abs(pred - real)

    return exceso.sum(), faltante.sum(), np.sum(error_abs * costo)


# =========================================================
# EVALUACIÓN 2025 Y FORECAST 2026
# =========================================================
def evaluar_sku(product_id, ventas, forecast_empresa, maestro):
    ventas_sku = ventas[ventas["product_id"] == product_id].copy()
    ventas_sku = ventas_sku.sort_values("date")

    train = ventas_sku[ventas_sku["date"].dt.year < 2025]
    real_2025 = ventas_sku[ventas_sku["date"].dt.year == 2025]

    if len(train) < 6 or len(real_2025) == 0:
        return None

    fc_empresa_2025 = forecast_empresa[
        (forecast_empresa["product_id"] == product_id) &
        (forecast_empresa["date"].dt.year == 2025)
    ].copy()

    df_2025 = real_2025.merge(
        fc_empresa_2025,
        on=["product_id", "date"],
        how="inner"
    )

    if df_2025.empty:
        return None

    train_y = train["demand_real"].to_numpy(dtype=float)
    y_real = df_2025["demand_real"].to_numpy(dtype=float)
    y_empresa = df_2025["forecast_company"].to_numpy(dtype=float)
    fechas_2025 = df_2025["date"]

    costo = maestro.loc[maestro["product_id"] == product_id, "unit_cost"]
    costo = float(costo.iloc[0]) if len(costo) > 0 else 0

    resultados = []

    exceso_emp, faltante_emp, error_emp_soles = errores_valorizados(y_real, y_empresa, costo)

    resultados.append({
        "Modelo": "Forecast empresa",
        "wMAPE (%)": wmape(y_real, y_empresa),
        "Bias (%)": bias_pct(y_real, y_empresa),
        "Exceso und": exceso_emp,
        "Faltante und": faltante_emp,
        "Error valorizado S/": error_emp_soles,
    })

    predicciones_2025 = {}

    for metodo in METODOS:
        pred = generar_prediccion(train_y, metodo, len(y_real))
        predicciones_2025[metodo] = pred

        exceso, faltante, error_soles = errores_valorizados(y_real, pred, costo)

        resultados.append({
            "Modelo": metodo,
            "wMAPE (%)": wmape(y_real, pred),
            "Bias (%)": bias_pct(y_real, pred),
            "Exceso und": exceso,
            "Faltante und": faltante,
            "Error valorizado S/": error_soles,
        })

    tabla = pd.DataFrame(resultados)

    propuestos = tabla[tabla["Modelo"] != "Forecast empresa"].copy()
    mejor = propuestos.sort_values(["wMAPE (%)", "Bias (%)"]).iloc[0]
    mejor_modelo = mejor["Modelo"]

    empresa = tabla[tabla["Modelo"] == "Forecast empresa"].iloc[0]

    # Forecast 2026 con el mejor modelo, entrenando hasta 2025
    full_train = ventas_sku[ventas_sku["date"].dt.year <= 2025]["demand_real"].to_numpy(dtype=float)
    fechas_2026 = pd.date_range(start="2026-01-01", periods=12, freq="MS")
    pred_2026 = generar_prediccion(full_train, mejor_modelo, 12)

    forecast_2026 = pd.DataFrame({
        "product_id": product_id,
        "date": fechas_2026,
        "forecast_2026": pred_2026,
        "best_model": mejor_modelo,
    })

    return {
        "product_id": product_id,
        "fechas_2025": fechas_2025,
        "real_2025": y_real,
        "empresa_2025": y_empresa,
        "predicciones_2025": predicciones_2025,
        "tabla": tabla,
        "mejor_modelo": mejor_modelo,
        "error_empresa": empresa["Error valorizado S/"],
        "error_propuesto": mejor["Error valorizado S/"],
        "ahorro": empresa["Error valorizado S/"] - mejor["Error valorizado S/"],
        "wmape_empresa": empresa["wMAPE (%)"],
        "wmape_propuesto": mejor["wMAPE (%)"],
        "bias_empresa": empresa["Bias (%)"],
        "bias_propuesto": mejor["Bias (%)"],
        "forecast_2026": forecast_2026,
    }


@st.cache_data
def evaluar_todos(ventas, forecast_empresa, maestro):
    resumen = []
    detalles = {}
    pronosticos_2026 = []

    productos = sorted(set(ventas["product_id"]) & set(forecast_empresa["product_id"]))

    for p in productos:
        r = evaluar_sku(p, ventas, forecast_empresa, maestro)

        if r is None:
            continue

        detalles[p] = r
        pronosticos_2026.append(r["forecast_2026"])

        resumen.append({
            "product_id": p,
            "mejor_modelo": r["mejor_modelo"],
            "wMAPE empresa (%)": r["wmape_empresa"],
            "wMAPE propuesto (%)": r["wmape_propuesto"],
            "Bias empresa (%)": r["bias_empresa"],
            "Bias propuesto (%)": r["bias_propuesto"],
            "Error empresa S/": r["error_empresa"],
            "Error propuesto S/": r["error_propuesto"],
            "Ahorro potencial S/": r["ahorro"],
        })

    df_resumen = pd.DataFrame(resumen)

    if pronosticos_2026:
        df_2026 = pd.concat(pronosticos_2026, ignore_index=True)
    else:
        df_2026 = pd.DataFrame()

    return df_resumen, detalles, df_2026


# =========================================================
# VISUALIZACIONES
# =========================================================
def grafico_modelo(fechas, real, empresa, propuesto, modelo):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fechas, y=real, mode="lines+markers", name="Ventas reales 2025"))
    fig.add_trace(go.Scatter(x=fechas, y=empresa, mode="lines+markers", name="Forecast empresa 2025"))
    fig.add_trace(go.Scatter(x=fechas, y=propuesto, mode="lines+markers", name=f"Forecast propuesto - {modelo}"))

    fig.update_layout(
        title=f"Comparación 2025: Empresa vs {modelo}",
        xaxis_title="Mes",
        yaxis_title="Unidades",
        hovermode="x unified",
        height=420,
    )

    return fig


def grafico_2026(df_2026_sku):
    fig = px.line(
        df_2026_sku,
        x="date",
        y="forecast_2026",
        markers=True,
        title="Pronóstico 2026 con mejor modelo seleccionado"
    )
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Unidades pronosticadas",
        height=420,
    )
    return fig


# =========================================================
# APP
# =========================================================
st.sidebar.header("Carga de datos")
archivo = st.sidebar.file_uploader("Sube tu Excel", type=["xlsx", "xls"])

if archivo is None:
    st.info("Sube un Excel con hojas: Maestro_SKU, Ventas_Historicas y Forecast_Comercial.")
    st.stop()

try:
    maestro, ventas, forecast_empresa = leer_base(archivo)
except Exception as e:
    st.error(f"Error leyendo la base: {e}")
    st.stop()

resumen, detalles, df_forecast_2026 = evaluar_todos(ventas, forecast_empresa, maestro)

if resumen.empty:
    st.warning("No se pudo evaluar ningún SKU. Revisa que existan ventas reales 2025 y forecast comercial 2025 por SKU.")
    st.stop()

error_empresa_total = resumen["Error empresa S/"].sum()
error_propuesto_total = resumen["Error propuesto S/"].sum()
ahorro_total = resumen["Ahorro potencial S/"].sum()

reduccion_error_valorizado = (
    (error_empresa_total - error_propuesto_total) / error_empresa_total * 100
    if error_empresa_total > 0 else 0
)

modelo_mas_usado = resumen["mejor_modelo"].mode().iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs evaluados", f"{len(resumen):,.0f}")
c2.metric("Reducción error valorizado", f"{reduccion_error_valorizado:.1f}%")
c3.metric("Ahorro potencial total", f"S/ {ahorro_total:,.2f}")
c4.metric("Modelo más usado", modelo_mas_usado)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Evaluación 2025",
    "🔮 Pronóstico 2026",
    "📋 Tablas descargables",
])

with tab1:
    st.subheader("Comparación global por SKU")

    st.dataframe(
        resumen.sort_values("Ahorro potencial S/", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    fig_modelos = px.pie(
        resumen,
        names="mejor_modelo",
        title="Distribución de mejores modelos por SKU"
    )
    st.plotly_chart(fig_modelos, use_container_width=True)

    st.divider()

    producto_sel = st.selectbox("Seleccionar SKU", sorted(detalles.keys()))
    r = detalles[producto_sel]

    st.success(f"Mejor modelo para {producto_sel}: {r['mejor_modelo']}")

    for metodo in METODOS:
        col_graf, col_tabla = st.columns([2, 1])

        with col_graf:
            st.plotly_chart(
                grafico_modelo(
                    r["fechas_2025"],
                    r["real_2025"],
                    r["empresa_2025"],
                    r["predicciones_2025"][metodo],
                    metodo
                ),
                use_container_width=True
            )

        with col_tabla:
            tabla_modelo = r["tabla"][r["tabla"]["Modelo"].isin(["Forecast empresa", metodo])].copy()
            st.dataframe(
                tabla_modelo,
                use_container_width=True,
                hide_index=True
            )

with tab2:
    st.subheader("Pronóstico mensual 2026")

    producto_2026 = st.selectbox("Seleccionar SKU para 2026", sorted(df_forecast_2026["product_id"].unique()))

    df_sku_2026 = df_forecast_2026[df_forecast_2026["product_id"] == producto_2026].copy()

    st.plotly_chart(grafico_2026(df_sku_2026), use_container_width=True)

    st.dataframe(df_sku_2026, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Resumen económico 2025")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader("Forecast propuesto 2026")
    st.dataframe(df_forecast_2026, use_container_width=True, hide_index=True)

    csv_resumen = resumen.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar resumen 2025",
        data=csv_resumen,
        file_name="resumen_evaluacion_2025.csv",
        mime="text/csv",
    )

    csv_2026 = df_forecast_2026.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar forecast 2026",
        data=csv_2026,
        file_name="forecast_propuesto_2026.csv",
        mime="text/csv",
    )

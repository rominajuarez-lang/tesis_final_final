# Kroger Inventory App - Pronóstico mensual y mejor método por producto

Esta aplicación en Streamlit permite:

- Cargar datos de demanda histórica y convertirlos a nivel mensual.
- Comparar automáticamente varios métodos de pronóstico por producto.
- Elegir el mejor método según menor wMAPE.
- Simular inventario mensual.
- Optimizar el stock de seguridad.

## Métodos incluidos

- Naive
- Promedio móvil
- SES
- Regresión lineal
- ARIMA
- SARIMA
- Holt-Winters
- Croston

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

En GitHub Codespaces:

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```


## Corrección v2

Se corrigió el error de columnas duplicadas en la tabla de mejores métodos.


## Corrección v3

Se agregó pronóstico futuro mensual. Ahora la app permite seleccionar una fecha final, por defecto diciembre 2026, y proyecta la demanda futura por producto usando el mejor método seleccionado.

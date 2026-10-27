# Análisis de sentimientos en YouTube sobre salud mental

Pipeline en Python que extrae y analiza los comentarios de videos de YouTube sobre salud mental (ansiedad, depresión, TEPT, TOC y trastorno bipolar) para identificar expresiones de malestar y medir cómo responde la audiencia a cada tema.

## Resultados principales

| Indicador | Resultado |
|---|---|
| Videos analizados | 30 (11.7 millones de vistas en conjunto) |
| Comentarios analizados | 2,320 |
| Comentarios negativos detectados: modelo base → léxico propio | 229 → 341 (**+49%**) |
| Comentarios clasificados como neutros: modelo base → léxico propio | 83% → 75% |
| Proporción de comentarios negativos: videos de ansiedad vs. depresión | **23% vs. 14%** |

Las palabras más frecuentes en los comentarios negativos fueron *ansiedad, miedo, depresión, dolor* y *pánico*. Esto permite ubicar en qué temas se concentra el malestar de quienes podrían padecer estos trastornos y priorizar la atención.

## Qué hace

1. **Extracción:** conecta con la YouTube Data API v3 mediante OAuth 2.0 y descarga métricas del video y del canal, además de hasta 100 comentarios por video. Reintenta automáticamente con espera exponencial ante errores temporales (403, 429 y 5xx).
2. **Limpieza y enriquecimiento:** normaliza el texto y agrega variables como longitud, número de palabras, preguntas, enlaces, fecha, hora y comentarios duplicados.
3. **Análisis de sentimiento:** clasifica cada comentario con un modelo TF-IDF + regresión logística (scikit-learn). `analisis_sentimientos.py` recalcula los resultados con un léxico en español propio, con negadores e intensificadores, que corrige el sesgo de TextBlob hacia "neutro".
4. **Indicadores por video:** tasa de engagement, polaridad promedio y ponderada por likes, índice de satisfacción y palabras clave.
5. **Salidas automáticas:** base de datos SQLite, Excel con seis hojas (comentarios, resumen, comparativo, palabras clave, palabras por sentimiento y tendencia), CSV, gráfica y un reporte ejecutivo en PDF.

## Reporte

El reporte ejecutivo completo, con una página por video y un comparativo final, está en [`docs/reporte_mineria.pdf`](docs/reporte_mineria.pdf). La gráfica de distribución de sentimiento está en [`docs/grafica_sentimientos.png`](docs/grafica_sentimientos.png).

## Tecnologías

Python · pandas · scikit-learn · matplotlib · TextBlob · YouTube Data API v3 · OAuth 2.0 · SQLite · openpyxl

## Cómo ejecutarlo

1. Instala las dependencias:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Crea credenciales de YouTube: en Google Cloud Console habilita la **YouTube Data API v3**, crea un ID de cliente OAuth de tipo "App de escritorio" y descarga el JSON como `client-secret.json` en esta carpeta. El archivo está excluido del repositorio en `.gitignore`.
3. Ejecuta el análisis:
   ```bash
   # Videos por defecto, o tus propios IDs
   python demo.py --video-ids VIDEO_ID_1 VIDEO_ID_2 --max-comments 100

   # Recalcular sentimientos con el léxico en español
   python analisis_sentimientos.py --input mineria_buap_final.xlsx
   ```

## Estructura

| Archivo | Función |
|---|---|
| `demo.py` | Extracción, limpieza, modelo de sentimiento, indicadores y reportes |
| `analisis_sentimientos.py` | Léxico en español y recálculo de resultados |
| `Google.py` | Autenticación OAuth 2.0 con las APIs de Google |
| `docs/` | Reporte en PDF y gráfica de resultados |

## Privacidad

Los datos crudos (comentarios y nombres de sus autores) no se publican en este repositorio. Solo se incluyen el código y resultados agregados.

## Autor

**Alexis Flores Díaz** · [GitHub](https://github.com/AlexisDiaz72)

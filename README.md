# DataSight - Dashboard de Análisis de CSV con Streamlit y Gemini IA

Este es un dashboard interactivo en Python que permite cargar cualquier archivo `.csv`, explorar los datos, aplicar filtros dinámicos (según el tipo de dato de cada columna), generar gráficos interactivos en tiempo real y **realizar análisis automáticos con Inteligencia Artificial utilizando Gemini LLM**.

## Características

- 📁 **Carga de CSV:** Soporte para cargar cualquier archivo CSV mediante arrastrar y soltar.
- ⚡ **Datos de Demostración:** Opción de simular un dataset de ventas de ejemplo si no dispones de un archivo CSV de inmediato.
- 🔍 **Filtros Dinámicos Automáticos:**
  - Control de fechas (calendario de rango) para columnas temporales.
  - Sliders de rango para columnas numéricas.
  - Multiselección para variables categóricas o de texto.
  - Búsqueda por coincidencia de texto para columnas de alta cardinalidad.
- 📈 **Gráficos Dinámicos:** Generador de gráficos (Barras, Líneas, Dispersión, Histograma, Boxplot) utilizando Plotly Express, actualizándose según el filtro aplicado.
- <img width="1581" height="769" alt="image" src="https://github.com/user-attachments/assets/f8482758-99da-40fb-9d7d-56ab6722d5c8" />
  <img width="1523" height="556" alt="image" src="https://github.com/user-attachments/assets/6da138bc-f72b-4eeb-9267-2ba51f5934cf" />
  <img width="1919" height="920" alt="image" src="https://github.com/user-attachments/assets/23535ba9-44d9-41ab-9ea4-8142b4c1f810" />

- 🧠 **Análisis de Datos con IA (Gemini):**
  - Generación de resúmenes de datos, patrones de tendencias, alertas de anomalías y recomendaciones estratégicas en tiempo real.
  - Permite ingresar preguntas o instrucciones específicas al modelo de lenguaje (ej. *"¿Por qué bajaron las ventas en marzo?"*).
  - Admite los modelos `gemini-1.5-flash` y `gemini-1.5-pro`.
  - Descarga de reportes analíticos generados por la IA en formato Markdown.
- 📥 **Exportación:** Permite descargar el subconjunto de datos filtrados como un nuevo archivo `.csv`.
- 🎨 **Diseño Premium:** Estética limpia y profesional con soporte automático para modo claro y oscuro, tipografías modernas y tarjetas de métricas tipo KPI.

---

## Configuración de Gemini API Key

Para utilizar las funciones de Inteligencia Artificial necesitas una clave de API de Gemini. Sigue estos pasos para configurarla:

1. **Obtener tu clave gratuita:**
   - Entra a [Google AI Studio](https://aistudio.google.com/).
   - Haz clic en **Get API Key** y genera una nueva clave de API.

2. **Configurarla en el Dashboard:**
   - **Opción A (Recomendada):** Pégala directamente en el panel lateral de la aplicación en el campo **Gemini API Key**.
   - **Opción B (Automática):** Configúrala como una variable de entorno en tu terminal antes de arrancar la aplicación:
     ```bash
     export GEMINI_API_KEY="tu_clave_de_api_aqui"
     streamlit run app.py
     ```

---

## Requisitos de Instalación

Asegúrate de tener Python 3.8 o superior instalado en tu Mac. Luego, sigue estos pasos:

1. **Instalar dependencias:**
   Ejecuta el siguiente comando en la terminal para instalar las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación:**
   Inicia el servidor de Streamlit con:
   ```bash
   streamlit run app.py
   ```

3. **Acceder en el navegador:**
   La aplicación se abrirá automáticamente en tu navegador web en la dirección local (por defecto `http://localhost:8501`).

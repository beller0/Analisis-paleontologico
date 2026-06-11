from click import style
import streamlit as st
import pandas as pd
import plotly.express as px
import io
import google.generativeai as genai
import numpy as np
# Manejo de datos
import pandas as pd

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Gráficos interactivos
import plotly.express as px

# Interfaz de usuario de la App
import streamlit as st

# Configuración de la página del Dashboard
st.set_page_config(
    page_title="DataSight | Dashboard de Análisis de CSV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para un look moderno y premium
def load_css(path: str = 'style.css'):
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css()
# Helper function to load data
@st.cache_data

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        return None

# Helper function to generate mock data for demo
def get_mock_data():
    try:
        url = f"https://paleobiodb.org/data1.2/occs/list.csv?base_name=Carcharhinidae,Tyrannosauridae,Elephantidae&show=coords,classext&vocab=pbdb"
        df = pd.read_csv(
        url, 
        comment='#',          # IGNORA cualquier línea que empiece con # (metadatos de la API)
        on_bad_lines='skip',  # Si una fila sigue rota, se la salta en lugar de romper el programa
        quotechar='"'  )       # Agrupa correctamente los textos que llevan comas dentro de comillas
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        
    return pd.DataFrame()
def get_grouped_fossil_data(df):
    variables_clustering = ['lat', 'lng', 'max_ma', 'min_ma']
    df_ia = df.dropna(subset=variables_clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ia[variables_clustering])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Guardamos el resultado del clúster directamente en el DataFrame original
    df_ia['cluster'] = kmeans.fit_predict(X_scaled)
    return df_ia

# Helper to get available models using session state to cache results
def get_available_models(api_key):
    default_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if not api_key:
        return default_models
    
    # Check if we already fetched models for this API key
    if 'models_cache' in st.session_state and st.session_state.get('cached_api_key') == api_key:
        return st.session_state['models_cache']
        
    try:
        genai.configure(api_key=api_key)
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                available.append(name)
        if available:
            # Sort to put key models first
            priority = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            sorted_models = [p for p in priority if p in available]
            other_models = [a for a in available if a not in priority]
            final_list = sorted_models + other_models
            
            # Cache the results
            st.session_state['models_cache'] = final_list
            st.session_state['cached_api_key'] = api_key
            return final_list
        return default_models
    except Exception:
        return default_models

def analyze_data_with_gemini(dataframe, user_query, api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Compile metadata and statistics
        num_rows, num_cols = dataframe.shape
        col_info = []
        for col in dataframe.columns:
            null_count = dataframe[col].isnull().sum()
            col_info.append(f"- {col}: {dataframe[col].dtype} ({null_count} valores nulos)")
        col_info_str = "\n".join(col_info)
        
        # Descriptive statistics summary
        stats_summary = ""
        numeric_df = dataframe.select_dtypes(include=['number'])
        if not numeric_df.empty:
            stats_summary = numeric_df.describe().to_string()
        else:
            stats_summary = "No hay columnas numéricas para resumir."
            
        # Sample of rows
        sample_rows = dataframe.head(10).to_string()
        
        # Prompt construction
        prompt = f"""
Eres un analista de datos y estratega de negocios experto. Te he proporcionado un dataset que ha sido filtrado interactivamente en mi dashboard.
Tu tarea es analizar esta información y proporcionar insights profundos, patrones relevantes y recomendaciones basadas en datos.

### Información del Dataset:
- **Filas Totales en la vista actual:** {num_rows}
- **Columnas Totales:** {num_cols}

### Estructura de Columnas y Tipos de Datos:
{col_info_str}

### Resumen Estadístico (Columnas Numéricas):
```
{stats_summary}
```

### Muestra de los Datos (Primeras 10 filas de la vista actual):
```
{sample_rows}
```

### Consulta/Instrucción específica del usuario:
"{user_query}"

Por favor, estructura tu respuesta en formato markdown elegante. Usa encabezados claros, viñetas para los puntos clave, y secciones separadas para:
1. **Resumen Ejecutivo:** Una síntesis rápida de lo que muestran los datos.
2. **Patrones y Hallazgos Clave:** Tendencias y relaciones interesantes entre variables.
3. **Anomalías o Puntos de Alerta:** Si existen outliers o inconsistencias.
4. **Recomendaciones de Negocio:** Sugerencias concretas basadas en los hallazgos.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ **Error al llamar a la API de Gemini:** {e}\n\nPor favor, verifica que tu API Key sea correcta y que tengas conexión a internet."

# --- HEADER DE LA APP ---
st.markdown("""
<div class="header-container">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800;">📊 DataSight</h1>
    <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9;">Sube tus archivos CSV, aplica filtros inteligentes y analiza tus datos con Inteligencia Artificial.</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURACIÓN E INPUTS ---
with st.sidebar:
    
    st.image("https://img.icons8.com/clouds/150/database.png", width=100)
    st.header("Entrada de Datos")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo .csv aquí:", 
        type=["csv"],
        help="Arrastra o selecciona un archivo CSV con formato estándar de coma o punto y coma."
    )
    
    use_demo = False
    if uploaded_file is None:
        st.info("💡 ¿No tienes un CSV a la mano? Prueba con nuestro dataset de demostración:")
        use_demo = st.checkbox("Usar datos de demostración")
        
    st.markdown("---")
    
    # SECCIÓN DE GEMINI CONFIG
    st.header("🧠 Configuración de Gemini")
    
    # Intentar obtener API Key de variables de entorno primero
    import os
    env_key = os.environ.get("GEMINI_API_KEY", "")
    
    api_key_input = st.text_input(
        "Gemini API Key:",
        value=env_key,
        type="password",
        placeholder="AIzaSy...",
        help="Ingresa tu clave de API de Google AI Studio. Si has seteado la variable GEMINI_API_KEY, se leerá automáticamente."
    )
    
    # Obtener modelos disponibles de forma dinámica
    available_models = get_available_models(api_key_input)
    
    gemini_model = st.selectbox(
        "Modelo a utilizar:",
        options=available_models,
        index=0,
        help="Modelos de Gemini disponibles en tu cuenta. Si tu API Key es correcta, cargará los modelos automáticamente."
    )
    
    st.markdown("---")
    st.subheader("⚙️ Filtros Activos")

# --- CONTROLADOR DE FLUJO DE DATOS ---
df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)
elif use_demo:
    df = get_mock_data()
    st.sidebar.success("Cargados datos de aprendizaje IA.")

if df is not None:
    # Copia del dataframe original para los filtros
    filtered_df = df.copy()
    
    # Intentar parsear columnas que parezcan fechas
    for col in filtered_df.columns:
        if filtered_df[col].dtype == 'object':
            try:
                filtered_df[col] = pd.to_datetime(filtered_df[col])
            except (ValueError, TypeError):
                pass

    # --- MOTOR DE FILTRADO DINÁMICO EN EL SIDEBAR ---
    with st.sidebar:
        # Ofrecer al usuario seleccionar qué columnas filtrar para no saturar la pantalla
        columns_to_filter = st.multiselect(
            "Selecciona columnas para filtrar:",
            options=list(df.columns),
            default=list(df.columns)[:3] if len(df.columns) > 3 else list(df.columns)
        )
        
        # Generar controles interactivos para las columnas seleccionadas
        for col in columns_to_filter:
            st.markdown(f"**Filtrar por: {col}**")
            
            # Caso 1: Columnas Temporales (Datetime)
            if pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                min_date = filtered_df[col].min().to_pydatetime()
                max_date = filtered_df[col].max().to_pydatetime()
                if min_date != max_date:
                    date_range = st.date_input(
                        f"Rango de Fechas para {col}",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_df = filtered_df[
                            (filtered_df[col].dt.date >= start_date) & 
                            (filtered_df[col].dt.date <= end_date)
                        ]
                else:
                    st.info(f"Solo una fecha única disponible: {min_date.strftime('%Y-%m-%d')}")
            
            # Caso 2: Columnas Numéricas
            elif pd.api.types.is_numeric_dtype(filtered_df[col]):
                unique_vals_count = filtered_df[col].nunique()
                if unique_vals_count <= 10:
                    options = sorted(list(filtered_df[col].unique()))
                    selected_vals = st.multiselect(f"Valores de {col}", options=options, default=options)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                else:
                    min_val = float(filtered_df[col].min())
                    max_val = float(filtered_df[col].max())
                    if min_val != max_val:
                        val_range = st.slider(
                            f"Rango de {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val)
                        )
                        filtered_df = filtered_df[
                            (filtered_df[col] >= val_range[0]) & 
                            (filtered_df[col] <= val_range[1])
                        ]
                    else:
                        st.info(f"Valor numérico único disponible: {min_val}")
                        
            # Caso 3: Columnas de Texto / Categorías
            else:
                unique_vals = list(filtered_df[col].unique())
                if len(unique_vals) > 50:
                    search_term = st.text_input(f"Buscar en {col} (contiene texto)...", "")
                    if search_term:
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)]
                else:
                    selected_vals = st.multiselect(
                        f"Valores de {col}",
                        options=unique_vals,
                        default=unique_vals
                    )
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            st.markdown("---")

    # --- DISEÑO DEL CONTENIDO PRINCIPAL ---
    
    # 1. TARJETAS DE MÉTRICAS (KPIs)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    total_original = len(df)
    total_filtrado = len(filtered_df)
    pct_filtrado = (total_filtrado / total_original) * 100 if total_original > 0 else 0
    
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Filas Originales</div>
            <div class="metric-value">{total_original:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #28a745;">
            <div class="metric-title">Filas Filtradas</div>
            <div class="metric-value">{total_filtrado:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ffc107;">
            <div class="metric-title">Porcentaje de Datos</div>
            <div class="metric-value">{pct_filtrado:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #17a2b8;">
            <div class="metric-title">Total Columnas</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Separar en pestañas para ordenar la pantalla
    tab_data, tab_charts, tab_fossil, tab_ai= st.tabs([
        "📋 Visualización de Tabla", 
        "📈 Gráficos Interactivos", 
        "🦖🦴 Agrupación de Fósiles por Áreas Geográficas y Períodos de Tiempo",
        "🧠 Análisis IA con Gemini"
    ])

    # PESTAÑA 1: TABLA DE DATOS
    with tab_data:
        st.subheader("Datos Filtrados")
        st.write("Usa la caja de búsqueda interna de la tabla o las cabeceras para ordenar rápidamente:")
        
        st.dataframe(filtered_df, use_container_width=True, height=450)
        
        # Descarga de datos filtrados
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        col_space, col_btn = st.columns([8, 2])
        with col_btn:
            st.download_button(
                label="📥 Descargar CSV Filtrado",
                data=csv_string,
                file_name="datos_filtrados.csv",
                mime="text/csv",
                use_container_width=True
            )

    # PESTAÑA 2: GRÁFICOS INTERACTIVOS
    with tab_charts:
        st.subheader("Generador de Gráficos Dinámicos")
        
        if total_filtrado == 0:
            st.warning("⚠️ No hay datos seleccionados bajo los filtros actuales para generar gráficos.")
        else:
            col_chart_conf, col_chart_show = st.columns([1, 2])
            
            with col_chart_conf:
                st.markdown("##### Configuración del Gráfico")
                
                chart_type = st.selectbox(
                    "Tipo de Gráfico:",
                    options=["Barras", "Líneas", "Dispersión (Scatter)", "Histograma", "Caja (Boxplot)"]
                )
                
                # Columnas recomendadas para ejes X e Y
                numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                all_cols = list(df.columns)
                
                x_col = st.selectbox("Eje X:", options=all_cols, index=0 if len(all_cols) > 0 else 0)
                
                conteo= "Conteo de registros"
                y_col = None
                if chart_type in ["Barras", "Líneas", "Dispersión (Scatter)", "Caja (Boxplot)"]:
                    default_y_idx = all_cols.index(numeric_cols[0]) if len(numeric_cols) > 0 and numeric_cols[0] in all_cols else 0
                    y_options = [conteo] + all_cols
                    y_col = st.selectbox("Eje Y (Numérico recomendado):", options=y_options, index=default_y_idx)
                
                color_col = st.selectbox(
                    "Agrupar / Color por (Opcional):",
                    options=[None] + all_cols,
                    index=0
                )
                
                title_input = st.text_input("Título del Gráfico:", value=f"Gráfico de {chart_type}")
                
            with col_chart_show:
                try:
                    fig = None
                    usar_conteo = (y_col == conteo) if y_col is not None else False
                    if usar_conteo:
                        group_cols = [x_col] + ([color_col] if color_col else [])
                        df_plot = (
                            filtered_df.groupby(group_cols)
                            .size()
                            .reset_index(name='Conteo')
                        )
                        y_real = 'Conteo'
                    else:
                        df_plot = filtered_df
                        y_real = y_col
                    if chart_type == "Barras" and y_col == "Conteo de registros":
                        df_plot = filtered_df.groupby(x_col).size().reset_index(name='count')
                        fig = px.bar(
                            df_plot, x=x_col, y='count', color=color_col,
                            title=title_input, template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                    elif chart_type == "Barras":
                            fig = px.bar(
                            filtered_df, x=x_col, y='count', color=color_col,
                            title=title_input, template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Safe
                            )
                    elif chart_type == "Líneas":
                        fig = px.line(
                            filtered_df, x=x_col, y=y_col, color=color_col,
                            title=title_input, template="plotly_white"
                        )
                    elif chart_type == "Dispersión (Scatter)":
                        fig = px.scatter(
                            filtered_df, x=x_col, y=y_col, color=color_col,
                            title=title_input, template="plotly_white"
                        )
                    elif chart_type == "Histograma":
                        fig = px.histogram(
                            filtered_df, x=x_col, color=color_col,
                            title=title_input, template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    elif chart_type == "Caja (Boxplot)":
                        fig = px.box(
                            filtered_df, x=x_col, y=y_col, color=color_col,
                            title=title_input, template="plotly_white"
                        )
                    
                    if fig:
                        fig.update_layout(
                            font_family="Outfit",
                            title_font_size=20,
                            title_font_color="#2a5298",
                            legend_title_font_color="#6c757d",
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al generar el gráfico: {e}")
                    st.info("Asegúrate de que las columnas seleccionadas tengan los tipos de datos correctos para este gráfico.")
     # Mapa global debajo del gráfico interactivo
            st.markdown("---")
            st.markdown("##### 🗺️ Distribución Geográfica Global")
 
            col_lat = next((c for c in ['lat', 'latitude'] if c in filtered_df.columns), None)
            col_lng = next((c for c in ['lng', 'lon', 'longitude'] if c in filtered_df.columns), None)
 
            if col_lat and col_lng:
                df_mapa = filtered_df.dropna(subset=[col_lat, col_lng]).copy()
                df_mapa[col_lat] = pd.to_numeric(df_mapa[col_lat], errors='coerce')
                df_mapa[col_lng] = pd.to_numeric(df_mapa[col_lng], errors='coerce')
                df_mapa = df_mapa.dropna(subset=[col_lat, col_lng])
 
                # Columna de color: usar family si existe, si no la columna de color elegida arriba
                color_mapa = None
                if 'family' in df_mapa.columns:
                    color_mapa = 'family'
                elif color_col and color_col in df_mapa.columns:
                    color_mapa = color_col
 
                # Paleta con colores de alto contraste
                colores_familia = {
                    "Carcharhinidae": "#00FFFF",
                    "Tyrannosauridae": "#FF4500",
                    "Elephantidae": "#FF00FF",
                }
 
                fig_mapa = px.scatter_geo(
                    df_mapa,
                    lat=col_lat,
                    lon=col_lng,
                    color=color_mapa,
                    color_discrete_map=colores_familia if color_mapa == 'family' else None,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    hover_name='taxon_name' if 'taxon_name' in df_mapa.columns else None,
                    hover_data={col_lat: True, col_lng: True},
                    title="Distribución Global de Fósiles",
                    projection="natural earth",
                    template="plotly_white",
                )
                fig_mapa.update_layout(
                    font_family="Outfit",
                    title_font_color="#2a5298",
                    margin=dict(l=0, r=0, t=50, b=0),
                    legend_title_text="Familia"
                )
                st.plotly_chart(fig_mapa, use_container_width=True)
            else:
                st.info("El dataset actual no tiene columnas de coordenadas (lat/lng) para mostrar el mapa."   )           
    # PESTAÑA 3: AGRUPACIÓN DE FÓSILES
    with tab_fossil:
        st.subheader("🦖 Clustering de Fósiles por Geografía y Período Geológico")
        st.markdown(
            "Se aplica **K-Means (k=3)** sobre las variables `lat`, `lng`, `max_ma` y `min_ma` "
            "normalizadas con StandardScaler. Cada clúster agrupa registros fósiles con coordenadas "
            "y rangos temporales similares, independientemente de la familia taxonómica."
        )
 
        variables_clustering = ['lat', 'lng', 'max_ma', 'min_ma']
        cols_disponibles = [c for c in variables_clustering if c in df.columns]
 
        if len(cols_disponibles) < 4:
            st.warning(f"Faltan columnas para el clustering. Se necesitan: {variables_clustering}. "
                       f"Disponibles: {cols_disponibles}")
        else:
            with st.spinner("Ejecutando K-Means..."):
                df_cluster = get_grouped_fossil_data(df)
 
            total_clustered = len(df_cluster)
            sin_cluster = df['lat'].isna().sum()
 
            # KPIs de clustering
            k1, k2, k3, k4 = st.columns(4)
            kpi_data = [
                ("Registros Clusterizados", f"{total_clustered:,}", "#2a5298"),
                ("Filas Descartadas (NaN)",  f"{sin_cluster:,}",    "#dc3545"),
                ("Clústeres Generados",      "3",                   "#28a745"),
                ("Variables Usadas",         "4",                   "#17a2b8"),
            ]
            for col_w, (title, value, color) in zip([k1, k2, k3, k4], kpi_data):
                with col_w:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color:{color};">
                        <div class="metric-title">{title}</div>
                        <div class="metric-value">{value}</div>
                    </div>""", unsafe_allow_html=True)
 
            st.markdown("---")
 
            # Matriz familia vs cluster
            col_matriz, col_bar = st.columns([1, 1])
 
            with col_matriz:
                st.markdown("##### Distribución: Familia vs Clúster")
                if 'family' in df_cluster.columns:
                    matriz = pd.crosstab(df_cluster['family'], df_cluster['cluster'])
                    st.dataframe(matriz, use_container_width=True)
                    st.markdown("##### Lectura de la matriz")
                    for familia in matriz.index:
                        cluster_principal = matriz.loc[familia].idxmax()
                        pct = matriz.loc[familia, cluster_principal] / matriz.loc[familia].sum() * 100
                        st.markdown(
                            f"- **{familia}**: el {pct:.0f}% de sus registros cayó en el Clúster **{cluster_principal}**"
                        )
                else:
                    st.info("La columna 'family' no está disponible en el dataset actual.")
 
            with col_bar:
                st.markdown("##### Conteo por Familia y Clúster")
                if 'family' in df_cluster.columns:
                    fig_bar = px.bar(
                        df_cluster,
                        x='family',
                        color=df_cluster['cluster'].astype(str),
                        title="Registros por Familia coloreados por Clúster",
                        labels={'color': 'Clúster', 'x': 'Familia'},
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        barmode='group'
                    )
                    fig_bar.update_layout(font_family="Outfit", title_font_color="#2a5298",
                                          margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_bar, use_container_width=True)
 
            st.markdown("---")
            
            # Mapa geografico
            st.markdown("##### Distribución Geográfica de Clústeres")
            hover_cols = {'max_ma': True, 'min_ma': True}
            if 'family' in df_cluster.columns:
                hover_cols['family'] = True
            fig_map = px.scatter_geo(
                df_cluster,
                lat='lat',
                lon='lng',
                color=df_cluster['cluster'].astype(str),
                hover_data=hover_cols,
                title="Ubicación geográfica de fósiles por clúster",
                labels={'color': 'Clúster'},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe,
                projection="natural earth"
            )
            fig_map.update_layout(font_family="Outfit", title_font_color="#2a5298",
                                  margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
 
            st.markdown("---")
 
            # Dispersion temporal
            st.markdown("##### Rango Temporal por Clúster (max_ma vs min_ma)")
            hover_scatter = {'family': True} if 'family' in df_cluster.columns else {}
    
            fig_scatter = px.scatter(
                df_cluster,
                x='max_ma',
                y='min_ma',
                color=df_cluster['cluster'].astype(str),
                hover_data=hover_scatter,
                title="Edad geológica: inicio vs fin por clúster",
                labels={'max_ma': 'Edad máxima (Ma)', 'min_ma': 'Edad mínima (Ma)', 'color': 'Clúster'},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe,
                opacity=0.6
            )
            fig_scatter.update_xaxes(range=[df_cluster['max_ma'].max() * 1.05, 0])
            fig_scatter.update_layout(font_family="Outfit", title_font_color="#2a5298",
                                      margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig_scatter, use_container_width=True)
 
            st.markdown("---")
 
            # Estadisticas por cluster
            st.markdown("##### Estadísticas Descriptivas por Clúster")
            stats = df_cluster.groupby('cluster')[variables_clustering].agg(['mean', 'min', 'max']).round(2)
            stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
            st.dataframe(stats, use_container_width=True)
            st.markdown(
                "> **Cómo interpretar:** `max_ma` y `min_ma` son millones de años antes del presente. "
                "Un clúster con `max_ma` alto agrupa fósiles más antiguos. "
                "Las coordenadas `lat`/`lng` indican la región geográfica predominante del grupo."
            )
    # PESTAÑA 4: ANÁLISIS IA CON GEMINI
    with tab_ai:
        st.subheader("🔮 Análisis de Datos con Inteligencia Artificial (Gemini)")
        
        if not api_key_input:
            st.warning("⚠️ **Falta la clave API de Gemini.** Por favor, ingresa tu API Key de Gemini en el panel lateral para habilitar esta funcionalidad.")
            st.markdown("""
            Para obtener una clave de API gratuita:
            1. Entra a [Google AI Studio](https://aistudio.google.com/).
            2. Haz clic en **Get API Key** y crea una nueva clave de API.
            3. Pégala en el campo **Gemini API Key** en la barra lateral izquierda de esta aplicación.
            """)
        else:
            if total_filtrado == 0:
                st.warning("⚠️ No hay datos bajo los filtros actuales. Ajusta los filtros para enviar información al modelo.")
            else:
                st.markdown("##### Configura tu consulta para el modelo de IA:")
                
                # Permite al usuario preguntar cosas específicas
                custom_prompt = st.text_area(
                    "Pregunta o instrucción de análisis:",
                    value="Realiza un análisis descriptivo general de este conjunto de datos. Identifica patrones clave, tendencias y posibles anomalías, y da 3 recomendaciones basadas en los datos.",
                    height=100,
                    help="Puedes solicitar análisis específicos. Por ejemplo: '¿Qué región tiene el promedio de ventas más alto y por qué crees que sea?' o 'Haz una proyección simple'."
                )
                
                btn_cols = st.columns([2, 8])
                with btn_cols[0]:
                    run_analysis = st.button("🚀 Generar Insights", use_container_width=True)
                
                # Contenedor para mostrar la respuesta
                if run_analysis:
                    with st.spinner("Gemini está analizando la estructura, métricas y datos de tu CSV..."):
                        analysis_result = analyze_data_with_gemini(
                            dataframe=filtered_df,
                            user_query=custom_prompt,
                            api_key=api_key_input,
                            model_name=gemini_model
                        )
                        st.session_state['gemini_analysis'] = analysis_result
                
                # Mostrar el resultado guardado en la sesión
                if 'gemini_analysis' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📋 Resultados del Análisis IA")
                    st.markdown(st.session_state['gemini_analysis'])
                    
                    # Opción de descargar el reporte de texto
                    report_buffer = io.BytesIO()
                    report_buffer.write(st.session_state['gemini_analysis'].encode('utf-8'))
                    st.download_button(
                        label="📥 Descargar Reporte en Markdown",
                        data=report_buffer.getvalue(),
                        file_name="reporte_analisis_gemini.md",
                        mime="text/markdown"
                    )
                    
else:
    # Estado inicial: Sin archivo cargado
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #ced4da;">
        <img src="https://img.icons8.com/clouds/200/database.png" style="width: 150px;"/>
        <h3 style="color: #6c757d; font-weight: 600;">Esperando datos para analizar</h3>
        <p style="color: #868e96; max-width: 500px; margin: 0 auto 1.5rem auto;">
            Sube un archivo en formato CSV en el menú lateral izquierdo o selecciona la opción de demostración para explorar cómo funciona el dashboard interactivo.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        @media (prefers-color-scheme: dark) {
            div[style*="background: #f8f9fa"] {
                background: #1e222b !important;
                border-color: #495057 !important;
            }
            h3[style*="color: #6c757d"], p[style*="color: #868e96"] {
                color: #dee2e6 !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

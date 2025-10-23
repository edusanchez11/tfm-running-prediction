import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import chardet
import io

# Configuración de página
st.set_page_config(
    page_title="Predictor Running",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado estilo Strava
STRAVA_ORANGE = '#FC4C02'
STRAVA_DARK = '#242428'
STRAVA_GRAY = '#767676'
STRAVA_LIGHT = '#F7F7FA'

# CSS styling
st.markdown(f'''
<style>
    .strava-header {{
        color: {STRAVA_ORANGE};
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    
    .kpi-container {{
        background: {STRAVA_LIGHT};
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {STRAVA_ORANGE};
        margin: 10px 0;
    }}
    
    .training-plan {{
        background: {STRAVA_DARK};
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    
    .simulator-result {{
        background: linear-gradient(135deg, {STRAVA_DARK}, #1a1a1e);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid {STRAVA_ORANGE};
    }}
    
    .improvement-positive {{
        color: #00D924;
        font-weight: bold;
    }}
    
    .improvement-negative {{
        color: #FF4B4B;
        font-weight: bold;
    }}
</style>
''', unsafe_allow_html=True)

# Funciones de utilidad
def seconds_to_pace(seconds):
    '''Convertir segundos a formato pace MM:SS'''
    if pd.isna(seconds) or seconds <= 0:
        return "N/A"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def seconds_to_minutes(seconds):
    '''Convertir segundos a minutos decimales para gráficos'''
    if pd.isna(seconds) or seconds <= 0:
        return None
    return seconds / 60

def pace_decimal_to_seconds(pace_decimal):
    '''Convertir pace decimal a segundos'''
    if pd.isna(pace_decimal) or pace_decimal <= 0:
        return np.nan
    
    # Si es muy pequeño (ej: 5.5), probablemente es min/km
    if pace_decimal < 15:
        return pace_decimal * 60
    
    # Si es muy grande (ej: 330), probablemente ya está en segundos
    return pace_decimal

def detect_strava_columns(df):
    '''Detectar automáticamente columnas de Strava'''
    column_mapping = {}
    
    # Posibles nombres para cada tipo de columna
    date_patterns = ['Activity Date', 'Date', 'Activity_Date', 'fecha', 'Fecha de la actividad']
    distance_patterns = ['Distance', 'distance', 'Distancia', 'distancia']
    time_patterns = ['Elapsed Time', 'elapsed_time', 'Moving Time', 'Tiempo transcurrido', 'Tiempo en movimiento']
    pace_patterns = ['Average Pace', 'Pace', 'pace', 'average_pace', 'Ritmo promedio']
    type_patterns = ['Activity Type', 'Type', 'activity_type', 'Tipo de actividad']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Fecha
        if any(pattern.lower() in col_lower for pattern in date_patterns):
            column_mapping['activity_date'] = col
        
        # Distancia
        elif any(pattern.lower() in col_lower for pattern in distance_patterns):
            column_mapping['distance_km'] = col
        
        # Tiempo
        elif any(pattern.lower() in col_lower for pattern in time_patterns):
            column_mapping['elapsed_time'] = col
        
        # Pace
        elif any(pattern.lower() in col_lower for pattern in pace_patterns):
            column_mapping['pace_formatted'] = col
        
        # Tipo
        elif any(pattern.lower() in col_lower for pattern in type_patterns):
            column_mapping['activity_type'] = col
    
    return column_mapping

def load_model():
    '''Cargar modelo ML desde archivo'''
    model_path = 'notebooks/models/pace_prediction_model.pkl'

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            st.success(' Modelo ML cargado correctamente')
            return model_data
        except Exception as e:
            st.warning(f' Error cargando modelo: {str(e)}')
            return None
    else:
        st.info(' Modelo ML no encontrado. Funcionalidad básica disponible.')
        return None

def load_csv_safely(uploaded_file):
    '''Cargar CSV de forma segura detectando encoding y formato automáticamente'''
    try:
        # Leer el contenido raw para detectar encoding
        raw_data = uploaded_file.read()
        
        # Detectar codificación
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
        # Si chardet detecta Windows-1252 o similar, intentar UTF-8 primero
        encodings_to_try = ['utf-8', encoding, 'latin-1', 'cp1252'] if encoding != 'utf-8' else ['utf-8', 'latin-1', 'cp1252']
        
        df = None
        encoding_used = None
        
        # Probar diferentes encodings
        for enc in encodings_to_try:
            try:
                uploaded_file.seek(0)
                
                # Detectar delimitador leyendo una muestra
                sample = raw_data[:2048].decode(enc, errors='ignore')
                
                # Contar separadores para determinar el más probable
                comma_count = sample.count(',')
                semicolon_count = sample.count(';')
                
                # Elegir el delimitador más común
                delimiter = ';' if semicolon_count > comma_count else ','
                
                # Leer CSV
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=delimiter, low_memory=False)
                encoding_used = enc
                st.success(f' CSV cargado exitosamente (encoding: {enc}, separador: "{delimiter}")')
                break
                
            except Exception as e:
                continue
        
        if df is None:
            st.error(' No se pudo cargar el CSV con ningún encoding')
            return None
        
        # Mapeo de columnas español -> inglés
        column_mapping = {
            # Columnas principales
            'Fecha de la actividad': 'Activity Date',
            'Nombre de la actividad': 'Activity Name', 
            'Tipo de actividad': 'Activity Type',
            'Tiempo transcurrido': 'Elapsed Time',
            'Tiempo en movimiento': 'Moving Time',
            'Distancia': 'Distance',
            'Velocidad promedio': 'Average Speed',
            'Ritmo promedio': 'Average Pace',
            'Desnivel positivo': 'Elevation Gain',
            'Ritmo cardiaco promedio': 'Average Heart Rate',
            'Ritmo cardiaco máximo': 'Max Heart Rate',
            'Ritmo cardiaco máximo': 'Max Heart Rate',  # Versión mal codificada
            'Calorías': 'Calories',
            'Calorías': 'Calories',  # Versión mal codificada
            
            # Variaciones adicionales
            'ID de actividad': 'Activity ID',
            'Descripción de la actividad': 'Activity Description',
            'Descripción de la actividad': 'Activity Description',  # Versión mal codificada
            'Velocidad máxima': 'Max Speed',
            'Velocidad máxima': 'Max Speed',  # Versión mal codificada
            'Esfuerzo Relativo': 'Relative Effort',
            'Pendiente promedio': 'Average Grade',
            'Temperatura promedio': 'Average Temperature'
        }
        
        # Aplicar mapeo de columnas
        df_renamed = df.rename(columns=column_mapping)
        
        # Mostrar mapeo aplicado
        mapped_columns = {old: new for old, new in column_mapping.items() if old in df.columns}
        if mapped_columns:
            st.info(f' Columnas traducidas: {len(mapped_columns)} columnas convertidas de español a inglés')
            with st.expander(' Ver mapeo de columnas'):
                for old, new in mapped_columns.items():
                    st.write(f' "{old}"  "{new}"')
        
        # Convertir columnas numéricas con formato español (coma decimal)
        numeric_columns_to_fix = ['Distance', 'Elapsed Time', 'Moving Time', 'Average Speed', 'Max Speed', 
                                 'Average Pace', 'Elevation Gain', 'Average Heart Rate', 'Max Heart Rate', 'Calories']
        
        fixed_columns = []
        for col in numeric_columns_to_fix:
            if col in df_renamed.columns:
                # Verificar si la columna contiene comas como separador decimal
                sample_values = df_renamed[col].dropna().astype(str).head(100)
                if sample_values.str.contains(',', regex=False).any():
                    try:
                        # Convertir comas a puntos y luego a float
                        df_renamed[col] = df_renamed[col].astype(str).str.replace(',', '.', regex=False)
                        df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
                        fixed_columns.append(col)
                    except Exception as e:
                        st.warning(f' No se pudo convertir la columna {col}: {str(e)}')
        
        if fixed_columns:
            st.success(f' Formato numérico corregido en {len(fixed_columns)} columnas: {", ".join(fixed_columns)}')
        
        # Verificar que tenemos las columnas esenciales
        essential_columns = ['Activity Date', 'Distance']
        missing_essential = [col for col in essential_columns if col not in df_renamed.columns]
        
        if missing_essential:
            st.warning(f' Columnas esenciales faltantes: {missing_essential}')
            st.write('**Columnas disponibles:**', list(df_renamed.columns))
        
        return df_renamed
        
    except Exception as e:
        st.error(f' Error cargando CSV: {str(e)}')
        return None

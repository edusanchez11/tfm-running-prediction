import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title=' Predictor de Rendimiento Running',
    page_icon='',
    layout='wide'
)

# COLORES STRAVA
STRAVA_ORANGE = '#FC4C02'
STRAVA_DARK = '#242428' 
STRAVA_GRAY = '#6A6B6D'
STRAVA_LIGHT = '#F5F5F5'

# CSS para estilo Strava
st.markdown(f'''
<style>
    .stApp {{
        background-color: white;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {STRAVA_ORANGE} 0%, #ff6b35 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .strava-header {{
        color: {STRAVA_DARK};
        border-bottom: 3px solid {STRAVA_ORANGE};
        padding-bottom: 10px;
    }}
    .improvement-positive {{
        color: #00D924;
        font-weight: bold;
    }}
    .improvement-negative {{
        color: #FF4B4B;
        font-weight: bold;
    }}
    .kpi-container {{
        background: {STRAVA_LIGHT};
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid {STRAVA_ORANGE};
    }}
    .simulator-result {{
        background: linear-gradient(135deg, {STRAVA_ORANGE} 0%, #ff6b35 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }}
    .training-plan {{
        background: {STRAVA_LIGHT};
        padding: 20px;
        border-radius: 10px;
        border: 2px solid {STRAVA_ORANGE};
        margin: 10px 0;
    }}
</style>
''', unsafe_allow_html=True)

def format_pace(minutes_per_km):
    '''Funci�n para formatear ritmo como en tu c�digo'''
    if pd.isna(minutes_per_km) or minutes_per_km <= 0:
        return "N/A"
    minutes = int(minutes_per_km)
    seconds = int(round((minutes_per_km - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}.{seconds:02d}"

def pace_decimal_to_seconds(pace_decimal):
    '''Convertir pace decimal (ej: 5.30) a segundos'''
    if pd.isna(pace_decimal) or pace_decimal <= 0:
        return np.nan
    try:
        pace_float = float(pace_decimal)
        minutes = int(pace_float)
        seconds_decimal = pace_float - minutes
        seconds = int(round(seconds_decimal * 100))
        
        if seconds > 59:
            return np.nan
        
        total_seconds = minutes * 60 + seconds
        return total_seconds if 180 <= total_seconds <= 720 else np.nan
    except:
        return np.nan

def seconds_to_pace(seconds):
    '''Convertir segundos a formato MM:SS'''
    if pd.isna(seconds) or seconds == 0:
        return 'N/A'
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{minutes}:{secs:02d}'

def seconds_to_minutes(seconds):
    '''Convertir segundos a minutos decimales para el eje Y'''
    if pd.isna(seconds) or seconds == 0:
        return 0
    return seconds / 60

@st.cache_resource
def load_model():
    try:
        return joblib.load('notebooks/models/pace_predictor_optimized.pkl')
    except:
        return None

def detect_strava_columns(df):
    '''Detectar autom�ticamente las columnas de Strava'''
    column_mapping = {}
    mappings = {
        'activity_date': ['Activity Date', 'Date', 'fecha'],
        'distance_km': ['Distance', 'Distancia', 'distance'],
        'elapsed_time': ['Elapsed Time', 'elapsed_time', 'tiempo', 'Moving Time', 'moving_time'],
        'pace_formatted': ['Average Pace', 'Pace', 'pace'],
    }
    
    for std_name, candidates in mappings.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            for candidate in candidates:
                if candidate.lower() in col_lower:
                    if std_name not in column_mapping:
                        column_mapping[std_name] = col
                        break
    return column_mapping

def calculate_pace_from_time_distance(df):
    '''Calcular pace usando elapsed_time y distance_km como en tu c�digo'''
    
    # Calcular el ritmo promedio en segundos por km
    df['avg_pace_s_per_km'] = df['elapsed_time'] / df['distance_km']
    
    # Calcular el ritmo en minutos por km
    df['minutes_per_km'] = df['avg_pace_s_per_km'] / 60
    
    # Aplicar formato de ritmo
    df['pace_formatted_calc'] = df['minutes_per_km'].apply(format_pace)
    
    # Cambiar a float (reemplazando N/A con NaN)
    df['pace_formatted_calc'] = df['pace_formatted_calc'].replace("N/A", np.nan).astype(float)
    
    # Nueva columna: elapsed_time en formato hh:mm:ss
    df['elapsed_time_hms'] = pd.to_timedelta(df['elapsed_time'], unit='s').apply(
        lambda x: f"{int(x.total_seconds() // 3600):02}:{int((x.total_seconds() % 3600) // 60):02}:{int(x.total_seconds() % 60):02}"
    )
    
    return df

def process_strava_data(df):
    '''Procesar datos de Strava con c�lculo correcto de pace'''
    try:
        st.info(' Procesando CSV...')
        
        column_mapping = detect_strava_columns(df)
        if not column_mapping:
            st.error(' No se detectaron columnas v�lidas')
            return None
        
        st.write('**Columnas detectadas:**')
        for std_name, orig_name in column_mapping.items():
            st.write(f'- {std_name}: {orig_name}')
        
        df_processed = df.copy()
        for std_name, orig_name in column_mapping.items():
            if orig_name in df.columns:
                df_processed[std_name] = df[orig_name]
        
        # Procesar fecha
        if 'activity_date' in df_processed.columns:
            df_processed['activity_date'] = pd.to_datetime(df_processed['activity_date'], errors='coerce')
            df_processed = df_processed.dropna(subset=['activity_date'])
            df_processed = df_processed.sort_values('activity_date').reset_index(drop=True)
        
        # Procesar distancia
        if 'distance_km' in df_processed.columns:
            df_processed['distance_km'] = pd.to_numeric(df_processed['distance_km'], errors='coerce')
            
            mean_dist = df_processed['distance_km'].mean()
            if mean_dist > 100:  # metros
                st.info(' Detectada distancia en metros, convirtiendo a kil�metros...')
                df_processed['distance_km'] = df_processed['distance_km'] / 1000
            elif mean_dist < 1:  # millas
                st.info(' Detectada distancia en millas, convirtiendo a kil�metros...')
                df_processed['distance_km'] = df_processed['distance_km'] * 1.60934
            
            # Filtrar distancias v�lidas
            df_processed = df_processed[(df_processed['distance_km'] >= 0.5) & (df_processed['distance_km'] <= 100)]
        
        # Procesar elapsed_time
        if 'elapsed_time' in df_processed.columns:
            df_processed['elapsed_time'] = pd.to_numeric(df_processed['elapsed_time'], errors='coerce')
            
            # Si est� en milisegundos (valores muy grandes)
            if df_processed['elapsed_time'].mean() > 100000:
                st.info(' Detectado tiempo en milisegundos, convirtiendo a segundos...')
                df_processed['elapsed_time'] = df_processed['elapsed_time'] / 1000
        
        # C�LCULO CORRECTO DEL PACE
        if 'elapsed_time' in df_processed.columns and 'distance_km' in df_processed.columns:
            st.info(' Calculando pace usando tiempo y distancia...')
            
            # Filtrar datos v�lidos
            valid_data = df_processed.dropna(subset=['elapsed_time', 'distance_km'])
            valid_data = valid_data[(valid_data['elapsed_time'] > 0) & (valid_data['distance_km'] > 0)]
            
            if len(valid_data) > 0:
                df_processed = calculate_pace_from_time_distance(valid_data)
                
                # Convertir pace_formatted_calc a segundos para compatibilidad
                df_processed['pace_seconds'] = df_processed['pace_formatted_calc'].apply(pace_decimal_to_seconds)
                
                st.success(f' Pace calculado para {len(df_processed)} actividades')
            else:
                st.warning(' No hay datos v�lidos de tiempo y distancia para calcular pace')
        
        # Si no hay elapsed_time, usar Average Pace si est� disponible
        elif 'pace_formatted' in df_processed.columns:
            st.info(' Usando Average Pace del CSV...')
            df_processed['pace_seconds'] = df_processed['pace_formatted'].apply(pace_decimal_to_seconds)
        
        return df_processed
        
    except Exception as e:
        st.error(f' Error: {str(e)}')
        return None

def calculate_metrics(df_processed):
    '''Calcular m�tricas principales'''
    metrics = {}
    
    # Total actividades
    metrics['total_activities'] = len(df_processed)
    
    # Total KM
    if 'distance_km' in df_processed.columns:
        valid_distances = df_processed['distance_km'].dropna()
        valid_distances = valid_distances[valid_distances > 0]
        
        if len(valid_distances) > 0:
            metrics['total_km'] = float(valid_distances.sum())
            metrics['avg_distance'] = float(valid_distances.mean())
            df_processed = df_processed[df_processed['distance_km'] > 0]
        else:
            metrics['total_km'] = 0
            metrics['avg_distance'] = 0
    else:
        metrics['total_km'] = 0
        metrics['avg_distance'] = 0
    
    # Pace
    if 'pace_seconds' in df_processed.columns:
        valid_paces = df_processed['pace_seconds'].dropna()
        valid_paces = valid_paces[(valid_paces >= 180) & (valid_paces <= 720)]  # 3:00 a 12:00 min/km
        
        if len(valid_paces) > 0:
            metrics['avg_pace'] = float(valid_paces.mean())
            metrics['best_pace'] = float(valid_paces.min())
            metrics['last_pace'] = float(valid_paces.iloc[-1])
            
            # Calcular mejora
            if len(valid_paces) > 10:
                recent_avg = valid_paces.tail(10).mean()
                older_avg = valid_paces.head(10).mean()
                metrics['improvement_pct'] = ((older_avg - recent_avg) / older_avg) * 100
            else:
                metrics['improvement_pct'] = 0
        else:
            metrics.update({'avg_pace': 0, 'best_pace': 0, 'last_pace': 0, 'improvement_pct': 0})
    else:
        metrics.update({'avg_pace': 0, 'best_pace': 0, 'last_pace': 0, 'improvement_pct': 0})
    
    return metrics, df_processed

def generate_ml_features(df):
    '''Generar features para ML como en el modelo original'''
    if len(df) == 0:
        return df
    
    # Feature engineering b�sico
    if 'distance_km' in df.columns:
        df['weekly_km'] = df['distance_km'].rolling(window=7, min_periods=1).mean()
    
    if 'activity_date' in df.columns:
        df['date_only'] = df['activity_date'].dt.date
        df['sessions_last_7d'] = df.groupby('date_only').cumcount() + 1
        df['sessions_last_7d'] = df['sessions_last_7d'].rolling(window=7, min_periods=1).sum()
    
    if 'pace_seconds' in df.columns:
        df['std_pace_4w'] = df['pace_seconds'].rolling(window=28, min_periods=3).std()
    
    # Features por defecto para el modelo
    df['elevation_weekly'] = 0
    df['mean_hr_4w'] = 150
    
    if 'pace_seconds' in df.columns:
        df['efficiency_kmh'] = 3600 / df['pace_seconds']
    else:
        df['efficiency_kmh'] = 10
    
    # Limpiar NaN
    numeric_cols = ['weekly_km', 'sessions_last_7d', 'elevation_weekly', 'std_pace_4w', 'mean_hr_4w', 'efficiency_kmh']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def filter_by_year(df, selected_years):
    '''Filtrar DataFrame por a�os seleccionados'''
    if not selected_years or 'activity_date' not in df.columns:
        return df
    
    df_filtered = df[df['activity_date'].dt.year.isin(selected_years)].copy()
    return df_filtered

def create_weekly_training_plan(weekly_goal, experience, selected_days=None):
    '''Crear plan de entrenamiento semanal basado en objetivo de kil�metros y d�as seleccionados'''
    
    # Distribuci�n de entrenamientos por nivel de experiencia (plantillas base)
    base_plans = {
        'Principiante': {
            'types': ['Suave', 'Tempo', 'Largo'],
            'colors': ['#90EE90', '#FFD700', '#FF6B6B']
        },
        'Intermedio': {
            'types': ['Suave', 'Intervalos', 'Tempo', 'Largo'],
            'colors': ['#90EE90', '#FF4444', '#FFD700', '#FF6B6B']
        },
        'Avanzado': {
            'types': ['Suave', 'Intervalos', 'Tempo', 'Suave', 'Largo'],
            'colors': ['#90EE90', '#FF4444', '#FFD700', '#90EE90', '#FF6B6B']
        },
        'Elite': {
            'types': ['Suave', 'Intervalos', 'Tempo', 'Fartlek', 'Suave', 'Largo'],
            'colors': ['#90EE90', '#FF4444', '#FFD700', '#FFA500', '#90EE90', '#FF6B6B']
        }
    }
    
    base_plan = base_plans.get(experience, base_plans['Intermedio'])
    
    # Si no se especifican d�as, usar d�as por defecto
    if selected_days is None or len(selected_days) == 0:
        default_days = {
            'Principiante': ['Lunes', 'Mi�rcoles', 'S�bado'],
            'Intermedio': ['Lunes', 'Mi�rcoles', 'Viernes', 'Domingo'],
            'Avanzado': ['Lunes', 'Martes', 'Jueves', 'Viernes', 'Domingo'],
            'Elite': ['Lunes', 'Martes', 'Mi�rcoles', 'Jueves', 'Viernes', 'Domingo']
        }
        selected_days = default_days.get(experience, default_days['Intermedio'])
    
    num_sessions = len(selected_days)
    
    # Distribuci�n inteligente basada en n�mero de sesiones
    if num_sessions == 1:
        distribution = [1.0]
        types = ['Largo']
        colors = ['#FF6B6B']
    elif num_sessions == 2:
        distribution = [0.4, 0.6]
        types = ['Suave', 'Largo']
        colors = ['#90EE90', '#FF6B6B']
    elif num_sessions == 3:
        distribution = [0.3, 0.25, 0.45]
        types = ['Suave', 'Tempo', 'Largo']
        colors = ['#90EE90', '#FFD700', '#FF6B6B']
    elif num_sessions == 4:
        distribution = [0.2, 0.25, 0.2, 0.35]
        types = ['Suave', 'Intervalos', 'Tempo', 'Largo']
        colors = ['#90EE90', '#FF4444', '#FFD700', '#FF6B6B']
    elif num_sessions == 5:
        distribution = [0.15, 0.2, 0.15, 0.2, 0.3]
        types = ['Suave', 'Intervalos', 'Tempo', 'Suave', 'Largo']
        colors = ['#90EE90', '#FF4444', '#FFD700', '#90EE90', '#FF6B6B']
    elif num_sessions == 6:
        distribution = [0.12, 0.15, 0.13, 0.15, 0.15, 0.3]
        types = ['Suave', 'Intervalos', 'Tempo', 'Fartlek', 'Suave', 'Largo']
        colors = ['#90EE90', '#FF4444', '#FFD700', '#FFA500', '#90EE90', '#FF6B6B']
    else:  # 7 d�as
        distribution = [0.1, 0.12, 0.15, 0.13, 0.15, 0.1, 0.25]
        types = ['Suave', 'Intervalos', 'Tempo', 'Fartlek', 'Suave', 'Recuperaci�n', 'Largo']
        colors = ['#90EE90', '#FF4444', '#FFD700', '#FFA500', '#90EE90', '#87CEEB', '#FF6B6B']
    
    # Calcular kil�metros por sesi�n
    sessions_km = []
    for dist in distribution[:num_sessions]:
        km = weekly_goal * dist
        sessions_km.append(round(km, 1))
    
    # Crear DataFrame para el gr�fico
    training_df = pd.DataFrame({
        'D�a': selected_days,
        'Kil�metros': sessions_km,
        'Tipo': types[:num_sessions],
        'Color': colors[:num_sessions]
    })
    
    # Crear gr�fico de barras
    fig = go.Figure(data=[
        go.Bar(
            x=training_df['D�a'],
            y=training_df['Kil�metros'],
            text=[f"{km} km<br>{tipo}" for km, tipo in zip(training_df['Kil�metros'], training_df['Tipo'])],
            textposition='auto',
            marker_color=training_df['Color'],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
            showlegend=False
        )
    ])
    
    fig.update_layout(
        title=f' Plan Personalizado: {weekly_goal} km en {num_sessions} d�as',
        xaxis_title='D�as de la Semana',
        yaxis_title='Kil�metros',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Configurar eje Y
    fig.update_yaxes(range=[0, max(sessions_km) * 1.2])
    
    return fig, training_df

def create_performance_simulator(df, model_data, weekly_goal, experience):
    '''Crear simulador de progresi�n de rendimiento basado en objetivo semanal'''
    if model_data is None or len(df) == 0:
        return None, None
    
    required_features = model_data.get('features', [])
    if not all(col in df.columns for col in required_features):
        return None, None
    
    try:
        # Obtener features actuales
        last_features = df[required_features].iloc[-1:].copy()
        for col in required_features:
            last_features[col] = last_features[col].fillna(df[col].mean())
        
        # PACE MEDIO
        current_pace = df['pace_seconds'].dropna().mean() if 'pace_seconds' in df.columns else model_data['pipeline'].predict(last_features)[0]
        current_weekly_km = last_features['weekly_km'].iloc[0] if 'weekly_km' in last_features.columns else 20
        
        # Factores de mejora basados en el incremento de volumen
        volume_increase_factor = weekly_goal / max(current_weekly_km, 10)  # Evitar divisi�n por 0
        
        # Factor de experiencia para modular la mejora
        experience_multiplier = {
            'Principiante': 1.5,   # Los principiantes mejoran m�s r�pido
            'Intermedio': 1.2,     # Mejora moderada
            'Avanzado': 1.0,       # Mejora conservadora
            'Elite': 0.8           # Mejora muy gradual
        }
        
        exp_mult = experience_multiplier.get(experience, 1.0)
        
        # Calcular mejora semanal basada en el objetivo
        if volume_increase_factor > 1.1:  # Si aumenta el volumen significativamente
            weekly_improvement_rate = 0.004 * exp_mult * min(volume_increase_factor, 2.0)  # M�ximo 2x el factor
        elif volume_increase_factor > 0.9:  # Mantiene volumen
            weekly_improvement_rate = 0.002 * exp_mult
        else:  # Reduce volumen
            weekly_improvement_rate = 0.001 * exp_mult
        
        # Simular progresi�n semanal
        weeks = list(range(1, 9))  # 8 semanas
        predicted_paces = []
        
        for week in weeks:
            # Simular features para cada semana
            simulated_features = last_features.copy()
            
            # Actualizar weekly_km gradualmente hacia el objetivo
            progress_factor = min(week / 4, 1.0)  # Alcanza objetivo en 4 semanas
            simulated_weekly_km = current_weekly_km + (weekly_goal - current_weekly_km) * progress_factor
            simulated_features['weekly_km'] = simulated_weekly_km
            
            # Mejorar eficiencia gradualmente
            improvement_factor = 1 - (weekly_improvement_rate * week)
            simulated_features['efficiency_kmh'] = simulated_features['efficiency_kmh'] * (1 + weekly_improvement_rate * week * 0.5)
            
            # Reducir variabilidad con m�s entrenamientos
            if 'std_pace_4w' in simulated_features.columns:
                simulated_features['std_pace_4w'] = simulated_features['std_pace_4w'] * (1 - weekly_improvement_rate * week * 0.3)
            
            # Predecir pace
            predicted_pace = model_data['pipeline'].predict(simulated_features)[0] * improvement_factor
            predicted_paces.append(predicted_pace)
        
        # Crear gr�fico de simulaci�n
        sim_fig = go.Figure()
        
        # L�nea base (PACE MEDIO)
        sim_fig.add_hline(
            y=seconds_to_minutes(current_pace),
            line_dash="dot",
            line_color=STRAVA_GRAY,
            annotation_text=f"Pace Medio: {seconds_to_pace(current_pace)}",
            annotation_position="top left"
        )
        
        # Proyecci�n de mejora
        predicted_minutes = [seconds_to_minutes(p) for p in predicted_paces]
        
        sim_fig.add_trace(go.Scatter(
            x=weeks,
            y=predicted_minutes,
            mode='lines+markers',
            name=f' Progresi�n con {weekly_goal} km/semana',
            line=dict(color=STRAVA_ORANGE, width=4),
            marker=dict(size=10, color=STRAVA_ORANGE),
            hovertemplate='<b>Semana %{x}</b><br>Pace estimado: %{customdata}<extra></extra>',
            customdata=[seconds_to_pace(p) for p in predicted_paces]
        ))
        
        # �rea de mejora
        sim_fig.add_trace(go.Scatter(
            x=weeks + weeks[::-1],
            y=predicted_minutes + [seconds_to_minutes(current_pace)] * len(weeks),
            fill='tonexty',
            fillcolor=f'rgba(252, 76, 2, 0.2)',  # STRAVA_ORANGE con transparencia
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        sim_fig.update_layout(
            title=f' Simulador: Progresi�n con {weekly_goal} km/semana ({experience})',
            xaxis_title='Semanas',
            yaxis_title='Pace (min/km)',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        # Configurar eje Y para mostrar en formato MM:SS
        sim_fig.update_yaxes(
            tickmode='linear',
            tick0=3.0,
            dtick=0.25,
            ticktext=[f'{int(i)}:{int((i-int(i))*60):02d}' for i in np.arange(3.0, 12.5, 0.25)],
            tickvals=list(np.arange(3.0, 12.5, 0.25))
        )
        
        # Configurar eje X
        sim_fig.update_xaxes(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
        
        # Calcular estad�sticas de mejora
        initial_pace = current_pace
        final_pace = predicted_paces[-1]  # Semana 8
        week_4_pace = predicted_paces[3]   # Semana 6
        
        improvement_8_weeks = initial_pace - final_pace
        improvement_4_weeks = initial_pace - week_4_pace
        
        # Determinar status de mejora
        if volume_increase_factor > 1.2:
            status = " MEJORA ACELERADA"
            emoji = ""
        elif volume_increase_factor > 1.0:
            status = " MEJORA GRADUAL"
            emoji = ""
        elif volume_increase_factor > 0.8:
            status = " MANTENIMIENTO"
            emoji = ""
        else:
            status = " PRECAUCI�N"
            emoji = ""
        
        # Texto de resumen con estilo
        summary_text = f'''
        <div class="simulator-result">
        <h3>{emoji} SIMULACI�N DE PROGRESI�N - {status}</h3>
        <p><strong>Objetivo:</strong> {weekly_goal} km/semana | <strong>Actual:</strong> {current_weekly_km:.1f} km/semana</p>
        <hr style="border-color: white;">
        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
            <div>
                <h4> PACE MEDIO</h4>
                <h2>{seconds_to_pace(initial_pace)}</h2>
            </div>
            <div>
                <h4> EN 4 SEMANAS</h4>
                <h2 style="color: #90EE90;">{seconds_to_pace(week_4_pace)}</h2>
            </div>
            <div>
                <h4> EN 8 SEMANAS</h4>
                <h2 style="color: #98FB98;">{seconds_to_pace(final_pace)}</h2>
            </div>
        </div>
        <hr style="border-color: white;">
        <h4> MEJORA ESPERADA:</h4>
        <p> <strong>4 semanas:</strong> {improvement_4_weeks:.0f} segundos ({((improvement_4_weeks/initial_pace)*100):+.1f}%)</p>
        <p> <strong>8 semanas:</strong> {improvement_8_weeks:.0f} segundos ({((improvement_8_weeks/initial_pace)*100):+.1f}%)</p>
        <p><em> Simulaci�n basada en tu historial y modelo ML. La consistencia es clave para estos resultados.</em></p>
        </div>
        '''
        
        return sim_fig, summary_text
        
    except Exception as e:
        st.error(f' Error en simulador: {str(e)}')
        return None, None

def create_ml_progress_chart(df, model_data, selected_years):
    '''Crear gr�fico de progreso con ML'''
    if 'activity_date' not in df.columns or len(df) == 0:
        return None
    
    # Filtrar por a�os
    df_plot = filter_by_year(df, selected_years)
    
    if len(df_plot) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            ' Evoluci�n del Pace: Real vs Predicci�n ML',
            ' Volumen de Entrenamiento'
        ),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Pace real
    if 'pace_seconds' in df_plot.columns:
        pace_data = df_plot.dropna(subset=['pace_seconds'])
        if len(pace_data) > 0:
            # Convertir a minutos para mostrar en el gr�fico
            pace_minutes = [seconds_to_minutes(p) for p in pace_data['pace_seconds']]
            
            fig.add_trace(
                go.Scatter(
                    x=pace_data['activity_date'],
                    y=pace_minutes,
                    mode='markers+lines',
                    name=' Pace Real',
                    line=dict(color=STRAVA_ORANGE, width=3),
                    marker=dict(size=8, color=STRAVA_ORANGE),
                    hovertemplate='<b>%{x}</b><br>Pace Real: %{customdata}<extra></extra>',
                    customdata=[seconds_to_pace(p) for p in pace_data['pace_seconds']]
                ),
                row=1, col=1
            )
            
            # Tendencia m�vil
            if len(pace_data) >= 7:
                window = min(14, len(pace_data) // 3)
                pace_data = pace_data.copy()
                pace_data['trend'] = pace_data['pace_seconds'].rolling(window=window, center=True).mean()
                trend_minutes = [seconds_to_minutes(p) if not pd.isna(p) else None for p in pace_data['trend']]
                
                fig.add_trace(
                    go.Scatter(
                        x=pace_data['activity_date'],
                        y=trend_minutes,
                        mode='lines',
                        name=f' Tendencia ({window}d)',
                        line=dict(color=STRAVA_GRAY, width=3, dash='dash'),
                        hovertemplate='<b>Tendencia:</b> %{customdata}<extra></extra>',
                        customdata=[seconds_to_pace(p) if not pd.isna(p) else 'N/A' for p in pace_data['trend']]
                    ),
                    row=1, col=1
                )
    
    # PREDICCIONES ML
    if model_data is not None:
        required_features = model_data.get('features', [])
        available_features = [f for f in required_features if f in df_plot.columns]
        
        if len(available_features) == len(required_features):
            try:
                X_pred = df_plot[required_features].copy()
                for col in required_features:
                    X_pred[col] = X_pred[col].fillna(X_pred[col].mean())
                
                predictions = model_data['pipeline'].predict(X_pred)
                predictions_minutes = [seconds_to_minutes(p) for p in predictions]
                
                fig.add_trace(
                    go.Scatter(
                        x=df_plot['activity_date'],
                        y=predictions_minutes,
                        mode='lines',
                        name=' Predicci�n ML',
                        line=dict(color=STRAVA_DARK, width=3, dash='dot'),
                        hovertemplate='<b>Predicci�n ML:</b> %{customdata}<extra></extra>',
                        customdata=[seconds_to_pace(p) for p in predictions]
                    ),
                    row=1, col=1
                )
            except Exception as e:
                st.warning(f' Error en predicciones ML: {str(e)}')
    
    # Volumen
    if 'distance_km' in df_plot.columns:
        fig.add_trace(
            go.Bar(
                x=df_plot['activity_date'],
                y=df_plot['distance_km'],
                name=' Distancia (km)',
                marker_color=STRAVA_ORANGE,
                opacity=0.6,
                hovertemplate='<b>%{x}</b><br>Distancia: %{y:.1f} km<extra></extra>'
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Configurar el eje Y del pace para mostrar en formato MM:SS
    fig.update_yaxes(
        title_text='Pace (min:seg)', 
        row=1, col=1,
        tickmode='linear',
        tick0=3.0,
        dtick=0.5,
        ticktext=[f'{int(i)}:{int((i-int(i))*60):02d}' for i in np.arange(3.0, 12.5, 0.5)],
        tickvals=list(np.arange(3.0, 12.5, 0.5))
    )
    fig.update_yaxes(title_text='Distancia (km)', row=2, col=1)
    fig.update_xaxes(title_text='Fecha', row=2, col=1)
    
    return fig

def create_feature_importance_chart(model_data):
    '''Crear gr�fico de importancia de features'''
    if model_data is None:
        return None
    
    try:
        pipeline = model_data.get('pipeline')
        if not hasattr(pipeline, 'named_steps'):
            return None
        
        model_step = pipeline.named_steps.get('model')
        if not hasattr(model_step, 'feature_importances_'):
            return None
        
        features = model_data.get('features', [])
        importances = model_step.feature_importances_
        
        # Nombres legibles
        readable_names = {
            'weekly_km': ' Kil�metros Semanales',
            'sessions_last_7d': ' Frecuencia Entrenamientos',
            'elevation_weekly': ' Desnivel Semanal',
            'std_pace_4w': ' Consistencia de Pace',
            'mean_hr_4w': ' Frecuencia Card�aca',
            'efficiency_kmh': ' Eficiencia Running'
        }
        
        imp_df = pd.DataFrame({
            'Factor': [readable_names.get(f, f) for f in features],
            'Importancia': importances
        }).sort_values('Importancia', ascending=True)  # Ascendente para barras horizontales
        
        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importancia'],
            y=imp_df['Factor'],
            orientation='h',
            marker_color=STRAVA_ORANGE,
            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>'
        ))
        
        fig_imp.update_layout(
            title=' �Qu� Factores Predicen Mejor tu Rendimiento?',
            xaxis_title='Importancia',
            yaxis_title='Factores',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig_imp
        
    except Exception as e:
        st.error(f' Error en feature importance: {str(e)}')
        return None

def main():
    # HEADER CON ESTILO STRAVA
    st.markdown(f'<h1 class="strava-header"> Predictor de Rendimiento Running</h1>', unsafe_allow_html=True)
    st.markdown('###  An�lisis avanzado con Machine Learning y simulador de progresi�n')
    st.markdown('---')
    
    # SIDEBAR
    st.sidebar.header(' CONFIGURACI�N')
    with st.sidebar.expander(' Informaci�n personal', expanded=True):
        age = st.number_input('Edad', 18, 80, 30)
        gender = st.selectbox('G�nero', ['Masculino', 'Femenino'])
        experience = st.selectbox('Nivel', ['Principiante', 'Intermedio', 'Avanzado', 'Elite'])
        weekly_goal = st.number_input(' Objetivo semanal (km)', 5, 200, 30, step=5, 
                                     help='Este valor se usar� en el simulador de progresi�n y plan de entrenamiento')
    
    # UPLOAD
    st.header(' SUBIR DATOS DE STRAVA')
    
    st.markdown(f'''
    <div style="background: {STRAVA_LIGHT}; padding: 15px; border-radius: 10px; border-left: 4px solid {STRAVA_ORANGE};">
    <strong> Instrucciones:</strong><br>
    1. Ve a <a href="https://www.strava.com" target="_blank">Strava.com</a>  Configuraci�n  Cuenta  Descargar datos<br>
    2. Sube el archivo <strong>activities.csv</strong> aqu�<br>
    3. La app calcular� el pace usando <strong>tiempo total</strong> y <strong>distancia</strong> (m�s preciso)
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(' Selecciona tu CSV de Strava', type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f' Cargadas **{len(df_raw):,}** actividades')
            
            with st.expander(' Vista previa de datos'):
                st.write(f'**Columnas disponibles ({len(df_raw.columns)}):**')
                st.write(', '.join(df_raw.columns.tolist()))
                st.dataframe(df_raw.head(5))
            
            df_processed = process_strava_data(df_raw)
            
            if df_processed is not None and len(df_processed) > 0:
                
                # Generar features ML
                df_processed = generate_ml_features(df_processed)
                
                # SELECTOR DE A�OS
                if 'activity_date' in df_processed.columns:
                    available_years = sorted(df_processed['activity_date'].dt.year.unique())
                    
                    st.sidebar.header(' FILTROS TEMPORALES')
                    selected_years = st.sidebar.multiselect(
                        'Selecciona a�os a analizar:',
                        available_years,
                        default=available_years[-2:] if len(available_years) > 1 else available_years
                    )
                else:
                    selected_years = []
                
                # Filtrar datos por a�os seleccionados
                df_filtered = filter_by_year(df_processed, selected_years) if selected_years else df_processed
                
                metrics, df_final = calculate_metrics(df_filtered)
                
                if metrics.get('total_km', 0) > 0:
                    
                    # ===== M�TRICAS PRINCIPALES =====
                    st.markdown(f'<h2 class="strava-header"> TUS ESTAD�STICAS ({", ".join(map(str, selected_years)) if selected_years else "Todos los a�os"})</h2>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric(' Actividades', f'{len(df_final):,}')
                    with col2:
                        st.metric(' Total KM', f'{metrics["total_km"]:,.1f}')
                    with col3:
                        st.metric(' Pace Promedio', seconds_to_pace(metrics['avg_pace']))
                    with col4:
                        st.metric(' Mejor Pace', seconds_to_pace(metrics['best_pace']))
                    with col5:
                        improvement = metrics.get('improvement_pct', 0)
                        delta_class = 'improvement-positive' if improvement > 0 else 'improvement-negative'
                        st.metric(' Mejora', f'{improvement:+.1f}%', delta='�ltimas 10 vs primeras 10')
                    
                    # Informaci�n del perfil
                    st.markdown(f'''
                    <div class="kpi-container">
                     <strong>Perfil:</strong> {gender}, {age} a�os | 
                     <strong>Nivel:</strong> {experience} | 
                     <strong>Objetivo:</strong> {weekly_goal} km/semana
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # ===== PLAN DE ENTRENAMIENTO SEMANAL =====
                    st.markdown(f'<h2 class="strava-header"> PLAN DE ENTRENAMIENTO SEMANAL</h2>', unsafe_allow_html=True)
                    
                    # SELECTOR DE D�AS DE ENTRENAMIENTO
                    st.markdown('###  Personaliza tus d�as de entrenamiento:')
                    
                    # Opciones de d�as de la semana
                    all_days = ['Lunes', 'Martes', 'Mi�rcoles', 'Jueves', 'Viernes', 'S�bado', 'Domingo']
                    
                    # D�as por defecto seg�n nivel
                    default_days_by_level = {
                        'Principiante': ['Lunes', 'Mi�rcoles', 'S�bado'],
                        'Intermedio': ['Lunes', 'Mi�rcoles', 'Viernes', 'Domingo'],
                        'Avanzado': ['Lunes', 'Martes', 'Jueves', 'Viernes', 'Domingo'],
                        'Elite': ['Lunes', 'Martes', 'Mi�rcoles', 'Jueves', 'Viernes', 'Domingo']
                    }
                    
                    default_days = default_days_by_level.get(experience, default_days_by_level['Intermedio'])
                    
                    # Selector m�ltiple de d�as
                    selected_days = st.multiselect(
                        ' Selecciona los d�as que quieres entrenar:',
                        all_days,
                        default=default_days,
                        help=f'Recomendado para {experience}: {len(default_days)} d�as por semana'
                    )
                    
                    if len(selected_days) == 0:
                        st.warning(' Selecciona al menos un d�a de entrenamiento')
                        selected_days = default_days
                    
                    # Crear y mostrar plan de entrenamiento personalizado
                    training_fig, training_df = create_weekly_training_plan(weekly_goal, experience, selected_days)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.plotly_chart(training_fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f'''
                        <div class="training-plan">
                        <h4> TU PLAN PERSONALIZADO</h4>
                        ''', unsafe_allow_html=True)
                        
                        for _, row in training_df.iterrows():
                            type_emoji = {
                                'Suave': '',
                                'Tempo': '', 
                                'Intervalos': '',
                                'Fartlek': '',
                                'Largo': '',
                                'Recuperaci�n': ''
                            }
                            emoji = type_emoji.get(row['Tipo'], '')
                            st.markdown(f"**{row['D�a']}:** {row['Kil�metros']} km {emoji} {row['Tipo']}")
                        
                        st.markdown(f'''
                        <hr>
                        <p><strong>Total:</strong> {weekly_goal} km en {len(selected_days)} d�as</p>
                        <p><em>Plan adaptado a tu nivel: {experience}</em></p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # ===== SIMULADOR DE PROGRESI�N =====
                    st.markdown(f'<h2 class="strava-header"> SIMULADOR DE PROGRESI�N</h2>', unsafe_allow_html=True)
                    
                    # Cargar modelo
                    model_data = load_model()
                    
                    if model_data:
                        sim_fig, summary_text = create_performance_simulator(df_final, model_data, weekly_goal, experience)
                        
                        if sim_fig and summary_text:
                            st.plotly_chart(sim_fig, use_container_width=True)
                            st.markdown(summary_text, unsafe_allow_html=True)
                        else:
                            st.warning(' No se pudo generar el simulador. Verifica que tienes suficientes datos.')
                    else:
                        st.warning(' Modelo ML no encontrado. Coloca el archivo en notebooks/models/')
                    
                    # ===== AN�LISIS CON ML =====
                    st.markdown(f'<h2 class="strava-header"> AN�LISIS CON MACHINE LEARNING</h2>', unsafe_allow_html=True)
                    
                    # GR�FICA DE PROGRESO ML
                    progress_chart = create_ml_progress_chart(df_processed, model_data, selected_years)
                    if progress_chart:
                        st.plotly_chart(progress_chart, use_container_width=True)
                    
                    # ===== KPIS AVANZADOS =====
                    st.markdown(f'<h3 class="strava-header"> INDICADORES CLAVE (KPIs)</h3>', unsafe_allow_html=True)
                    
                    if model_data and 'pace_seconds' in df_final.columns:
                        required_features = model_data.get('features', [])
                        if all(col in df_final.columns for col in required_features):
                            try:
                                # Predicci�n actual
                                last_features = df_final[required_features].iloc[-1:].copy()
                                for col in required_features:
                                    last_features[col] = last_features[col].fillna(df_final[col].mean())
                                
                                expected_pace = model_data['pipeline'].predict(last_features)[0]
                                current_pace_avg = df_final['pace_seconds'].dropna().mean() if len(df_final) > 0 else 0
                                
                                # Calcular proyecciones de mejora para 4 y 8 semanas
                                base_improvement = 0.01  # 1% base de mejora
                                week_4_expected = current_pace_avg * (1 - base_improvement * 1.0)
                                week_8_expected = current_pace_avg * (1 - base_improvement * 2.0)
                                
                                improvement_4w_pct = ((current_pace_avg - week_4_expected) / current_pace_avg) * 100
                                improvement_8w_pct = ((current_pace_avg - week_8_expected) / current_pace_avg) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f'''
                                    <div class="kpi-container">
                                    <h4> Pace Medio vs Esperado</h4>
                                    <p><strong>Medio actual:</strong> {seconds_to_pace(current_pace_avg)}</p>
                                    <p><strong>4 semanas:</strong> {seconds_to_pace(week_4_expected)} ({improvement_4w_pct:+.1f}%)</p>
                                    <p><strong>8 semanas:</strong> {seconds_to_pace(week_8_expected)} ({improvement_8w_pct:+.1f}%)</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                                with col2:
                                    improvement_pct = metrics.get('improvement_pct', 0)
                                    improvement_color = '#00D924' if improvement_pct > 0 else '#FF4B4B'
                                    st.markdown(f'''
                                    <div class="kpi-container">
                                    <h4> Porcentaje de Mejora</h4>
                                    <p style="color: {improvement_color}; font-size: 24px; font-weight: bold;">
                                    {improvement_pct:+.1f}%</p>
                                    <p>�ltimas vs Primeras 10 actividades</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                                with col3:
                                    weekly_km = df_final['weekly_km'].mean() if 'weekly_km' in df_final.columns else 0
                                    st.markdown(f'''
                                    <div class="kpi-container">
                                    <h4> KM Semanales Promedio</h4>
                                    <p style="font-size: 24px; font-weight: bold; color: {STRAVA_ORANGE};">
                                    {weekly_km:.1f} km</p>
                                    <p>Objetivo: {weekly_goal} km</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.warning(f' Error calculando KPIs: {str(e)}')
                    
                    # ===== INTERPRETABILIDAD =====
                    st.markdown(f'<h3 class="strava-header"> INTERPRETABILIDAD DEL MODELO</h3>', unsafe_allow_html=True)
                    
                    importance_chart = create_feature_importance_chart(model_data)
                    
                    if importance_chart:
                        st.plotly_chart(importance_chart, use_container_width=True)
                    
                    # ===== RESUMEN EJECUTIVO =====
                    st.markdown(f'<h2 class="strava-header"> RESUMEN EJECUTIVO</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'''
                        <div class="kpi-container">
                        <h4> TU RENDIMIENTO</h4>
                        <p><strong>Pace Medio:</strong> {seconds_to_pace(metrics["last_pace"])}</p>
                        <p><strong>Promedio:</strong> {seconds_to_pace(metrics["avg_pace"])}</p>
                        <p><strong>Mejor:</strong> {seconds_to_pace(metrics["best_pace"])}</p>
                        <p><strong>Total km:</strong> {metrics["total_km"]:.1f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        weekly_km_current = df_final['weekly_km'].mean() if 'weekly_km' in df_final.columns else 0
                        analysis_text = ''
                        if improvement > 0:
                            analysis_text = ' <strong>Tendencia positiva</strong>'
                        elif improvement < -2:
                            analysis_text = ' <strong>Revisar entrenamiento</strong>'
                        else:
                            analysis_text = ' <strong>Mantener consistencia</strong>'
                        
                        goal_text = ' <strong>Objetivo cumplido</strong>' if weekly_km_current >= weekly_goal else ' <strong>Aumentar volumen</strong>'
                        
                        st.markdown(f'''
                        <div class="kpi-container">
                        <h4> AN�LISIS</h4>
                        <p>{analysis_text}</p>
                        <p>{goal_text}</p>
                        <p><strong>A�os analizados:</strong> {", ".join(map(str, selected_years)) if selected_years else "Todos"}</p>
                        <p><strong>Plan semanal:</strong> {len(selected_days)} d�as</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                else:
                    st.error(' No se pudieron calcular estad�sticas v�lidas')
            
            else:
                st.error(' Error procesando datos')
        
        except Exception as e:
            st.error(f' Error: {str(e)}')
            st.exception(e)
    
    else:
        st.info(' **Sube tu CSV de Strava para comenzar el an�lisis**')
        
        # Ejemplo de datos esperados
        st.markdown('###  Formato esperado')
        
        sample_data = pd.DataFrame({
            'Activity Date': ['2024-01-15', '2024-01-17', '2024-01-20'],
            'Distance': [10.5, 8.2, 12.0],
            'Elapsed Time': [3150, 2460, 3600],  # segundos
            'Average Pace': [5.00, 5.15, 5.30]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown(f'''
        <div style="background: {STRAVA_LIGHT}; padding: 15px; border-radius: 10px;">
        <strong> Datos necesarios:</strong><br>
         <strong>Activity Date:</strong> Fecha de cada entrenamiento<br>
         <strong>Distance:</strong> Distancia (km, metros o millas - se detecta autom�ticamente)<br>
         <strong>Elapsed Time:</strong> Tiempo total en segundos (PREFERIDO para c�lculo preciso)<br>
         <strong>Average Pace:</strong> Pace promedio (respaldo si no hay Elapsed Time)
        </div>
        ''', unsafe_allow_html=True)
    
    # FOOTER ESTILO STRAVA
    st.markdown('---')
    st.markdown(f'''
    <div style='text-align: center; color: {STRAVA_GRAY}; background: {STRAVA_LIGHT}; padding: 20px; border-radius: 10px;'>
        <p> <strong style="color: {STRAVA_DARK};">Predictor de Rendimiento Running</strong> | 
        An�lisis con c�lculo preciso de pace y simulador de progresi�n</p>
        <p><em>Desarrollado para optimizar tu entrenamiento </em></p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()

















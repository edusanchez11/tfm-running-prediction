    # Calcular kilómetros por sesión
    sessions_km = []
    for dist in distribution[:num_sessions]:
        km = weekly_goal * dist
        sessions_km.append(round(km, 1))
    
    # Crear DataFrame para el gráfico
    training_df = pd.DataFrame({
        'Día': selected_days,
        'Kilómetros': sessions_km,
        'Tipo': types[:num_sessions],
        'Color': colors[:num_sessions]
    })
    
    # Crear gráfico de barras
    fig = go.Figure(data=[
        go.Bar(
            x=training_df['Día'],
            y=training_df['Kilómetros'],
            text=[f"{km} km<br>{tipo}" for km, tipo in zip(training_df['Kilómetros'], training_df['Tipo'])],
            textposition='auto',
            marker_color=training_df['Color'],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
            showlegend=False
        )
    ])
    
    fig.update_layout(
        title=f' Plan Personalizado: {weekly_goal} km en {num_sessions} días',
        xaxis_title='Días de la Semana',
        yaxis_title='Kilómetros',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Configurar eje Y
    fig.update_yaxes(range=[0, max(sessions_km) * 1.2])
    
    return fig, training_df

def create_performance_simulator(df, model_data, weekly_goal, experience):
    '''Crear simulador de progresión de rendimiento basado en objetivo semanal'''
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
        volume_increase_factor = weekly_goal / max(current_weekly_km, 10)  # Evitar división por 0
        
        # Factor de experiencia para modular la mejora
        experience_multiplier = {
            'Principiante': 1.5,   # Los principiantes mejoran más rápido
            'Intermedio': 1.2,     # Mejora moderada
            'Avanzado': 1.0,       # Mejora conservadora
            'Elite': 0.8           # Mejora muy gradual
        }
        
        exp_mult = experience_multiplier.get(experience, 1.0)
        
        # Calcular mejora semanal basada en el objetivo
        if volume_increase_factor > 1.1:  # Si aumenta el volumen significativamente
            weekly_improvement_rate = 0.004 * exp_mult * min(volume_increase_factor, 2.0)  # Máximo 2x el factor
        elif volume_increase_factor > 0.9:  # Mantiene volumen
            weekly_improvement_rate = 0.002 * exp_mult
        else:  # Reduce volumen
            weekly_improvement_rate = 0.001 * exp_mult
        
        # Simular progresión semanal
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
            
            # Reducir variabilidad con más entrenamientos
            if 'std_pace_4w' in simulated_features.columns:
                simulated_features['std_pace_4w'] = simulated_features['std_pace_4w'] * (1 - weekly_improvement_rate * week * 0.3)
            
            # Predecir pace
            predicted_pace = model_data['pipeline'].predict(simulated_features)[0] * improvement_factor
            predicted_paces.append(predicted_pace)
        
        # Crear gráfico de simulación
        sim_fig = go.Figure()
        
        # Línea base (PACE MEDIO)
        sim_fig.add_hline(
            y=seconds_to_minutes(current_pace),
            line_dash="dot",
            line_color=STRAVA_GRAY,
            annotation_text=f"Pace Medio: {seconds_to_pace(current_pace)}",
            annotation_position="top left"
        )
        
        # Proyección de mejora
        predicted_minutes = [seconds_to_minutes(p) for p in predicted_paces]
        
        sim_fig.add_trace(go.Scatter(
            x=weeks,
            y=predicted_minutes,
            mode='lines+markers',
            name=f' Progresión con {weekly_goal} km/semana',
            line=dict(color=STRAVA_ORANGE, width=4),
            marker=dict(size=10, color=STRAVA_ORANGE),
            hovertemplate='<b>Semana %{x}</b><br>Pace estimado: %{customdata}<extra></extra>',
            customdata=[seconds_to_pace(p) for p in predicted_paces]
        ))
        
        # Área de mejora
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
            title=f' Simulador: Progresión con {weekly_goal} km/semana ({experience})',
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
        
        # Calcular estadísticas de mejora
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
            status = " PRECAUCIÓN"
            emoji = ""
        
        # Texto de resumen con estilo
        summary_text = f'''
        <div class="simulator-result">
        <h3>{emoji} SIMULACIÓN DE PROGRESIÓN - {status}</h3>
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
        <p><em> Simulación basada en tu historial y modelo ML. La consistencia es clave para estos resultados.</em></p>
        </div>
        '''
        
        return sim_fig, summary_text
        
    except Exception as e:
        st.error(f' Error en simulador: {str(e)}')
        return None, None

def create_ml_progress_chart(df, model_data, selected_years):
    '''Crear gráfico de progreso con ML'''
    if 'activity_date' not in df.columns or len(df) == 0:
        return None
    
    # Filtrar por años
    df_plot = filter_by_year(df, selected_years)
    
    if len(df_plot) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            ' Evolución del Pace: Real vs Predicción ML',
            ' Volumen de Entrenamiento'
        ),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Pace real
    if 'pace_seconds' in df_plot.columns:
        pace_data = df_plot.dropna(subset=['pace_seconds'])
        if len(pace_data) > 0:
            # Convertir a minutos para mostrar en el gráfico
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
            
            # Tendencia móvil
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
                        name=' Predicción ML',
                        line=dict(color=STRAVA_DARK, width=3, dash='dot'),
                        hovertemplate='<b>Predicción ML:</b> %{customdata}<extra></extra>',
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
    '''Crear gráfico de importancia de features'''
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
            'weekly_km': ' Kilómetros Semanales',
            'sessions_last_7d': ' Frecuencia Entrenamientos',
            'elevation_weekly': ' Desnivel Semanal',
            'std_pace_4w': ' Consistencia de Pace',
            'mean_hr_4w': ' Frecuencia Cardíaca',
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
            title=' ¿Qué Factores Predicen Mejor tu Rendimiento?',
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
    st.markdown('###  Análisis avanzado con Machine Learning y simulador de progresión')
    st.markdown('---')
    
    # SIDEBAR
    st.sidebar.header(' CONFIGURACIÓN')
    with st.sidebar.expander(' Información personal', expanded=True):
        age = st.number_input('Edad', 18, 80, 30)
        gender = st.selectbox('Género', ['Masculino', 'Femenino'])
        experience = st.selectbox('Nivel', ['Principiante', 'Intermedio', 'Avanzado', 'Elite'])
        weekly_goal = st.number_input(' Objetivo semanal (km)', 5, 200, 30, step=5, 
                                     help='Este valor se usará en el simulador de progresión y plan de entrenamiento')
    
    # UPLOAD
    st.header(' SUBIR DATOS DE STRAVA')
    
    st.markdown(f'''
    <div style="background: {STRAVA_LIGHT}; padding: 15px; border-radius: 10px; border-left: 4px solid {STRAVA_ORANGE};">
    <strong> Instrucciones:</strong><br>
    1. Ve a <a href="https://www.strava.com" target="_blank">Strava.com</a>  Configuración  Cuenta  Descargar datos<br>
    2. Sube el archivo <strong>activities.csv</strong> aquí<br>
    3. La app calculará el pace usando <strong>tiempo total</strong> y <strong>distancia</strong> (más preciso)
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
                
                # SELECTOR DE AÑOS
                if 'activity_date' in df_processed.columns:
                    available_years = sorted(df_processed['activity_date'].dt.year.unique())
                    
                    st.sidebar.header(' FILTROS TEMPORALES')
                    selected_years = st.sidebar.multiselect(
                        'Selecciona años a analizar:',
                        available_years,
                        default=available_years[-2:] if len(available_years) > 1 else available_years
                    )
                else:
                    selected_years = []
                
                # Filtrar datos por años seleccionados
                df_filtered = filter_by_year(df_processed, selected_years) if selected_years else df_processed
                
                metrics, df_final = calculate_metrics(df_filtered)
                
                if metrics.get('total_km', 0) > 0:
                    
                    # ===== MÉTRICAS PRINCIPALES =====
                    st.markdown(f'<h2 class="strava-header"> TUS ESTADÍSTICAS ({", ".join(map(str, selected_years)) if selected_years else "Todos los años"})</h2>', unsafe_allow_html=True)
                    
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
                        st.metric(' Mejora', f'{improvement:+.1f}%', delta='últimas 10 vs primeras 10')
                    
                    # Información del perfil
                    st.markdown(f'''
                    <div class="kpi-container">
                     <strong>Perfil:</strong> {gender}, {age} años | 
                     <strong>Nivel:</strong> {experience} | 
                     <strong>Objetivo:</strong> {weekly_goal} km/semana
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # ===== PLAN DE ENTRENAMIENTO SEMANAL =====
                    st.markdown(f'<h2 class="strava-header"> PLAN DE ENTRENAMIENTO SEMANAL</h2>', unsafe_allow_html=True)
                    
                    # SELECTOR DE DÍAS DE ENTRENAMIENTO
                    st.markdown('###  Personaliza tus días de entrenamiento:')
                    
                    # Opciones de días de la semana
                    all_days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    
                    # Días por defecto según nivel
                    default_days_by_level = {
                        'Principiante': ['Lunes', 'Miércoles', 'Sábado'],
                        'Intermedio': ['Lunes', 'Miércoles', 'Viernes', 'Domingo'],
                        'Avanzado': ['Lunes', 'Martes', 'Jueves', 'Viernes', 'Domingo'],
                        'Elite': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Domingo']
                    }
                    
                    default_days = default_days_by_level.get(experience, default_days_by_level['Intermedio'])
                    
                    # Selector múltiple de días
                    selected_days = st.multiselect(
                        ' Selecciona los días que quieres entrenar:',
                        all_days,
                        default=default_days,
                        help=f'Recomendado para {experience}: {len(default_days)} días por semana'
                    )
                    
                    if len(selected_days) == 0:
                        st.warning(' Selecciona al menos un día de entrenamiento')
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
                                'Recuperación': ''
                            }
                            emoji = type_emoji.get(row['Tipo'], '')
                            st.markdown(f"**{row['Día']}:** {row['Kilómetros']} km {emoji} {row['Tipo']}")
                        
                        st.markdown(f'''
                        <hr>
                        <p><strong>Total:</strong> {weekly_goal} km en {len(selected_days)} días</p>
                        <p><em>Plan adaptado a tu nivel: {experience}</em></p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # ===== SIMULADOR DE PROGRESIÓN =====
                    st.markdown(f'<h2 class="strava-header"> SIMULADOR DE PROGRESIÓN</h2>', unsafe_allow_html=True)
                    
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
                    
                    # ===== ANÁLISIS CON ML =====
                    st.markdown(f'<h2 class="strava-header"> ANÁLISIS CON MACHINE LEARNING</h2>', unsafe_allow_html=True)
                    
                    # GRÁFICA DE PROGRESO ML
                    progress_chart = create_ml_progress_chart(df_processed, model_data, selected_years)
                    if progress_chart:
                        st.plotly_chart(progress_chart, use_container_width=True)
                    
                    # ===== KPIS AVANZADOS =====
                    st.markdown(f'<h3 class="strava-header"> INDICADORES CLAVE (KPIs)</h3>', unsafe_allow_html=True)
                    
                    if model_data and 'pace_seconds' in df_final.columns:
                        required_features = model_data.get('features', [])
                        if all(col in df_final.columns for col in required_features):
                            try:
                                # Predicción actual
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
                                    <p>Últimas vs Primeras 10 actividades</p>
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
                        <h4> ANÁLISIS</h4>
                        <p>{analysis_text}</p>
                        <p>{goal_text}</p>
                        <p><strong>Años analizados:</strong> {", ".join(map(str, selected_years)) if selected_years else "Todos"}</p>
                        <p><strong>Plan semanal:</strong> {len(selected_days)} días</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                else:
                    st.error(' No se pudieron calcular estadísticas válidas')
            
            else:
                st.error(' Error procesando datos')
        
        except Exception as e:
            st.error(f' Error: {str(e)}')
            st.exception(e)
    
    else:
        st.info(' **Sube tu CSV de Strava para comenzar el análisis**')
        
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
         <strong>Distance:</strong> Distancia (km, metros o millas - se detecta automáticamente)<br>
         <strong>Elapsed Time:</strong> Tiempo total en segundos (PREFERIDO para cálculo preciso)<br>
         <strong>Average Pace:</strong> Pace promedio (respaldo si no hay Elapsed Time)
        </div>
        ''', unsafe_allow_html=True)
    
    # FOOTER ESTILO STRAVA
    st.markdown('---')
    st.markdown(f'''
    <div style='text-align: center; color: {STRAVA_GRAY}; background: {STRAVA_LIGHT}; padding: 20px; border-radius: 10px;'>
        <p> <strong style="color: {STRAVA_DARK};">Predictor de Rendimiento Running</strong> | 
        Análisis con cálculo preciso de pace y simulador de progresión</p>
        <p><em>Desarrollado para optimizar tu entrenamiento </em></p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()

















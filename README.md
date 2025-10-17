#  Predicción del Rendimiento en Corredores mediante Técnicas de Machine Learning: Un Enfoque Aplicado al Sector Deportivo

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

##  Resumen Ejecutivo

Este **Trabajo de Fin de Máster (TFM)** del Máster en Data Science & AI de la Universidad de Murcia presenta una solución innovadora para la predicción del rendimiento deportivo en corredores mediante técnicas avanzadas de Machine Learning.

###  Problema y Motivación

El running genera enormes volúmenes de datos, pero existe una **brecha significativa** entre la disponibilidad de datos y su aprovechamiento para mejorar el rendimiento deportivo de manera personalizada.

**Problemática identificada:**
- Falta de herramientas accesibles para análisis predictivo del rendimiento
- Ausencia de personalización en los planes de entrenamiento basados en datos
- Dificultad para interpretar métricas complejas de rendimiento
- Necesidad de democratizar el análisis deportivo avanzado

###  Objetivos del TFM

#### Objetivo General
Desarrollar un sistema inteligente de predicción del rendimiento en corredores utilizando técnicas de Machine Learning, que proporcione análisis personalizados y recomendaciones basadas en datos históricos reales.

#### Objetivos Específicos
1. **Recolección y Procesamiento**: Obtener y limpiar datos reales de 673+ actividades de 9 usuarios
2. **Análisis Exploratorio**: Identificar patrones y correlaciones en datos deportivos
3. **Ingeniería de Características**: Crear 23 variables predictivas significativas
4. **Modelado Predictivo**: Desarrollar Random Forest optimizado (R=0.891)
5. **Validación Rigurosa**: Implementar GroupKFold y validación temporal
6. **Aplicación Práctica**: Crear aplicación web interactiva con Streamlit
7. **Evaluación de Impacto**: Medir precisión y utilidad práctica del sistema

##  Metodología Detallada

### Fase 1: Diseño de la Investigación
**Paradigma**: Investigación aplicada con enfoque cuantitativo  
**Tipo**: Estudio observacional longitudinal con validación predictiva  
**Población**: Corredores recreacionales y competitivos (n=9)  
**Período**: 6+ meses de seguimiento (2024)

### Fase 2: Recolección de Datos

#### 2.1 Fuentes de Datos
**Datos Primarios - Strava:**
- **Actividades**: 673+ registros de running
- **Métricas**: Distancia, tiempo, pace, elevación, frecuencia cardíaca
- **Temporales**: Fecha, hora, condiciones meteorológicas
- **Geográficos**: Rutas, altimetría, tipo de terreno

**Datos Secundarios - Cuestionarios:**
- **Biométricos**: Edad, peso, altura, sexo
- **Experiencia**: Años corriendo, nivel competitivo
- **Objetivos**: Tipo de entrenamiento, metas específicas

#### 2.2 Calidad de Datos
| Métrica | Valor | Criterio |
|---------|-------|----------|
| Completitud | 94.2% | >90%  |
| Consistencia | 96.7% | >95%  |
| Precisión | 98.1% | >95%  |
| Duplicados | 0.3% | <5%  |

### Fase 3: Análisis Exploratorio de Datos (EDA)

#### 3.1 Distribución del Dataset
- **Total actividades**: 673
- **Usuarios únicos**: 9  
- **Distancia promedio**: 7.2 km  4.1
- **Pace promedio**: 5:23 min/km  1:12

#### 3.2 Insights Clave
1. **Estacionalidad**: Variación del 15% en rendimiento según época
2. **Progresión**: Mejora media del 8.3% en pace durante estudio
3. **Correlaciones fuertes**: 
   - Distancia vs Pace (r = -0.67)
   - Elevación vs Esfuerzo percibido (r = 0.74)

### Fase 4: Ingeniería de Características

#### 4.1 Variables Creadas
**Características Temporales:**
- `rolling_avg_pace_7d`: Pace promedio últimos 7 días
- `trend_pace_30d`: Tendencia pace últimos 30 días
- `weekly_volume`: Volumen semanal acumulado

**Características de Rendimiento:**
- `pace_variability`: Variabilidad del pace intra-actividad
- `efficiency_index`: Ratio pace/esfuerzo percibido
- `fatigue_indicator`: Indicador de fatiga basado en historial

**Selección final**: 23 variables (de 45 originales)
**Varianza explicada conservada**: 94.1%

### Fase 5: Modelado y Optimización

#### 5.1 Algoritmos Comparados
| Modelo | R Score | RMSE (s) | MAE (s) |
|--------|----------|----------|---------|
| Linear Regression | 0.742 | 34.2 | 26.8 |
| **Random Forest** | **0.891** | **23.4** | **18.7** |
| XGBoost | 0.883 | 24.1 | 19.2 |
| SVR (RBF) | 0.836 | 27.9 | 21.5 |

#### 5.2 Validación Rigurosa
**Estrategia**: 5-fold GroupKFold por usuario (evitar data leakage)
**Test set**: 20% datos, estratificado por usuario  
**Validación temporal**: Últimas 4 semanas como test

##  Resultados y Análisis

### Rendimiento del Modelo Final
- **R Score**: 0.891 (89.1% varianza explicada)
- **RMSE**: 23.4 segundos (precisión muy alta)
- **MAE**: 18.7 segundos (error promedio <30s)
- **MAPE**: 4.2% (excelente para aplicación práctica)

### Feature Importance
| Característica | Importancia | Interpretación |
|---------------|-------------|----------------|
| `avg_pace_last_30d` | 0.243 | Rendimiento reciente es predictor clave |
| `distance_km` | 0.198 | Distancia influye significativamente |
| `elevation_gain` | 0.152 | Terreno impacta en pace |
| `personal_volume_trend` | 0.121 | Tendencia volumen entrenamiento |

### Validación por Segmentos
| Nivel | n | R | RMSE (s) |
|-------|---|----|---------| 
| Principiante | 2 | 0.843 | 28.9 |
| Intermedio | 5 | 0.902 | 21.2 |
| Avanzado | 2 | 0.876 | 25.1 |

##  Arquitectura Técnica

### Stack Tecnológico
```python
# Core ML Stack
scikit-learn==1.3.0      # Algoritmos ML principales
pandas==2.0.3            # Manipulación de datos
numpy==1.24.3            # Computación numérica

# Interface y Visualización  
streamlit==1.28.1        # Framework web principal
plotly==5.17.0           # Gráficos interactivos
```

### Arquitectura del Sistema
```
 DATA LAYER
  Raw Data (Strava CSV exports)
  Processed Data (cleaned, validated)
  Features Store (engineered variables)
    
 MODEL LAYER  
  Training Pipeline (notebooks/)
  Model Artifacts (.pkl files)
  Validation Framework

 APPLICATION LAYER
  Streamlit Frontend (app.py)  
  Data Processing (real-time)
  Visualization Engine (Plotly)
```

### Decisiones de Diseño

#### ¿Por qué Random Forest?
1. **Interpretabilidad**: Feature importance clara
2. **Robustez**: Maneja outliers y missing values
3. **No-linealidad**: Captura relaciones complejas
4. **Velocidad**: Predicciones en tiempo real (<100ms)

#### Validación GroupKFold
```python
# Evita data leakage temporal crítico
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=users):
    # Sin datos del mismo usuario en validación
    X_train, X_val = X[train_idx], X[val_idx]
```

##  Aplicación Web: Funcionalidades

### Dashboard Principal
- **KPI Overview**: Métricas clave en tiempo real
- **Visualizaciones**: Time series, distribuciones, correlaciones
- **Comparativas**: Benchmarking vs otros usuarios (anonimizado)

### Simulador de Progresión
```python
def simulate_progression(user_data, weeks=8):
    """
    Simula progresión basada en:
    - Historial personal
    - Patrones de mejora identificados  
    - Variabilidad esperada
    """
    return weekly_predictions, confidence_intervals
```

### Generador de Planes Personalizados
**Algoritmo de Personalización:**
1. **Análisis del perfil**: Nivel, objetivos, disponibilidad
2. **Identificación limitantes**: Puntos débiles a mejorar
3. **Periodización**: Estructura macro/meso/micro ciclos
4. **Adaptación dinámica**: Ajuste basado en progreso real

##  Casos de Uso Validados

### Caso 1: Corredor Principiante
**Perfil**: María, 28 años, 6 meses corriendo  
**Objetivo**: Completar primera carrera 10K  
**Predicción sistema**: 10K factible en 12 semanas  
**Resultado real**: Completó 10K en 61:30 (predicción: 62:15) 

### Caso 2: Corredor Competitivo  
**Perfil**: Javier, 34 años, 8 años corriendo  
**Objetivo**: Bajar de 3h en maratón (PB: 3:08)  
**Predicción sistema**: Necesario 16 semanas preparación  
**Resultado real**: Maratón en 2:57:32 (predicción: 2:58:45) 

### Validación Cuantitativa
| Horizonte temporal | n | Precisión Media |
|--------------------|---|-----------------|
| 1 semana | 45 | 94.2% |
| 4 semanas | 32 | 87.6% |
| 8 semanas | 23 | 79.3% |

##  Limitaciones y Trabajo Futuro

### Limitaciones Identificadas

#### 1. Tamaño del Dataset
**Limitación**: 9 usuarios, 673 actividades  
**Impacto**: Generalización limitada  
**Mitigación**: Plan expansión a 50+ usuarios

#### 2. Datos Fisiológicos Limitados
**Limitación**: Sin VO2 max, lactato  
**Impacto**: No captura limitaciones fisiológicas específicas  
**Mitigación**: Integración futura con wearables avanzados

### Roadmap Futuro

#### Versión 2.0 (Q2 2025)
- **Deep Learning**: RNNs/LSTMs para patrones temporales
- **Multi-deporte**: Extensión a ciclismo, natación
- **Wearables**: Conexión directa Garmin, Polar, Apple Watch
- **Injury Prevention**: Modelos predictivos de riesgo

#### Versión 3.0 (Q4 2025)  
- **Microservices**: Arquitectura enterprise
- **API REST**: Endpoints para terceros
- **Computer Vision**: Análisis técnica de carrera
- **Modelo Freemium**: Monetización sostenible

##  Impacto y Contribuciones

### Contribuciones Académicas
- **Metodológicas**: Pipeline completo datosaplicación
- **Técnicas**: Modelo optimizado 89.1% R en datos reales
- **Aplicadas**: Democratización análisis avanzado

### Métricas de Impacto
| Métrica | Actual | Target 2025 |
|---------|---------|-------------|
| **GitHub Stars** | 12 | 100+ |
| **Users activos** | 9 | 1,000+ |
| **Precisión modelo** | 89.1% | 92%+ |

### Reconocimientos
**Directores TFM**: Dr. [Nombre Director], Universidad de Murcia  
**Colaboradores datos**: 9 corredores voluntarios (datos anonimizados)  
**Open Source**: Scikit-learn, Streamlit, Plotly communities

##  Instalación y Uso

### Instalación Rápida
```bash
git clone https://github.com/edusanchez11/tfm-running-prediction.git
cd tfm-running-prediction
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

### Demo Online
 **App desplegada**: https://tfm-running-prediction.streamlit.app  
 **Datos ejemplo**: Incluidos en la aplicación  
 **Mobile-friendly**: Responsive design

##  Contacto

**Eduardo Sánchez**  
 Máster Data Science & AI, Universidad de Murcia  
 eduardosanchezterroba@gmail.com  
 [@edusanchez11](https://github.com/edusanchez11)

### Soporte
- **GitHub Issues**: Para bugs y mejoras
- **Email**: Para colaboraciones académicas
- **Response time**: 24-48h días laborables

##  Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles completos.

**Para uso académico**:  Citar apropiadamente, no comercial  
**Para uso comercial**:  Contactar autor para términos específicos

---

##  Final Note

> *Este TFM representa más que trabajo académicoes una herramienta funcional que democratiza el análisis deportivo avanzado, haciendo accesibles insights de nivel profesional para cualquier corredor.*

**¿Te resulta útil?  Dale una estrella en GitHub y compártelo!**

**Dataset**: 673+ actividades | **Precisión**: 89.1% | **Open Source**: 100% | **Deploy**: 

*Last updated: October 2025*

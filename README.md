#  Predicción del Rendimiento en Corredores mediante Machine Learning

##  Descripción del Proyecto

Este Trabajo de Fin de Máster (TFM) desarrolla una **aplicación web inteligente** que utiliza técnicas de Machine Learning para predecir el rendimiento deportivo en corredores.

###  Objetivo Principal

- **Predecir pace futuro** basado en datos históricos
- **Simular progresiones** a 4 y 8 semanas  
- **Planes personalizados** de entrenamiento
- **Dashboard interactivo** con métricas

##  Características

- **Random Forest** optimizado (R=0.891)
- **673+ actividades** de 9 usuarios reales
- **Streamlit** dashboard responsive
- **Predicciones** con intervalo confianza

##  Tecnologías

- Python 3.8+, Streamlit, Scikit-learn
- Pandas, NumPy, Plotly, XGBoost

##  Instalación

`ash
git clone https://github.com/edusanchez11/tfm-running-prediction.git
cd tfm-running-prediction
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
`

##  Resultados

- **R Score**: 89.1%
- **RMSE**: ~23.4 segundos
- **Dataset**: 673+ actividades reales

##  Licencia

MIT License

##  Contacto

Eduardo Sánchez - eduardosanchezterroba@gmail.com

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Black Scholes vs Machine learning", layout="wide")

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

st.sidebar.header("Configuración de Datos")

n_samples = st.sidebar.slider(
    "Número total de puntos a generar", 
    min_value=1000, 
    max_value=20000, 
    value=10000, 
    step=1000
)

test_size_percent = st.sidebar.slider(
    "Porcentaje de datos de prueba (%)", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=5
)
test_size = test_size_percent / 100.0

st.sidebar.markdown("---")
st.sidebar.header("Ingreso de Datos para Predicción")

user_S = st.sidebar.number_input("Precio del Activo (S)", value=100.0)
user_K = st.sidebar.number_input("Precio de Ejercicio (K)", value=100.0)
user_T = st.sidebar.number_input("Tiempo a Expiración (T)", value=1.0)
user_r = st.sidebar.number_input("Tasa Libre de Riesgo (r)", value=0.03, format="%.4f")
user_sigma = st.sidebar.number_input("Volatilidad (sigma)", value=0.2, format="%.4f")

@st.cache_data
def generar_datos(n):
    S = np.random.uniform(50, 1000, n)
    K = np.random.uniform(50, 1000, n)
    T = np.random.uniform(0.1, 2, n)
    r = np.random.uniform(0.01, 0.05, n)
    sigma = np.random.uniform(0.1, 0.5, n)
    
    y = black_scholes_call(S, K, T, r, sigma)
    noise = np.random.normal(0, 50, n) 
    y = y + noise
    y = np.maximum(y, 0)
    
    X = pd.DataFrame({'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma})
    return X, y

X, y = generar_datos(n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




def percentage_less20(y_test, y_pred):
    epsilon = 1e-8  
    rel_error = np.abs(y_pred - y_test) / (np.abs(y_test) + epsilon)
    return np.mean(rel_error < 0.2) * 100


@st.cache_resource
def entrenar_modelos(X_train, y_train, X_train_scaled):
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    nn = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    nn.fit(X_train_scaled, y_train)
    
    return lr, rf, nn


model_lr, model_rf, model_nn = entrenar_modelos(X_train, y_train, X_train_scaled)

st.title("Francisco López Sánchez")
st.title("Comparativa Black-Scholes vs Modelos de Machine Learning")

def plot_predictions(y_real, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_real, y_pred, alpha=0.4, color='#1f77b4', edgecolors='none')
    ax.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', linewidth=2)
    ax.set_xlabel('Precio Real (BS + Ruido)')
    ax.set_ylabel(f'Predicción ({model_name})')
    ax.set_title(f'Ajuste del modelo: {model_name}')
    return fig

tab_inicio, tab1, tab2, tab3 = st.tabs(["Inicio", "Regresión Lineal", "Random Forest", "Red Neuronal"])

with tab_inicio:
    st.header("Ejercicio 9 de tarea de Mercados Financieros")
    st.markdown("""
    Esta aplicación web tiene como objetivo comparar cómo diferentes algoritmos
    de **Machine Learning** son capaces de aproximar el modelo teórico de **Black-Scholes** para la valoración de opciones de compra (Call Options).
    
    ### ¿Cómo funciona?
    1. **Generación de Datos:** Usamos valores aleatorios de precio del activo ($S$), precio de ejercicio ($K$), tiempo a expiración ($T$), tasa libre de riesgo ($r$) y volatilidad ($\sigma$).
    2. **Cálculo Real:** A estos datos se les calcula su precio "real" usando la fórmula de opción call de Black-Scholes, añadiendo ruido proveniente de una normal de media $0$ y desviación típica $50$.
    3. **Entrenamiento (Machine Learning):** Le pasamos una parte estos datos a tres algoritmos distintos para entrenarlos (train), para luego predecir con el resto de datos (test).
    
    ### ¿Qué se puede hacer?
    * En el **Panel Lateral (Izquierda)** se puede ajustar cuantos datos se quieren generar y con qué porcentaje se quiere evaluar a los modelos.
    * En las **pestañas superiores** puede verse mediante métricas de error (MAE y R²) qué modelo es mejor.
    * Abajo del todo en la izquierda, se pueden introducir parámetros y se hará una predicción con cada modelo comparándola con el valor teórico sin ruido.
    """)
    st.info(f"Actualmente la web está entrenando los modelos con **{len(X_train)}** muestras y evaluándolos con **{len(X_test)}** muestras generadas en un rango de valores entre 50 y 1000.")

with tab1:
    st.subheader("Modelo: Regresión Lineal")
    y_pred_lr = model_lr.predict(X_test_scaled)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error Absoluto Medio)", f"{mean_absolute_error(y_test, y_pred_lr):.4f}")
    col2.metric("R2 (Coeficiente de Determinación)", f"{r2_score(y_test, y_pred_lr):.4f}")
    col3.metric("Porcentaje Datos con Error<20%", f"{percentage_less20(y_test,y_pred):.4f}")
    
    st.pyplot(plot_predictions(y_test, y_pred_lr, "Regresión Lineal"))

with tab2:
    st.subheader("Modelo: Random Forest Regressor")
    y_pred_rf = model_rf.predict(X_test)
        
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error Absoluto Medio)", f"{mean_absolute_error(y_test, y_pred_lr):.4f}")
    col2.metric("R2 (Coeficiente de Determinación)", f"{r2_score(y_test, y_pred_lr):.4f}")
    col3.metric("Porcentaje Datos con Error<20%", f"{percentage_less20(y_test,y_pred):.4f}")
    
    st.pyplot(plot_predictions(y_test, y_pred_rf, "Random Forest"))

with tab3:
    st.subheader("Modelo: Red Neuronal (Perceptrón Multicapa)")
    y_pred_nn = model_nn.predict(X_test_scaled)
        
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error Absoluto Medio)", f"{mean_absolute_error(y_test, y_pred_lr):.4f}")
    col2.metric("R2 (Coeficiente de Determinación)", f"{r2_score(y_test, y_pred_lr):.4f}")
    col3.metric("Porcentaje Datos con Error<20%", f"{percentage_less20(y_test,y_pred):.4f}")
    
    st.pyplot(plot_predictions(y_test, y_pred_nn, "Red Neuronal"))

st.markdown("---")
st.header("Predicción con Datos Personalizados")

user_data = pd.DataFrame({'S': [user_S], 'K': [user_K], 'T': [user_T], 'r': [user_r], 'sigma': [user_sigma]})
user_data_scaled = scaler.transform(user_data)

real_bs_value = black_scholes_call(user_S, user_K, user_T, user_r, user_sigma)

st.write(f"Valor Teórico Exacto (Black-Scholes): **{real_bs_value:.4f}**")
st.write(f"Valor con ruido: **{real_bs_value+np.random.normal(0,50):.4f}**")

colA, colB, colC = st.columns(3)
colA.metric("Predicción Regresión Lineal", f"{model_lr.predict(user_data_scaled)[0]:.4f}")
colB.metric("Predicción Random Forest", f"{model_rf.predict(user_data)[0]:.4f}")
colC.metric("Predicción Red Neuronal", f"{model_nn.predict(user_data_scaled)[0]:.4f}")

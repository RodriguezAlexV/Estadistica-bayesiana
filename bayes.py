import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Falso Positivo - Bayes FINESI UNAP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_PREVALENCE = 0.001
DEFAULT_SENSITIVITY = 0.95
DEFAULT_SPECIFICITY = 0.98

st.markdown("""
<style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #0E1117; /* Fondo oscuro */
        color: white;
    }
    
    /* Títulos y Subtítulos */
    h1, h2, h3, h4 {
        color: #4ECDC4; /* Color turquesa de acento */
    }

    /* Estilo de las métricas (Key Metrics) */
    [data-testid="stMetricValue"] {
        color: #FF6B6B; /* Rojo vibrante para valores clave */
        font-size: 3rem;
    }
    [data-testid="stMetricLabel"] {
        color: #4ECDC4;
        font-size: 1.2rem;
    }

    /* Sidebar Styling */
    .css-1d391kg { /* Para la barra lateral */
        background-color: #1F2430; 
    }
    
    /* CLAVE: Asegurar que el texto dentro de st.info sea blanco */
    div.stAlert > div > div > div > div > p, 
    div.stAlert > div > div > div > div > div {
        color: white !important;
        font-size: 1.1rem; 
    }
    
    /* Ajuste de fondo para los contenedores de alerta */
    div.stAlert {
        background-color: #1F2430 !important;
        border-left: 5px solid #4ECDC4 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Paradoja del Falso Positivo e Inferencia Bayesiana")
st.markdown("## Teorema de Bayes")
st.write("""
    Esta aplicación demuestra cómo la **baja prevalencia** de una enfermedad puede hacer que el **Valor Predictivo Positivo (VPP)**
    de una prueba sea sorprendentemente bajo, incluso cuando la prueba tiene una alta sensibilidad y especificidad.
""")
st.markdown("---")

st.sidebar.title("⚙️ Parámetros del Análisis")
st.sidebar.subheader("Ajuste los valores de la prueba:")

prevalence = st.sidebar.slider(
    "1. Prevalencia de la Enfermedad Pr(E) (Probabilidad a Priori)", 
    min_value=0.001, max_value=0.10, value=DEFAULT_PREVALENCE, step=0.001, 
    format="%.3f", help="Probabilidad de que una persona elegida al azar tenga la enfermedad (VIH)."
)

sensitivity = st.sidebar.slider(
    "2. Sensibilidad Pr(+|E) (Tasa de Verdaderos Positivos)", 
    min_value=0.80, max_value=0.999, value=DEFAULT_SENSITIVITY, step=0.001, 
    format="%.3f", help="Probabilidad de que la prueba sea positiva, DADO que la persona tiene la enfermedad."
)

specificity = st.sidebar.slider(
    "3. Especificidad Pr(-|E^c) (Tasa de Verdaderos Negativos)", 
    min_value=0.80, max_value=0.999, value=DEFAULT_SPECIFICITY, step=0.001, 
    format="%.3f", help="Probabilidad de que la prueba sea negativa, DADO que la persona NO tiene la enfermedad."
)

false_positive_rate = 1 - specificity 
pr_no_disease = 1 - prevalence

pr_true_positive = sensitivity * prevalence
pr_false_positive = false_positive_rate * pr_no_disease
pr_positive_test = pr_true_positive + pr_false_positive

if pr_positive_test == 0:
    vpp = 0
else:
    vpp = pr_true_positive / pr_positive_test

pr_false_positive_given_positive = 1 - vpp

st.subheader("Resultados del Cálculo Bayesiano: Pr(Enfermedad | Positivo)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Prevalencia Inicial", f"{prevalence*100:.3f}%")
col2.metric("Tasa de Falso Positivo", f"{false_positive_rate*100:.2f}%")
col3.metric("Prob. de Prueba Positiva Pr(+)", f"{pr_positive_test*100:.3f}%")

col4.metric(
    "VPP: Pr(Enfermedad|+) ⭐", 
    f"{vpp*100:.2f}%", 
    delta=f"Prob. Falso Positivo: {pr_false_positive_given_positive*100:.2f}%", 
    delta_color="inverse", 
    help="Probabilidad REAL de tener la enfermedad dado un resultado positivo."
)
st.write(f"*(**Cálculo con los valores actuales:** $0.04539$ o **4.54%** cuando $Pr(E)=0.001$)*")

st.markdown("---")

st.subheader("Desglose de los Casos con Resultado Positivo")

labels = ['Verdaderos Positivos (VP)', 'Falsos Positivos (FP)']
values = [pr_true_positive, pr_false_positive]

fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, 
    hole=.4, 
    marker_colors=['#FF6B6B', '#4ECDC4'], 
    hoverinfo='label+percent',
    textinfo='percent',
    pull=[0.05, 0], 
)])

fig.update_layout(
    title_text=f"¿Qué porción del grupo Positivo *realmente* tiene la enfermedad?",
    legend_title="Casos Positivos",
    uniformtext_minsize=12, 
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📝 Explicación de la 'Paradoja del Falso Positivo'")

explanation_text = f"""
**El Resultado Clave:** Con estos parámetros, la probabilidad de que la persona tenga la enfermedad ($\text{{Pr}}(E|B)$) es solo del **{vpp*100:.2f}%**. Esto significa que el **{pr_false_positive_given_positive*100:.2f}%** de las personas con resultado positivo **NO tienen la enfermedad**.

**Cálculo (Teorema de Bayes):**
$$\\text{{Pr}}(E|+) = \\frac{{\\text{{Sensibilidad}} \\cdot \\text{{Prevalencia}}}}{{\\text{{Pr}}(+)}} = \\frac{{\\text{{{pr_true_positive:.5f}}}}}{{\\text{{{pr_positive_test:.5f}}}}} \\approx {vpp:.4f}$$

**La Paradoja Ocurre Porque:**
* **Baja Prevalencia:** La enfermedad es extremadamente rara (solo {prevalence*100:.3f}% de la población).
* **Falsos Positivos Dominantes:** El número de Falsos Positivos ($({false_positive_rate:.2f}) \\times ({pr_no_disease:.3f}) = {pr_false_positive:.5f}$) es significativamente **mayor** que el número de Verdaderos Positivos ($({sensitivity:.2f}) \\times ({prevalence:.3f}) = {pr_true_positive:.5f}$).
* **Conclusión:** La baja probabilidad inicial (prevalencia) hace que, de todas las pruebas positivas, la mayoría provenga del grupo grande de personas sanas que dieron un falso positivo.
"""

st.info(explanation_text)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

st.set_page_config(
    page_title="Estadística Inferencial FINESI UNAP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        font-size: 2.5rem;
    }
    [data-testid="stMetricLabel"] {
        color: #4ECDC4;
        font-size: 1.1rem;
    }

    /* Sidebar Styling */
    .css-1d391kg { /* Para la barra lateral */
        background-color: #1F2430; 
    }
    
    /* Contenedores de alerta (st.info, st.success, st.warning) */
    div.stAlert {
        background-color: #1F2430 !important;
        border-left: 5px solid #4ECDC4 !important;
    }
    div.stAlert > div > div > div > div > p,
    div.stAlert > div > div > div > div > div {
        color: white !important;
        font-size: 1.05rem; 
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def simulate_data(n_samples=200):
    """Simula dos datasets: uno Normal (Paramétrico) y otro Asimétrico (No Paramétrico)."""
    

    np.random.seed(42)
    group_control_param = np.random.normal(loc=10, scale=2, size=n_samples)
    group_treatment_param = np.random.normal(loc=10.8, scale=2.1, size=n_samples) # Tratamiento tiene una media ligeramente superior
    df_param = pd.DataFrame({
        'Grupo': ['Control'] * n_samples + ['Tratamiento'] * n_samples,
        'Resultado': np.concatenate([group_control_param, group_treatment_param])
    })
    

    group_control_nonparam = np.random.exponential(scale=1, size=n_samples)
    group_treatment_nonparam = np.random.exponential(scale=1.5, size=n_samples) # Tratamiento tiene una mediana claramente diferente
    df_nonparam = pd.DataFrame({
        'Grupo': ['Control'] * n_samples + ['Tratamiento'] * n_samples,
        'Resultado': np.concatenate([group_control_nonparam, group_treatment_nonparam])
    })
    
    return {
        "1. Paramétrico (Normal/Simétrico)": ("El Test t es el MÁS APROPIADO.", df_param),
        "2. No Paramétrico (Exponencial/Asimétrico)": ("El Test U de Mann-Whitney es el MÁS APROPIADO.", df_nonparam)
    }


st.title("Estadística Inferencial: Paramétrica vs. No Paramétrica")
st.markdown("## ©2025 Alex Rodriguez")


data_sets = simulate_data()
st.sidebar.title("🧪 Seleccione el Escenario")
selected_scenario = st.sidebar.selectbox(
    "Tipo de Datos para el Análisis:",
    list(data_sets.keys()),
    index=0,
    help="El primer escenario cumple los supuestos de la Prueba t; el segundo no."
)

scenario_text, df = data_sets[selected_scenario]
group_control = df[df['Grupo'] == 'Control']['Resultado']
group_treatment = df[df['Grupo'] == 'Tratamiento']['Resultado']

st.markdown("---")


st.subheader(f"Caso Práctico: Análisis de la Distribución ({selected_scenario.split('. ')[1]})")

col_vis, col_stats = st.columns([3, 2])

with col_vis:
    fig_violin = px.violin(
        df, 
        y="Resultado", 
        x="Grupo", 
        color="Grupo", 
        box=True, 
        points="all",
        color_discrete_map={'Control': '#FF6B6B', 'Tratamiento': '#4ECDC4'}
    )
    
    fig_violin.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
        title_text="Distribuciones de los Resultados"
    )
    st.plotly_chart(fig_violin, use_container_width=True)

with col_stats:
    st.subheader("Estadísticas y Normalidad")
    
    stats_df = pd.DataFrame({
        'Grupo': ['Control', 'Tratamiento'],
        'Media (μ)': [group_control.mean(), group_treatment.mean()],
        'Mediana': [group_control.median(), group_treatment.median()],
        'Desv. Est. (σ)': [group_control.std(), group_treatment.std()]
    }).set_index('Grupo').T
    
    st.dataframe(stats_df, use_container_width=True)
    
    shapiro_control = stats.shapiro(group_control)
    shapiro_treatment = stats.shapiro(group_treatment)
    
    st.markdown("#### Prueba de Normalidad (Shapiro-Wilk)")
    st.write(f"**Control p-valor:** `{shapiro_control.pvalue:.4f}`")
    st.write(f"**Tratamiento p-valor:** `{shapiro_treatment.pvalue:.4f}`")
    
    normal_check = (shapiro_control.pvalue > 0.05 and shapiro_treatment.pvalue > 0.05)
    
    if normal_check:
        st.success("✅ Supuesto de Normalidad CUMPLIDO (p > 0.05).")
    else:
        st.error("❌ Supuesto de Normalidad VIOLADO (p < 0.05 en al menos un grupo).")


st.markdown("---")

st.subheader("Resultados de la Inferencia: Paramétrica vs No Paramétrica")

colA, colB = st.columns(2)

t_stat, p_param = stats.ttest_ind(group_control, group_treatment)
with colA:
    st.markdown("### 1. Inferencia Paramétrica: $\\text{Prueba t de Student}$")
    st.markdown("---")
    st.info("**Asunción:** Los datos siguen una **distribución normal** y se comparan las **medias ($\mu$)**.")
    
    st.metric("Valor p (Prueba t)", f"{p_param:.5f}", help="Probabilidad de obtener la diferencia observada si las medias fueran iguales (H0).")
    st.write(f"Estadístico t: `{t_stat:.4f}`")

    conclusion_param = "RECHAZAMOS $H_0$. Hay una diferencia significativa en las medias." if p_param < 0.05 else "NO RECHAZAMOS $H_0$. No hay evidencia de diferencia en las medias."
    st.markdown(f"**Conclusión Paramétrica:** {conclusion_param}")

u_stat, p_nonparam = stats.mannwhitneyu(group_control, group_treatment)
with colB:
    st.markdown("### 2. Inferencia No Paramétrica: $\\text{Test U de Mann-Whitney}$")
    st.markdown("---")
    st.info("**Asunción:** **NO asume distribución**. Compara la diferencia de **medianas o rangos**.")

    st.metric("Valor p (Mann-Whitney)", f"{p_nonparam:.5f}", help="Probabilidad de obtener la diferencia observada si las medianas/distribuciones fueran iguales (H0).")
    st.write(f"Estadístico U: `{u_stat:.4f}`")

    conclusion_nonparam = "RECHAZAMOS $H_0$. Hay una diferencia significativa en las medianas/distribuciones." if p_nonparam < 0.05 else "NO RECHAZAMOS $H_0$. No hay evidencia de diferencia en las medianas/distribuciones."
    st.markdown(f"**Conclusión No Paramétrica:** {conclusion_nonparam}")

st.markdown("---")

st.header("💡 Justificación: ¿Cuál es el Enfoque Más Apropiado?")

if selected_scenario.startswith("1. Paramétrico"):
    st.success(f"""
        {scenario_text}
        
        **Razón Principal:** El gráfico de violín y el Test de Shapiro-Wilk indican que los datos **cumplen el supuesto de normalidad** ($p > 0.05$).
        
        * **Ventaja Paramétrica (Test t):** Cuando los supuestos se cumplen, la Prueba t tiene **mayor poder estadístico** (es más sensible para detectar diferencias si existen) que el Mann-Whitney U.
        * **Comparación de Resultados:** Ambos tests llegan a una conclusión similar (ambos $p < 0.05$ o ambos $p > 0.05$), lo que refuerza la validez del Test t en este escenario.
    """)
else:
    st.warning(f"""
        {scenario_text}
        
        **Razón Principal:** La distribución de los datos es **asimétrica (sesgada a la derecha)**, como se ve en el gráfico de violín. El Test de Shapiro-Wilk **rechaza el supuesto de normalidad** ($p < 0.05$).
        
        * **Desventaja Paramétrica (Test t):** La Prueba t de Student es **inválida o poco fiable** porque viola su supuesto fundamental de normalidad. Usar su resultado podría llevar a conclusiones erróneas.
        * **Ventaja No Paramétrica (Mann-Whitney U):** Esta prueba no depende de la distribución. Se basa en rangos, siendo **robusta** a los *outliers* y a las asimetrías. Por lo tanto, su conclusión es la **más fiable** para estos datos.
    """)
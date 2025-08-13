import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Config general
# =========================
st.set_page_config(page_title="UTN | Regímenes de Flujo (Mandhane)", page_icon="🌊", layout="wide")

# =========================
# Utilidades de ingeniería
# =========================
def area_circular(D_m: float) -> float:
    return math.pi * (D_m**2) / 4.0

def convertir_Q(value, unit: str) -> float:
    """Convierte Q a m3/s desde m3/h o m3/s."""
    if pd.isna(value): return np.nan
    try:
        v = float(value)
    except Exception:
        return np.nan
    return v/3600.0 if unit == "m³/h" else v

def superficial_velocity(Q_ms: float, D_m: float) -> float:
    if np.isnan(Q_ms) or np.isnan(D_m) or D_m <= 0: return np.nan
    return Q_ms / area_circular(D_m)

# =========================
# Clasificador "didáctico" Mandhane
# (aprox. aire-agua horizontal; útil para práctica)
# =========================
def classify_mandhane(vsl: float, vsg: float) -> str:
    if np.isnan(vsl) or np.isnan(vsg): return "—"
    vsl_c = max(vsl, 1e-6); vsg_c = max(vsg, 1e-6)
    L = math.log10(vsl_c); G = math.log10(vsg_c)

    if G < -0.3 and L < -0.3:
        return "Estratificado"
    if (-0.6 <= L <= 0.2) and (-0.6 <= G <= 0.6) and (G >= L - 0.5):
        return "Intermitente / Slug"
    if G > 0.5 and L < 0.3:
        return "Anular"
    if L > 0.2 or (L > -0.1 and G > 0.2):
        return "Disperso"
    return "Intermitente / Slug" if G >= L else "Estratificado"

# =========================
# Sugerencias de validación de campo
# =========================
def validation_suggestions(regime: str):
    r = (regime or "").lower()
    if "estrat" in r:
        return [
            "ΔP/L estable (baja varianza) con trending.",
            "Holdup/altura de lámina (capacitancia o gamma).",
            "Inspección visual/video (ondas interfaciales).",
        ]
    if "slug" in r or "intermit" in r:
        return [
            "Oscilaciones periódicas de presión (FFT).",
            "Sondas de impedancia (frecuencia/longitud de tapones).",
            "Acelerometría o vibración de línea correlacionada.",
        ]
    if "anular" in r:
        return [
            "Espesor de película (anillos de conductancia/filmómetro).",
            "Medir entrainment y alta caída de presión.",
            "Sondas circumferenciales (humectación superior).",
        ]
    if "dispers" in r or "burbu" in r:
        return [
            "Fracción de vacío por impedancia/capacitancia.",
            "Distribución de tamaños de burbuja (imagenología si aplica).",
            "ΔP/L con baja varianza.",
        ]
    return ["ΔP/L + varianza", "Holdup", "Observación visual"]

# =========================
# Sugerencia de VARIABLE a controlar (P o T) para CRUDO
# =========================
def control_variable_suggestion(regime: str, T_c: float | None, WAT_c: float | None) -> tuple[str, str]:
    """
    Devuelve ('Presión' o 'Temperatura', explicación breve) orientado a petróleo crudo.
    Criterio simple:
      - Slug/intermitente: priorizar Presión (back-pressure) para amortiguar slugging.
      - Anular: priorizar Temperatura (viscosidad/interfacial) y vigilar ΔP alto.
      - Estratificado: Presión (estabilidad) + si T < WAT, Temperatura por riesgo de cera.
      - Disperso/burbujeante: Temperatura si T≈WAT (reducir μ y evitar cera), si no Presión secundaria.
    """
    r = (regime or "").lower()

    def near_or_below_wat(T, WAT):
        if T is None or WAT is None: return False
        return T <= WAT + 2.0  # margen didáctico

    if "slug" in r or "intermit" in r:
        return ("Presión",
                "Aplicar control de back‑pressure / válvula de salida o estabilización de caudal para amortiguar la formación de slugs y reducir la varianza de ΔP.")
    if "anular" in r:
        return ("Temperatura",
                "Elevar T reduce μ del crudo y la tensión superficial, estabilizando la película y disminuyendo ΔP; mantener T bien por encima de la WAT para evitar cera/ensuciamiento.")
    if "estrat" in r:
        if near_or_below_wat(T_c, WAT_c):
            return ("Temperatura",
                    "Operar por encima de la WAT para bajar viscosidad y evitar deposición de cera; luego ajustar back‑pressure para mantener ΔP estable.")
        return ("Presión",
                "Mantener ΔP estable y bajo mediante control de back‑pressure; monitorear holdup para evitar inundación.")
    # Disperso / burbujeante
    if near_or_below_wat(T_c, WAT_c):
        return ("Temperatura",
                "Alejarse de la WAT reduce μ y el riesgo de cera; favorece atomización estable y menor ΔP.")
    return ("Presión",
            "Con patrones dispersos, priorizar estabilidad de ΔP y evitar oscilaciones; la temperatura queda como variable secundaria salvo proximidad a WAT.")

# =========================
# Portada Institucional (logo fijo ARRIBA)
# =========================
logo_path = Path("logoutn.png")
if logo_path.exists():
    st.image(str(logo_path), width=160)
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0;'>UNIVERSIDAD TECNOLÓGICA NACIONAL</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;'>Cátedra: Flujos Multifásicos</div>",
    unsafe_allow_html=True
)
st.markdown(
    """
<div style='text-align:center; margin-top:6px;'>
<b>Profesor:</b> Ezequiel Arturo Krumrick<br>
<b>Alumnos:</b> Catereniuc Federico / Rioseco Juan Manuel
</div>
""",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# Presentación didáctica
# =========================
with st.expander("📚 Qué hace la app y por qué es importante (crudo)", expanded=True):
    st.markdown(
        """
La app calcula **velocidades superficiales** \\(j_L, j_G\\) a partir de \\((Q_L, Q_G, D)\\), 
ubica los puntos en un **mapa tipo Mandhane** y sugiere el **régimen** esperado. 
Incluye recomendaciones de **qué variable controlar** (Presión o Temperatura) en contextos de **petróleo crudo**, 
considerando la **WAT** si la indicás.

**Relevancia:** conocer el régimen permite anticipar **ΔP**, **slugging**, **arrastre**, **humectación**, y **riesgo de cera**. 
Esto guía decisiones de **operación** (back‑pressure, calentamiento, aislamiento) y **medición** (ΔP, holdup, impedancia).
        """
    )

# =========================
# Sidebar: Parámetros de ducto y de crudo
# =========================
st.sidebar.header("Parámetros del ducto")
D = st.sidebar.number_input("Diámetro interno D [m]", min_value=0.001, value=0.10, step=0.001, format="%.3f")
unit = st.sidebar.selectbox("Unidades de Q", ["m³/h", "m³/s"], index=0)

st.sidebar.header("Propiedades de crudo (para recomendaciones)")
T_c = st.sidebar.number_input("Temperatura de operación T [°C]", value=25.0, step=0.5)
WAT_c = st.sidebar.number_input("WAT (Wax Appearance Temperature) [°C]", value=20.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Carga por CSV (opcional)**")
csv_example = "tag,QL,QG\nCaso 1,2.0,200.0\nCaso 2,5.0,50.0\nCaso 3,0.5,10.0"
st.sidebar.download_button("Descargar plantilla CSV", csv_example, file_name="plantilla_Q.csv", mime="text/csv")
uploaded = st.sidebar.file_uploader("Subí CSV (tag, QL, QG) con las mismas unidades elegidas", type=["csv"])

# =========================
# 1) Datos de entrada
# =========================
st.header("1) Datos de entrada")
st.markdown("Completá hasta **tres casos** (o subí CSV). Elegí unidades en la barra lateral.")

default_df = pd.DataFrame({
    "tag": ["Caso 1", "Caso 2", "Caso 3"],
    "QL": [2.0, 5.0, 0.5],
    "QG": [200.0, 50.0, 10.0],
})
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        df_in = df_in.rename(columns={c: c.strip() for c in df_in.columns})
        for col in ["tag", "QL", "QG"]:
            if col not in df_in.columns:
                st.error(f"Falta columna '{col}' en el CSV."); st.stop()
        df = df_in.copy()
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        df = default_df.copy()
else:
    df = default_df.copy()

df = st.data_editor(
    df, num_rows="fixed",
    column_config={
        "tag": st.column_config.TextColumn("Identificador"),
        "QL": st.column_config.NumberColumn(f"Q_L [{unit}]"),
        "QG": st.column_config.NumberColumn(f"Q_G [{unit}]"),
    }
)

# =========================
# 2) Cálculos y clasificación
# =========================
st.header("2) Cálculo de velocidades superficiales y régimen")
rows = []
for _, row in df.iterrows():
    tag = str(row.get("tag", "Caso"))
    QL_u = convertir_Q(row.get("QL", np.nan), unit)  # m3/s
    QG_u = convertir_Q(row.get("QG", np.nan), unit)  # m3/s
    vsl = superficial_velocity(QL_u, D)  # m/s
    vsg = superficial_velocity(QG_u, D)  # m/s
    regime = classify_mandhane(vsl, vsg)
    ctrl_var, ctrl_note = control_variable_suggestion(regime, T_c, WAT_c)
    rows.append({
        "tag": tag,
        "QL [m³/s]": QL_u,
        "QG [m³/s]": QG_u,
        "jL = Vsl [m/s]": vsl,
        "jG = Vsg [m/s]": vsg,
        "Régimen (estimado)": regime,
        "Control prioritario": ctrl_var,
        "Justificación control": ctrl_note
    })

res = pd.DataFrame(rows)
st.dataframe(res.style.format({
    "QL [m³/s]": "{:.6f}",
    "QG [m³/s]": "{:.6f}",
    "jL = Vsl [m/s]": "{:.4f}",
    "jG = Vsg [m/s]": "{:.4f}",
}))

st.download_button("⬇️ Descargar resultados (CSV)",
                   res.to_csv(index=False).encode("utf-8"),
                   file_name="resultados_mandhane_crudo.csv",
                   mime="text/csv")

# =========================
# 3) Mapa: puntos sobre imagen 'regimenes.png'
# =========================
st.header("3) Mapa de Mandhane con imagen de fondo")

def draw_points_over_image(points, img_path: str):
    # Límites reales del Mandhane de tu imagen
    x_min, x_max = 1e-2, 2e1   # VSG
    y_min, y_max = 1e-2, 3e0   # VSL
    
    fig, ax = plt.subplots(figsize=(7,6))
    
    # Leer imagen y mostrarla en coordenadas log–log
    img = plt.imread(img_path)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.imshow(
        img,
        extent=[x_min, x_max, y_min, y_max],
        aspect='auto',
        origin='upper',   # usa 'lower' si quieres invertir vertical
        zorder=0
    )
    
    # Etiquetas
    ax.set_xlabel(r"$V_{SG}$ [m/s]")
    ax.set_ylabel(r"$V_{SL}$ [m/s]")
    ax.set_title("Ubicación sobre mapa Mandhane")
    
    # Puntos
    for vsl, vsg, tag, regime in points:
        if np.isnan(vsl) or np.isnan(vsg): 
            continue
        ax.scatter(vsg, vsl, s=60, zorder=5)
        ax.annotate(f"{tag}: {regime}", xy=(vsg, vsl), xytext=(5,5),
                    textcoords="offset points", fontsize=9, zorder=6)
    
    ax.grid(False)
    return fig


# =========================
# 4) Validación + Variable de control sugerida
# =========================
st.header("4) Validación de campo y variable de control prioritaria (crudo)")
for _, r in res.iterrows():
    st.subheader(f"🔎 {r['tag']}: {r['Régimen (estimado)']}")
    st.markdown(f"**Variable prioritaria a controlar:** {r['Control prioritario']}")
    st.markdown(f"_Motivo:_ {r['Justificación control']}")
    st.markdown("**Mediciones sugeridas para validar:**")
    for tip in validation_suggestions(r["Régimen (estimado)"]):
        st.markdown(f"- {tip}")

st.markdown("---")
st.markdown(
    "Nota: Clasificador didáctico. Para proyectos con crudo real podemos incorporar límites digitalizados, "
    "Taitel–Dukler y correcciones por propiedades (ρ, μ, σ) y por WAT para mover fronteras de transición."
)


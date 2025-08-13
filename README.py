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
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except Exception:
        return np.nan
    return v/3600.0 if unit == "m³/h" else v

def superficial_velocity(Q_ms: float, D_m: float) -> float:
    if np.isnan(Q_ms) or np.isnan(D_m) or D_m <= 0:
        return np.nan
    return Q_ms / area_circular(D_m)

# =========================
# Clasificador Mandhane (didáctico)
# =========================
def classify_mandhane(vsl: float, vsg: float) -> str:
    if np.isnan(vsl) or np.isnan(vsg):
        return "—"
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
# Validación de campo
# =========================
def validation_suggestions(regime: str):
    r = (regime or "").lower()
    if "estrat" in r:
        return ["ΔP/L estable (baja varianza) con trending.",
                "Holdup/altura de lámina (capacitancia o gamma).",
                "Inspección visual/video (ondas interfaciales)."]
    if "slug" in r or "intermit" in r:
        return ["Oscilaciones periódicas de presión (FFT).",
                "Sondas de impedancia (frecuencia/longitud de tapones).",
                "Acelerometría/vibración de línea correlacionada."]
    if "anular" in r:
        return ["Espesor de película (conductancia/filmómetro).",
                "Medir entrainment; ΔP elevado.",
                "Sondas circumferenciales (humectación superior)."]
    if "dispers" in r or "burbu" in r:
        return ["Fracción de vacío (impedancia/capacitancia).",
                "Distribución de tamaños de burbuja (imagenología).",
                "ΔP/L con baja varianza."]
    return ["ΔP/L + varianza", "Holdup", "Observación visual"]

# =========================
# Variable de control prioritaria (crudo)
# =========================
def control_variable_suggestion(regime: str, T_c: float | None, WAT_c: float | None) -> tuple[str, str]:
    def near_or_below_wat(T, W):
        if T is None or W is None: return False
        return T <= W + 2.0  # margen didáctico
    r = (regime or "").lower()
    if "slug" in r or "intermit" in r:
        return ("Presión", "Back‑pressure/estabilización de caudal para amortiguar slugging y reducir varianza de ΔP.")
    if "anular" in r:
        return ("Temperatura", "Elevar T reduce μ y σ; estabiliza la película y baja ΔP. Mantener T > WAT para evitar cera.")
    if "estrat" in r:
        if near_or_below_wat(T_c, WAT_c):
            return ("Temperatura", "Operar por encima de WAT para bajar μ y evitar cera; luego ajustar back‑pressure.")
        return ("Presión", "Mantener ΔP estable/baja con back‑pressure y controlar holdup.")
    if near_or_below_wat(T_c, WAT_c):
        return ("Temperatura", "Alejar la operación de la WAT reduce μ y riesgo de cera; favorece atomización.")
    return ("Presión", "Priorizar estabilidad de ΔP; Temperatura secundaria salvo proximidad a WAT.")

# =========================
# Portada UTN (logo fijo ARRIBA)
# =========================
logo_path = Path("logoutn.png")
if logo_path.exists():
    st.image(str(logo_path), width=160)
st.markdown("<h2 style='text-align:center;margin-bottom:0;'>UNIVERSIDAD TECNOLÓGICA NACIONAL</h2>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Cátedra: Flujos Multifásicos</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; margin-top:6px;'><b>Profesor:</b> Ezequiel Arturo Krumrick<br>"
    "<b>Alumnos:</b> Catereniuc Federico / Rioseco Juan Manuel</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# Presentación didáctica
# =========================
with st.expander("📚 ¿Qué hace la app y por qué importa (crudo)?", expanded=True):
    st.markdown(
        """
Calcula **j_L, j_G** a partir de \\((Q_L, Q_G, D)\\), ubica los puntos en **Mandhane** (fondo real) y sugiere el **régimen**.
Incluye recomendación de **qué variable controlar** (Presión/Temperatura) considerando **WAT** del crudo.
        """
    )

# =========================
# Sidebar: Ducto & crudo
# =========================
st.sidebar.header("Parámetros del ducto")
D = st.sidebar.number_input("Diámetro interno D [m]", min_value=0.001, value=0.10, step=0.001, format="%.3f")
unit = st.sidebar.selectbox("Unidades de Q", ["m³/h", "m³/s"], index=0)

st.sidebar.header("Propiedades de crudo (para recomendación)")
T_c = st.sidebar.number_input("Temperatura de operación T [°C]", value=25.0, step=0.5)
WAT_c = st.sidebar.number_input("WAT [°C]", value=20.0, step=0.5)

# >>> NUEVO: Calibración del fondo <<<
st.sidebar.header("Calibración del mapa (si no coincide)")
rotate90 = st.sidebar.checkbox("Rotar 90° (si la imagen está 'parada')", value=False)
origin_upper = st.sidebar.checkbox("Origin arriba (True) / abajo (False)", value=True)
st.sidebar.caption("Recorte de márgenes (fracción de ancho/alto).")
left_f  = st.sidebar.slider("Recorte izquierdo", 0.0, 0.4, 0.08, 0.005)
right_f = st.sidebar.slider("Recorte derecho",  0.0, 0.4, 0.03, 0.005)
top_f   = st.sidebar.slider("Recorte superior", 0.0, 0.4, 0.05, 0.005)
bot_f   = st.sidebar.slider("Recorte inferior", 0.0, 0.4, 0.12, 0.005)

st.sidebar.caption("Rango de ejes Mandhane")
x_min = float(st.sidebar.text_input("VSG min", "1e-2"))
x_max = float(st.sidebar.text_input("VSG max", "2e1"))
y_min = float(st.sidebar.text_input("VSL min", "1e-2"))
y_max = float(st.sidebar.text_input("VSL max", "3e0"))

# =========================
# 1) Datos de entrada
# =========================
st.header("1) Datos de entrada")
default_df = pd.DataFrame({
    "tag": ["Caso 1", "Caso 2", "Caso 3"],
    "QL": [2.0, 5.0, 0.5],
    "QG": [200.0, 50.0, 10.0],
})
uploaded = st.sidebar.file_uploader("Subí CSV (tag, QL, QG) con las mismas unidades elegidas", type=["csv"])
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
# 2) Cálculos + régimen + control recomendado
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
# 3) Mapa: puntos sobre imagen 'regimenes.png' (con recorte)
# =========================
st.header("3) Mapa de Mandhane con imagen de fondo")

def load_and_crop(path_str: str, rotate90: bool,
                  left_f: float, right_f: float, top_f: float, bot_f: float) -> np.ndarray | None:
    p = Path(path_str)
    if not p.exists():
        return None
    img = plt.imread(str(p))
    if rotate90:
        img = np.rot90(img, k=1)
    h, w = img.shape[0], img.shape[1]
    l = int(w * left_f)
    r = w - int(w * right_f)
    t = int(h * top_f)
    b = h - int(h * bot_f)
    l = max(0, min(l, w - 2)); r = max(l + 1, min(r, w))
    t = max(0, min(t, h - 2)); b = max(t + 1, min(b, h))
    return img[t:b, l:r]

def draw_points_over_image(points, img: np.ndarray,
                           x_min: float, x_max: float, y_min: float, y_max: float,
                           origin_upper: bool):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
    ax.imshow(img,
              extent=[x_min, x_max, y_min, y_max],
              aspect='auto',
              origin='upper' if origin_upper else 'lower',
              zorder=0)
    ax.set_xlabel(r"$V_{SG}$ [m/s]")
    ax.set_ylabel(r"$V_{SL}$ [m/s]")
    ax.set_title("Ubicación sobre mapa de régimen (imagen calibrada)")
    for vsl, vsg, tag, regime in points:
        if np.isnan(vsl) or np.isnan(vsg): 
            continue
        ax.scatter(vsg, vsl, s=60, zorder=5)
        ax.annotate(f"{tag}: {regime}", xy=(vsg, vsl), xytext=(5,5),
                    textcoords="offset points", fontsize=9, zorder=6)
    ax.grid(False)
    return fig

bg_img = load_and_crop("regimenes.png", rotate90, left_f, right_f, top_f, bot_f)
if bg_img is None:
    st.error("No se encontró 'regimenes.png' en la raíz del repo. Subilo con ese nombre.")
else:
    pts = [(r["jL = Vsl [m/s]"], r["jG = Vsg [m/s]"], r["tag"], r["Régimen (estimado)"])
           for _, r in res.iterrows()]
    fig = draw_points_over_image(pts, bg_img, x_min, x_max, y_min, y_max, origin_upper)
    st.pyplot(fig, use_container_width=True)

# =========================
# 4) Validación + variable de control
# =========================
st.header("4) Validación de campo y variable de control prioritaria (crudo)")
for _, r in res.iterrows():
    st.subheader(f"🔎 {r['tag']}: {r['Régimen (estimado)']}")
    st.markdown(f"**Variable prioritaria:** {r['Control prioritario']}")
    st.markdown(f"_Motivo:_ {r['Justificación control']}")
    st.markdown("**Mediciones sugeridas para validar:**")
    for tip in validation_suggestions(r["Régimen (estimado)"]):
        st.markdown(f"- {tip}")

st.markdown("---")
st.caption("Clasificador didáctico. Para trabajo profesional: límites digitalizados, Taitel–Dukler y correcciones por propiedades (ρ, μ, σ) y WAT.")


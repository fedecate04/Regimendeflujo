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
# Clasificador Mandhane (didáctico, aprox. aire-agua horizontal)
# =========================
def classify_mandhane(vsl: float, vsg: float) -> str:
    if np.isnan(vsl) or np.isnan(vsg):
        return "—"
    vsl_c = max(vsl, 1e-6); vsg_c = max(vsg, 1e-6)
    L = math.log10(vsl_c); G = math.log10(vsg_c)

    # Zonas aproximadas coherentes con el mapa dibujado abajo
    if G < -0.3 and L < -0.3:
        return "Estratificado"
    if (-0.6 <= L <= 0.2) and (-0.6 <= G <= 0.6) and (G >= L - 0.5):
        return "Intermitente / Slug"
    if G > 0.6 and (L < 0.3):
        return "Anular / Niebla anular"
    if L > 0.2 or (L > -0.1 and G > 0.2):
        return "Burbujas dispersas"
    return "Ondas / Intermedio"

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
    if "anular" in r or "niebla" in r:
        return ["Espesor de película (conductancia/filmómetro).",
                "Medir entrainment; ΔP elevado.",
                "Sondas circumferenciales (humectación superior)."]
    if "burbu" in r or "dispers" in r:
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
    if "anular" in r or "niebla" in r:
        return ("Temperatura", "Elevar T reduce μ y σ; estabiliza la película y baja ΔP. Mantener T > WAT para evitar cera.")
    if "estrat" in r:
        if near_or_below_wat(T_c, WAT_c):
            return ("Temperatura", "Operar por encima de WAT para bajar μ y evitar cera; luego ajustar back‑pressure.")
        return ("Presión", "Mantener ΔP estable/baja con back‑pressure y controlar holdup.")
    if near_or_below_wat(T_c, WAT_c):
        return ("Temperatura", "Alejar la operación de la WAT reduce μ y riesgo de cera; favorece atomización.")
    return ("Presión", "Priorizar estabilidad de ΔP; Temperatura secundaria salvo proximidad a WAT.")

# ===== Portada UTN con logo (robusta) =====
from pathlib import Path

def find_logo():
    """
    Busca el logo 'logoutn' en la raíz del repo admitiendo varias extensiones.
    Devuelve la ruta como string o None si no existe.
    """
    # nombres candidatos en raíz (podés agregar más rutas si lo guardas en /assets, /img, etc.)
    candidates = [
        Path("logoutn"),                   # por si lo subiste sin extensión (poco común)
        Path("logoutn.png"),
        Path("logoutn.jpg"),
        Path("logoutn.jpeg"),
        Path("logoutn.svg"),
        Path("logoutn.webp"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

logo_path = find_logo()

# Encabezado institucional
if logo_path:
    # centrar el logo usando columnas
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(logo_path, use_column_width=False, width=180)  # ajustá el ancho si querés
else:
    st.warning("No se encontró el archivo de logo 'logoutn.(png|jpg|svg|...)' en la raíz del repo.")

st.markdown(
    "<h2 style='text-align:center;margin:6px 0 0 0;'>UNIVERSIDAD TECNOLÓGICA NACIONAL</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center; font-size:16px;'>Cátedra: Flujos Multifásicos</div>",
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
with st.expander("📚 ¿Qué hace la app y por qué importa (crudo)?", expanded=True):
    st.markdown(
        """
Calcula **j_L, j_G** a partir de \\((Q_L, Q_G, D)\\), ubica los puntos en un **mapa tipo Mandhane** generado
programáticamente y sugiere el **régimen**. Incluye recomendación de **qué variable controlar** (Presión/Temperatura)
considerando **WAT** del crudo.  
El mapa y las fronteras son **aproximadas didácticas**, coherentes con literatura clásica.
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
# 3) Mapa tipo Mandhane (generado por código)
# =========================
st.header("3) Mapa tipo Mandhane (programático)")

def mandhane_boundaries():
    """
    Genera fronteras aproximadas parecidas a la figura de referencia.
    Devuelve dict con arrays (x=VSG, y=VSL) para distintas curvas.
    """
    # Rango de ejes
    x = np.logspace(-2, 1.3, 300)  # VSG: 1e-2 → ~20
    # Curva azul central (transición estratificado/ondas ↔ slug ↔ burbujas dispersas)
    # Construimos por tramos para imitar la forma:
    y1 = []
    for xi in x:
        if xi < 0.7:
            # tramo izquierdo casi vertical hasta ~0.35 m/s
            y1.append(0.35 * (xi/0.7)**(-0.02))
        elif xi < 1.0:
            # codo hacia abajo
            y1.append(0.35 * (xi/0.7)**(-1.6))
        elif xi < 3.0:
            # tramo inclinado
            y1.append(0.35 * (xi/1.0)**(-1.2))
        else:
            # baja fuerte hasta 0.01 en ~10
            y1.append(0.12 * (xi/3.0)**(-2.2))
    y1 = np.array(y1)
    # Curva naranja derecha (hacia niebla anular)
    x2 = np.logspace(0.2, 1.3, 120)  # ~1.6 a 20
    y2 = 0.08 + 0.6*(x2/2.0)**0.8   # sube con VSG

    # Líneas horizontales (como tu imagen)
    y_h1 = 0.10  # línea púrpura (~0.1 m/s)
    y_h2 = 0.20  # línea amarilla (~0.2 m/s)

    return {
        "x": x, "y1": y1,
        "x2": x2, "y2": y2,
        "y_h1": y_h1, "y_h2": y_h2
    }

def draw_mandhane(points):
    b = mandhane_boundaries()

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim([1e-2, 2e1]); ax.set_ylim([1e-2, 3e0])

    # Región gris tenue (muy bajos VSL/VSG)
    ax.fill_between([1e-2, 2e1], 1e-2, 2e-2, color='0.85', alpha=0.6, linewidth=0)

    # Curvas
    ax.plot(b["x"], b["y1"], color="#1f77b4", linewidth=2.5)          # azul
    ax.plot(b["x2"], b["y2"], color="#ff7f0e", linewidth=2.5, ls="--") # naranja
    ax.hlines(b["y_h1"], 1e-2, 2e1, colors="#6a3d9a", linestyles="-.", linewidth=2)  # púrpura
    ax.hlines(b["y_h2"], 1e-2, 2e1, colors="#b39b00", linestyles="--", linewidth=2)  # amarilla

    # Etiquetas de zonas (posiciones elegidas para que no molesten)
    ax.text(0.015, 0.018, "Estratificado", fontsize=10, alpha=0.8)
    ax.text(0.05, 0.5, "Burbuja\nelongada", fontsize=10, alpha=0.8)
    ax.text(0.8, 1.8, "Burbujas dispersas", fontsize=10, alpha=0.8)
    ax.text(2.2, 0.8, "Tapón", fontsize=10, alpha=0.8)
    ax.text(6.5, 0.06, "Ondas", fontsize=10, alpha=0.8)
    ax.text(12, 0.18, "Niebla anular", fontsize=10, alpha=0.8)

    # Puntos del usuario (x=VSG, y=VSL)
    for vsl, vsg, tag, regime in points:
        if np.isnan(vsl) or np.isnan(vsg): 
            continue
        ax.scatter(vsg, vsl, s=70, zorder=5)
        ax.annotate(f"{tag}: {regime}", xy=(vsg, vsl), xytext=(5,5),
                    textcoords="offset points", fontsize=9, zorder=6)

    ax.set_xlabel(r"$V_{SG}$  [m/s]")
    ax.set_ylabel(r"$V_{SL}$  [m/s]")
    ax.set_title("Mapa tipo Mandhane (horizontal) — versión didáctica generada por código")
    ax.grid(True, which='both', alpha=0.2)
    return fig

points = [(r["jL = Vsl [m/s]"], r["jG = Vsg [m/s]"], r["tag"], r["Régimen (estimado)"]) 
          for _, r in res.iterrows()]
fig = draw_mandhane(points)
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
st.caption("Clasificador y fronteras aproximadas para práctica. Para uso profesional: límites digitalizados, Taitel–Dukler y correcciones por propiedades (ρ, μ, σ) y WAT.")

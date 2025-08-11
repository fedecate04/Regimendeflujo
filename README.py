import math
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Utilidades de ingeniería
# =========================

def area_circular(D_m):
    return math.pi * (D_m**2) / 4.0

def convertir_Q(value, unit):
    # Admite m3/h y m3/s -> devuelve m3/s
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except:
        return np.nan
    return v/3600.0 if unit == "m³/h" else v

def superficial_velocity(Q_ms, D_m):
    if np.isnan(Q_ms) or np.isnan(D_m) or D_m <= 0:
        return np.nan
    A = area_circular(D_m)
    return Q_ms / A

# =========================
# Clasificador Mandhane "didáctico"
# (aproximado para horizontal aire-agua)
# =========================
# Basado en reglas simples sobre Vsl, Vsg (m/s) en escala logarítmica.
# Sirve para ubicar y entrenar criterio; no reemplaza límites oficiales.
def classify_mandhane(vsl, vsg):
    if np.isnan(vsl) or np.isnan(vsg):
        return "—"
    # Evitar log de cero:
    vsl_c = max(vsl, 1e-6)
    vsg_c = max(vsg, 1e-6)
    L = math.log10(vsl_c)
    G = math.log10(vsg_c)

    # Reglas aproximadas (pueden refinarse):
    # 1) Estratificado: Vsg bajo y Vsl bajo
    if G < -0.3 and L < -0.3:
        return "Estratificado"

    # 2) Intermitente/Slug: Vsg medio y Vsl bajo-medio
    if (-0.6 <= L <= 0.2) and (-0.6 <= G <= 0.6) and (G >= L - 0.5):
        return "Intermitente / Slug"

    # 3) Anular: Vsg alto y Vsl bajo a medio
    if G > 0.5 and L < 0.3:
        return "Anular"

    # 4) Disperso (burbujeante / disperso): Vsl alto y/o ambos moderados-altos
    if L > 0.2 or (L > -0.1 and G > 0.2):
        return "Disperso"

    # Fallback: elegir por predominio gas/líquido
    return "Intermitente / Slug" if G >= L else "Estratificado"

# =========================================
# Sugerencias de validación de campo
# =========================================
def validation_suggestions(regime):
    regime = (regime or "").lower()
    if "estrat" in regime:
        return [
            "Perfil de presión (ΔP/L) estable y bajo; registrar varianza de dP.",
            "Medición de holdup/altura de lámina (Gamma densitometría o capacitancia).",
            "Inspección visual/video endoscópico: ondas interfaciales gravitacionales.",
        ]
    if "slug" in regime or "intermit" in regime:
        return [
            "Oscilaciones periódicas de presión (FFT/frecuencia de slugs).",
            "Sondas de impedancia para longitud/frecuencia de tapones.",
            "Acelerometría de línea (vibración) correlacionada con pasaje de slugs.",
        ]
    if "anular" in regime:
        return [
            "Espesor de película en pared (anillos de conductancia o filmómetro).",
            "Fracción de arrastre (entrainment) y caída de presión elevada.",
            "Sondas ópticas/capacitivas circumferenciales para humectación superior.",
        ]
    if "dispers" in regime or "burbu" in regime:
        return [
            "Fracción de vacío (void fraction) por impedancia/capacitancia.",
            "Distribución de tamaños de burbuja (PIV/imagenología si aplica).",
            "Perfil de presión con baja varianza temporal.",
        ]
    return [
        "Perfil de presión con análisis de varianza/FFT.",
        "Alguna técnica de holdup (capacitancia, gamma) según disponibilidad.",
        "Observación visual/filmación si la línea de prueba lo permite.",
    ]

# =========================
# Gráfico tipo Mandhane
# =========================
def draw_mandhane_style(points):
    fig, ax = plt.subplots(figsize=(7,6))

    # Rango típico de trabajo (m/s)
    vmin, vmax = 1e-2, 30
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])

    ax.set_xlabel(r"$V_{sl}$  (m/s)")
    ax.set_ylabel(r"$V_{sg}$  (m/s)")
    ax.set_title("Mapa tipo Mandhane (horizontal) — aproximado para práctica")

    # Dibujar zonas aproximadas (polígonos)
    # Estratificado (bajo-bajo)
    strat_poly = np.array([
        [1e-2, 1e-2], [2e-1, 1e-2], [2e-1, 5e-1], [1e-2, 5e-1]
    ])
    ax.fill(strat_poly[:,0], strat_poly[:,1], alpha=0.08, label="Estratificado")

    # Intermitente/Slug (zona media)
    slug_poly = np.array([
        [5e-2, 5e-2], [2e-1, 2e0], [8e-1, 1e0], [3e-1, 2e-2], [5e-2, 3e-2]
    ])
    ax.fill(slug_poly[:,0], slug_poly[:,1], alpha=0.08, label="Intermitente/Slug")

    # Anular (Vsg alto)
    ann_poly = np.array([
        [1e-2, 3e0], [3e-1, 3e0], [3e-1, 3e1], [1e-2, 3e1]
    ])
    ax.fill(ann_poly[:,0], ann_poly[:,1], alpha=0.08, label="Anular")

    # Disperso (Vsl alto y/o altos ambos)
    disp_poly = np.array([
        [2e-1, 1e-2], [3e1, 1e-2], [3e1, 3e1], [2e-1, 3e1]
    ])
    ax.fill(disp_poly[:,0], disp_poly[:,1], alpha=0.08, label="Disperso")

    # Puntos del usuario
    for i, p in enumerate(points, start=1):
        vsl, vsg, tag, regime = p
        if np.isnan(vsl) or np.isnan(vsg):
            continue
        ax.scatter(vsl, vsg, s=60)
        ax.annotate(f"{tag}: {regime}",
                    xy=(vsl, vsg), xytext=(5,5),
                    textcoords="offset points", fontsize=9)

    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    return fig

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Mapa de Regímenes — Mandhane (Didáctico)",
                   page_icon="🌊", layout="wide")

st.title("🌊 Flujos Multifásicos — Ubicación en Mandhane y Validación de Campo")
st.caption("Cálculo de velocidades superficiales, ubicación en mapa tipo Mandhane (horizontal), "
           "clasificación de régimen y sugerencias de validación de campo.")

with st.expander("📚 ¿Qué hace esta aplicación? (presentación)", expanded=True):
    st.markdown("""
Esta herramienta te permite:
- **Ingresar hasta 3 pares** de caudales \\((Q_L, Q_G)\\) para un **ducto horizontal** de diámetro \\(D\\).
- Calcular **velocidades superficiales** \\(j_L = V_{sl}\\) y \\(j_G = V_{sg}\\).
- **Ubicar los puntos** en un **mapa tipo Mandhane** (ejes log–log de \\(V_{sl}\\) vs \\(V_{sg}\\)).
- **Estimar el régimen** de flujo (Estratificado, Intermitente/Slug, Anular, Disperso) con un **clasificador aproximado**.
- Sugerir **mediciones de campo** para **validar** la selección del régimen.

> Nota técnica: El mapa y clasificador incluidos son **didácticos** (aire–agua horizontal).
> Para trabajo profesional con otros fluidos/condiciones, podemos **refinar límites** o 
> implementar **Taitel–Dukler** u otro modelo mecanístico.
""")

st.sidebar.header("Parámetros")
D = st.sidebar.number_input("Diámetro interno D [m]", min_value=0.001, value=0.10, step=0.001, format="%.3f")
unit = st.sidebar.selectbox("Unidades de Q", ["m³/h", "m³/s"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Carga por CSV (opcional)**")
csv_example = "tag,QL,QG\nCaso 1,2.0,200.0\nCaso 2,5.0,50.0\nCaso 3,0.5,10.0"
st.sidebar.download_button("Descargar plantilla CSV", csv_example, file_name="plantilla_Q.csv", mime="text/csv")
uploaded = st.sidebar.file_uploader("Subí un CSV (tag, QL, QG) con las mismas unidades seleccionadas", type=["csv"])

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
        # normalizar columnas
        df_in = df_in.rename(columns={c: c.strip() for c in df_in.columns})
        for col in ["tag", "QL", "QG"]:
            if col not in df_in.columns:
                st.error(f"Falta columna '{col}' en el CSV.")
                st.stop()
        df = df_in.copy()
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        df = default_df.copy()
else:
    df = default_df.copy()

df = st.data_editor(df, num_rows="fixed",
                    column_config={
                        "tag": st.column_config.TextColumn("Identificador"),
                        "QL": st.column_config.NumberColumn(f"Q_L [{unit}]"),
                        "QG": st.column_config.NumberColumn(f"Q_G [{unit}]"),
                    })

st.header("2) Cálculo de velocidades superficiales")
rows = []
for _, row in df.iterrows():
    tag = str(row.get("tag", "Caso"))
    QL_u = convertir_Q(row.get("QL", np.nan), unit)  # m3/s
    QG_u = convertir_Q(row.get("QG", np.nan), unit)  # m3/s
    vsl = superficial_velocity(QL_u, D)  # m/s
    vsg = superficial_velocity(QG_u, D)  # m/s
    regime = classify_mandhane(vsl, vsg)
    rows.append({
        "tag": tag,
        "QL [m³/s]": QL_u,
        "QG [m³/s]": QG_u,
        "jL = Vsl [m/s]": vsl,
        "jG = Vsg [m/s]": vsg,
        "Régimen (estimado)": regime
    })

res = pd.DataFrame(rows)
st.dataframe(res.style.format({
    "QL [m³/s]": "{:.6f}",
    "QG [m³/s]": "{:.6f}",
    "jL = Vsl [m/s]": "{:.4f}",
    "jG = Vsg [m/s]": "{:.4f}",
}))

csv_out = res.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resultados (CSV)", csv_out, file_name="resultados_mandhane.csv", mime="text/csv")

st.header("3) Mapa tipo Mandhane (log–log)")
points = [(r["jL = Vsl [m/s]"], r["jG = Vsg [m/s]"], r["tag"], r["Régimen (estimado)"]) for _, r in res.iterrows()]
fig = draw_mandhane_style(points)
st.pyplot(fig, use_container_width=True)

st.header("4) Validación de campo sugerida")
for _, r in res.iterrows():
    st.subheader(f"🔎 {r['tag']}: {r['Régimen (estimado)']}")
    tips = validation_suggestions(r["Régimen (estimado)"])
    st.markdown("\n".join([f"- {t}" for t in tips]))

st.markdown("---")
st.markdown("""
### Notas y próximos pasos
- Este clasificador es **aproximado** para fines **didácticos**.  
- Si querés **refinar límites** (p. ej., digitalizar curvas de Mandhane o implementar **Taitel–Dukler**), lo integramos.
- Podemos agregar **propiedades de fluido** y **correcciones** para otras combinaciones distintas de aire–agua.
""")

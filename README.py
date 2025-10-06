# app.py — Modelo 1D segmentado (GL) con criterios de corte, explicación y reporte PDF
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# ====== PDF backends (reportlab -> preferido; fpdf2 -> fallback) ======
REPORT_BACKEND = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import cm
    REPORT_BACKEND = "reportlab"
except Exception:
    try:
        from fpdf import FPDF  # pip install fpdf2
        REPORT_BACKEND = "fpdf"
    except Exception:
        REPORT_BACKEND = None

# --------- Modelos físicos básicos ---------
def churchill_f(Re, eps_over_D):
    Re = max(Re, 1.0)
    A = (2.457*np.log((7/Re)**0.9 + 0.27*eps_over_D))**16
    B = (37530/Re)**16
    f = 8 * ( (8/Re)**12 + 1/((A + B)**1.5) )**(1/12)
    return f  # Darcy

def rho_gas_ideal(rho_ref, P, T, P_ref, T_ref):
    # z = 1 (gas ideal); isotermo si T se mantiene constante
    return rho_ref * (P/P_ref) * (T_ref/T)

def mixture_props(rhoL, muL, rhoG, muG, HL, a=0.6, b=0.4):
    rhoM = HL*rhoL + (1-HL)*rhoG
    muM  = a*muL + b*muG
    return rhoM, muM

# ---- Hold-up según modo ----
def holdup_from_mode(mode, QL, QG, mL, mG, rhoL, rhoG, HL_user):
    if mode == "Ingresado":
        return float(np.clip(HL_user, 0.01, 0.99))
    elif mode == "No-slip volumétrico":
        denom = QL + QG
        return float(np.clip(QL/denom if denom > 0 else 0.5, 0.01, 0.99))
    elif mode == "Desde calidad másica (homogéneo)":
        mtot = mL + mG
        x = mG/mtot if mtot > 0 else 0.5  # calidad másica gas
        alpha = (x/rhoG) / ((x/rhoG) + ((1-x)/rhoL))  # fracción volumétrica de gas
        HL = 1 - alpha
        return float(np.clip(HL, 0.01, 0.99))
    else:
        return float(np.clip(HL_user, 0.01, 0.99))

# ---- Límite de tamaño de tramo por criterios ----
def step_size_limits(Pi, g, vSG_i, A, QG_i, max_dpfrac, max_dvfrac, iters=24):
    # Límite por % de caída de presión: Δx_Δp = (γ_p * p_i) / g
    dx_p = max_dpfrac * Pi / max(g, 1e-12)

    # Límite por % de cambio de vSG (compresibilidad ideal) — bisección
    lo, hi = 0.0, max(dx_p*10, 1e-6)
    vSG0 = vSG_i
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        P2 = Pi - g*mid
        if P2 <= 1.0:  # evita presiones no físicas
            hi = mid
            continue
        QG2 = QG_i * (Pi/P2)  # ideal isotermo
        vSG2 = QG2 / A
        dvfrac = abs(vSG2 - vSG0)/max(vSG0, 1e-12)
        if dvfrac > max_dvfrac:
            hi = mid
        else:
            lo = mid
    dx_v = lo
    return dx_p, dx_v

# ---- Núcleo iterativo por tramos ----
def solve_pipe(D, Ltot, eps, Pin, T,
               rhoL, muL, muG, rhoG_ref, P_ref, T_ref,
               a_mu=0.6, b_mu=0.4,
               use_mass=False, QL=5e-4, QG0=2e-3, mL=0.0, mG=0.0,
               HL_mode="Ingresado", HL_user=0.6,
               max_dp_pct=3.0, max_dv_pct=5.0, safety=0.95):

    A = np.pi*D**2/4
    # Si el usuario ingresa másicos, construimos Q a Pin
    if use_mass:
        rhoG0 = rho_gas_ideal(rhoG_ref, Pin, T, P_ref, T_ref)
        QL = mL / max(rhoL, 1e-12)
        QG0 = mG / max(rhoG0, 1e-12)

    vSL = QL / A

    rows, i = [], 1
    x, Lrem, Pi = 0.0, Ltot, Pin
    QG_i = QG0
    vSG_i = QG_i / A

    while Lrem > 1e-9 and Pi > 2e3:  # corta si la presión cae demasiado
        rhoG_i = rho_gas_ideal(rhoG_ref, Pi, T, P_ref, T_ref)

        HL_i = holdup_from_mode(
            HL_mode, QL, QG_i,
            mL if use_mass else rhoL*QL,
            mG if use_mass else rhoG_i*QG_i,
            rhoL, rhoG_i, HL_user
        )

        rhoM_i, muM_i = mixture_props(rhoL, muL, rhoG_i, muG, HL_i, a_mu, b_mu)
        vM_i = vSL + vSG_i
        Re_i = rhoM_i * vM_i * D / max(muM_i, 1e-12)
        f_i = churchill_f(Re_i, eps/D)
        g_i = 2 * f_i * rhoM_i * vM_i**2 / D  # Pa/m

        dx_p, dx_v = step_size_limits(
            Pi, g_i, vSG_i, A, QG_i,
            max_dp_pct/100.0, max_dv_pct/100.0
        )

        dx = min(Lrem, safety*min(dx_p, dx_v))
        if dx <= 0:
            break

        P2 = Pi - g_i*dx
        QG_2 = QG_i * (Pi/max(P2, 1.0))
        vSG_2 = QG_2 / A

        dp = Pi - P2
        dpfrac = dp / max(Pi, 1e-12)
        dvfrac = abs(vSG_2 - vSG_i) / max(vSG_i, 1e-12)

        rows.append({
            "i": i, "x0[m]": x, "x1[m]": x+dx, "dx[m]": dx,
            "dx_p_lim[m]": dx_p, "dx_v_lim[m]": dx_v, "limit": "Δp" if dx_p <= dx_v else "ΔvSG",
            "Pin[bar]": Pi/1e5, "Pout[bar]": P2/1e5, "dp[bar]": dp/1e5, "dp/p[%]": 100*dpfrac,
            "HL[-]": HL_i, "rhoG[kg/m3]": rhoG_i, "rhoM[kg/m3]": rhoM_i,
            "muM[mPa·s]": 1e3*muM_i, "Re[-]": Re_i, "f_Darcy[-]": f_i,
            "g[Pa/m]": g_i, "vSL[m/s]": vSL, "vSG_in[m/s]": vSG_i, "vSG_out[m/s]": vSG_2,
            "ΔvSG/vSG[%]": 100*dvfrac
        })

        x += dx; Lrem -= dx; i += 1
        Pi = P2; QG_i = QG_2; vSG_i = vSG_2

    return pd.DataFrame(rows)

# ----- Gráfico Matplotlib P vs L (para ver y para PDF) -----
def make_pressure_plot(df):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
    ax.plot(df["x1[m]"], df["Pout[bar]"], lw=2, color="#0E6BA8")
    ax.set_xlabel("Longitud L [m]")
    ax.set_ylabel("Presión [bar]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Perfil de presión a lo largo del ducto")
    fig.tight_layout()
    return fig

# ----- Generación del PDF (usa reportlab o fpdf2 según disponibilidad) -----
def build_pdf(df, fig, meta):
    if REPORT_BACKEND == "reportlab":
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        W, H = A4

        y = H - 2*cm
        # Logo
        if meta.get("logo_path"):
            try:
                c.drawImage(ImageReader(meta["logo_path"]), 1.5*cm, y-2.2*cm,
                            width=4.0*cm, height=2.2*cm, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        # Encabezado
        c.setFont("Helvetica-Bold", 14); c.drawString(6.0*cm, y, meta.get("titulo",""))
        c.setFont("Helvetica", 11)
        c.drawString(6.0*cm, y-0.8*cm, f"Cátedra: {meta.get('catedra','')}")
        c.drawString(6.0*cm, y-1.5*cm, f"Profesor: {meta.get('profesor','')}")
        c.drawString(6.0*cm, y-2.2*cm, f"Año: {meta.get('anio','')}")
        c.drawRightString(W-1.5*cm, y-2.2*cm, datetime.now().strftime("Fecha y hora: %Y-%m-%d %H:%M"))
        c.line(1.5*cm, y-2.5*cm, W-1.5*cm, y-2.5*cm)

        # Resumen
        y2 = y - 3.2*cm
        c.setFont("Helvetica-Bold", 12); c.drawString(1.5*cm, y2, "Resumen del cálculo")
        c.setFont("Helvetica", 11)
        c.drawString(1.5*cm, y2-0.7*cm, f"Tramos calculados: {meta['ntramos']}")
        c.drawString(1.5*cm, y2-1.3*cm, f"P_in: {meta['pin_bar']:.2f} bar")
        c.drawString(1.5*cm, y2-1.9*cm, f"P_out: {meta['pout_bar']:.2f} bar")
        c.drawString(1.5*cm, y2-2.5*cm, f"Δp total: {meta['dptot_bar']:.2f} bar")

        # Gráfico
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=160, bbox_inches="tight"); img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 1.5*cm, 3.0*cm, width=W-3.0*cm, height=8.0*cm,
                    preserveAspectRatio=True, mask='auto')

        c.setFont("Helvetica-Oblique", 9)
        c.drawRightString(W-1.5*cm, 1.5*cm, "Generado automáticamente por la app Streamlit — Flujos Multifásicos (GL)")
        c.showPage(); c.save(); buffer.seek(0)
        return buffer.read()

    elif REPORT_BACKEND == "fpdf":
        pdf = FPDF(format='A4', unit='mm')
        pdf.add_page()

        # Logo
        try:
            pdf.image(meta.get("logo_path",""), x=10, y=10, w=35)
        except Exception:
            pass

        # Encabezado
        pdf.set_xy(55, 12); pdf.set_font('Helvetica', 'B', 14)
        pdf.multi_cell(0, 7, meta.get("titulo",""))
        pdf.set_xy(55, 26); pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 6,
            f"Cátedra: {meta.get('catedra','')}\n"
            f"Profesor: {meta.get('profesor','')}\n"
            f"Año: {meta.get('anio','')}\n"
            f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        # Resumen
        pdf.set_xy(10, 60); pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 7, "Resumen del cálculo", ln=1)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 6, f"Tramos calculados: {meta['ntramos']}", ln=1)
        pdf.cell(0, 6, f"P_in: {meta['pin_bar']:.2f} bar", ln=1)
        pdf.cell(0, 6, f"P_out: {meta['pout_bar']:.2f} bar", ln=1)
        pdf.cell(0, 6, f"Δp total: {meta['dptot_bar']:.2f} bar", ln=1)

        # Gráfico
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', dpi=160, bbox_inches='tight'); img_buf.seek(0)
        pdf.image(img_buf, x=10, y=105, w=190)

        pdf.set_y(285); pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 6, "Generado automáticamente por la app Streamlit — Flujos Multifásicos (GL)", align='R')

        return pdf.output(dest='S').encode('latin1')

    else:
        raise RuntimeError("No hay backend de PDF disponible. Instala 'reportlab' o 'fpdf2'.")

# --------- Interfaz Streamlit ---------
def main():
    st.set_page_config(page_title="Flujos Multifásicos — Segmentación Δp (Gas-Líquido)", layout="wide")

    # ----- Encabezado institucional (sin warning) -----
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        st.image("logoutn.png", caption=None, use_container_width=True)
    with col_title:
        st.markdown(
            """
            ### Cátedra: **FLUJOS MULTIFÁSICOS EN LA INDUSTRIA DEL PETRÓLEO**  
            **Profesor:** Ezequiel Arturo Krumrick  
            **Año:** 2025
            """
        )

    st.title("Modelo 1D segmentado con criterios de corte (Gas–Líquido)")

    # ----- Introducción (texto plano) -----
    st.markdown(
        """
**Introducción.** En líneas multifásicas, la caída de presión impacta el patrón de flujo, el holdup,
la capacidad de transporte, la operabilidad (slugging/inestabilidades) y el aseguramiento de flujo (cera/hidratos).
Una disminución rápida de presión expande el gas, aumenta la velocidad superficial del gas y puede gatillar
cambios de régimen. Acotar la variación local de presión y de velocidad superficial ayuda a evitar transiciones bruscas.
"""
    )

    st.markdown("**Modelo y fórmulas clave**")
    st.latex(r"g \equiv \left(\frac{dp}{dx}\right)_f = \frac{2\,f\,\rho_M\,v_M^2}{D},\quad v_M=v_{SL}+v_{SG},\quad \rho_M=H_L\rho_L+(1-H_L)\rho_G")
    st.latex(r"\Delta x_{\Delta p} = \frac{\gamma_p\,p_i}{g}")
    st.latex(r"\text{Gas ideal isotermo: } Q_G \propto \frac{1}{p}\Rightarrow v_{SG}\propto \frac{1}{p}")
    st.latex(r"p_{i+1}=p_i-g\,\Delta x,\qquad \Delta x = \min\big(L_{\text{restante}},\, s\cdot\min\{\Delta x_{\Delta p},\,\Delta x_{\Delta v}\}\big)")

    # ----- Datos de entrada (sidebar) -----
    with st.sidebar:
        st.header("Datos de entrada")
        D   = st.number_input("Diámetro D [m]", value=0.05, min_value=0.005, step=0.005)
        L   = st.number_input("Longitud total L [m]", value=500.0, min_value=10.0, step=10.0)
        eps = st.number_input("Rugosidad ε [m]", value=1e-4, format="%.1e")
        Pin = st.number_input("P_in [bar]", value=60.0, step=5.0) * 1e5
        T   = st.number_input("Temperatura T [K]", value=318.15)

        st.subheader("Caudales (elige modo)")
        use_mass = st.radio("Modo de entrada de caudales", ["Volumétricos Q", "Másicos m·"], horizontal=True) == "Másicos m·"
        if use_mass:
            mL = st.number_input("m·_L [kg/s]", value=0.41, format="%.3f")
            mG = st.number_input("m·_G [kg/s]", value=0.24, format="%.3f")
            QL = st.number_input("Q_L [m³/s] (opcional, autocalcula)", value=5e-4, format="%.2e")
            QG0= st.number_input("Q_G0 [m³/s] (autocalcula con Pin)", value=2e-3, format="%.2e")
        else:
            QL  = st.number_input("Q_L [m³/s]", value=5e-4, format="%.2e")
            QG0 = st.number_input("Q_G en línea [m³/s]", value=2e-3, format="%.2e")
            mL = mG = 0.0

        st.subheader("Propiedades")
        rhoL= st.number_input("ρ_L [kg/m³]", value=820.0)
        muL = st.number_input("μ_L [Pa·s]", value=3.5e-3, format="%.2e")
        muG = st.number_input("μ_G [Pa·s]", value=2.0e-5, format="%.1e")
        rhoG_ref = st.number_input("ρ_G,ref [kg/m³] a P_ref,T_ref", value=120.0)
        P_ref = st.number_input("P_ref [bar]", value=1.0) * 1e5
        T_ref = st.number_input("T_ref [K]", value=288.15)

        st.subheader("Hold-up HL")
        HL_mode = st.selectbox("Modo HL", ["Ingresado", "No-slip volumétrico", "Desde calidad másica (homogéneo)"])
        HL_user = st.slider("HL (si 'Ingresado')", 0.05, 0.95, 0.60)

        st.subheader("Reglas y numérico")
        maxdp = st.slider("Máx. Δp/p por tramo [%]", 0.5, 10.0, 3.0)
        maxdv = st.slider("Máx. ΔvSG/vSG por tramo [%]", 1.0, 10.0, 5.0)
        safety = st.slider("Factor de seguridad tamaño de tramo", 0.5, 1.0, 0.95)
        a_mu = st.number_input("a (μ_M = a μ_L + b μ_G)", value=0.6)
        b_mu = st.number_input("b (μ_M = a μ_L + b μ_G)", value=0.4)

    # ----- Resolver -----
    df = solve_pipe(
        D, L, eps, Pin, T,
        rhoL, muL, muG, rhoG_ref, P_ref, T_ref,
        a_mu, b_mu,
        use_mass, QL, QG0, mL, mG,
        HL_mode, HL_user,
        maxdp, maxdv, safety
    )

    # ----- Salidas -----
    with st.expander("¿Qué datos entrega esta aplicación? (salidas y significado)"):
        st.markdown(
            """
- **Por tramo**: `x0, x1, dx`, `dx_p_lim`, `dx_v_lim`, `limit`, `Pin, Pout, dp, dp/p`,
  `vSG_in, vSG_out, ΔvSG/vSG`, `HL, ρG, ρM, μM, Re, f_Darcy, g`.  
- **Global**: número de tramos, Δp total y gráficos de presión vs longitud.
            """
        )

    # ----- Resultados (tabla y gráfico P vs L) -----
    st.subheader("Resultados por tramo")
    st.dataframe(df, use_container_width=True, height=430)

    fig = None
    if not df.empty:
        fig = make_pressure_plot(df)
        st.pyplot(fig, clear_figure=False)

        ntramos = len(df)
        dptot_bar = float(df["dp[bar]"].sum())
        pin_bar = float(df.iloc[0]["Pin[bar]"])
        pout_bar = float(df.iloc[-1]["Pout[bar]"])
        st.caption(f"Tramos: {ntramos} | Δp total ≈ {dptot_bar:.2f} bar | P_out ≈ {pout_bar:.2f} bar")

        st.download_button(
            "Descargar resultados (CSV)",
            df.to_csv(index=False).encode(),
            file_name="resultados_tramos.csv",
            mime="text/csv"
        )

        # ----- Reporte PDF -----
        st.subheader("Exportar reporte PDF")
        meta = {
            "logo_path": "logoutn.png",
            "titulo": "FLUJOS MULTIFÁSICOS — Modelo 1D segmentado (Gas–Líquido)",
            "catedra": "FLUJOS MULTIFÁSICOS EN LA INDUSTRIA DEL PETRÓLEO",
            "profesor": "Ezequiel Arturo Krumrick",
            "anio": "2025",
            "pin_bar": pin_bar,
            "pout_bar": pout_bar,
            "dptot_bar": dptot_bar,
            "ntramos": ntramos
        }
        try:
            pdf_bytes = build_pdf(df, fig, meta)
            st.download_button(
                "Descargar reporte (PDF)",
                data=pdf_bytes,
                file_name=f"Reporte_FlujosMultifasicos_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"No se pudo generar el PDF ({e}). Instala 'reportlab' o 'fpdf2' en requirements.txt.")

if __name__ == "__main__":
    main()

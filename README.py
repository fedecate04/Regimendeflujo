import streamlit as st
import numpy as np
import pandas as pd

# --------- Modelos físicos básicos ---------
def churchill_f(Re, eps_over_D):
    Re = max(Re, 1.0)
    A = (2.457*np.log((7/Re)**0.9 + 0.27*eps_over_D))**16
    B = (37530/Re)**16
    f = 8 * ( (8/Re)**12 + 1/((A + B)**1.5) )**(1/12)
    return f  # Darcy

def rho_gas_ideal(rho_ref, P, T, P_ref, T_ref):
    # z=1, isotermo si T constante (igual usamos T por si cambia)
    return rho_ref * (P/P_ref) * (T_ref/T)

def mixture_props(rhoL, muL, rhoG, muG, HL, a=0.6, b=0.4):
    rhoM = HL*rhoL + (1-HL)*rhoG
    muM  = a*muL + b*muG
    return rhoM, muM

# ---- Hold-up según modo ----
def holdup_from_mode(mode, QL, QG, mL, mG, rhoL, rhoG, HL_user):
    if mode == "Ingresado":
        return HL_user
    elif mode == "No-slip volumétrico":
        denom = QL + QG
        return np.clip(QL/denom if denom>0 else 0.5, 0.01, 0.99)
    elif mode == "Desde calidad másica (homogéneo)":
        mtot = mL + mG
        x = mG/mtot if mtot>0 else 0.5  # calidad másica gas
        # fracción de gas en volumen (homogéneo)
        alpha = (x/rhoG) / ( (x/rhoG) + ((1-x)/rhoL) )
        HL = 1 - alpha
        return float(np.clip(HL, 0.01, 0.99))
    else:
        return HL_user

# ---- Límite de tamaño de tramo por criterios ----
def step_size_limits(Pi, g, vSG_i, A, QG_i, max_dpfrac, max_dvfrac, iters=24):
    # límite por % de caída de presión
    dx_p = max_dpfrac * Pi / max(g, 1e-12)

    # límite por % de cambio de vSG (compresibilidad ideal)
    lo, hi = 0.0, max(dx_p*10, 1e-6)
    vSG0 = vSG_i
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        P2 = Pi - g*mid
        if P2 <= 1.0:  # evita negativos
            hi = mid
            continue
        QG2 = QG_i * (Pi/P2)
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
    # Consistencias de entrada
    if use_mass:
        # si vienen másicos, construir volumétricos iniciales a Pin
        rhoG0 = rho_gas_ideal(rhoG_ref, Pin, T, P_ref, T_ref)
        QL = mL / max(rhoL, 1e-12)
        QG0 = mG / max(rhoG0, 1e-12)
    vSL = QL / A

    rows, i = [], 1
    x, Lrem, Pi = 0.0, Ltot, Pin
    QG_i = QG0
    vSG_i = QG_i / A

    while Lrem > 1e-9 and Pi > 2e3:  # corta si presión cae en exceso
        # propiedades con la presión local
        rhoG_i = rho_gas_ideal(rhoG_ref, Pi, T, P_ref, T_ref)

        # Recalcular HL según el modo
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

        # límites de tamaño de tramo
        dx_p, dx_v = step_size_limits(
            Pi, g_i, vSG_i, A, QG_i,
            max_dp_pct/100.0, max_dv_pct/100.0
        )
        dx = min(Lrem, safety*min(dx_p, dx_v))
        if dx <= 0: break

        # Integración (Euler)
        P2 = Pi - g_i*dx

        # Actualizar caudal de gas por compresibilidad
        QG_2 = QG_i * (Pi/max(P2, 1.0))
        vSG_2 = QG_2 / A

        # Métricas
        dp = Pi - P2
        dpfrac = dp / Pi
        dvfrac = abs(vSG_2 - vSG_i) / max(vSG_i, 1e-12)

        rows.append({
            "i": i, "x0[m]": x, "x1[m]": x+dx, "dx[m]": dx,
            "Pin[bar]": Pi/1e5, "Pout[bar]": P2/1e5, "dp[bar]": dp/1e5,
            "dp/p[%]": 100*dpfrac, "limit": "Δp" if dx_p <= dx_v else "ΔvSG",
            "HL[-]": HL_i, "rhoG[kg/m3]": rhoG_i, "rhoM[kg/m3]": rhoM_i,
            "muM[mPa·s]": 1e3*muM_i, "Re[-]": Re_i, "f_Darcy[-]": f_i,
            "g[Pa/m]": g_i, "vSL[m/s]": vSL, "vSG_in[m/s]": vSG_i, "vSG_out[m/s]": vSG_2,
            "ΔvSG/vSG[%]": 100*dvfrac
        })

        # Avanzar tramo
        x += dx; Lrem -= dx; i += 1
        Pi = P2; QG_i = QG_2; vSG_i = vSG_2

    return pd.DataFrame(rows)

# --------- Interfaz Streamlit ---------
def main():
    st.set_page_config(page_title="Segmentación Δp (Gas-Líquido) — Flujos Multifásicos", layout="wide")
    st.title("Modelo 1D segmentado con criterios de corte (GL)")

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

    df = solve_pipe(
        D, L, eps, Pin, T,
        rhoL, muL, muG, rhoG_ref, P_ref, T_ref,
        a_mu, b_mu,
        use_mass, QL, QG0, mL, mG,
        HL_mode, HL_user,
        maxdp, maxdv, safety
    )

    st.dataframe(df, use_container_width=True, height=420)

    if not df.empty:
        c1, c2 = st.columns(2)
        with c1: st.line_chart(df.set_index("x1[m]")[["Pout[bar]"]], height=260)
        with c2: st.line_chart(df.set_index("x1[m]")[["vSG_out[m/s]"]], height=260)
        st.caption(f"Tramos: {len(df)} | Δp total ≈ {df['dp[bar]'].sum():.2f} bar")

        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode(),
                           file_name="resultados_tramos.csv", mime="text/csv")

if __name__ == "__main__":
    main()

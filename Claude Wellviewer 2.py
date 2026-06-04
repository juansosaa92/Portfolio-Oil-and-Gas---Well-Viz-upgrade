"""
WellViewer — Visualizador 3D Generico de Pozos  v1.1
=====================================================
Configura casings, tubing, completacion y fracturamiento
desde el sidebar. Vista 3D interactiva con corte transversal.
v1.1: Columna estratigrafica con topes/fondos de formaciones.

Ejecutar: streamlit run wellviewer.py
Dependencias: pip install streamlit plotly numpy pandas
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WellViewer 3D",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
.wv-header{font-size:1.9rem;font-weight:800;letter-spacing:-.5px;margin-bottom:.1rem}
.wv-sub{font-size:.9rem;color:#888;margin-bottom:1rem}
.info-box{background:#1a1a2e;border-left:4px solid #4a9eff;
    padding:8px 13px;border-radius:5px;margin-bottom:6px;font-size:.84rem;color:#ddd}
.warn-box{background:#2d1a00;border-left:4px solid #ff9900;
    padding:9px 13px;border-radius:5px;margin-top:8px;font-size:.88rem;color:#ffcc88}
.pill{display:inline-block;padding:2px 9px;border-radius:12px;
    font-size:.78rem;font-weight:600;margin-right:4px}
.section-title{font-size:1.05rem;font-weight:700;
    border-bottom:1px solid #333;padding-bottom:4px;margin:10px 0 8px 0}
.strat-row{display:flex;align-items:center;gap:8px;padding:5px 0;
    border-bottom:1px solid #1e1e2e;font-size:.82rem}
.strat-swatch{width:14px;height:14px;border-radius:3px;flex-shrink:0}
.strat-name{color:#e6edf3;font-weight:600;min-width:100px}
.strat-depth{color:#8b949e;font-size:.78rem}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES Y PALETA
# ═══════════════════════════════════════════════════════════════════════════

ESCALA_R  = 28
NT_FULL   = 36
CEMENTO_COLOR = "#BDC3C7"
FLUID_COLOR   = "#1A6B3C"

# Paleta de colores por string (hasta 6 casings)
CASING_PALETTE = [
    ("#7B3F00", "#A0522D"),   # Conductor   — cafe
    ("#1A5276", "#2E86C1"),   # Surface     — azul
    ("#1E8449", "#27AE60"),   # Intermediate— verde
    ("#6C3483", "#9B59B6"),   # Liner 1     — violeta
    ("#7D6608", "#D4AC0D"),   # Liner 2     — dorado
    ("#7B241C", "#E74C3C"),   # Liner 3     — rojo
]

# Columna estratigrafica default — Cuenca Neuquina (editable desde sidebar)
STRAT_DEFAULT = [
    {"name": "Neuquén",     "top":    0, "base": 1200, "color": "#8b6f4e"},
    {"name": "Rayoso",      "top": 1200, "base": 1800, "color": "#4a9eff"},
    {"name": "Agrio",       "top": 1800, "base": 2500, "color": "#a0c4ff"},
    {"name": "Quintuco",    "top": 2500, "base": 3400, "color": "#ffd700"},
    {"name": "VM Superior", "top": 3400, "base": 4200, "color": "#3fb950"},
    {"name": "VM Inferior", "top": 4200, "base": 5000, "color": "#f78166"},
    {"name": "Tordillo",    "top": 5000, "base": 5500, "color": "#d2a8ff"},
    {"name": "Precuyano",   "top": 5500, "base": 5868, "color": "#ff6b6b"},
]

COMP_TYPES = [
    "Produccion Natural",
    "Gas Lift Continuo",
    "Bombeo Electrosumergible (BES)",
    "Inyector de Agua",
    "Inyector de Gas",
]

GRADOS_CASING = ["H-40","J-55","K-55","N-80","L-80","C-90","T-95","P-110","Q-125"]
GRADOS_TUBING = ["J-55","K-55","N-80","L-80","C-90","P-110"]

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS DE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def pulg_a_mm(p): return p * 25.4
def r(mm):        return mm / 1000 * ESCALA_R

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CILINDROS (reutilizado del v2.1, parametrizado)
# ═══════════════════════════════════════════════════════════════════════════

def add_tube(fig, z_top, z_bot, r_ext, r_int,
             color, name, opacity=1.0,
             corte=True, angulo_corte=180,
             hover_extra=""):
    """Agrega un tubo anular vertical. Z crece hacia abajo (profundidad)."""
    if r_ext <= r_int or z_top >= z_bot:
        return
    ang_rad = np.radians(angulo_corte if corte else 360)
    nt      = max(4, int(NT_FULL * (angulo_corte if corte else 360) / 360))
    th      = np.linspace(0, ang_rad, nt, endpoint=(not corte))
    is_full = not corte

    for idx_r, (rad, show_leg) in enumerate([(r_ext, True), (r_int, False)]):
        xt = rad * np.cos(th);  yt = rad * np.sin(th)
        x_v = np.concatenate([xt, xt])
        y_v = np.concatenate([yt, yt])
        z_v = np.concatenate([np.full(nt, z_top), np.full(nt, z_bot)])

        il, jl, kl = [], [], []
        n_loop = nt if is_full else nt - 1
        for i in range(n_loop):
            i0=i; i1=(i+1)%nt; j0=i+nt; j1=(i+1)%nt+nt
            il+=[i0,i0]; jl+=[i1,j0]; kl+=[j0,j1]
        if corte and nt > 2:
            for i in range(nt-2):
                il.append(0);  jl.append(i+1); kl.append(i+2)
            b=nt
            for i in range(nt-2):
                il.append(b); jl.append(b+i+1); kl.append(b+i+2)

        ht = f"<b>{name}</b>{hover_extra}<extra></extra>"
        fig.add_trace(go.Mesh3d(
            x=x_v, y=y_v, z=z_v, i=il, j=jl, k=kl,
            color=color, opacity=opacity,
            name=name, showlegend=show_leg,
            lighting=dict(ambient=0.55,diffuse=0.85,
                          specular=0.4,roughness=0.3,fresnel=0.2),
            flatshading=False,
            hovertemplate=ht,
        ))


def add_disk(fig, z_depth, r_out, r_in,
             color, name, opacity=0.85,
             corte=True, angulo_corte=180, hover_extra=""):
    """Agrega un disco (tapa horizontal) a la profundidad indicada."""
    ang_rad = np.radians(angulo_corte if corte else 360)
    nt = max(6, int(NT_FULL * (angulo_corte if corte else 360) / 360))
    th = np.linspace(0, ang_rad, nt, endpoint=(not corte))
    x_out = r_out * np.cos(th); y_out = r_out * np.sin(th)
    x_in  = r_in  * np.cos(th); y_in  = r_in  * np.sin(th)
    x_v = np.concatenate([x_out, x_in])
    y_v = np.concatenate([y_out, y_in])
    z_v = np.full(2*nt, z_depth)
    il,jl,kl = [],[],[]
    nl = nt if not corte else nt-1
    for i in range(nl):
        i0=i; i1=(i+1)%nt; j0=i+nt; j1=(i+1)%nt+nt
        il+=[i0,i0]; jl+=[i1,j0]; kl+=[j0,j1]
    fig.add_trace(go.Mesh3d(
        x=x_v, y=y_v, z=z_v, i=il, j=jl, k=kl,
        color=color, opacity=opacity,
        name=name, showlegend=False, flatshading=True,
        hovertemplate=f"<b>{name}</b>{hover_extra}<extra></extra>",
    ))

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🛢️ WellViewer")
    st.markdown("---")

    # ── 1. DATOS GENERALES ─────────────────────────────────────────────────
    with st.expander("📋 Datos generales del pozo", expanded=True):
        nombre_pozo = st.text_input("Nombre del pozo", value="POZO-001")
        formacion   = st.text_input("Formacion", value="Vaca Muerta")
        cuenca      = st.text_input("Cuenca", value="Neuquina")
        md_total    = st.number_input("MD Total (m)", 100, 8000, 4200, 50)
        tvd_total   = st.number_input("TVD Total (m)", 100, 6000, 3100, 50)
        orientacion = st.selectbox("Orientacion", ["Vertical","Horizontal (tramo vertical visualizado)"])

    # ── 2. CASINGS ─────────────────────────────────────────────────────────
    with st.expander("🪨 Diseño de Casings", expanded=True):
        st.caption("Definir de afuera hacia adentro (conductor primero)")
        n_casings = st.number_input("Numero de strings de casing", 1, 6, 3, 1)

        casings_cfg = []
        for i in range(int(n_casings)):
            idx_color = min(i, len(CASING_PALETTE)-1)
            col_ext, col_int = CASING_PALETTE[idx_color]

            nombre_default = ["Conductor","Surface Casing","Int./Prod. Casing",
                              "Liner 1","Liner 2","Liner 3"][i]
            od_default     = [20.0, 13.375, 9.625, 7.0, 5.5, 4.5][i]
            id_default     = [od_default * 0.95] * 6  # approx
            zapato_default = [60, 600, 2800, 3200, 3800, 4100][i]

            st.markdown(f"**String {i+1} — {nombre_default}**")
            c1, c2 = st.columns(2)
            with c1:
                od_pulg = st.number_input(f'OD (pulg) #{i+1}', 3.0, 30.0,
                                           od_default, 0.125,
                                           key=f"od_{i}")
                grado   = st.selectbox(f'Grado #{i+1}', GRADOS_CASING,
                                        index=GRADOS_CASING.index("K-55") if i<2
                                        else GRADOS_CASING.index("P-110"),
                                        key=f"grado_{i}")
                cemento = st.checkbox(f'Cemento #{i+1}', value=True,
                                      key=f"cem_{i}")
            with c2:
                od_mm   = pulg_a_mm(od_pulg)
                id_mm   = st.number_input(f'ID (mm) #{i+1}',
                                           od_mm*0.5, od_mm*0.99,
                                           round(od_mm*0.944,1), 0.5,
                                           key=f"id_{i}")
                zapato  = st.number_input(f'Zapato (m MD) #{i+1}',
                                           10, int(md_total),
                                           min(zapato_default, int(md_total)),
                                           10, key=f"zap_{i}")
            casings_cfg.append({
                "nombre": nombre_default,
                "od_pulg": od_pulg,
                "od_mm":   od_mm,
                "id_mm":   id_mm,
                "grado":   grado,
                "zapato":  zapato,
                "cemento": cemento,
                "color_ext": col_ext,
                "color_int": col_int,
            })
            if i < int(n_casings)-1:
                st.markdown("---")

    # ── 3. TUBING ──────────────────────────────────────────────────────────
    with st.expander("🔧 Tubing", expanded=True):
        tbg_od_pulg = st.number_input("OD Tubing (pulg)", 1.5, 7.0, 3.5, 0.125)
        tbg_od_mm   = pulg_a_mm(tbg_od_pulg)
        tbg_id_mm   = st.number_input("ID Tubing (mm)",
                                       tbg_od_mm*0.5, tbg_od_mm*0.99,
                                       round(tbg_od_mm*0.855,1), 0.5)
        tbg_grado   = st.selectbox("Grado Tubing", GRADOS_TUBING,
                                    index=GRADOS_TUBING.index("L-80"))
        tbg_fondo   = st.number_input("Prof. fondo tubing (m MD)",
                                       100, int(md_total),
                                       min(3150, int(md_total)), 50)
        show_fluido = st.checkbox("Mostrar columna de fluido", value=True)
        show_tubing = st.checkbox("Mostrar tubing", value=True)

    # ── 4. COMPLETACION ────────────────────────────────────────────────────
    with st.expander("⚙️ Tipo de completacion", expanded=True):
        comp_tipo = st.selectbox("Tipo de completacion", COMP_TYPES)
        show_packer = st.checkbox("Packer de produccion", value=True)
        packer_md   = st.number_input("Profundidad packer (m MD)",
                                       100, int(md_total),
                                       min(3000, int(md_total)), 50) if show_packer else None

        # ── Gas Lift ───────────────────────────────────────────────────
        glv_list = []
        if comp_tipo == "Gas Lift Continuo":
            st.markdown("**Valvulas Gas Lift**")
            show_gas_annulus = st.checkbox("Gas en annulus", value=True)
            n_glv = st.number_input("Numero de valvulas", 1, 8, 4, 1)
            glv_defaults_md  = [850, 1550, 2300, 2900, 3200, 3500, 3700, 3900]
            glv_defaults_psi = [1250,1050, 850,  720,  600,  500,  420,  360]
            for g in range(int(n_glv)):
                ca, cb = st.columns(2)
                with ca:
                    gmd = st.number_input(f"Profundidad GLV-{g+1} (m)",
                                           100, int(md_total),
                                           min(glv_defaults_md[g], int(md_total)),
                                           50, key=f"gmd_{g}")
                with cb:
                    gpsi = st.number_input(f"P apertura GLV-{g+1} (psi)",
                                            100, 3000,
                                            glv_defaults_psi[g], 10,
                                            key=f"gpsi_{g}")
                es_op = (g == int(n_glv)-1)
                glv_list.append({
                    "id": g+1, "md": gmd, "psi": gpsi,
                    "tipo": "Operating Valve" if es_op else "Unloading Valve",
                    "estado": "ABIERTA - Inyeccion activa" if es_op else "Cerrada",
                    "color": "#27AE60" if es_op else "#E67E22",
                    "nota": "Valvula operativa" if es_op else "Valvula de descarga",
                })
        else:
            show_gas_annulus = False

        # ── BES ────────────────────────────────────────────────────────
        bes_cfg = {}
        if comp_tipo == "Bombeo Electrosumergible (BES)":
            st.markdown("**Configuracion BES**")
            bes_cfg["bomba_md"]  = st.number_input("Prof. bomba (m MD)",
                                                    100, int(md_total),
                                                    min(2800, int(md_total)), 50)
            bes_cfg["motor_len"] = st.number_input("Long. motor+bomba (m)", 10, 40, 20, 1)
            bes_cfg["od_mm"]     = st.number_input("OD BES (mm)", 50.0, 200.0, 114.0, 1.0)
            bes_cfg["color"]     = "#C0392B"

        # ── Fracturamiento ─────────────────────────────────────────────
        frac_cfg = {}
        with st.expander("💥 Fracturamiento hidraulico", expanded=False):
            tiene_frac = st.checkbox("El pozo tiene fracturamiento", value=False)
            if tiene_frac:
                frac_cfg["n"]          = st.number_input("Numero de etapas", 1, 60, 12, 1)
                frac_cfg["prof_inicio"]= st.number_input("Prof. inicio (m MD)",
                                                          100, int(md_total),
                                                          min(2800, int(md_total)), 50)
                frac_cfg["espac"]      = st.number_input("Espaciamiento (m)", 10, 200, 60, 5)
                frac_cfg["radio"]      = st.number_input("Radio visual fractura (m)", 5, 80, 30, 5)
                frac_cfg["color"]      = "#E74C3C"

        # ── Punzados ───────────────────────────────────────────────────
        punzados_cfg = []
        with st.expander("🎯 Punzados (Perforaciones)", expanded=False):
            tiene_punzados = st.checkbox("El pozo tiene punzados", value=False)
            if tiene_punzados:
                n_pz = st.number_input("Numero de intervalos punzados", 1, 10, 1, 1)
                st.caption("Definir cada intervalo: tope, base y densidad de punzado")
                for pz_i in range(int(n_pz)):
                    st.markdown(f"**Intervalo {pz_i+1}**")
                    pza, pzb, pzc = st.columns(3)
                    with pza:
                        pz_top = st.number_input(
                            f"Tope (m) #{pz_i+1}", 0, int(md_total),
                            min(2900 + pz_i*50, int(md_total)), 10,
                            key=f"pztop_{pz_i}")
                    with pzb:
                        pz_bot = st.number_input(
                            f"Base (m) #{pz_i+1}", 0, int(md_total),
                            min(2950 + pz_i*50, int(md_total)), 10,
                            key=f"pzbot_{pz_i}")
                    with pzc:
                        pz_dens = st.number_input(
                            f"Disparos/m #{pz_i+1}", 1, 24, 12, 1,
                            key=f"pzdens_{pz_i}")
                    punzados_cfg.append({
                        "top":   pz_top,
                        "base":  pz_bot,
                        "dens":  pz_dens,
                        "n_dis": max(1, int((pz_bot - pz_top) * pz_dens)),
                        "color": "#FFD700",
                    })

    # ── 5. ESTRATIGRAFIA ───────────────────────────────────────────────────
    with st.expander("🪨 Columna Estratigrafica", expanded=False):
        show_strat = st.checkbox("Mostrar columna estratigrafica", value=True)
        op_strat   = st.slider("Opacidad estratigrafia", 0.05, 0.45, 0.18, 0.05)
        show_strat_labels = st.checkbox("Etiquetas de formaciones", value=True)
        show_contact_lines = st.checkbox("Lineas de contacto (topes)", value=True)

        st.markdown("**Formaciones** — ajustar topes y bases:")
        st.caption("Los valores se actualizan en tiempo real en el 3D")

        strat_cfg = []
        for si, sd in enumerate(STRAT_DEFAULT):
            col = sd["color"]
            nom = sd["name"]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"margin-bottom:2px'>"
                f"<div style='width:10px;height:10px;border-radius:2px;"
                f"background:{col};flex-shrink:0'></div>"
                f"<b style='font-size:.82rem;color:#e6edf3'>{nom}</b></div>",
                unsafe_allow_html=True)
            sa, sb = st.columns(2)
            with sa:
                s_top  = st.number_input(f"Tope (m) #{si+1}", 0, 9000,
                                          sd["top"],  50, key=f"st_{si}")
            with sb:
                s_base = st.number_input(f"Base (m) #{si+1}", 0, 9000,
                                          sd["base"], 50, key=f"sb_{si}")
            strat_cfg.append({
                "name":  sd["name"],
                "top":   s_top,
                "base":  s_base,
                "color": sd["color"],
            })

    # ── 6. VISIBILIDAD ─────────────────────────────────────────────────────
    with st.expander("👁️ Visibilidad", expanded=False):
        show_cemento   = st.checkbox("Cemento", value=True)
        show_casings   = st.checkbox("Casings", value=True)
        show_valvulas  = st.checkbox("Valvulas / accesorios", value=True)
        show_zapatos   = st.checkbox("Zapatos de casing", value=True)
        show_labels    = st.checkbox("Etiquetas 3D", value=True)
        op_casing      = st.slider("Opacidad casings",  0.3, 1.0, 0.90, 0.05)
        op_tubing_sl   = st.slider("Opacidad tubing",   0.3, 1.0, 0.92, 0.05)
        op_fluido_sl   = st.slider("Opacidad fluidos",  0.1, 1.0, 0.60, 0.05)

    # ── 7. CORTE TRANSVERSAL ───────────────────────────────────────────────
    with st.expander("✂️ Corte transversal", expanded=True):
        corte        = st.checkbox("Aplicar corte", value=True)
        angulo_corte = st.slider("Angulo del corte (°)", 0, 360, 180, 10) if corte else 360

    # ── 8. INTERVENCION ────────────────────────────────────────────────────
    with st.expander("🔴 Intervencion", expanded=False):
        mi = st.checkbox("Marcar zona de intervencion", value=False)
        pi, ti = int(md_total*0.7), "Wire-line"
        if mi:
            pi = st.slider("Profundidad intervencion (m MD)",
                            100, int(md_total), int(md_total*0.7), 50)
            ti = st.selectbox("Tipo de operacion", [
                "Cambio de valvula GLV", "Wire-line",
                "Coiled Tubing", "Pesca de herramienta",
                "Caioneo", "Stimulacion", "Medicion / registro"])

    # ── 9. VISTA ───────────────────────────────────────────────────────────
    with st.expander("📷 Vista y display", expanded=False):
        vp = st.selectbox("Vista inicial", [
            "Isometrica / Corte", "Frontal", "Lateral", "Superior"])
        altura_fig = st.slider("Altura del grafico (px)", 500, 1100, 800, 50)

    st.markdown("---")
    st.caption("WellViewer v1.0 | Streamlit + Plotly\nPortfolio — Ingenieria de Yacimientos")

# ═══════════════════════════════════════════════════════════════════════════
# CALCULAR RADIOS A PARTIR DE LA CONFIGURACION
# ═══════════════════════════════════════════════════════════════════════════

VIS_DEPTH  = min(int(tvd_total * 1.05), int(md_total))
Z0         = 0   # boca de pozo
Z1         = VIS_DEPTH

# Tubing
R_TBG_EXT = r(tbg_od_mm)
R_TBG_INT = r(tbg_id_mm)

# Referencia para escala de terreno: radio externo del primer casing
R_REF_EXT = r(casings_cfg[0]["od_mm"]) if casings_cfg else 0.5

# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCCION DE LA FIGURA
# ═══════════════════════════════════════════════════════════════════════════

fig = go.Figure()

kw = dict(corte=corte, angulo_corte=angulo_corte)  # shorthand

# ─── TERRENO ─────────────────────────────────────────────────────────────────
xs = np.linspace(-R_REF_EXT*3, R_REF_EXT*3, 4)
ys = np.linspace(-R_REF_EXT*3, R_REF_EXT*3, 4)
XS, YS = np.meshgrid(xs, ys)
fig.add_trace(go.Surface(
    x=XS, y=YS, z=np.zeros_like(XS),
    colorscale=[[0,"#2d5a27"],[1,"#4a8a40"]],
    opacity=0.22, showscale=False,
    name="Superficie", hoverinfo="skip"))

# ─── WELLHEAD / BOP ──────────────────────────────────────────────────────────
wh_h = 55
add_tube(fig, -wh_h, Z0,
         R_REF_EXT * 0.88, r(casings_cfg[1]["od_mm"]) * 0.96 if len(casings_cfg)>1 else R_REF_EXT*0.5,
         "#E67E22", "Wellhead / BOP", 1.0, **kw)
if show_labels:
    fig.add_trace(go.Scatter3d(
        x=[R_REF_EXT*1.6], y=[0], z=[-wh_h*0.4],
        mode="text", text=["WELLHEAD / BOP"],
        textfont=dict(size=10, color="white"),
        showlegend=False, hoverinfo="skip"))

# Lineas de flujo y gas (superficiales)
lf_len = R_REF_EXT * 3.2
for y_off, col, nm in [
    ( R_REF_EXT, "#E74C3C", "Linea de flujo"),
    (-R_REF_EXT, "#2ECC71", "Linea de gas inyeccion"),
]:
    fig.add_trace(go.Scatter3d(
        x=[0, lf_len], y=[y_off, y_off], z=[-wh_h*0.5, -wh_h*0.5],
        mode="lines", line=dict(color=col, width=7),
        name=nm, hovertemplate=f"<b>{nm}</b><extra></extra>"))

# ─── CASINGS Y CEMENTO ────────────────────────────────────────────────────────
if show_casings:
    for i, c in enumerate(casings_cfg):
        r_ext = r(c["od_mm"])
        r_int = r(c["id_mm"])
        z_bot = min(c["zapato"], Z1)

        add_tube(fig, Z0, z_bot,
                 r_ext, r_int,
                 c["color_ext"],
                 f'{c["nombre"]} {c["od_pulg"]}"',
                 op_casing, **kw)

        # Cemento: entre este casing y el siguiente (hacia adentro)
        if show_cemento and c["cemento"] and i + 1 < len(casings_cfg):
            next_c   = casings_cfg[i+1]
            r_cem_in = r(next_c["od_mm"])
            z_cem_bot= min(next_c["zapato"], Z1)
            if r_int > r_cem_in:
                add_tube(fig,
                         min(c["zapato"], next_c["zapato"]),
                         z_cem_bot,
                         r_int, r_cem_in,
                         CEMENTO_COLOR,
                         f"Cemento {i+1}-{i+2}",
                         0.50, **kw)

        # Etiqueta
        if show_labels:
            fig.add_trace(go.Scatter3d(
                x=[-r_ext*1.35], y=[0], z=[z_bot*0.35],
                mode="text",
                text=[f'{c["od_pulg"]}" {c["grado"]}'],
                textfont=dict(size=9, color=c["color_int"]),
                showlegend=False, hoverinfo="skip"))

# ─── ZAPATOS DE CASING ────────────────────────────────────────────────────────
if show_zapatos:
    for c in casings_cfg:
        if c["zapato"] > Z1:
            continue
        add_disk(fig, c["zapato"],
                 r(c["od_mm"]), r(c["id_mm"]),
                 c["color_int"], f'Zapato {c["od_pulg"]}"',
                 0.95, **kw,
                 hover_extra=f"<br>Prof: {c['zapato']} m")

# ─── GAS EN ANNULUS (Gas Lift) ────────────────────────────────────────────────
if show_gas_annulus and glv_list:
    # Gas ocupa el anular del ultimo casing interno hasta la ultima valvula
    inner_c = casings_cfg[-1]
    depth_gas = min(glv_list[-1]["md"], Z1)
    add_tube(fig, Z0, depth_gas,
             r(inner_c["id_mm"]), R_TBG_EXT,
             "#D4E6F1", "Gas en annulus", op_fluido_sl, **kw)

# ─── TUBING ───────────────────────────────────────────────────────────────────
if show_tubing:
    add_tube(fig, Z0, min(tbg_fondo, Z1),
             R_TBG_EXT, R_TBG_INT,
             "#D4AC0D", f'Tubing {tbg_od_pulg}" {tbg_grado}',
             op_tubing_sl, **kw)

# ─── COLUMNA DE FLUIDO ────────────────────────────────────────────────────────
if show_fluido:
    add_tube(fig, Z0, min(tbg_fondo, Z1),
             R_TBG_INT, 0.0,
             FLUID_COLOR, "Columna de fluido",
             op_fluido_sl, **kw)

# ─── PACKER ───────────────────────────────────────────────────────────────────
if show_packer and packer_md and packer_md <= Z1:
    inner_c = casings_cfg[-1]
    h_pk    = 85
    add_tube(fig,
             packer_md - h_pk/2, packer_md + h_pk/2,
             r(inner_c["id_mm"]) * 0.97, R_TBG_EXT * 1.02,
             "#8E44AD", "Packer de produccion",
             0.95, **kw)
    if show_labels:
        fig.add_trace(go.Scatter3d(
            x=[r(inner_c["id_mm"])*1.6], y=[0], z=[packer_md],
            mode="text",
            text=[f"PACKER  {packer_md} m"],
            textfont=dict(size=10, color="#C39BD3"),
            showlegend=False, hoverinfo="skip"))

# ─── VALVULAS GAS LIFT ────────────────────────────────────────────────────────
if show_valvulas and glv_list:
    for v in glv_list:
        if v["md"] > Z1:
            continue
        is_open = "ABIERTA" in v["estado"]
        nm      = f"GLV-{v['id']} {v['tipo']} ({v['md']}m)"
        r_mnd   = R_TBG_EXT * 1.55

        add_tube(fig, v["md"]-25, v["md"]+25,
                 r_mnd, R_TBG_INT,
                 v["color"], nm, 1.0, **kw,
                 hover_extra=(f"<br>Prof: {v['md']} m MD"
                              f"<br>P apertura: {v['psi']} psi"
                              f"<br>Estado: {v['estado']}"))
        if show_labels:
            lbl = ("✅ " if is_open else "⭕ ") + f"GLV-{v['id']}  {v['md']} m"
            fig.add_trace(go.Scatter3d(
                x=[r_mnd*1.7], y=[0], z=[v["md"]],
                mode="text", text=[lbl],
                textfont=dict(size=10,
                              color="#27AE60" if is_open else "#E67E22"),
                showlegend=False, hoverinfo="skip"))

# ─── BES ─────────────────────────────────────────────────────────────────────
if show_valvulas and bes_cfg:
    bmd   = bes_cfg["bomba_md"]
    blen  = bes_cfg["motor_len"]
    bod   = bes_cfg.get("od_mm", 114.0)
    r_bes = r(bod)
    add_tube(fig, bmd, bmd + blen,
             r_bes, R_TBG_INT,
             bes_cfg["color"], "BES — Motor + Bomba",
             0.95, **kw,
             hover_extra=f"<br>Prof. bomba: {bmd} m MD<br>Long: {blen} m")
    add_disk(fig, bmd, r_bes, 0, bes_cfg["color"],
             "BES — tapa superior", 1.0, **kw)
    add_disk(fig, bmd+blen, r_bes, 0, bes_cfg["color"],
             "BES — tapa inferior", 1.0, **kw)
    if show_labels:
        fig.add_trace(go.Scatter3d(
            x=[r_bes*2.2], y=[0], z=[bmd + blen/2],
            mode="text", text=[f"BES  {bmd} m"],
            textfont=dict(size=10, color="#E74C3C"),
            showlegend=False, hoverinfo="skip"))

# ─── FRACTURAMIENTO ───────────────────────────────────────────────────────────
if frac_cfg:
    r_inner = r(casings_cfg[-1]["id_mm"]) if casings_cfg else R_TBG_EXT
    r_frac  = r(frac_cfg["radio"] * 1000)  # radio en mm → escala
    n_fr    = int(frac_cfg["n"])
    pi_frac = frac_cfg["prof_inicio"]
    esp     = frac_cfg["espac"]

    for f_i in range(n_fr):
        z_f = pi_frac + f_i * esp
        if z_f > Z1:
            break

        # Disco de fractura (horizontal)
        ang_rad  = np.radians(360)
        nt_f     = NT_FULL
        th_f     = np.linspace(0, ang_rad, nt_f, endpoint=False)
        x_f = np.concatenate([r_frac*np.cos(th_f), r_inner*np.cos(th_f)])
        y_f = np.concatenate([r_frac*np.sin(th_f), r_inner*np.sin(th_f)])
        z_f_v = np.full(2*nt_f, z_f)
        il_f,jl_f,kl_f = [],[],[]
        for fi in range(nt_f):
            f0=fi; f1=(fi+1)%nt_f; g0=fi+nt_f; g1=(fi+1)%nt_f+nt_f
            il_f+=[f0,f0]; jl_f+=[f1,g0]; kl_f+=[g0,g1]

        nm_f = f"Fractura {f_i+1}" if f_i==0 else f"Fractura {f_i+1}"
        fig.add_trace(go.Mesh3d(
            x=x_f, y=y_f, z=z_f_v,
            i=il_f, j=jl_f, k=kl_f,
            color=frac_cfg["color"], opacity=0.55,
            name="Etapas de fractura" if f_i==0 else nm_f,
            showlegend=(f_i == 0),
            flatshading=True,
            hovertemplate=(f"<b>{nm_f}</b><br>"
                           f"Prof: {z_f:.0f} m MD<extra></extra>"),
        ))
        # Linea de perforation en el centro
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[z_f-8, z_f+8],
            mode="lines",
            line=dict(color="#FF6B6B", width=5),
            showlegend=False, hoverinfo="skip"))

    if show_labels and n_fr > 0:
        z_label = pi_frac + (n_fr//2) * esp
        fig.add_trace(go.Scatter3d(
            x=[r_frac*1.1], y=[0], z=[z_label],
            mode="text",
            text=[f"  {n_fr} etapas Frac"],
            textfont=dict(size=10, color="#FF6B6B"),
            showlegend=False, hoverinfo="skip"))

# ─── COLUMNA ESTRATIGRAFICA — PLANOS HORIZONTALES ────────────────────────────
if show_strat and strat_cfg:
    # Cada formacion = losa rectangular horizontal que atraviesa el pozo
    # Ancho del plano: 2x el diámetro exterior del casing mas externo
    hw = R_REF_EXT * 2.8   # half-width del plano en X e Y

    for si, s in enumerate(strat_cfg):
        z_top_s = max(s["top"],  0)
        z_bot_s = min(s["base"], Z1)
        if z_top_s >= z_bot_s:
            continue

        # 8 vertices del paralelepipedo (caja plana):
        # 0-3 = tope,  4-7 = base
        #   3──2        7──6
        #   │  │   →    │  │
        #   0──1        4──5
        x_v = np.array([-hw,  hw,  hw, -hw, -hw,  hw,  hw, -hw])
        y_v = np.array([-hw, -hw,  hw,  hw, -hw, -hw,  hw,  hw])
        z_v = np.array([z_top_s]*4 + [z_bot_s]*4)

        # 6 caras × 2 triangulos cada una
        il_s = [0,0, 4,4, 0,0, 1,1, 0,0, 2,2]
        jl_s = [1,2, 5,6, 4,1, 5,2, 3,2, 6,3]
        kl_s = [2,3, 6,7, 1,5, 2,6, 2,6, 7,7]

        nm_s = s["name"]
        fig.add_trace(go.Mesh3d(
            x=x_v, y=y_v, z=z_v,
            i=il_s, j=jl_s, k=kl_s,
            color=s["color"], opacity=op_strat,
            name=nm_s,
            showlegend=True,
            legendgroup="estratigrafia",
            legendgrouptitle_text="Estratigrafía" if si == 0 else None,
            flatshading=True,
            hovertemplate=(
                f"<b>{s['name']}</b><br>"
                f"Tope: {s['top']} m<br>"
                f"Base: {s['base']} m<br>"
                f"Espesor: {s['base'] - s['top']} m"
                f"<extra></extra>"),
        ))

        # Linea de contacto en el tope (borde superior del plano)
        if show_contact_lines:
            fig.add_trace(go.Scatter3d(
                x=[-hw,  hw,  hw, -hw, -hw],
                y=[-hw, -hw,  hw,  hw, -hw],
                z=[z_top_s] * 5,
                mode="lines",
                line=dict(color=s["color"], width=2),
                showlegend=False, hoverinfo="skip",
                legendgroup="estratigrafia",
            ))

        # Etiqueta centrada en el plano
        if show_strat_labels:
            z_mid = (z_top_s + z_bot_s) / 2
            fig.add_trace(go.Scatter3d(
                x=[hw * 1.05], y=[0], z=[z_mid],
                mode="text",
                text=[f"  {s['name']}  ({s['base']-s['top']} m)"],
                textfont=dict(size=9, color=s["color"]),
                showlegend=False, hoverinfo="skip",
                legendgroup="estratigrafia",
            ))


# ─── PUNZADOS ─────────────────────────────────────────────────────────────────
if punzados_cfg:
    inner_c  = casings_cfg[-1]
    r_csg_in = r(inner_c["id_mm"])   # ID del ultimo casing = pared de perf.

    for pz_idx, pz in enumerate(punzados_cfg):
        z_top_pz = min(pz["top"],  Z1)
        z_bot_pz = min(pz["base"], Z1)
        if z_top_pz >= z_bot_pz:
            continue

        espesor_pz = z_bot_pz - z_top_pz
        n_dis      = pz["n_dis"]

        # ── Franja de fondo: resalta el intervalo punzado sobre el casing ──
        add_tube(fig, z_top_pz, z_bot_pz,
                 r_csg_in * 1.01, r_csg_in * 0.99,
                 pz["color"], f"Punzado {pz_idx+1}",
                 0.0, **kw)   # solo para la leyenda; invisible

        # Barra lateral resaltada sobre el casing (franja dorada visible)
        add_tube(fig, z_top_pz, z_bot_pz,
                 r_csg_in + 0.012, r_csg_in,
                 pz["color"],
                 f"Intervalo punzado {pz_idx+1}  ({z_top_pz}–{z_bot_pz} m)",
                 0.9, **kw,
                 hover_extra=(f"<br>Intervalo: {z_top_pz}–{z_bot_pz} m<br>"
                              f"Espesor: {espesor_pz} m<br>"
                              f"Disparos: {n_dis}  ({pz['dens']}/m)"))

        # ── Disparos individuales: lineas radiales desde casing hacia afuera ──
        # Distribuidos uniformemente en profundidad, alternando angulo (4 cuadrantes)
        ang_offsets = [0, 90, 45, 135]   # grados, patron tipico de 4 fases
        for d_i in range(n_dis):
            if d_i > 80:   # cap visual para no saturar
                break
            z_d   = z_top_pz + (d_i + 0.5) * (espesor_pz / n_dis)
            ang_d = np.radians(ang_offsets[d_i % 4] + (d_i // 4) * (360 / max(n_dis//4, 1)))
            x0 = r_csg_in * np.cos(ang_d)
            y0 = r_csg_in * np.sin(ang_d)
            x1 = (r_csg_in + 0.06) * np.cos(ang_d)
            y1 = (r_csg_in + 0.06) * np.sin(ang_d)
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z_d, z_d],
                mode="lines",
                line=dict(color=pz["color"], width=3),
                showlegend=False, hoverinfo="skip",
            ))

        # ── Marcadores de tope y base del intervalo ──
        for z_lim, lbl in [(z_top_pz, "TOPE"), (z_bot_pz, "BASE")]:
            fig.add_trace(go.Scatter3d(
                x=[r_csg_in * 1.8], y=[0], z=[z_lim],
                mode="markers+text",
                marker=dict(size=6, color=pz["color"],
                            symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"  {'▲' if lbl=='TOPE' else '▼'} Pz{pz_idx+1} {lbl} {z_lim}m"],
                textfont=dict(size=9, color=pz["color"]),
                showlegend=False, hoverinfo="skip",
            ))

        # ── Linea vertical que une tope y base ──
        fig.add_trace(go.Scatter3d(
            x=[r_csg_in * 1.6, r_csg_in * 1.6],
            y=[0, 0],
            z=[z_top_pz, z_bot_pz],
            mode="lines",
            line=dict(color=pz["color"], width=4),
            showlegend=False, hoverinfo="skip",
        ))

# ─── EJE DE PROFUNDIDAD ───────────────────────────────────────────────────────
step = 200 if Z1 <= 2000 else (500 if Z1 > 4000 else 200)
depth_marks = list(range(0, Z1+1, step))
fig.add_trace(go.Scatter3d(
    x=[-R_REF_EXT*3.5]*len(depth_marks),
    y=[0]*len(depth_marks),
    z=depth_marks,
    mode="text",
    text=[f"{d} m" for d in depth_marks],
    textfont=dict(size=8, color="#777777"),
    showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter3d(
    x=[-R_REF_EXT*3.3, -R_REF_EXT*3.3],
    y=[0, 0], z=[0, Z1],
    mode="lines",
    line=dict(color="#444", width=1, dash="dot"),
    showlegend=False, hoverinfo="skip"))

# ─── ZONA DE INTERVENCION ─────────────────────────────────────────────────────
if mi and pi <= Z1:
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[pi],
        mode="markers+text",
        marker=dict(size=18, color="red", symbol="x",
                    line=dict(color="yellow", width=3)),
        text=[f"  ⚠️ {ti}"],
        textfont=dict(size=10, color="red"),
        name=f"Intervencion: {ti}",
        hovertemplate=(f"<b>INTERVENCION</b><br>{ti}<br>"
                       f"Prof: {pi} m<extra></extra>")))

# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

CP = {
    "Isometrica / Corte": dict(
        eye=dict(x=1.8, y=0.9, z=-0.6),
        up =dict(x=0,   y=0,   z=-1)),
    "Frontal": dict(
        eye=dict(x=0,   y=2.5, z=0),
        up =dict(x=0,   y=0,   z=-1)),
    "Lateral": dict(
        eye=dict(x=2.5, y=0,   z=0),
        up =dict(x=0,   y=0,   z=-1)),
    "Superior": dict(
        eye=dict(x=0,   y=0,   z=-2.5),
        up =dict(x=0,   y=1,   z=0)),
}

titulo_fig = (
    f"🛢️ {nombre_pozo} — {comp_tipo} — {formacion}"
    + (f"  |  CORTE {angulo_corte}°" if corte else "")
)

fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    scene=dict(
        bgcolor="#0d1117",
        xaxis=dict(title="", showgrid=False, zeroline=False,
                   showticklabels=False, backgroundcolor="#0d1117"),
        yaxis=dict(title="", showgrid=False, zeroline=False,
                   showticklabels=False, backgroundcolor="#0d1117"),
        zaxis=dict(
            title="Profundidad (m MD)",
            showgrid=True, gridcolor="#1a1a2e",
            backgroundcolor="#0d1117", color="#aaa",
            autorange="reversed",
        ),
        camera=CP[vp],
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=5.0),
    ),
    legend=dict(
        bgcolor="rgba(10,10,20,0.88)",
        bordercolor="#333", borderwidth=1,
        font=dict(color="white", size=10),
        x=0.01, y=0.98,
    ),
    margin=dict(l=0, r=0, t=45, b=0),
    title=dict(
        text=titulo_fig,
        font=dict(color="white", size=14),
        x=0.5,
    ),
    height=altura_fig,
)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
with h1:
    st.markdown(
        f'<div class="wv-header">🛢️ WellViewer — {nombre_pozo}</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="wv-sub">{comp_tipo} | {formacion} | {cuenca}'
        f' | {orientacion}</div>',
        unsafe_allow_html=True)
with h2:
    st.metric("MD Total", f"{md_total} m")
with h3:
    st.metric("TVD Total", f"{tvd_total} m")
with h4:
    st.metric("Casings", f"{len(casings_cfg)}")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🌐 Vista 3D",
    "📋 Datos tecnicos",
    "💡 Guia de uso",
])

with tab1:
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"wellviewer_{nombre_pozo}",
            "scale": 2,
        },
    })
    if mi:
        st.markdown(
            f'<div class="warn-box">⚠️ <b>INTERVENCION MARCADA:</b> '
            f'{ti} @ {pi} m MD — Verificar presiones y estado de '
            f'completacion antes de iniciar trabajos.</div>',
            unsafe_allow_html=True)

with tab2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-title">Casings</div>', unsafe_allow_html=True)
        for c in casings_cfg:
            st.markdown(
                f'<div class="info-box">'
                f'<b>{c["nombre"]}</b><br>'
                f'OD: {c["od_pulg"]}" ({c["od_mm"]:.1f} mm) | ID: {c["id_mm"]:.1f} mm<br>'
                f'Grado: {c["grado"]} | Zapato: {c["zapato"]} m MD<br>'
                f'Cemento: {"✅" if c["cemento"] else "❌"}'
                f'</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-title">Tubing</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">'
            f'OD: {tbg_od_pulg}" ({tbg_od_mm:.1f} mm) | ID: {tbg_id_mm:.1f} mm<br>'
            f'Grado: {tbg_grado} | Fondo: {tbg_fondo} m MD'
            f'</div>',
            unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="section-title">Completacion: {comp_tipo}</div>',
                    unsafe_allow_html=True)

        if show_packer and packer_md:
            st.markdown(
                f'<div class="info-box">🟣 <b>Packer de produccion</b><br>'
                f'Prof: {packer_md} m MD</div>',
                unsafe_allow_html=True)

        if glv_list:
            for v in glv_list:
                color_pill = "#27AE60" if "ABIERTA" in v["estado"] else "#E67E22"
                st.markdown(
                    f'<div class="info-box">'
                    f'<b>GLV-{v["id"]} — {v["tipo"]}</b><br>'
                    f'Prof: {v["md"]} m | P apertura: {v["psi"]} psi<br>'
                    f'Estado: {v["estado"]}'
                    f'</div>',
                    unsafe_allow_html=True)

        if bes_cfg:
            st.markdown(
                f'<div class="info-box">🔴 <b>BES</b><br>'
                f'Prof. bomba: {bes_cfg["bomba_md"]} m MD<br>'
                f'Long. motor+bomba: {bes_cfg["motor_len"]} m<br>'
                f'OD: {bes_cfg["od_mm"]} mm'
                f'</div>',
                unsafe_allow_html=True)

        if frac_cfg:
            st.markdown(
                f'<div class="info-box">💥 <b>Fracturamiento</b><br>'
                f'{int(frac_cfg["n"])} etapas | Inicio: {frac_cfg["prof_inicio"]} m<br>'
                f'Espaciamiento: {frac_cfg["espac"]} m | Radio: {frac_cfg["radio"]} m'
                f'</div>',
                unsafe_allow_html=True)

        if punzados_cfg:
            st.markdown('<div class="section-title">Punzados</div>',
                        unsafe_allow_html=True)
            total_disparos = sum(pz["n_dis"] for pz in punzados_cfg)
            for pz_idx, pz in enumerate(punzados_cfg):
                esp_pz = pz["base"] - pz["top"]
                # Formacion donde cae el intervalo
                form_pz = next(
                    (s["name"] for s in strat_cfg
                     if s["top"] <= (pz["top"]+pz["base"])//2 <= s["base"]),
                    "—")
                st.markdown(
                    f'<div class="info-box" style="border-color:#FFD700">'
                    f'🎯 <b>Intervalo {pz_idx+1}</b><br>'
                    f'Tope: {pz["top"]} m  |  Base: {pz["base"]} m<br>'
                    f'Espesor: {esp_pz} m  |  Densidad: {pz["dens"]} disp/m<br>'
                    f'Disparos totales: <b>{pz["n_dis"]}</b><br>'
                    f'Formacion: <b>{form_pz}</b>'
                    f'</div>',
                    unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-box" style="border-color:#FFD700;background:#1a1500">'
                f'<b>Total disparos del pozo: {total_disparos}</b>'
                f'</div>',
                unsafe_allow_html=True)

        # ── Relacion zapatos / formaciones ─────────────────────────────
        if strat_cfg and casings_cfg:
            st.markdown('<div class="section-title">Zapatos vs Estratigrafia</div>',
                        unsafe_allow_html=True)
            for c in casings_cfg:
                zap = c["zapato"]
                form_zap = next(
                    (s["name"] for s in strat_cfg if s["top"] <= zap <= s["base"]),
                    "—")
                st.markdown(
                    f'<div class="info-box" style="border-color:{c["color_int"]}">'
                    f'Zapato <b>{c["od_pulg"]}"</b> @ {zap} m<br>'
                    f'Formacion: <b>{form_zap}</b>'
                    f'</div>',
                    unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-title">Columna Estratigrafica</div>',
                    unsafe_allow_html=True)
        # Tabla con espesor y si algun zapato/packer cae en esa formación
        rows = []
        for s in strat_cfg:
            espesor = s["base"] - s["top"]
            # Chequear si hay zapatos en esta formacion
            zaps_en = [f'{c["od_pulg"]}"' for c in casings_cfg
                       if s["top"] <= c["zapato"] <= s["base"]]
            pack_en = "✅" if (show_packer and packer_md
                               and s["top"] <= packer_md <= s["base"]) else ""
            rows.append({
                "Formacion": s["name"],
                "Tope (m)": s["top"],
                "Base (m)": s["base"],
                "Espesor (m)": espesor,
                "Zapatos": ", ".join(zaps_en) if zaps_en else "—",
                "Packer": pack_en if pack_en else "—",
            })
        df_strat = pd.DataFrame(rows)
        st.dataframe(df_strat, hide_index=True, use_container_width=True)

        # Leyenda de colores de formaciones
        st.markdown("**Leyenda de colores:**")
        for s in strat_cfg:
            st.markdown(
                f'<div class="strat-row">'
                f'<div class="strat-swatch" style="background:{s["color"]}"></div>'
                f'<span class="strat-name">{s["name"]}</span>'
                f'<span class="strat-depth">{s["top"]}–{s["base"]} m '
                f'({s["base"]-s["top"]} m)</span>'
                f'</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-title">Resumen del pozo</div>',
                    unsafe_allow_html=True)
        df_sum = pd.DataFrame({
            "Parametro": [
                "Nombre","Formacion","Cuenca","Orientacion",
                "MD Total","TVD Total",
                "N° casings","Completacion",
                "Fondo tubing",
            ],
            "Valor": [
                nombre_pozo, formacion, cuenca, orientacion,
                f"{md_total} m", f"{tvd_total} m",
                str(len(casings_cfg)), comp_tipo,
                f"{tbg_fondo} m MD",
            ],
        })
        st.dataframe(df_sum, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("""
## Guia de Uso — WellViewer v1.1

### Como armar un pozo
1. **Datos generales**: nombre, formacion, profundidades MD y TVD
2. **Casings**: elegir cuantos strings y configurar cada uno (OD, ID, grado, zapato)
   - El orden es de afuera hacia adentro (conductor primero)
   - El cemento se dibuja automaticamente entre strings consecutivos
3. **Tubing**: definir OD, ID, grado y profundidad de fondo
4. **Completacion**: elegir el tipo y configurar los accesorios
5. **Estratigrafia**: ajustar los topes y bases de cada formacion

### Tipos de completacion
| Tipo | Accesorios disponibles |
|------|----------------------|
| Produccion Natural | Packer |
| Gas Lift Continuo | Packer + valvulas GLV + gas en annulus |
| BES | Motor + bomba submergible |
| Inyector | Solo casings + tubing |

### Columna estratigrafica
- Se activa desde **"🪨 Columna Estratigrafica"** en el sidebar
- Ajustar los topes y bases de cada formacion con los inputs
- Las **lineas de contacto** marcan cada limite de formacion
- En la tab **Datos tecnicos** se muestra que zapatos y el packer caen en cada formacion
- La columna usa la estratigrafia tipica de la Cuenca Neuquina (editable)

### Fracturamiento hidraulico
Activar en la seccion "Fracturamiento" del sidebar. Configura:
- Numero de etapas
- Profundidad de inicio
- Espaciamiento entre etapas
- Radio visual (referencia geometrica, no a escala real)

### Corte transversal
- Activar **"Aplicar corte"** para ver el interior
- El angulo controla cuanto se muestra (180° = mitad del pozo)
- Revela cemento → casings → gas/fluido → tubing

### Controles 3D
| Accion | Control |
|--------|---------|
| Rotar | Click + arrastrar |
| Zoom | Scroll / pellizco |
| Pan | Click derecho |
| Info elemento | Hover |
| Exportar imagen | Boton camara en la barra |

### Para intervenciones
1. Activar **"Intervencion"** en el sidebar
2. Fijar la profundidad del trabajo
3. Seleccionar el tipo de operacion
4. La cruz roja aparece en el lugar exacto del pozo

---
*WellViewer v1.1 | Portfolio Ingenieria de Yacimientos | Streamlit + Plotly*
""")

st.markdown("---")
st.markdown(
    "<center><small>WellViewer v1.1 | Generador generico de pozos 3D | "
    "Streamlit + Plotly</small></center>",
    unsafe_allow_html=True)

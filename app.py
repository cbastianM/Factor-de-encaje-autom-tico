import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from PIL import Image
import io
import base64

st.set_page_config(page_title="Morfología de Agregados", page_icon="🪨", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 10px; padding: 0.8rem 1rem 0.6rem 1rem;
}
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #94a3b8; }
/* Radio buttons como pills */
div[role="radiogroup"] { display: flex; gap: 0.5rem; }
div[role="radiogroup"] label {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 0.45rem 1rem;
    cursor: pointer; font-size: 0.88rem; transition: all 0.15s;
}
div[role="radiogroup"] label:hover { border-color: #64748b; }
.leyenda {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.55rem 0.9rem;
    font-size: 0.8rem; margin-bottom: 0.5rem; line-height: 2;
}
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("🪨 Análisis Morfológico de Agregados")
st.caption(
    "**Redondez** — círculo inscrito / circunscrito  ·  "
    "**Angularidad** — desviación vs. elipse ajustada  ·  ASTM C136 / INVIAS"
)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
archivo = st.file_uploader("📷 Sube la imagen del ensayo", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.info("💡 Fotografía desde arriba, fondo oscuro liso, iluminación uniforme.")
    st.stop()

pil_img = Image.open(archivo).convert("RGB")
img_np  = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
H, W    = img_np.shape[:2]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
TAMICES = {
    '3"   – 75.0 mm': 75.0, '2½" – 63.0 mm': 63.0, '2"   – 50.0 mm': 50.0,
    '1½" – 37.5 mm': 37.5, '1"   – 25.0 mm': 25.0, '¾"  – 19.0 mm': 19.0,
    '½"  – 12.5 mm': 12.5, '⅜"  –  9.5 mm':  9.5, 'N°4  –  4.75 mm': 4.75,
}

st.markdown("**🔧 Configuración del ensayo**")
col_tam, col_esc = st.columns(2)
with col_tam:
    tamiz_sel = st.selectbox("Tamiz de análisis", list(TAMICES.keys()), index=4)
    d_mm = TAMICES[tamiz_sel]
    st.info(f"Apertura: **{d_mm} mm** — partículas con diámetro ≥ {d_mm} mm")
with col_esc:
    ancho_real_cm = st.number_input("Ancho real de la imagen (cm)", 1.0, 500.0, 30.0, 0.5)
    px_por_mm = W / (ancho_real_cm * 10)
    area_min  = math.pi * (d_mm / 2) ** 2 * px_por_mm ** 2
    st.success(f"Escala: **{px_por_mm:.2f} px/mm** — área mínima: **{area_min:.0f} px²**")

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN
# ─────────────────────────────────────────────────────────────────────────────
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray    = clahe.apply(gray)
blur    = cv2.GaussianBlur(gray, (7, 7), 0)
_, msk  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
msk     = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel, iterations=2)
msk     = cv2.morphologyEx(msk, cv2.MORPH_OPEN,  kernel, iterations=1)
contornos, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contornos    = [c for c in contornos if cv2.contourArea(c) >= area_min]

if not contornos:
    st.warning("No se detectaron partículas. Verifica el ancho real o mejora la iluminación.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ramanujan_perimeter(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

def circle_pts(cx: float, cy: float, r: float):
    t  = np.linspace(0, 2 * math.pi, 120)
    xs = (cx + r * np.cos(t)).tolist()
    ys = ((H - cy) + r * np.sin(t)).tolist()   # Y invertida para Plotly
    return xs, ys

def ellipse_pts(cx: float, cy: float, a: float, b: float, angle_deg: float):
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    t   = np.linspace(0, 2 * math.pi, 140)
    xl  = a * np.cos(t);  yl = b * np.sin(t)
    xe  = xl * ca - yl * sa + cx
    ye  = xl * sa + yl * ca + cy
    return xe.tolist(), (H - ye).tolist()       # Y invertida para Plotly

# Paletas
COLORES_FORMA = {
    "Redondeada":     ("rgba(74,222,128,0.20)",  "rgb(74,222,128)"),
    "Sub-redondeada": ("rgba(250,204,21,0.20)",  "rgb(250,204,21)"),
    "Angular":        ("rgba(248,113,113,0.20)", "rgb(248,113,113)"),
}
COLORES_ANG = {
    "Baja":  ("rgba(74,222,128,0.20)",  "rgb(74,222,128)"),
    "Media": ("rgba(250,204,21,0.20)",  "rgb(250,204,21)"),
    "Alta":  ("rgba(248,113,113,0.20)", "rgb(248,113,113)"),
}

def clas_forma(r: float) -> str:
    if r >= 0.70: return "Redondeada"
    if r >= 0.50: return "Sub-redondeada"
    return "Angular"

def clas_ang(a: float) -> str:
    if a < 5:  return "Baja"
    if a < 15: return "Media"
    return "Alta"

def clas_elong(e: float) -> str:
    if e < 1.5: return "Isométrica"
    if e < 2.0: return "Elongada"
    return "Muy elongada"

PLOT_BG = dict(
    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE MÉTRICAS (una sola vez)
# ─────────────────────────────────────────────────────────────────────────────
particles = []
filas     = []

for i, cnt in enumerate(contornos):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if area == 0 or perim == 0:
        continue

    # ── Alto y Ancho (caja orientada mínima)
    rect = cv2.minAreaRect(cnt)
    (_, _), (wr, hr), _ = rect
    ancho_mm  = min(wr, hr) / px_por_mm
    alto_mm   = max(wr, hr) / px_por_mm
    elongacion = alto_mm / ancho_mm if ancho_mm > 0 else 1.0

    # ── REDONDEZ: círculo inscrito (transformada de distancia)
    #             / círculo circunscrito (minEnclosingCircle)
    xb, yb, wb, hb = cv2.boundingRect(cnt)
    pad = 4
    x0, y0 = max(0, xb - pad), max(0, yb - pad)
    x1, y1 = min(W, xb + wb + pad), min(H, yb + hb + pad)
    cnt_loc = cnt - np.array([[[x0, y0]]])
    msk_loc = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(msk_loc, [cnt_loc], -1, 255, cv2.FILLED)
    dist_t  = cv2.distanceTransform(msk_loc, cv2.DIST_L2, 5)
    r_in_px = float(dist_t.max())
    _, _, _, loc_in = cv2.minMaxLoc(dist_t)
    cx_in   = float(x0 + loc_in[0])
    cy_in   = float(y0 + loc_in[1])
    r_in_mm = r_in_px / px_por_mm

    (cxo, cyo), r_out_px = cv2.minEnclosingCircle(cnt)
    r_out_px = float(r_out_px)
    r_out_mm = r_out_px / px_por_mm
    redondez = r_in_px / r_out_px if r_out_px > 0 else 0.0

    # ── ANGULARIDAD: (P_real / P_elipse_ajustada − 1) × 100 %
    tiene_elipse = len(cnt) >= 5
    ex = ey = a_ell = b_ell = ell_ang = 0.0
    if tiene_elipse:
        try:
            elipse_cv           = cv2.fitEllipse(cnt)
            (ex, ey), (MA, ma), ell_ang = elipse_cv
            a_ell, b_ell        = MA / 2, ma / 2
            p_ell               = ramanujan_perimeter(a_ell, b_ell)
            angularidad         = max(0.0, (perim / p_ell - 1) * 100) if p_ell > 0 else 0.0
        except Exception:
            tiene_elipse = False
            angularidad  = 0.0
    if not tiene_elipse:
        angularidad = 0.0

    # Clasificaciones
    forma_cls = clas_forma(redondez)
    ang_cls   = clas_ang(angularidad)
    elong_cls = clas_elong(elongacion)

    # ── Geometría Plotly (Y invertida)
    pts = cnt.squeeze()
    if pts.ndim == 1: pts = pts[np.newaxis, :]
    xs_cnt = pts[:, 0].tolist() + [pts[0, 0]]
    ys_cnt = (H - pts[:, 1]).tolist() + [H - pts[0, 1]]

    xci, yci = circle_pts(cx_in, cy_in, r_in_px)       # círculo inscrito
    xco, yco = circle_pts(cxo, cyo, r_out_px)           # círculo circunscrito

    xe, ye = [], []
    if tiene_elipse and a_ell > 0 and b_ell > 0:
        xe, ye = ellipse_pts(ex, ey, a_ell, b_ell, ell_ang)

    # Centroide para la etiqueta
    M_mom = cv2.moments(cnt)
    if M_mom["m00"] != 0:
        lcx = M_mom["m10"] / M_mom["m00"]
        lcy = H - M_mom["m01"] / M_mom["m00"]
    else:
        lcx = xb + wb / 2
        lcy = H - (yb + hb / 2)

    area_mm2 = area  / px_por_mm ** 2
    perim_mm = perim / px_por_mm

    hover = (
        f"<b>Partícula {i + 1}</b><br>"
        f"──────────────────────────────<br>"
        f"↔️ Ancho: <b>{ancho_mm:.1f} mm</b>  ↕️ Alto: <b>{alto_mm:.1f} mm</b><br>"
        f"📐 Área: <b>{area_mm2:.1f} mm²</b>  📏 Per.: <b>{perim_mm:.1f} mm</b><br>"
        f"<br>"
        f"⊙ r inscrito:  <b>{r_in_mm:.2f} mm</b><br>"
        f"⊙ R circunsc.: <b>{r_out_mm:.2f} mm</b><br>"
        f"⭕ <b>Redondez = {redondez:.3f}</b>  → {forma_cls}<br>"
        f"<br>"
        f"⬡ P elipse: <b>{ramanujan_perimeter(a_ell, b_ell)/px_por_mm:.1f} mm</b><br>"
        f"⬡ <b>Angularidad = {angularidad:.1f}%</b>  → {ang_cls}<br>"
        f"<br>"
        f"📊 Elongación: <b>{elongacion:.2f}</b>  → {elong_cls}"
    )

    particles.append(dict(
        idx=i,
        xs_cnt=xs_cnt, ys_cnt=ys_cnt,
        xci=xci, yci=yci,
        xco=xco, yco=yco,
        xe=xe,   ye=ye,
        lcx=lcx, lcy=lcy,
        redondez=redondez,
        angularidad=angularidad,
        forma_cls=forma_cls,
        ang_cls=ang_cls,
        elong_cls=elong_cls,
        elongacion=elongacion,
        area_mm2=area_mm2,
        perim_mm=perim_mm,
        ancho_mm=ancho_mm,
        alto_mm=alto_mm,
        r_in_mm=r_in_mm,
        r_out_mm=r_out_mm,
        hover=hover,
        tiene_elipse=tiene_elipse,
    ))

    filas.append({
        "N°":              i + 1,
        "Ancho (mm)":      round(ancho_mm,    2),
        "Alto (mm)":       round(alto_mm,     2),
        "Área (mm²)":      round(area_mm2,    2),
        "Perímetro (mm)":  round(perim_mm,    2),
        "r inscrito (mm)": round(r_in_mm,     2),
        "R circunsc.(mm)": round(r_out_mm,    2),
        "Redondez":        round(redondez,     3),
        "Angularidad (%)": round(angularidad,  2),
        "Elongación":      round(elongacion,   3),
        "Forma":           forma_cls,
        "Ang. Clase":      ang_cls,
        "Elong. Clase":    elong_cls,
    })

df      = pd.DataFrame(filas)
img_b64 = pil_to_b64(pil_img)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTOR DE FIGURA  (se llama cada vez que cambia el modo)
# ─────────────────────────────────────────────────────────────────────────────
def build_fig(modo: str) -> go.Figure:
    fig = go.Figure()
    fig.add_layout_image(
        source=img_b64, x=0, y=H, xref="x", yref="y",
        sizex=W, sizey=H, sizing="stretch", layer="below",
    )

    for p in particles:
        f_fill, f_line = COLORES_FORMA[p["forma_cls"]]
        a_fill, a_line = COLORES_ANG[p["ang_cls"]]

        hcfg_forma = dict(bgcolor="#0f172a", bordercolor=f_line,
                          font=dict(color="white", size=12, family="'Courier New', monospace"))
        hcfg_ang   = dict(bgcolor="#0f172a", bordercolor=a_line,
                          font=dict(color="white", size=12, family="'Courier New', monospace"))

        # ── MODO: REDONDEZ ──────────────────────────────────────────────
        if modo == "redondez":
            # 1. Círculo circunscrito (anillo exterior, color de forma)
            fig.add_trace(go.Scatter(
                x=p["xco"], y=p["yco"], mode="lines",
                line=dict(color=f_line, width=2.5),
                fill="toself", fillcolor="rgba(0,0,0,0)",
                hoverinfo="skip", showlegend=False,
            ))
            # 2. Contorno de la partícula (transparente, para ver círculos)
            fig.add_trace(go.Scatter(
                x=p["xs_cnt"], y=p["ys_cnt"], mode="lines",
                fill="toself", fillcolor=f_fill,
                line=dict(color=f_line, width=1.5),
                hoverinfo="text", hovertext=p["hover"], hoverlabel=hcfg_forma,
                showlegend=False,
            ))
            # 3. Círculo inscrito (cian, interior)
            fig.add_trace(go.Scatter(
                x=p["xci"], y=p["yci"], mode="lines",
                line=dict(color="#38bdf8", width=2.5),
                fill="toself", fillcolor="rgba(56,189,248,0.08)",
                hoverinfo="skip", showlegend=False,
            ))
            label_txt  = f"{p['redondez']:.2f}"
            label_fill = f_fill
            label_line = f_line
            label_sz   = 30

        # ── MODO: ANGULARIDAD ────────────────────────────────────────────
        elif modo == "angularidad":
            # 1. Elipse ajustada (color de clase de angularidad)
            if p["tiene_elipse"] and len(p["xe"]) > 0:
                fig.add_trace(go.Scatter(
                    x=p["xe"], y=p["ye"], mode="lines",
                    line=dict(color=a_line, width=2.5),
                    fill="toself", fillcolor="rgba(0,0,0,0)",
                    hoverinfo="skip", showlegend=False,
                ))
            # 2. Contorno de la partícula (coloreado por angularidad)
            fig.add_trace(go.Scatter(
                x=p["xs_cnt"], y=p["ys_cnt"], mode="lines",
                fill="toself", fillcolor=a_fill,
                line=dict(color=a_line, width=1.5),
                hoverinfo="text", hovertext=p["hover"], hoverlabel=hcfg_ang,
                showlegend=False,
            ))
            label_txt  = f"{p['angularidad']:.0f}%"
            label_fill = a_fill
            label_line = a_line
            label_sz   = 28

        # ── MODO: ÁREA Y PERÍMETRO ───────────────────────────────────────
        else:
            fig.add_trace(go.Scatter(
                x=p["xs_cnt"], y=p["ys_cnt"], mode="lines",
                fill="toself", fillcolor=f_fill,
                line=dict(color=f_line, width=2),
                hoverinfo="text", hovertext=p["hover"], hoverlabel=hcfg_forma,
                showlegend=False,
            ))
            label_txt  = str(p["idx"] + 1)
            label_fill = f_fill
            label_line = f_line
            label_sz   = 20

        # Etiqueta central (común a los tres modos)
        fig.add_trace(go.Scatter(
            x=[p["lcx"]], y=[p["lcy"]],
            mode="text+markers",
            text=[label_txt],
            textfont=dict(size=9, color="white", family="'Courier New', Bold"),
            marker=dict(size=label_sz, color=label_fill,
                        line=dict(color=label_line, width=1.5)),
            hoverinfo="text", hovertext=p["hover"],
            hoverlabel=dict(bgcolor="#0f172a", bordercolor=label_line,
                            font=dict(color="white", size=12,
                                      family="'Courier New', monospace")),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#000", paper_bgcolor="#000",
        hovermode="closest", dragmode="pan",
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LEYENDAS POR MODO
# ─────────────────────────────────────────────────────────────────────────────
LEYENDAS = {
    "redondez": (
        '<div class="leyenda">'
        '<b>Redondez = r_inscrito / R_circunscrito</b><br>'
        '<span class="dot" style="background:#38bdf8"></span>Círculo inscrito (cian)&emsp;'
        '<span style="display:inline-block;width:18px;height:3px;background:#fff;vertical-align:middle;margin-right:4px;border-radius:2px"></span>Círculo circunscrito (color de forma)<br>'
        '<span class="dot" style="background:#4ade80"></span>Redondeada ≥ 0.70&emsp;'
        '<span class="dot" style="background:#facc15"></span>Sub-redondeada 0.50–0.70&emsp;'
        '<span class="dot" style="background:#f87171"></span>Angular &lt; 0.50&emsp;'
        '<span style="color:#94a3b8">· Etiqueta = valor de redondez</span>'
        '</div>'
    ),
    "angularidad": (
        '<div class="leyenda">'
        '<b>Angularidad = (P_real / P_elipse − 1) × 100 %</b><br>'
        '<span style="display:inline-block;width:18px;height:3px;background:#fff;vertical-align:middle;margin-right:4px;border-radius:2px"></span>Elipse ajustada (borde sólido = clase de angularidad)<br>'
        '<span class="dot" style="background:#4ade80"></span>Baja &lt; 5 %&emsp;'
        '<span class="dot" style="background:#facc15"></span>Media 5–15 %&emsp;'
        '<span class="dot" style="background:#f87171"></span>Alta &gt; 15 %&emsp;'
        '<span style="color:#94a3b8">· Etiqueta = angularidad en %</span>'
        '</div>'
    ),
    "area": (
        '<div class="leyenda">'
        '<b>Área y Perímetro por partícula</b><br>'
        '<span class="dot" style="background:#4ade80"></span>Redondeada (R ≥ 0.70)&emsp;'
        '<span class="dot" style="background:#facc15"></span>Sub-redondeada (0.50–0.70)&emsp;'
        '<span class="dot" style="background:#f87171"></span>Angular (&lt; 0.50)&emsp;'
        '<span style="color:#94a3b8">· Hover → área y perímetro exactos</span>'
        '</div>'
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_det, tab_dash, tab_tabla = st.tabs(["🔍 Detección", "📊 Dashboard", "📋 Tabla"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  DETECCIÓN + SELECTOR DE MODO
# ══════════════════════════════════════════════════════════════════════════════
with tab_det:
    # ── Selector de modo
    modo_lbl = st.radio(
        "Vista de análisis",
        ["⊙  Redondez", "⬡  Angularidad", "📐  Área y Perímetro"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if "Redondez" in modo_lbl:
        modo_code = "redondez"
    elif "Angularidad" in modo_lbl:
        modo_code = "angularidad"
    else:
        modo_code = "area"

    # ── Columnas: original | detección
    c_orig, c_det = st.columns(2)

    with c_orig:
        st.subheader("Imagen original")
        st.image(img_np, use_container_width=True)

    with c_det:
        st.subheader(f"Detectadas: **{len(particles)}** partículas  ·  Tamiz {tamiz_sel.strip()}")
        st.markdown(LEYENDAS[modo_code], unsafe_allow_html=True)
        st.plotly_chart(
            build_fig(modo_code),
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": False},
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    n       = len(df)
    pct_red = round((df["Forma"] == "Redondeada").sum() / n * 100, 1)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🪨 Partículas",    n)
    m2.metric("⭕ Redondez μ",    f"{df['Redondez'].mean():.3f}")
    m3.metric("⬡ Angularidad μ", f"{df['Angularidad (%)'].mean():.1f} %")
    m4.metric("📊 Elongación μ", f"{df['Elongación'].mean():.2f}")
    m5.metric("r inscrito μ",    f"{df['r inscrito (mm)'].mean():.1f} mm")
    m6.metric("✅ % Redondeadas",f"{pct_red} %")

    st.divider()

    SHAPE_C = {"Redondeada": "#4ade80", "Sub-redondeada": "#facc15", "Angular": "#f87171"}
    ANG_C   = {"Baja": "#4ade80", "Media": "#facc15", "Alta": "#f87171"}
    ELONG_C = {"Isométrica": "#4ade80", "Elongada": "#facc15", "Muy elongada": "#f87171"}

    c1, c2 = st.columns(2)
    with c1:
        fig_sc = go.Figure()
        for shape, color in SHAPE_C.items():
            sub = df[df["Forma"] == shape]
            if not sub.empty:
                fig_sc.add_trace(go.Scatter(
                    x=sub["Ancho (mm)"], y=sub["Alto (mm)"],
                    mode="markers+text", text=sub["N°"].astype(str),
                    textposition="top center", textfont=dict(size=8, color="#e2e8f0"),
                    marker=dict(size=12, color=color, opacity=0.85,
                                line=dict(color="#0f172a", width=0.8)),
                    name=shape,
                    hovertemplate="<b>P%{text}</b>  Ancho %{x:.1f} mm  Alto %{y:.1f} mm<extra></extra>",
                ))
        max_d = max(df["Alto (mm)"].max(), df["Ancho (mm)"].max()) * 1.15
        fig_sc.add_trace(go.Scatter(x=[0, max_d], y=[0, max_d], mode="lines",
                                    line=dict(dash="dash", color="#475569", width=1),
                                    name="1:1", showlegend=True))
        fig_sc.update_layout(title="Alto vs Ancho por partícula",
                             xaxis_title="Ancho (mm)", yaxis_title="Alto (mm)",
                             height=360, legend=dict(bgcolor="rgba(0,0,0,0)"), **PLOT_BG)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        shape_vc = df["Forma"].value_counts()
        fig_pie  = go.Figure(go.Pie(
            labels=shape_vc.index, values=shape_vc.values,
            marker_colors=[SHAPE_C.get(k, "#999") for k in shape_vc.index],
            hole=0.52, textinfo="label+percent", textfont=dict(size=12),
            hovertemplate="%{label}: %{value} partícula(s)<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{n}</b><br>partículas", x=0.5, y=0.5,
                               font=dict(size=15, color="#e2e8f0"), showarrow=False)
        fig_pie.update_layout(title="Distribución de Forma", height=360,
                              showlegend=False, plot_bgcolor="#0f172a",
                              paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_pie, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        mu_r   = df["Redondez"].mean()
        fig_hr = go.Figure()
        fig_hr.add_vrect(x0=0.0, x1=0.5,  fillcolor="rgba(248,113,113,0.07)", line_width=0,
                         annotation_text="Angular", annotation_position="top left",
                         annotation_font=dict(color="#f87171", size=9))
        fig_hr.add_vrect(x0=0.5, x1=0.7,  fillcolor="rgba(250,204,21,0.07)", line_width=0,
                         annotation_text="Sub-red.", annotation_position="top left",
                         annotation_font=dict(color="#facc15", size=9))
        fig_hr.add_vrect(x0=0.7, x1=1.01, fillcolor="rgba(74,222,128,0.07)", line_width=0,
                         annotation_text="Redondeada", annotation_position="top left",
                         annotation_font=dict(color="#4ade80", size=9))
        fig_hr.add_trace(go.Histogram(x=df["Redondez"], nbinsx=15,
                                      marker=dict(color="#38bdf8", opacity=0.85,
                                                  line=dict(color="#0f172a", width=0.5)),
                                      hovertemplate="Redondez %{x:.2f}: %{y} partículas<extra></extra>"))
        fig_hr.add_vline(x=mu_r, line_dash="dash", line_color="#facc15",
                         annotation_text=f"μ = {mu_r:.3f}",
                         annotation_font_color="#facc15", annotation_position="top right")
        fig_hr.update_layout(title="Distribución de Redondez  (r / R)",
                             xaxis_title="Redondez (0–1)", yaxis_title="N° partículas",
                             height=300, showlegend=False, **PLOT_BG)
        st.plotly_chart(fig_hr, use_container_width=True)

    with c4:
        mu_a   = df["Angularidad (%)"].mean()
        max_a  = max(df["Angularidad (%)"].max() * 1.1 + 1, 20)
        fig_ha = go.Figure()
        fig_ha.add_vrect(x0=0,  x1=5,    fillcolor="rgba(74,222,128,0.07)", line_width=0,
                         annotation_text="Baja", annotation_position="top left",
                         annotation_font=dict(color="#4ade80", size=9))
        fig_ha.add_vrect(x0=5,  x1=15,   fillcolor="rgba(250,204,21,0.07)", line_width=0,
                         annotation_text="Media", annotation_position="top left",
                         annotation_font=dict(color="#facc15", size=9))
        fig_ha.add_vrect(x0=15, x1=max_a, fillcolor="rgba(248,113,113,0.07)", line_width=0,
                         annotation_text="Alta", annotation_position="top left",
                         annotation_font=dict(color="#f87171", size=9))
        fig_ha.add_trace(go.Histogram(x=df["Angularidad (%)"], nbinsx=15,
                                      marker=dict(color="#fb923c", opacity=0.85,
                                                  line=dict(color="#0f172a", width=0.5)),
                                      hovertemplate="Angularidad %{x:.1f}%%: %{y} partículas<extra></extra>"))
        fig_ha.add_vline(x=mu_a, line_dash="dash", line_color="#facc15",
                         annotation_text=f"μ = {mu_a:.1f}%",
                         annotation_font_color="#facc15", annotation_position="top right")
        fig_ha.update_layout(title="Distribución de Angularidad  (P_real / P_elipse − 1) × 100",
                             xaxis_title="Angularidad (%)", yaxis_title="N° partículas",
                             height=300, showlegend=False, **PLOT_BG)
        st.plotly_chart(fig_ha, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig_box = go.Figure()
        for col_n, color, lbl in [
            ("r inscrito (mm)", "#38bdf8", "r inscrito"),
            ("R circunsc.(mm)", "#818cf8", "R circunscrito"),
            ("Ancho (mm)",      "#4ade80", "Ancho"),
            ("Alto (mm)",       "#f87171", "Alto"),
        ]:
            fig_box.add_trace(go.Box(y=df[col_n], name=lbl, marker_color=color, boxmean="sd",
                                     hovertemplate=f"{lbl}: %{{y:.1f}} mm<extra></extra>"))
        fig_box.update_layout(title="Radios y Dimensiones (mm)",
                              yaxis_title="mm", height=300, showlegend=False, **PLOT_BG)
        st.plotly_chart(fig_box, use_container_width=True)

    with c6:
        ang_vc   = df["Ang. Clase"].value_counts().reindex(["Baja","Media","Alta"]).fillna(0)
        elong_vc = df["Elong. Clase"].value_counts().reindex(
                       ["Isométrica","Elongada","Muy elongada"]).fillna(0)
        fig_bar  = go.Figure()
        for lbl, val, clr in zip(ang_vc.index, ang_vc.values, [ANG_C[k] for k in ang_vc.index]):
            fig_bar.add_trace(go.Bar(x=["Angularidad"], y=[val], name=lbl, marker_color=clr,
                                     text=[int(val)], textposition="auto",
                                     hovertemplate=f"{lbl}: %{{y}}<extra></extra>"))
        for lbl, val, clr in zip(elong_vc.index, elong_vc.values,
                                  [ELONG_C[k] for k in elong_vc.index]):
            fig_bar.add_trace(go.Bar(x=["Elongación"], y=[val], name=lbl, marker_color=clr,
                                     text=[int(val)], textposition="auto", showlegend=False,
                                     hovertemplate=f"{lbl}: %{{y}}<extra></extra>"))
        fig_bar.update_layout(title="Clasificación: Angularidad y Elongación",
                              barmode="stack", yaxis_title="N° partículas",
                              height=300, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                              **PLOT_BG)
        st.plotly_chart(fig_bar, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        cmpct = (4 * math.pi * df["Área (mm²)"].mean() / df["Perímetro (mm)"].mean() ** 2
                 if df["Perímetro (mm)"].mean() > 0 else 0)
        radar = {
            "Redondez":      min(1.0, df["Redondez"].mean()),
            "1−Angularidad": max(0.0, 1.0 - df["Angularidad (%)"].mean() / 40.0),
            "Isometría":     max(0.0, min(1.0, 2.0 - df["Elongación"].mean())),
            "% Redondeadas": pct_red / 100.0,
            "Compacidad":    min(1.0, cmpct),
        }
        cats = list(radar.keys())
        vals = list(radar.values()) + [list(radar.values())[0]]
        fig_rad = go.Figure(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill="toself",
            fillcolor="rgba(56,189,248,0.15)",
            line=dict(color="#38bdf8", width=2.5), marker=dict(size=6, color="#38bdf8"),
        ))
        fig_rad.update_layout(
            title="Perfil Morfológico Promedio",
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e293b",
                                color="#94a3b8", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="#1e293b", color="#e2e8f0", tickfont=dict(size=10)),
                bgcolor="#0f172a",
            ),
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=360, showlegend=False,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    with c8:
        fig_rv = go.Figure()
        for shape, color in SHAPE_C.items():
            sub = df[df["Forma"] == shape]
            if not sub.empty:
                fig_rv.add_trace(go.Scatter(
                    x=sub["Redondez"], y=sub["Angularidad (%)"],
                    mode="markers+text", text=sub["N°"].astype(str),
                    textposition="top center", textfont=dict(size=8, color="#e2e8f0"),
                    marker=dict(size=12, color=color, opacity=0.85,
                                line=dict(color="#0f172a", width=0.8)),
                    name=shape,
                    hovertemplate="<b>P%{text}</b>  Redondez %{x:.3f}  Angularidad %{y:.1f}%<extra></extra>",
                ))
        fig_rv.update_layout(title="Redondez vs Angularidad",
                             xaxis_title="Redondez (r/R)", yaxis_title="Angularidad (%)",
                             height=360, legend=dict(bgcolor="rgba(0,0,0,0)"), **PLOT_BG)
        st.plotly_chart(fig_rv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  —  TABLA
# ══════════════════════════════════════════════════════════════════════════════
with tab_tabla:
    st.subheader(f"📋 Resultados — {n} partículas")
    with st.expander("📈 Estadísticas descriptivas"):
        cols_num = ["Ancho (mm)","Alto (mm)","Área (mm²)","Perímetro (mm)",
                    "r inscrito (mm)","R circunsc.(mm)","Redondez","Angularidad (%)","Elongación"]
        st.dataframe(df[cols_num].describe().round(3), use_container_width=True)

    def color_r(v):
        r = max(0.0, min(1.0, v))
        return f"color: rgb({int(248*(1-r)+74*r)},{int(113*(1-r)+222*r)},{int(113*(1-r)+128*r)})"

    fmt = {"Ancho (mm)":"{:.2f}","Alto (mm)":"{:.2f}","Área (mm²)":"{:.2f}",
           "Perímetro (mm)":"{:.2f}","r inscrito (mm)":"{:.2f}","R circunsc.(mm)":"{:.2f}",
           "Redondez":"{:.3f}","Angularidad (%)":"{:.2f}","Elongación":"{:.3f}"}
    try:
        styled = df.style.applymap(color_r, subset=["Redondez"]).format(fmt)
    except AttributeError:
        styled = df.style.map(color_r, subset=["Redondez"]).format(fmt)

    st.dataframe(styled, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv,
                       file_name="morfologia_agregados.csv", mime="text/csv")

st.divider()
st.caption(
    "**Redondez** = r_inscrito / R_circunscrito  (transformada de distancia · minEnclosingCircle)  ·  "
    "**Angularidad (%)** = (P_real / P_elipse − 1)×100, elipse por cv2.fitEllipse + Ramanujan  ·  "
    "**Elongación** = Alto / Ancho sobre minAreaRect"
)

import math, io, base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from skimage import exposure, filters, morphology, measure
from skimage.color import rgb2gray

st.set_page_config(page_title="Análisis de Agregados", page_icon="🪨", layout="wide")
st.title("🪨 Análisis de Agregados Gruesos")
st.caption("Pasa el ratón sobre cada partícula para ver su perímetro y área.")

# ── CARGA ─────────────────────────────────────────────────────────────────────
archivo = st.file_uploader("📷 Sube la imagen", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.info("Recomendación: agrega sobre un fondo oscuro liso con buena iluminación.")
    st.stop()

pil_img = Image.open(archivo).convert("RGB")
img_np  = np.array(pil_img)
H, W    = img_np.shape[:2]

# ── TAMICES ───────────────────────────────────────────────────────────────────
TAMICES = {
    '3"   – 75.0 mm':  75.0,
    '2½" – 63.0 mm':  63.0,
    '2"   – 50.0 mm':  50.0,
    '1½" – 37.5 mm':  37.5,
    '1"   – 25.0 mm':  25.0,
    '¾"  – 19.0 mm':  19.0,
    '½"  – 12.5 mm':  12.5,
    '⅜"  –  9.5 mm':   9.5,
    'N°4  –  4.75 mm':  4.75,
}

st.markdown("**🔧 Configuración del ensayo**")
col_tam, col_esc = st.columns(2)

with col_tam:
    st.caption("Selecciona el tamiz sobre el que quedaron retenidas las partículas.")
    tamiz_sel = st.selectbox("Tamiz de análisis", list(TAMICES.keys()), index=4)
    d_mm = TAMICES[tamiz_sel]
    st.info(f"Apertura: **{d_mm} mm** — se ignora todo objeto más pequeño que este diámetro.")

with col_esc:
    st.caption("Indica cuánto mide la foto en la realidad para convertir a milímetros.")
    ancho_real_cm = st.number_input(
        "Ancho real de la imagen (cm)",
        min_value=1.0, max_value=500.0, value=30.0, step=0.5,
        help="Mide con una regla cuántos cm abarca la foto de lado a lado."
    )
    px_por_mm   = W / (ancho_real_cm * 10)
    area_min_px = math.pi * (d_mm / 2) ** 2 * px_por_mm ** 2
    st.success(f"Escala: **{px_por_mm:.2f} px/mm** → área mínima = **{area_min_px:.0f} px²**")

# ── DETECCIÓN (solo skimage + numpy) ─────────────────────────────────────────
gray   = rgb2gray(img_np)
gray_e = exposure.equalize_adapthist(gray, clip_limit=0.03)
smooth = filters.gaussian(gray_e, sigma=2)

thresh = filters.threshold_otsu(smooth)
binary = smooth < thresh   # True = partícula (oscuro sobre fondo claro)
                           # Si fondo oscuro, lo invertimos automáticamente:
if binary.mean() > 0.5:
    binary = ~binary

# Morfología para limpiar
disk3 = morphology.disk(3)
disk5 = morphology.disk(5)
binary = morphology.binary_closing(binary, disk5)
binary = morphology.binary_opening(binary, disk3)
binary = morphology.remove_small_holes(binary, area_threshold=int(area_min_px))

# Etiquetar regiones
labeled = measure.label(binary)
props   = measure.regionprops(labeled)
props   = [p for p in props if p.area >= area_min_px]

if not props:
    st.warning(
        f"No se detectaron partículas del tamiz **{tamiz_sel}**. "
        "Verifica que el ancho real sea correcto o mejora el contraste de la foto."
    )
    st.stop()

# ── DOS COLUMNAS ──────────────────────────────────────────────────────────────
col_orig, col_det = st.columns(2)
with col_orig:
    st.subheader("Imagen original")
    st.image(img_np, use_container_width=True)

# ── PLOTLY ────────────────────────────────────────────────────────────────────
def pil_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def colores(r):
    if r <= 1.2:   return "rgba(74,222,128,0.28)",  "rgb(74,222,128)"
    elif r <= 1.5: return "rgba(250,204,21,0.28)",  "rgb(250,204,21)"
    else:          return "rgba(248,113,113,0.28)", "rgb(248,113,113)"

fig = go.Figure()
fig.add_layout_image(
    source=pil_b64(pil_img),
    x=0, y=H, xref="x", yref="y",
    sizex=W, sizey=H, sizing="stretch", layer="below"
)

filas = []
for i, prop in enumerate(props):
    area_px  = prop.area
    perim_px = prop.perimeter
    r_idx    = (perim_px ** 2) / (4 * math.pi * area_px) if area_px > 0 else 1.0
    clas     = "Favorable" if r_idx <= 1.2 else ("Moderado" if r_idx <= 1.5 else "Desfavorable")
    fill, line = colores(r_idx)

    area_mm2 = area_px  / (px_por_mm ** 2)
    perim_mm = perim_px / px_por_mm

    # Contorno de la región → coordenadas Plotly (Y invertido)
    contour_pts = measure.find_contours(labeled == prop.label, level=0.5)
    if not contour_pts:
        continue
    pts = contour_pts[0]            # (row, col)
    xs  = pts[:, 1].tolist() + [pts[0, 1]]
    ys  = (H - pts[:, 0]).tolist() + [H - pts[0, 0]]

    cy_plot = H - prop.centroid[0]
    cx_plot = prop.centroid[1]

    hover = (
        f"<b>Partícula {i+1}</b><br>"
        f"──────────────────────<br>"
        f"📐 Área: <b>{area_mm2:.1f} mm²</b><br>"
        f"📏 Perímetro: <b>{perim_mm:.1f} mm</b><br>"
        f"🔷 R: <b>{r_idx:.3f}</b><br>"
        f"✅ {clas}"
    )
    hlabel = dict(bgcolor="#0f172a", bordercolor=line,
                  font=dict(color="white", size=13, family="'Courier New'"))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, fill="toself", fillcolor=fill,
        line=dict(color=line, width=2), mode="lines",
        hoverinfo="text", hovertext=hover, hoverlabel=hlabel,
        showlegend=False, name=f"Partícula {i+1}",
    ))
    fig.add_trace(go.Scatter(
        x=[cx_plot], y=[cy_plot],
        mode="text+markers",
        text=[str(i + 1)],
        textfont=dict(size=11, color="white", family="'Courier New'"),
        marker=dict(size=22, color=fill, line=dict(color=line, width=2)),
        hoverinfo="text", hovertext=hover, hoverlabel=hlabel,
        showlegend=False,
    ))

    filas.append({
        "N°": i + 1,
        "Área (mm²)": round(area_mm2, 2),
        "Perímetro (mm)": round(perim_mm, 2),
        "R": round(r_idx, 3),
        "Clasificación": clas,
    })

fig.update_layout(
    xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
               visible=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(range=[0, H], showgrid=False, zeroline=False, visible=False),
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#000", paper_bgcolor="#000",
    hovermode="closest", dragmode="pan",
)

with col_det:
    st.subheader(f"{len(props)} partículas  ·  Tamiz {tamiz_sel.strip()}")
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": False})

# ── MÉTRICAS + TABLA ──────────────────────────────────────────────────────────
st.divider()
df      = pd.DataFrame(filas)
n_total = len(df)
n_fav   = (df["Clasificación"] == "Favorable").sum()
fe      = round(n_fav / n_total * 100, 1)

m1, m2, m3 = st.columns(3)
m1.metric("Partículas totales", n_total)
m2.metric("R promedio", round(df["R"].mean(), 3))
m3.metric("Factor de Encaje", f"{fe}%")

st.dataframe(df, use_container_width=True, hide_index=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV", data=csv,
                   file_name="agregados.csv", mime="text/csv")

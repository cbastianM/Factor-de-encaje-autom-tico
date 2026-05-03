import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from PIL import Image
import io
import base64

st.set_page_config(page_title="Análisis de Agregados", page_icon="🪨", layout="wide")

st.title("🪨 Análisis de Agregados Gruesos")
st.caption("Pasa el ratón sobre cada partícula para ver su perímetro y área.")

# ── Carga
archivo = st.file_uploader("📷 Sube la imagen", type=["jpg", "jpeg", "png"])

if archivo is None:
    st.info("Recomendación: coloca los agregados sobre un fondo oscuro liso con buena iluminación.")
    st.stop()

pil_img = Image.open(archivo).convert("RGB")
img_np  = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
H, W    = img_np.shape[:2]

# Tamices de agregado grueso (ASTM C136 / INVIAS)
TAMICES = {
    '3"   – 75.0 mm':   75.0,
    '2½" – 63.0 mm':   63.0,
    '2"   – 50.0 mm':   50.0,
    '1½" – 37.5 mm':   37.5,
    '1"   – 25.0 mm':   25.0,
    '¾"  – 19.0 mm':   19.0,
    '½"  – 12.5 mm':   12.5,
    '⅜"  –  9.5 mm':    9.5,
    'N°4  –  4.75 mm':   4.75,
}

st.markdown("**🔧 Configuración del ensayo**")
col_tam, col_esc = st.columns(2)

with col_tam:
    st.caption("Selecciona el tamiz sobre el que quedaron retenidas las partículas que quieres analizar.")
    tamiz_sel = st.selectbox(
        "Tamiz de análisis",
        options=list(TAMICES.keys()),
        index=4,   # 1" por defecto
    )
    d_mm = TAMICES[tamiz_sel]
    st.info(f"Apertura del tamiz: **{d_mm} mm** → partículas con diámetro ≥ {d_mm} mm")

with col_esc:
    st.caption("Para convertir píxeles a milímetros la app necesita saber cuánto mide la imagen en la realidad.")
    ancho_real_cm = st.number_input(
        "Ancho real de la imagen (cm)",
        min_value=1.0, max_value=500.0, value=30.0, step=0.5,
        help="Mide con una regla cuántos centímetros abarca la foto de lado a lado."
    )
    px_por_mm  = W / (ancho_real_cm * 10)           # píxeles por mm
    area_min   = math.pi * (d_mm / 2) ** 2 * px_por_mm ** 2   # área mínima en px²
    st.success(f"Escala: **{px_por_mm:.2f} px/mm** → área mínima = **{area_min:.0f} px²**")

# ── DETECCIÓN
gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray  = clahe.apply(gray)
blur  = cv2.GaussianBlur(gray, (7, 7), 0)
_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contornos    = [c for c in contornos if cv2.contourArea(c) >= area_min]

if not contornos:
    st.warning(
        f"No se detectaron partículas del tamiz **{tamiz_sel}**. "
        "Verifica que el ancho real de la imagen sea correcto, "
        "o mejora el contraste de la foto (fondo oscuro liso)."
    )
    st.stop()

# ── DOS COLUMNAS
col_orig, col_detect = st.columns(2)

with col_orig:
    st.subheader("Imagen original")
    st.image(img_np, use_container_width=True)

# ── Convertir imagen a base64 para Plotly
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

img_b64 = pil_to_b64(pil_img)

# ── FIGURA PLOTLY
fig = go.Figure()

fig.add_layout_image(
    source=img_b64,
    x=0, y=H,
    xref="x", yref="y",
    sizex=W, sizey=H,
    sizing="stretch",
    layer="below"
)

def colores(r):
    if r <= 1.2:
        return "rgba(74,222,128,0.30)", "rgb(74,222,128)"
    elif r <= 1.5:
        return "rgba(250,204,21,0.30)", "rgb(250,204,21)"
    else:
        return "rgba(248,113,113,0.30)", "rgb(248,113,113)"

filas = []

for i, cnt in enumerate(contornos):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    r_idx = (perim ** 2) / (4 * math.pi * area) if area > 0 else 0
    clas  = "Favorable" if r_idx <= 1.2 else ("Moderado" if r_idx <= 1.5 else "Desfavorable")
    fill, line = colores(r_idx)

    pts = cnt.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    xs = pts[:, 0].tolist() + [pts[0, 0]]
    ys = (H - pts[:, 1]).tolist() + [H - pts[0, 1]]   # invertir Y

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = H - M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w / 2, H - (y + h / 2)

    area_mm2  = area  / (px_por_mm ** 2)
    perim_mm  = perim / px_por_mm
    hover = (
        f"<b>Partícula {i+1}</b><br>"
        f"─────────────────────<br>"
        f"📐 Área: <b>{area_mm2:.1f} mm²</b><br>"
        f"📏 Perímetro: <b>{perim_mm:.1f} mm</b><br>"
        f"🔷 R: <b>{r_idx:.3f}</b><br>"
        f"✅ {clas}"
    )

    # Polígono relleno (contorno)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill="toself",
        fillcolor=fill,
        line=dict(color=line, width=2),
        mode="lines",
        hoverinfo="text",
        hovertext=hover,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor=line,
            font=dict(color="white", size=13, family="'Courier New', monospace")
        ),
        showlegend=False,
        name=f"Partícula {i+1}",
    ))

    # Número en el centro (con hover)
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode="text+markers",
        text=[str(i + 1)],
        textfont=dict(size=11, color="white", family="'Courier New'"),
        marker=dict(size=22, color=fill, line=dict(color=line, width=2)),
        hoverinfo="text",
        hovertext=hover,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor=line,
            font=dict(color="white", size=13, family="'Courier New', monospace")
        ),
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
    plot_bgcolor="#000",
    paper_bgcolor="#000",
    hovermode="closest",
    dragmode="pan",
)

with col_detect:
    st.subheader(f"Partículas detectadas: {len(contornos)}  ·  Tamiz {tamiz_sel.strip()}")
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": False})

# ── MÉTRICAS + TABLA
st.divider()
df = pd.DataFrame(filas)

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
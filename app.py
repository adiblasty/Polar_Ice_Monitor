import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# === CONFIG ===
CSV_PATH = "climate_output/melt_report.csv"
OVERLAY_DIR = "climate_output/overlays"
PLOT_PATH = "climate_output/melt_trend_plot.png"
MODEL_PATH = "runs/segment/train8/weights/best.pt"
DANGER_THRESHOLD = 60.0  # Alert threshold %

st.set_page_config(page_title="Polar Ice Monitor", layout="wide")

# === TITLE ===
st.title("🧊 Polar Ice Melt Monitoring Dashboard")
st.markdown("Using Vision AI to monitor climate impact in real time.")

from streamlit_folium import st_folium
import folium

# === MAP OF POLAR ZONES ===
st.subheader("🗺️ Polar Observation Regions")

# Define key polar observation points (approximate)
locations = {
    "Beaufort Sea (Arctic)": [72.0, -145.0],
    "Baffin Bay (Arctic)": [75.0, -70.0],
    "Ross Sea (Antarctica)": [-75.0, 160.0],
    "Weddell Sea (Antarctica)": [-75.0, -45.0]
}

m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB dark_matter")

# Add region markers
for name, coords in locations.items():
    folium.Marker(
        location=coords,
        popup=name,
        tooltip=f"Load sample for {name}",
        icon=folium.Icon(color='blue' if 'Arctic' in name else 'red')
    ).add_to(m)

# Render map in dashboard
st_data = st_folium(m, width=800, height=400)
selected = st_data['last_object_clicked']
sample_images = {
    "Beaufort Sea (Arctic)": "samples/beaufort.jpg",
    "Baffin Bay (Arctic)": "samples/baffin.jpg",
    "Ross Sea (Antarctica)": "samples/ross.jpg",
    "Weddell Sea (Antarctica)": "samples/weddell.jpg"
}

if selected:
    lat, lon = selected['lat'], selected['lng']
    st.info(f"You selected lat={lat:.2f}, lon={lon:.2f}")

    # Match region by proximity
    clicked_name = None
    for name, coords in locations.items():
        if abs(lat - coords[0]) < 1 and abs(lon - coords[1]) < 1:
            clicked_name = name
            break

    if clicked_name and clicked_name in sample_images:
        st.success(f"Auto-loading sample for **{clicked_name}**")
        st.image(sample_images[clicked_name], caption=f"{clicked_name} - Sample Image", use_container_width=True)


    st.info(f"You selected lat={lat:.2f}, lon={lon:.2f}")
    # Optionally: map this to a folder/image and run prediction


# === LOAD DATA ===
if not os.path.exists(CSV_PATH):
    st.warning("Melt report not found. Please run the analysis script first.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')

# === PLOT COVERAGE OVER TIME ===
st.subheader("Ice Coverage Trend For Hudson Bay Sea")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['date'], df['coverage_percent'], label='Ice Coverage (%)', color='blue')
danger = df[df['danger'] == 1]
ax.scatter(danger['date'], danger['coverage_percent'], color='red', label='Danger', zorder=5)
ax.set_ylabel("Coverage %")
ax.set_xlabel("Date")
ax.set_ylim(0, 100)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# === LATEST IMAGE & STATS ===
st.subheader("Latest Analysis")

latest = df.iloc[-1]
latest_file = latest['filename']
latest_path = os.path.join(OVERLAY_DIR, latest_file)

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Date", latest['date'].strftime("%Y-%m-%d %H:%M"))
    st.metric("Coverage %", f"{latest['coverage_percent']:.2f}%")
    status = "🚨 Danger" if latest['danger'] else "✅ Safe"
    st.metric("Status", status)

with col2:
    if os.path.exists(latest_path):
        st.image(Image.open(latest_path), caption=f"Overlay: {latest_file}", use_column_width=True)
    else:
        st.warning(f"No overlay found for {latest_file}")

# === MODEL LOADING ===
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# === Upload Analysis Section ===
st.subheader("📥 Upload New Image for Live Analysis")
uploaded = st.file_uploader("Upload a polar satellite image (.jpg or .png)", type=["jpg", "png"])

# === Required Imports (once, top or bottom of file) ===
from fpdf import FPDF
import base64
import datetime

def generate_upload_pdf(coverage_pct, danger_flag, overlay_np, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Polar Ice Melt - Uploaded Image Report", ln=1, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Image Name: {filename}", ln=1)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Coverage: {coverage_pct:.2f}%", ln=1)
    status = "DANGER " if danger_flag else "Safe "
    pdf.cell(0, 10, f"Status: {status}", ln=1)
    pdf.ln(10)

    temp_overlay = os.path.join(tempfile.gettempdir(), "overlay.jpg")
    cv2.imwrite(temp_overlay, overlay_np)
    pdf.image(temp_overlay, w=180)

    return pdf.output(dest="S")  # Ensure bytes for base64

# === Analysis and PDF ===
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        temp_img.write(uploaded.read())
        temp_path = temp_img.name

    st.image(temp_path, caption="Uploaded Image", use_column_width=True)
    results = model.predict(source=temp_path, imgsz=416, save=False)
    result = results[0]

    image = cv2.imread(temp_path)
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks:
        for m in result.masks.data:
            m_resized = cv2.resize(m.cpu().numpy(), (w, h))
            combined_mask = np.logical_or(combined_mask, m_resized > 0.5)

        ice_pixels = np.sum(combined_mask)
        total_pixels = h * w
        coverage_pct = 100 * ice_pixels / total_pixels
        danger_flag = int(coverage_pct < DANGER_THRESHOLD)

        overlay_color = (0, 0, 255) if danger_flag else (0, 255, 0)
        color_mask = np.zeros_like(image)
        color_mask[combined_mask] = overlay_color
        overlay = cv2.addWeighted(image, 1.0, color_mask, 0.4, 0)

        st.metric("Coverage %", f"{coverage_pct:.2f}%")
        st.metric("Status", "🚨 Danger" if danger_flag else "✅ Safe")
        st.image(overlay, caption="Overlay Preview", use_column_width=True)

        # PDF download button
        if st.button("📄 Download Report for This Image"):
            report_bytes = generate_upload_pdf(coverage_pct, danger_flag, overlay, uploaded.name)
            b64 = base64.b64encode(report_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Upload_Report.pdf">📥 Click here to download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("No ice detected in this image.")

    os.remove(temp_path)


from PIL import Image
import time

st.subheader("⏳ Timelapse: Ice Melt Over Time")

overlay_files = sorted([
    f for f in os.listdir(OVERLAY_DIR)
    if f.lower().endswith(('.jpg', '.png'))
])

if overlay_files:
    # Initialize session state
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    # UI controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Play"):
            st.session_state.playing = True
    with col2:
        if st.button("⏸️ Pause"):
            st.session_state.playing = False
    with col3:
        if st.button("🔁 Reset"):
            st.session_state.frame_index = 0
            st.session_state.playing = False

    speed = st.slider("Frame delay (seconds)", 0.1, 2.0, 0.5, 0.1)

    # Show image
    index = st.session_state.frame_index
    img = Image.open(os.path.join(OVERLAY_DIR, overlay_files[index]))
    st.image(img, caption=f"Frame {index+1}/{len(overlay_files)}: {overlay_files[index]}", use_column_width=True)

    # Autoplay logic
    if st.session_state.playing:
        time.sleep(speed)
        st.session_state.frame_index = (index + 1) % len(overlay_files)
        st.rerun()
else:
    st.warning("No overlays found. Run inference first.")

from fpdf import FPDF
import base64

def generate_rich_pdf_report(df, latest_row, trend_plot_path, overlay_path):
    pdf = FPDF()
    pdf.add_page()

    # === Title ===
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Hudson Bay Ice Melt Report", ln=1, align="C")
    pdf.ln(5)

    # === Latest Stats ===
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f" Date: {latest_row['date']}", ln=1)
    pdf.cell(0, 10, f" Coverage: {latest_row['coverage_percent']:.2f}%", ln=1)
    status = "DANGER " if latest_row['danger'] else "Safe ✅"
    pdf.cell(0, 10, f"Status: {status}", ln=1)
    pdf.ln(5)

    # === Summary Stats ===
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary Statistics", ln=1)
    pdf.set_font("Arial", "", 12)

    total = len(df)
    avg_coverage = df["coverage_percent"].mean()
    danger_count = df["danger"].sum()
    first_date = df["date"].min()
    last_date = df["date"].max()

    pdf.cell(0, 10, f"Total Samples: {total}", ln=1)
    pdf.cell(0, 10, f"Average Coverage: {avg_coverage:.2f}%", ln=1)
    pdf.cell(0, 10, f"Danger Events: {danger_count}", ln=1)
    pdf.cell(0, 10, f"Period: {first_date.date()} to {last_date.date()}", ln=1)
    pdf.ln(5)

    # === Add Trend Plot ===
    if os.path.exists(trend_plot_path):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Ice Coverage Trend", ln=1)
        pdf.image(trend_plot_path, w=180)
        pdf.ln(5)

    # === Add Latest Overlay ===
    if os.path.exists(overlay_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Latest Analysis Overlay", ln=1)
        pdf.image(overlay_path, w=180)

    # === Optional: List Danger Dates ===
    danger_dates = df[df["danger"] == 1]["date"].dt.strftime('%Y-%m-%d').tolist()
    if danger_dates:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Historical Danger Dates", ln=1)
        pdf.set_font("Arial", "", 11)
        for d in danger_dates[-10:]:
            pdf.cell(0, 10, f"- {d}", ln=1)

    # Export to bytes
    return pdf.output(dest="S")
st.subheader("📄 Export Summary Report")

if st.button("📥 Download Detailed PDF Report"):
    pdf_data = generate_rich_pdf_report(df, latest, PLOT_PATH, latest_path)
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="HudsonBay_Ice_Report.pdf">👉 Click here to download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)


# === FOOTER ===
st.markdown("---")
st.caption("Built by Vision AI + YOLOv8 — Fighting Climate Change One Pixel at a Time.")

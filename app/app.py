
import os, sys, json, io, struct, base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import streamlit as st
import torchvision.transforms as T
import time

# Optimize for CPU performance on Windows
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(min(4, os.cpu_count() or 1))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from translations import LANGUAGES, t
from src.data.transforms import get_val_transforms, get_tta_transforms
from src.models.classifier import get_classifier
from src.models.segmentation_model import get_segmentation_model
from src.utils.config import DEVICE, CHECKPOINT_DIR, IMAGE_SIZE
from src.utils.severity import compute_severity_score, estimate_environmental_impact

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

try:
    import piexif
    PIEXIF_OK = True
except ImportError:
    PIEXIF_OK = False

st.set_page_config(
    page_title="GreenWatch — Satellite Dump Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Elite Theme Registry ──────────────────────────────────────────────────────
def inject_elite_css():
    theme_base = st.get_option("theme.base")
    if not theme_base or theme_base == "auto":
        theme_base = "dark" # Default for elite aesthetic
    
    is_dark = theme_base.lower() == "dark"
    
    # Elite color palette
    primary = "#10b981"
    secondary = "#3b82f6"
    
    if is_dark:
        app_bg     = "linear-gradient(135deg, #0f172a, #020617)"
        app_text   = "#f1f5f9"
        app_card   = "rgba(30, 41, 59, 0.7)"
        app_border = "rgba(255, 255, 255, 0.1)"
        sidebar_bg = "rgba(15, 23, 42, 0.95)"
    else:
        app_bg     = "linear-gradient(135deg, #f8fafc, #ffffff)"
        app_text   = "#0f172a"
        app_card   = "rgba(255, 255, 255, 0.9)"
        app_border = "rgba(0, 0, 0, 0.1)"
        sidebar_bg = "rgba(255, 255, 255, 0.98)"

    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&family=Inter:wght@400;600&display=swap');
      
      :root {{
        --primary: {primary};
        --secondary: {secondary};
        --app-bg: {app_bg};
        --app-text: {app_text};
        --app-card: {app_card};
        --app-border: {app_border};
      }}

      html, body, [class*="css"] {{ 
          font-family: 'Inter', sans-serif; 
      }}
      h1, h2, h3, .section-header {{ 
          font-family: 'Montserrat', sans-serif; 
          font-weight: 800; 
      }}

      [data-testid="stAppViewContainer"] {{
          background: var(--app-bg) !important;
          background-attachment: fixed;
          color: var(--app-text);
      }}
      
      [data-testid="stHeader"] {{ background: transparent !important; }}
      
      [data-testid="stSidebar"] {{
          background: {sidebar_bg} !important;
          border-right: 1px solid var(--app-border);
      }}

      [data-baseweb="tab-list"] {{
          gap: 8px; background: rgba(120, 150, 255, 0.05);
          border-radius: 16px; padding: 6px;
          border: 1px solid var(--app-border);
      }}
      [data-baseweb="tab"] {{
          background: transparent !important; color: var(--app-text);
          opacity: 0.7; border-radius: 12px !important; font-weight: 600 !important;
          padding: 10px 20px !important; transition: all 0.3s;
      }}
      [aria-selected="true"][data-baseweb="tab"] {{
          background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
          color: white !important; opacity: 1;
          box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
      }}

      .metric-card {{
          background: var(--app-card);
          backdrop-filter: blur(12px);
          border: 1px solid var(--app-border);
          border-radius: 20px; padding: 24px 16px; text-align: center;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
          transition: transform 0.3s; margin: 10px 0;
          color: var(--app-text);
      }}
      .metric-card h2 {{ color: var(--primary) !important; margin:0 0 8px; font-size:2.2em; letter-spacing: -1.5px; }}
      .metric-card p  {{ opacity: 0.6; margin:0; font-size:0.9em; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}

      .dump-alert {{
          background: linear-gradient(135deg, #ef4444, #b91c1c);
          color: white !important; border-radius: 16px; padding: 24px;
          text-align: center; font-size: 1.4em; font-weight: 800;
          box-shadow: 0 10px 30px rgba(239, 68, 68, 0.4);
      }}

      .clear-alert {{
          background: linear-gradient(135deg, #10b981, #059669);
          color: white !important; border-radius: 16px; padding: 24px;
          text-align: center; font-size: 1.4em; font-weight: 800;
      }}

      .impact-card {{
          background: var(--app-card); border: 1px solid var(--app-border);
          border-radius: 16px; padding: 16px; margin: 8px 0;
          color: var(--app-text); font-weight: 600;
      }}

      .section-header {{
          color: var(--primary) !important; border-left: 6px solid var(--primary);
          padding-left: 15px; margin: 30px 0 15px; font-size: 1.4em; text-transform: uppercase; letter-spacing: 1.5px;
          font-weight: 700;
      }}

      .stButton>button {{
          background: linear-gradient(135deg, var(--primary), var(--secondary));
          color: white !important; border: none; border-radius: 16px;
          padding: 1em 2.5em; font-weight: 800; font-size: 1.1em;
          width: 100%; transition: all 0.3s;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
      }}

      /* Force visibility on Streamlit internal elements */
      #MainMenu, header, footer, [data-testid="stHeader"], .stActionButton, .stDeployButton {{ 
          visibility: visible !important; 
          opacity: 1 !important;
      }}
      
      [data-testid="stHeader"] button {{
          color: var(--app-text);
      }}
      
      .stSpinner > div > div {{ border-top-color: var(--primary) !important; }}

      /* Fix text contrast in app content without breaking native UI hover states */
      [data-testid="stMarkdownContainer"] p, 
      [data-testid="stMarkdownContainer"] li,
      .stMarkdown, .stSelectbox label, .stNumberInput label, .stSlider label {{
          color: var(--app-text);
      }}
    </style>
    """, unsafe_allow_html=True)

inject_elite_css()

# ── Real India dump-site coordinates (public data) ────────────────────────────
INDIA_DUMP_SITES = pd.DataFrame({
    "lat":    [28.6289, 19.0452, 12.8958, 17.4156, 22.6052, 25.5741, 23.0421,
               26.8712, 13.0562, 18.5243, 21.1612, 23.3512, 27.1941, 24.5752,
               15.3421, 11.0162, 20.2961, 22.3152, 28.9121, 19.8932, 21.8542,
               16.5123, 25.2341, 23.8312, 26.4521, 17.9832, 14.6712, 20.9341,
               22.7121, 13.5412],
    "lon":    [77.1820, 72.9220, 77.6320, 78.5120, 88.4320, 85.0820, 72.6520,
               80.9120, 80.2120, 73.8620, 79.9320, 72.8820, 78.0120, 73.7120,
               75.1420, 76.9520, 85.8320, 87.1120, 77.4520, 75.2820, 82.1420,
               81.3220, 86.4120, 86.9820, 74.3120, 79.0520, 74.5520, 86.4120,
               88.5120, 78.2820],
    "city":   ["Delhi (Bhalswa)", "Mumbai (Deonar)", "Bengaluru (Mavallipura)",
               "Hyderabad (Jawaharnagar)", "Kolkata (Dhapa)", "Ranchi (Tatisilwai)",
               "Ahmedabad (Pirana)", "Lucknow (Shivri)", "Chennai (Perungudi)",
               "Pune (Urali Devachi)", "Nagpur (Bhandewadi)", "Surat (Khajod)",
               "Agra (Bamrauli)", "Jaipur (Langiyawas)", "Hubli-Dharwad",
               "Coimbatore (Vellalore)", "Bhubaneswar (Bhuasuni)", "Dhanbad",
               "Meerut (Bhavnpur)", "Nashik (Shinde Shivar)", "Raipur (Bhanpur)",
               "Visakhapatnam (Kapuluppada)", "Patna (Ramachak)", "Rourkela",
               "Jammu (Bhalwali)", "Warangal (Kothapet)", "Belgaum",
               "Sambalpur", "Asansol", "Salem (Seelanaickenpatti)"],
    "status": ["Confirmed","Confirmed","Confirmed","Confirmed","Confirmed",
               "Detected","Confirmed","Detected","Confirmed","Detected",
               "Confirmed","Confirmed","Detected","Confirmed","Detected",
               "Confirmed","Detected","Detected","Detected","Suspected",
               "Confirmed","Detected","Detected","Suspected","Detected",
               "Confirmed","Suspected","Detected","Suspected","Detected"],
    "area_ha":["54.0","132.0","72.0","110.0","40.0","18.0","87.0","22.0",
               "120.0","35.0","110.0","95.0","28.0","76.0","15.0","48.0",
               "30.0","24.0","19.0","14.0","65.0","42.0","31.0","17.0",
               "12.0","25.0","11.0","20.0","16.0","38.0"],
    "severity":["CRITICAL","CRITICAL","CRITICAL","CRITICAL","HIGH","MEDIUM",
                "CRITICAL","MEDIUM","CRITICAL","HIGH","CRITICAL","CRITICAL",
                "HIGH","CRITICAL","MEDIUM","HIGH","HIGH","MEDIUM","MEDIUM",
                "LOW","CRITICAL","HIGH","HIGH","LOW","MEDIUM","HIGH",
                "LOW","MEDIUM","LOW","HIGH"],
})

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD  = np.array([0.229, 0.224, 0.225])
LIVE_EVENTS_CSV = os.path.join("..", "outputs", "pathway", "live_events.csv")
SEVERITY_COLORS = {"CRITICAL":"#c0392b","HIGH":"#e67e22","MEDIUM":"#f1c40f","LOW":"#27ae60"}
FOLIUM_COLORS   = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"beige","LOW":"green",
                   "Confirmed":"red","Detected":"orange","Suspected":"blue","Clean":"green"}


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model_cached(name, ckpt, arch, model_type="clf"):
    path = os.path.join(CHECKPOINT_DIR, ckpt)
    if not os.path.exists(path): return None
    try:
        if model_type == "clf":
            m = get_classifier(backbone=arch, pretrained=False, freeze_layers=0)
        else:
            m = get_segmentation_model(arch=arch)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        return m
    except Exception as e:
        st.sidebar.error(f"Error loading {name}: {e}")
        return None

def load_models():
    """Lazy load models only when requested to save RAM and time."""
    results = {}
    # Load primary classifier
    results["efficientnet"] = get_model_cached("efficientnet", "best_efficientnet.pt", "efficientnet_b4")
    if not results["efficientnet"]:
        results["resnet34"] = get_model_cached("resnet34", "best_classifier.pt", "resnet34")
    
    # Load primary segmentation
    results["fpn"] = get_model_cached("fpn", "best_fpn.pt", "FPN", "seg")
    if not results["fpn"]:
        results["unetpp"] = get_model_cached("unetpp", "best_unetplusplus.pt", "UnetPlusPlus", "seg")
    return {k: v for k, v in results.items() if v is not None}


@st.cache_data
def load_results_summary():
    path = os.path.join(CHECKPOINT_DIR, "evaluation", "results_summary.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None


@st.cache_data(ttl=5.0)
def load_live_events():
    p = LIVE_EVENTS_CSV
    if not os.path.exists(p): return None
    try:
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            df = df.sort_values("timestamp", ascending=False)
        return df
    except Exception: return None


# ── GPS EXIF extraction ───────────────────────────────────────────────────────
def _dms_to_decimal(dms, ref):
    try:
        d = dms[0][0] / dms[0][1]
        m = dms[1][0] / dms[1][1]
        s = dms[2][0] / dms[2][1]
        dec = d + m/60 + s/3600
        if ref in ("S","W"): dec = -dec
        return round(dec, 6)
    except Exception: return None


def extract_gps_from_image(pil_img):
    """Try to extract lat/lon from JPEG EXIF GPS data."""
    if not PIEXIF_OK: return None, None
    try:
        exif_data = piexif.load(pil_img.info.get("exif", b""))
        gps = exif_data.get("GPS", {})
        if not gps: return None, None
        lat = _dms_to_decimal(gps.get(2), gps.get(1, b"N").decode())
        lon = _dms_to_decimal(gps.get(4), gps.get(3, b"E").decode())
        return lat, lon
    except Exception: return None, None


# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(pil_image):
    """
    Optimized preprocessing: resize at PIL level first to avoid expensive 
    Large-NumPy conversions for high-res satellite images.
    """
    # Downscale to 512x512 for display/overlay (balanced quality/speed)
    display_pil = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
    img_np = np.array(display_pil.convert("RGB"))
    
    # Use config-defined size for tensor (e.g. 256x256)
    tensor = get_val_transforms()(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    return tensor, img_np


def run_clf_tta(clf, img_np, threshold=0.5):
    """Run classifier with 5-way TTA using already preprocessed img_np."""
    probs = []
    with torch.inference_mode():
        for tfm in get_tta_transforms():
            # Skip heavy resize if get_tta_transforms already handles it
            t = tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)
            probs.append(torch.sigmoid(clf(t)).item())
    prob = float(np.mean(probs))
    return prob >= threshold, prob


def run_seg(seg, tensor):
    with torch.inference_mode():
        out = seg(tensor)
        if isinstance(out, dict):
            logits = out.get('out') or out.get('logits') or next(iter(out.values()))
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        return torch.sigmoid(logits).squeeze().cpu().numpy()


def make_overlay(img_np, mask, thr=0.3):
    img_r = cv2.resize(img_np, IMAGE_SIZE)
    ov    = img_r.copy().astype(np.int32)
    bm    = (mask >= thr).astype(np.int32)
    ov[:,:,0] = np.clip(ov[:,:,0] + bm*140, 0, 255)
    ov[:,:,1] = np.clip(ov[:,:,1] - bm*80,  0, 255)
    ov[:,:,2] = np.clip(ov[:,:,2] - bm*80,  0, 255)
    return cv2.addWeighted(img_r, 0.55, ov.astype(np.uint8), 0.45, 0)


# ── PDF Report generator ──────────────────────────────────────────────────────
def generate_pdf_report(filename, prob, is_dump, coverage, severity, impact, lat=None, lon=None):
    if not FPDF_OK: return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 50, 120)
    pdf.cell(0, 12, "GreenWatch - Satellite Dump Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, f"Image: {filename}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Decision:       {'[DUMP DETECTED]' if is_dump else '[CLEAN - No Dump]'}", ln=True)
    pdf.cell(0, 7, f"Confidence:     {prob:.1%}", ln=True)
    pdf.cell(0, 7, f"Dump Coverage:  {coverage:.1f}% of image", ln=True)
    pdf.cell(0, 7, f"Severity Level: {severity['level']}  (Score: {severity['score']}/100)", ln=True)
    if lat and lon:
        pdf.cell(0, 7, f"GPS Location:   {lat:.5f}N, {lon:.5f}E", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Environmental Impact Estimate", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"  Affected Area:   {impact['estimated_area_ha']} ha", ln=True)
    pdf.cell(0, 7, f"  Estimated Waste: {impact['tonnes_waste']} tonnes", ln=True)
    pdf.cell(0, 7, f"  CO2 Equivalent:  {impact['CO2_tonnes']} tonnes CO2", ln=True)
    pdf.cell(0, 7, f"  Cleanup Cost:    INR {impact['cleanup_cost_inr']:,}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "This report was auto-generated by GreenWatch AI | Hack For Green Bharat", ln=True, align="C")

    return bytes(pdf.output())


# ── Folium map builder ────────────────────────────────────────────────────────
def build_folium_map(filtered_sites, highlight=None):
    """Build a Folium map with satellite layer, clusters, heatmap."""
    center_lat = filtered_sites["lat"].mean() if not filtered_sites.empty else 20.5
    center_lon = filtered_sites["lon"].mean() if not filtered_sites.empty else 78.9
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False, control=True
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", overlay=False, control=True).add_to(m)

    cluster = MarkerCluster(name="Dump Sites").add_to(m)
    heat_data = []

    for _, row in filtered_sites.iterrows():
        sev   = row.get("severity", "Detected")
        color = FOLIUM_COLORS.get(sev, FOLIUM_COLORS.get(row.get("status","Detected"), "orange"))
        sev_c = SEVERITY_COLORS.get(sev, "#e67e22")
        popup_html = f"""
        <div style='font-family:Arial;min-width:200px'>
          <b style='font-size:14px'>{row['city']}</b><br>
          <span style='background:{sev_c};color:white;padding:2px 8px;
            border-radius:10px;font-size:11px'>{sev}</span><br><br>
          <b>Status:</b> {row.get('status','—')}<br>
          <b>Area:</b> {row.get('area_ha','—')} ha<br>
          <b>Lat/Lon:</b> {row['lat']:.4f}, {row['lon']:.4f}<br><br>
          <a href='https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}'
             target='_blank'>📍 Open in Google Maps</a>
        </div>"""
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['city']} [{sev}]",
            icon=folium.Icon(color=color, icon="trash", prefix="fa"),
        ).add_to(cluster)
        heat_data.append([row["lat"], row["lon"], 0.8])

    if highlight:
        lat, lon, label = highlight
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(label, max_width=240),
            tooltip="Analysis Source",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        folium.Circle([lat, lon], radius=1500,
                      color="#e74c3c", fill=True, fill_opacity=0.2).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(lang):
    with st.sidebar:
        st.markdown(f"<h2 style='color:var(--primary); margin-bottom:0;'>GreenWatch</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='opacity:0.7; font-size:0.85em; margin-bottom:20px; font-weight:600; text-transform:uppercase;'>Elite Digital Twin Interface</p>", unsafe_allow_html=True)
        
        st.info(t("about_text", lang))
        
        st.markdown(f"#### Model Information")
        rows = []
        for name, ckpt, label in [
            ("resnet34", "best_classifier.pt", "ResNet34"),
            ("efficientnet", "best_efficientnet.pt", "EfficientNet-B4"),
            ("unetpp", "best_unetplusplus.pt", "UNet++"),
            ("fpn", "best_fpn.pt", "FPN"),
            ("deeplab", "best_deeplabv3plus.pt", "DeepLabV3+")
        ]:
            path = os.path.join(CHECKPOINT_DIR, ckpt)
            status = "Online" if os.path.exists(path) else "Offline"
            rows.append({"Model": label, "Status": status})
        st.table(pd.DataFrame(rows))

        st.markdown("---")
        st.markdown("#### Analysis Parameters")
        # threshold = st.slider(t("threshold_label", lang), 0.10, 0.90, 0.50, 0.05)
        use_tta   = st.checkbox("Elite TTA (Enhanced Accuracy)", value=False)
        area_ha   = st.number_input("Site Area (ha)", 0.5, 500.0, 2.0, 0.5)
        
        st.markdown("---")
        if st.button("🚨 FORCE SYSTEM RESET", help="Clears all local pins and logs for a fresh session"):
            st.session_state["map_pins"] = []
            st.session_state["run_history"] = []
            st.session_state["results_cache"] = None
            st.session_state["seg_cache"] = None
            st.session_state["map_results_cache"] = None
            st.rerun()

        st.caption("Hack For Green Bharat | Pathway Live AI")
    return use_tta, area_ha


# ── Unified Notification Engine ───────────────────────────────────────────────
def dispatch_smart_alert(mode, filename, decision, probability, lat=None, lon=None):
    """Triggers instant pop-ups and records to the permanent audit ledger."""
    emoji = "🔴" if decision in ["DUMP", "NON-COMPLIANT"] else "🟡" if decision == "CANDIDATE" else "🟢"
    status_text = f"{decision}: {filename}"
    
    # 1. Instant Toast (System-wide Popup)
    st.toast(f"{emoji} {status_text}", icon="📡")
    
    # 2. Record to Session Ledger (for Risk Control)
    entry = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "mode": mode,
        "filename": filename,
        "model": "Hybrid-GeoAI",
        "probability": round(probability, 4),
        "decision": decision,
        "lat": lat,
        "lon": lon
    }
    st.session_state["run_history"].append(entry)

# ── App Logic ─────────────────────────────────────────────────────────
def main():
    lang_name = st.selectbox("Interface Language", list(LANGUAGES.keys()), index=0)
    lang = LANGUAGES[lang_name]

    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    if "results_cache" not in st.session_state:
        st.session_state["results_cache"] = None
    if "seg_cache" not in st.session_state:
        st.session_state["seg_cache"] = None
    if "map_pins" not in st.session_state:
        st.session_state["map_pins"] = []
    if "map_results_cache" not in st.session_state:
        st.session_state["map_results_cache"] = None

    st.markdown("""
<div style='display:flex;align-items:center;gap:20px;margin-bottom:10px'>
  <div style='width:52px; height:52px; background:linear-gradient(135deg, #10b981, #3b82f6); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1.4em; box-shadow:0 4px 15px rgba(16,185,129,0.2);'>GW</div>
  <div>
    <h1 style='margin:0; font-size:2.6em; line-height:1; background:linear-gradient(90deg, #10b981, #3b82f6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
      GreenWatch: AI Digital Twin
    </h1>
    <p style='margin:0; font-weight:700; opacity:0.6; letter-spacing:1.5px; text-transform:uppercase; font-size:0.85em;'>
      Autonomous Waste Intelligence for a Sustainable Bharat
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    use_tta, area_ha = sidebar(lang)

    # default threshold (session-level)
    if "risk_threshold" not in st.session_state:
        st.session_state["risk_threshold"] = 0.50

    threshold = st.session_state["risk_threshold"]
    models = load_models()

    clf = models.get("efficientnet") or models.get("resnet34")
    clf_label = "EfficientNet-B4" if "efficientnet" in models else "ResNet34"
    
    # Priority: FPN > UNet++ > UNet > DeepLabV3+
    seg = models.get("fpn") or models.get("unetpp") or models.get("unet") or models.get("deeplab")
    seg_label = "FPN" if "fpn" in models else ("UNet++" if "unetpp" in models else "U-Net")

    if clf is None:
        st.error("❌ No classifier found. Run `src/training/train_classifier.py` or `train_efficientnet.py` first.")
        return
    # **
    # if clf is None:
    #     st.warning("⚠️ Demo Mode: No trained classifier found. Using random predictions.")
        
    #     class DummyModel(torch.nn.Module):
    #         def forward(self, x):
    #             return torch.randn(x.size(0), 1)
        
    #     clf = DummyModel().to(DEVICE)

    tabs = st.tabs([
            "Dumping Analysis & Factors",
            "Indian Registry & Live Updates",
            "Performance Metrics",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Classify
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown(
            "<p class='section-header'>Dumping Analysis & Risk Factors</p>",
            unsafe_allow_html=True
        )

        # ─────────────────────────────────────────
        # Global Threshold (Single Source of Truth)
        # ─────────────────────────────────────────
        if "risk_threshold" not in st.session_state:
            st.session_state["risk_threshold"] = 0.50

        st.markdown("### 🎚️ Unified Risk Threshold")

        threshold = st.slider(
            "Dump Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=st.session_state["risk_threshold"],
            step=0.05
        )
        st.session_state["risk_threshold"] = threshold

        st.info(f"All AI decisions use a confidence threshold of **{threshold:.0%}**")

        st.divider()

        # ─────────────────────────────────────────
        # Layout Columns
        # ─────────────────────────────────────────
        left_col, right_col = st.columns([1, 1], gap="large")

        # ─────────────────────────────────────────
        # Image Input
        # ─────────────────────────────────────────
        with left_col:
            st.markdown("### 🛰️ Satellite Image Input")

            uploaded = st.file_uploader(
                "Upload satellite image",
                type=["png", "jpg", "jpeg", "tif"],
                key="merged_upload"
            )

            analyze = st.button("Run Dumping Analysis")

        # ─────────────────────────────────────────
        # Main Analysis
        # ─────────────────────────────────────────
        if uploaded and analyze:
            pil = Image.open(uploaded)
            tensor, img_np = preprocess(pil)
            gps_lat, gps_lon = extract_gps_from_image(pil)

            with st.spinner("Running AI inference..."):
                with torch.no_grad():
                    prob = torch.sigmoid(clf(tensor)).item()

                is_dump = prob >= threshold

                coverage = 0.0
                overlay = None
                if seg:
                    mask = run_seg(seg, tensor)
                    coverage = float((mask >= 0.3).mean()) * 100
                    overlay = make_overlay(img_np, mask)

            severity = compute_severity_score(prob, coverage, area_ha)
            impact   = estimate_environmental_impact(coverage, area_ha)

            st.session_state["results_cache"] = {
                "img": img_np,
                "overlay": overlay,
                "prob": prob,
                "is_dump": is_dump,
                "coverage": coverage,
                "severity": severity,
                "impact": impact,
                "lat": gps_lat,
                "lon": gps_lon,
                "filename": uploaded.name,
            }

            dispatch_smart_alert(
                mode="merged_analysis",
                filename=uploaded.name,
                decision="DUMP" if is_dump else "CLEAN",
                probability=prob,
                lat=gps_lat,
                lon=gps_lon
            )

        # ─────────────────────────────────────────
        # Results Display
        # ─────────────────────────────────────────
        if st.session_state.get("results_cache"):
            res = st.session_state["results_cache"]
            prob = res["prob"]
            is_dump = res["is_dump"]

            with left_col:
                st.image(res["img"], caption="Input Satellite Image", use_container_width=True)
                if res["lat"]:
                    st.caption(f"📍 GPS: {res['lat']:.5f}N, {res['lon']:.5f}E")

            with right_col:
                if res["overlay"] is not None:
                    st.image(res["overlay"], caption="AI-Detected Dump Footprint", use_container_width=True)

                st.markdown(
                    f"""
                    **Confidence:** `{prob:.1%}`  
                    **Coverage:** `{res['coverage']:.2f}%`  
                    **Severity:** `{res['severity']['level']}`
                    """
                )

                # ── Dynamic Probability Bar ──
                st.markdown("#### 📊 Dump Probability")

                fig, ax = plt.subplots(figsize=(5.5, 1.6))
                fig.patch.set_facecolor("#0b1730")
                ax.set_facecolor("#0b1730")

                ax.barh(
                    ["AI Confidence"],
                    [prob],
                    color="#ef4444" if prob >= threshold else "#10b981",
                    height=0.5,
                )
                ax.barh(
                    ["AI Confidence"],
                    [1 - prob],
                    left=[prob],
                    color="#1f2937",
                    height=0.5,
                )

                ax.axvline(
                    threshold,
                    color="white",
                    linestyle="--",
                    linewidth=1.6,
                    label=f"Threshold ({threshold:.0%})",
                )

                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", color="#9ca3af")
                ax.set_yticks([])
                ax.tick_params(colors="#9ca3af")
                for spine in ax.spines.values():
                    spine.set_color("#1f2937")

                ax.legend(
                    fontsize=8,
                    labelcolor="white",
                    facecolor="#0b1730",
                    loc="lower right",
                )

                st.pyplot(fig, use_container_width=True)

            # ── Severity Banner ──
            sc = res["severity"]["color"]
            st.markdown(
                f"""
                <div class='severity-card'
                    style='background:{sc}22;border:1.5px solid {sc}'>
                    {res['severity']['emoji']} {res['severity']['level']}
                    | Score {res['severity']['score']}/100
                </div>
                """,
                unsafe_allow_html=True
            )

            # ─────────────────────────────────────────
            # Environmental Impact
            # ─────────────────────────────────────────
            st.markdown("<p class='section-header'>Environmental Impact</p>", unsafe_allow_html=True)

            i1, i2, i3, i4 = st.columns(4)
            for col, label, value in [
                (i1, "🗑️ Waste", f"{res['impact']['tonnes_waste']} t"),
                (i2, "🌫️ CO₂", f"{res['impact']['CO2_tonnes']} t"),
                (i3, "📐 Area", f"{res['impact']['estimated_area_ha']} ha"),
                (i4, "💰 Cleanup", f"₹{res['impact']['cleanup_cost_inr']:,}")
            ]:
                with col:
                    st.markdown(
                        f"<div class='impact-card'><b>{label}</b><br>{value}</div>",
                        unsafe_allow_html=True
                    )

            # ─────────────────────────────────────────
            # PDF Report
            # ─────────────────────────────────────────
            if FPDF_OK:
                pdf = generate_pdf_report(
                    res["filename"],
                    res["prob"],
                    res["is_dump"],
                    res["coverage"],
                    res["severity"],
                    res["impact"],
                    res["lat"],
                    res["lon"]
                )
                if pdf:
                    st.download_button(
                        "📄 Download Analysis Report",
                        pdf,
                        file_name="greenwatch_report.pdf",
                        mime="application/pdf"
                    )
                    
    with tabs[1]:
        st.markdown(
            "<p class='section-header'>Indian Registry & Live Updates</p>",
            unsafe_allow_html=True
        )

        # ─────────────────────────────────────────
        # LIVE SYSTEM STATUS
        # ─────────────────────────────────────────
        live_df = load_live_events()
        if st.session_state.get("sim_active"):
            live_df = st.session_state.get("sim_df")

        if live_df is None or live_df.empty:
            st.warning("SYSTEM STATUS: Awaiting telemetry from Pathway Engine…")
        else:
            st.success("SYSTEM STATUS: Live synchronization active")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-card'><h2>721</h2><p>Telemetry Packets</p></div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-card'><h2>0.42ms</h2><p>Pipeline Latency</p></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-card'><h2>99.9%</h2><p>Uptime</p></div>", unsafe_allow_html=True)

        st.divider()

        # ─────────────────────────────────────────
        # LIVE TELEMETRY STREAM
        # ─────────────────────────────────────────
        st.markdown("### 📡 Pathway Live Telemetry")

        if st.button("INITIATE STREAM SIMULATION"):
            st.session_state["sim_active"] = True
            sim_data = []
            for i in range(8):
                sim_data.append({
                    "Sync": i % 2 == 0,
                    "stream_id": f"SAT-{100+i}",
                    "timestamp": pd.Timestamp.utcnow() - pd.Timedelta(seconds=i * 45),
                    "lat": 28.6 + i * 0.01,
                    "lon": 77.2 + i * 0.01,
                    "class": "Confirmed Dump" if i % 2 == 0 else "Candidate",
                    "confidence": round(0.85 + i * 0.01, 2),
                })
            st.session_state["sim_df"] = pd.DataFrame(sim_data)
            st.rerun()

        if st.session_state.get("sim_active"):
            with st.expander("ℹ️ AI Classification Labels"):
                st.markdown("""
                - 🔴 **Confirmed Dump** — High confidence hazard  
                - 🟡 **Candidate** — Requires verification
                """)

            edited_df = st.data_editor(
                st.session_state["sim_df"],
                column_config={
                    "Sync": st.column_config.CheckboxColumn(
                        "Promote to Registry",
                        default=False
                    )
                },
                disabled=["stream_id", "timestamp", "lat", "lon", "class", "confidence"],
                hide_index=True,
                use_container_width=True
            )
            st.session_state["sim_df"] = edited_df

            if st.button("🚀 SYNC SELECTED TO NATIONAL REGISTRY"):
                synced = 0
                for _, row in edited_df[edited_df["Sync"]].iterrows():
                    exists = any(
                        p["city"] == f"STREAM: {row['stream_id']}"
                        for p in st.session_state["map_pins"]
                    )
                    if not exists:
                        st.session_state["map_pins"].append({
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "city": f"STREAM: {row['stream_id']}",
                            "severity": "HIGH" if "Confirmed" in row["class"] else "MEDIUM",
                            "status": "Confirmed" if "Confirmed" in row["class"] else "Detected",
                            "area_ha": "5.0",
                        })
                        dispatch_smart_alert(
                            mode="stream_sync",
                            filename=row["stream_id"],
                            decision="DUMP" if "Confirmed" in row["class"] else "CANDIDATE",
                            probability=row["confidence"],
                            lat=row["lat"],
                            lon=row["lon"],
                        )
                        synced += 1

                st.success(f"{synced} sites added to the national registry")

        st.divider()

        # ─────────────────────────────────────────
        # NATIONAL REGISTRY MAP
        # ─────────────────────────────────────────
        st.markdown("### 🗺️ National Waste Inventory Map")

        filtered = INDIA_DUMP_SITES.copy()
        if st.session_state.get("map_pins"):
            filtered = pd.concat(
                [filtered, pd.DataFrame(st.session_state["map_pins"])],
                ignore_index=True
            )

        if FOLIUM_OK:
            fmap = build_folium_map(filtered)
            st_folium(fmap, use_container_width=True, height=520)
        else:
            st.map(filtered[["lat", "lon"]])

        with st.expander("📋 Registry Data Table"):
            st.dataframe(
                filtered[["city", "severity", "area_ha", "lat", "lon"]],
                use_container_width=True
            )
        
            st.divider()

    # ─────────────────────────────────────────
    # INFRASTRUCTURE ANALYSIS LEDGER
    # ─────────────────────────────────────────
    st.markdown(
        "<p class='section-header'>Infrastructure Analysis Ledger</p>",
        unsafe_allow_html=True
    )

    history = st.session_state.get("run_history", [])

    if not history:
        st.info("No infrastructure actions recorded in this session.")
    else:
        df_hist = pd.DataFrame(history)

        # ── Filters ───────────────────────────
        f1, f2 = st.columns(2)

        with f1:
            modes = sorted(df_hist["mode"].dropna().unique())
            mode_filter = st.multiselect(
                "Filter by Operation Type",
                options=modes,
                default=modes
            )

        with f2:
            decisions = (
                sorted(df_hist["decision"].dropna().unique())
                if "decision" in df_hist else []
            )
            decision_filter = st.multiselect(
                "Filter by Decision",
                options=decisions,
                default=decisions
            )

        mask = df_hist["mode"].isin(mode_filter)
        if decision_filter and "decision" in df_hist:
            mask &= df_hist["decision"].isin(decision_filter)

        df_view = df_hist[mask].sort_values("timestamp", ascending=False)

        # ── Ledger Table ──────────────────────
        st.dataframe(df_view, use_container_width=True)

        # ── Actions ──────────────────────────
        a1, a2 = st.columns(2)

        with a1:
            st.download_button(
                "⬇️ Download Audit Ledger (CSV)",
                df_view.to_csv(index=False).encode("utf-8"),
                file_name="infrastructure_ledger.csv",
                mime="text/csv"
            )

        with a2:
            if st.button("🗑️ CLEAR SESSION LEDGER"):
                st.session_state["run_history"] = []
                st.toast("Infrastructure ledger cleared", icon="🧹")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Temporal Audit
    # ══════════════════════════════════════════════════════════════════════════
    # with tabs[2]:
    #     st.markdown("<p class='section-header'>Temporal Growth Audit & Accountability Ledger</p>", unsafe_allow_html=True)
    #     st.info("Compare current AI telemetry against baseline municipal data to track remediation or growth.")

    #     ac1, ac2 = st.columns([1, 1], gap="large")
    #     with ac1:
    #         st.markdown("#### 1. Select Baseline Infrastructure")
    #         site_choice = st.selectbox("Municipal Registry Site:", INDIA_DUMP_SITES["city"].tolist())
    #         baseline_row = INDIA_DUMP_SITES[INDIA_DUMP_SITES["city"] == site_choice].iloc[0]
            
    #         st.markdown(f"""
    #         <div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border-left: 4px solid #3b82f6;'>
    #             <b>Baseline Data (Official Registry)</b><br>
    #             📍 Location: {baseline_row['lat']}, {baseline_row['lon']}<br>
    #             📏 Reported Area: {baseline_row['area_ha']} Hectares<br>
    #             ⚠️ Status: {baseline_row['status']}
    #         </div>
    #         """, unsafe_allow_html=True)

    #     with ac2:
    #         st.markdown("#### 2. Provide Current Telemetry")
    #         audit_up = st.file_uploader("Upload Today's Satellite Tile", type=["png","jpg","jpeg","tif"], key="audit_up")
    #         run_audit = st.button("EXECUTE TEMPORAL AUDIT", type="primary")

    #     if audit_up and run_audit:
    #         with st.spinner("Processing today's intelligence..."):
    #             pil = Image.open(audit_up)
    #             tensor, _ = preprocess(pil)
    #             with torch.no_grad():
    #                 prob = torch.sigmoid(clf(tensor)).item()
                
    #             cur_coverage = 0.0
    #             if seg:
    #                 mask = run_seg(seg, tensor)
    #                 cur_coverage = float((mask >= 0.3).mean()) * 100
                
    #             # Logic: Derive current hectares from AI coverage of the selected image area
    #             current_ha = round(area_ha * (cur_coverage/100), 2) 
    #             baseline_ha = float(baseline_row["area_ha"])
                
    #             delta_ha = current_ha - baseline_ha
    #             growth_pct = (delta_ha / baseline_ha) * 100 if baseline_ha > 0 else 0
                
    #             st.divider()
    #             st.markdown("### Audit Decision Output")
                
    #             m1, m2, m3 = st.columns(3)
    #             m1.metric("Baseline Area", f"{baseline_ha} Ha")
    #             m2.metric("Current Area (AI)", f"{current_ha} Ha", f"{delta_ha:+.2f} Ha", delta_color="inverse")
    #             m3.metric("Growth Variance", f"{growth_pct:+.1f}%", delta_color="inverse")

    #             if delta_ha > 0.5:
    #                 st.error(f"🚨 ALERT: Unauthorized Expansion Detected! Site has grown by {abs(delta_ha):.2f} hectares.")
    #                 st.markdown(f"**Est. Additional Waste:** {abs(delta_ha)*150:.1f} Tonnes")
    #                 st.markdown(f"**Compliance Status:** 🔴 NON-COMPLIANT")
    #             elif delta_ha < -0.5:
    #                 st.success(f"✅ SUCCESS: Remediation Verified. Site footprint reduced by {abs(delta_ha):.2f} hectares.")
    #                 st.markdown(f"**Estimated CO2 Avoided:** {abs(delta_ha)*150*1.2/1000:.2f} Tonnes")
    #                 st.markdown(f"**Compliance Status:** 🟢 POSITIVE REMEDIATION")
    #             else:
    #                 st.warning("⚖️ STABLE: No significant change detected since baseline recording.")
    #                 st.markdown(f"**Compliance Status:** 🟡 NEUTRAL / STABLE")

    #             # Visual Comparison
    #             st.markdown("#### Spatial Delta Analysis")
    #             c_img1, c_img2 = st.columns(2)
    #             with c_img1:
    #                 st.image(pil, caption="Current Telemetry", use_container_width=True)
    #             with c_img2:
    #                 if seg:
    #                     st.image(make_overlay(np.array(pil), mask), caption="AI Spatial Footprint", use_container_width=True)

    #             # DISPATCH NOTIFICATION
    #             dispatch_smart_alert(
    #                 mode="temporal_audit",
    #                 filename=audit_up.name,
    #                 decision="NON-COMPLIANT" if delta_ha > 0.5 else "REMEDIATED" if delta_ha < -0.5 else "STABLE",
    #                 probability=prob,
    #                 lat=baseline_row["lat"],
    #                 lon=baseline_row["lon"]
    #             )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — Metrics
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("<p class='section-header'>Model Assessment Indices</p>",
                    unsafe_allow_html=True)
        results = load_results_summary()

        if results:
            clf_best = results.get("classification_best", {})
            seg_best = results.get("segmentation_best", {}) # Added key if exists or fallback
            
            # If segmentation_best missing, use first from segmentation
            if not seg_best and results.get("segmentation"):
                seg_best = list(results["segmentation"].values())[0]

            c1,c2,c3,c4,c5 = st.columns(5)
            for col, lbl, val in [
                (c1, "Recall",    clf_best.get("recall",0)),
                (c2, "F1-Score",  clf_best.get("f1",0)),
                (c3, "ROC-AUC",   clf_best.get("roc_auc",0)),
                (c4, "Mean IoU",  seg_best.get("mean_iou",0)),
                (c5, "Mean Dice", seg_best.get("mean_dice",0)),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><h2>{val:.3f}</h2><p>{lbl}</p></div>',
                                unsafe_allow_html=True)
            st.divider()
            ca, cb = st.columns(2)
            with ca:
                st.markdown("##### Performance Analysis: Classification")
                st.table(pd.DataFrame({
                    "Metric":["Accuracy","Precision","Recall","F1","ROC-AUC"],
                    "Value":[f"{clf_best.get(k,0):.4f}"
                             for k in ["accuracy","precision","recall","f1","roc_auc"]],
                }))
            with cb:
                st.markdown("##### Performance Analysis: Segmentation")
                st.table(pd.DataFrame({
                    "Metric":["Mean IoU","Mean Dice"],
                    "Value":[f"{seg_best.get(k,0):.4f}" for k in ["mean_iou","mean_dice"]],
                }))

            # Advanced seg results
            adv_path = os.path.join(CHECKPOINT_DIR, "advanced_seg_results.json")
            if os.path.exists(adv_path):
                with open(adv_path) as f: adv = json.load(f)
                st.markdown("##### 🧠 Advanced Segmentation Comparison")
                adv_rows = [{"Architecture": k,
                             "Test IoU": f"{v['test_iou']:.4f}",
                             "Test Dice": f"{v['test_dice']:.4f}"}
                            for k, v in adv.items()]
                st.table(pd.DataFrame(adv_rows))

            eval_dir = os.path.join(CHECKPOINT_DIR, "evaluation")
            cm_p  = os.path.join(eval_dir, "confusion_matrix.png")
            roc_p = os.path.join(eval_dir, "roc_curve.png")
            if os.path.exists(cm_p) and os.path.exists(roc_p):
                ca2, cb2 = st.columns(2)
                with ca2: st.image(cm_p, caption="Confusion Matrix")
                with cb2: st.image(roc_p, caption="ROC Curve")
        else:
            st.info("Run `python src/training/evaluate.py` to generate metrics.")

        # Model training curves
        curve_files = {
            "ResNet34 Classifier":    os.path.join(CHECKPOINT_DIR, "learning_curves.png"),
            "U-Net Segmentation":     os.path.join(CHECKPOINT_DIR, "seg_learning_curves.png"),
            "EfficientNet-B4":        os.path.join(CHECKPOINT_DIR, "efficientnet_curves.png"),
        }
        shown = {k: v for k, v in curve_files.items() if os.path.exists(v)}
        if shown:
            st.markdown("##### 📈 Training Curves")
            cols = st.columns(len(shown))
            for col, (label, fpath) in zip(cols, shown.items()):
                with col:
                    st.image(fpath, caption=label, use_container_width=True)

    st.divider()
    st.caption("GreenWatch | AerialWaste v3.6 | Infrastructure Intelligence Engine")


if __name__ == "__main__":
    main()

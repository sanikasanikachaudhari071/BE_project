import streamlit as st
import os
import tempfile
import cv2
from inference import load_models, predict_media

# Page config
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")

# CSS
st.markdown("""
<style>
    .main {background-color: #0d1117; color: #c9d1d9;}
    h1 {color: #58a6ff; font-family: 'Inter', sans-serif;}
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: 1px solid rgba(240, 246, 252, 0.1);
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {background-color: #2ea043;}
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    .fake-text {color: #f85149; font-size: 2.5em; font-weight: bold;}
    .real-text {color: #3fb950; font-size: 2.5em; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_models():
    return load_models()

st.title("Deepfake Detection System 🕵️")
st.markdown("Upload a video, image, or provide a URL to detect if it's REAL or a DEEPFAKE.")

# Initialize models
try:
    spatial_model, freq_model, fusion_model = init_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Missing weights warning
if not os.path.exists("freq_model.pth"):
    st.warning("⚠️ `freq_model.pth` not found! Please run your Colab script (with the saving fix) and download it.")
if not os.path.exists("deepfake_model.pth"):
    st.warning("⚠️ `deepfake_model.pth` not found! Please download it from Colab and place it in the project root.")

option = st.sidebar.selectbox("Choose Input Type", ["Video/Image Upload"])

def process_and_display(media_path):
    with st.spinner("Analyzing media... This might take a moment."):
        prob, meta = predict_media(media_path, spatial_model, freq_model, fusion_model)
    
    if prob is None:
        st.error(meta)
    else:
        is_fake = prob > 0.6
        label = "FAKE" if is_fake else "REAL"
        css_class = "fake-text" if is_fake else "real-text"
        
        st.markdown(f"""
        <div class="result-box">
            <h2>Prediction: <span class="{css_class}">{label}</span></h2>
            <p>Confidence (Fake Score): {prob*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"Successfully processed {len(meta)} faces/frames.")

if option == "Video/Image Upload":
    uploaded_file = st.file_uploader("Upload Media (MP4, AVI, JPG, PNG)", type=['mp4', 'avi', 'jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            st.video(tfile.name)
        else:
            st.image(tfile.name)
            
        if st.button("Detect Deepfake"):
            process_and_display(tfile.name)
        
        # Cleanup
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)


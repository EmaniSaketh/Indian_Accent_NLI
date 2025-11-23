# ===============================================================================
# PROJECT: Native Language Identification (NLI) & Cuisine Recommender
# SUBMISSION VERSION: Final Polish & Robustness
# ===============================================================================

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# -------------------------
# 1. CONFIGURATION
# -------------------------
DATA_DIR = "dataset"
HUBERT_NAME = "facebook/hubert-base-ls960"
MODEL_PATH = "best_nli_model.pth"

# Audio Settings
SAMPLE_RATE = 16000
DURATION_SEC = 6  # Requirement: Support at least 6 seconds
MAX_AUDIO_LEN = SAMPLE_RATE * DURATION_SEC
MIN_AUDIO_SAMPLES = 1600  # 0.1s minimum to prevent model crash

# Training Settings (Fast/Robust)
BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 1
NUM_SAMPLES_PER_L1 = 3 
MFCC_DIM = 40

device = torch.device("cpu")

# -------------------------
# 2. UI STYLING (Requirement 5: Attractive Interface)
# -------------------------
def set_custom_style():
    st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        .stAlert {
            padding: 0.5rem;
            margin-bottom: 1rem;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #262730;
            border: 1px solid #4B4B4B;
            margin-top: 20px;
            text-align: center;
        }
        .rec-box {
            background-color: #FF4B4B;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# 3. ML & AUDIO FUNCTIONS
# -------------------------

def find_audio_files(base_dir):
    rows = []
    try:
        if not os.path.exists(base_dir):
            st.error(f"❌ Error: '{base_dir}' folder not found.")
            return pd.DataFrame()
        classes = [d.name for d in os.scandir(base_dir) if d.is_dir()]
    except Exception as e:
        st.error(f"❌ Error scanning data: {e}")
        return pd.DataFrame()

    for c in classes:
        folder = os.path.join(base_dir, c)
        files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".wav")]
        # Limit samples for speed
        files = files[:NUM_SAMPLES_PER_L1] 
        for f in files:
            rows.append({"path": f, "l1": c})
    return pd.DataFrame(rows)

def preprocess_audio(path):
    """Robust audio loader that handles formats and lengths safely."""
    try:
        # Use librosa for most robust loading (handles resampling automatically)
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        # Guard: Too short
        if len(audio) < MIN_AUDIO_SAMPLES: 
            return None
            
        # Pad or Truncate to exactly 6 seconds (or max len)
        if len(audio) > MAX_AUDIO_LEN:
            audio = audio[:MAX_AUDIO_LEN]
        else:
            audio = np.pad(audio, (0, MAX_AUDIO_LEN - len(audio)))
            
        return audio.astype(np.float32)
    except Exception:
        return None

def extract_features(audio, extractor, model):
    # 1. MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_DIM), axis=1)
    
    # 2. HuBERT
    try:
        inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Use Layer 1 (Proven best performer)
        hubert = torch.mean(out.hidden_states[1].squeeze(0), dim=0).numpy()
    except RuntimeError:
        # Fallback if audio is weirdly shaped
        return None

    return np.concatenate([mfcc, hubert])

# --- Model Definition ---
class Classifier(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.fc1 = nn.Linear(dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, classes)
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

class NliDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.long)

# -------------------------
# 4. TRAINING PIPELINE
# -------------------------
def train_model_pipeline():
    status = st.status("⚙️ System Initialization: Training Model...", expanded=True)
    
    status.write("📂 Scanning dataset...")
    df = find_audio_files(DATA_DIR)
    if df.empty:
        status.update(label="❌ Training Failed: No Data", state="error")
        return False

    status.write(f"📊 Found {len(df)} samples across {len(df.l1.unique())} languages.")
    
    le = LabelEncoder()
    le.fit(df.l1)
    
    # Load HuBERT
    status.write("🤖 Loading HuBERT model...")
    extractor = AutoFeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = AutoModel.from_pretrained(HUBERT_NAME).to(device)

    X_train, y_train = [], []
    
    status.write("🧠 Extracting acoustic features (This happens once)...")
    progress_bar = status.progress(0)
    
    total = len(df)
    for idx, row in enumerate(tqdm(df.itertuples(), total=total)):
        audio = preprocess_audio(row.path)
        if audio is None: continue
        
        features = extract_features(audio, extractor, hubert)
        if features is not None:
            X_train.append(features)
            y_train.append(le.transform([row.l1])[0])
        
        progress_bar.progress(min((idx + 1) / total, 1.0))

    if not X_train:
        status.update(label="❌ Fatal Error: No usable audio extracted.", state="error")
        return False

    # Train Classifier
    status.write("🔥 Training Neural Network...")
    model = Classifier(len(X_train[0]), len(le.classes_)).to(device)
    optimz = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(NliDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(NUM_EPOCHS):
        for xb, yb in loader:
            optimz.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optimz.step()

    # Save
    torch.save({
        "model": model.state_dict(),
        "dim": len(X_train[0]),
        "classes": le.classes_.tolist()
    }, MODEL_PATH)
    
    status.update(label="✅ System Ready!", state="complete", expanded=False)
    time.sleep(1)
    return True

# -------------------------
# 5. MAIN APP
# -------------------------

# Cuisine Map
CUISINE_MAP = {
    'andhra_pradesh': 'Spicy Biryani, Gongura Pickle, Pesarattu',
    'gujrat': 'Dhokla, Thepla, Undhiyu (Sweet & Savory)',
    'jharkhand': 'Litti Chokha, Thekua, Rugda',
    'karnataka': 'Bisi Bele Bath, Mysore Pak, Ragi Mudde',
    'kerala': 'Appam with Stew, Puttu, Fish Curry',
    'tamil': 'Dosa, Idli, Sambar, Chettinad Chicken',
    'unknown': 'Classic Indian Thali'
}

def main():
    st.set_page_config(page_title="NLI Accent AI", layout="wide", page_icon="🗣️")
    set_custom_style()
    
    # Header
    st.title("🗣️ Native Language Identification AI")
    st.markdown("### *Detecting Regional Accents & Recommending Cuisine*")
    st.divider()

    # 1. Check Model
    if not os.path.exists(MODEL_PATH):
        if not train_model_pipeline():
            st.stop()
        st.experimental_rerun()

    # 2. Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    model = Classifier(checkpoint['dim'], len(classes)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    extractor = AutoFeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = AutoModel.from_pretrained(HUBERT_NAME).to(device)

    # 3. Layout
    col_input, col_result = st.columns([1, 1], gap="large")

    audio_path_to_process = None

    with col_input:
        st.subheader("1. Input Audio")
        tab1, tab2 = st.tabs(["🎤 Live Microphone", "xc2 Upload File"])

        # --- TAB 1: LIVE MIC (Requirement 1 & 2) ---
        with tab1:
            st.write("Click Start and speak for 3-6 seconds.")
            
            # WebRTC Streamer with Key
            ctx = webrtc_streamer(
                key="mic_input",
                mode=WebRtcMode.SENDONLY,
                audio_html_attrs={"autoPlay": True, "muted": True},
                media_stream_constraints={"audio": True, "video": False},
            )

            # Visual Indicator (Requirement 2)
            if ctx.state.playing:
                st.markdown('<div class="rec-box">🔴 MIC ACTIVE - LISTENING...</div>', unsafe_allow_html=True)

            # Processing Logic for Mic
            if ctx.audio_receiver:
                try:
                    frames = ctx.audio_receiver.get_frames(timeout=1)
                    if frames:
                        # Convert frames to a single valid audio array
                        sound_chunk = np.concatenate([f.to_ndarray() for f in frames])
                        
                        # Convert to valid WAV format (int16)
                        if sound_chunk.dtype != np.int16:
                             sound_chunk = (sound_chunk * 32767).astype(np.int16)

                        # Save to temp file
                        temp_name = "live_rec.wav"
                        wavfile.write(temp_name, 16000, sound_chunk)
                        
                        # Verify size (Requirement 4 compliance check)
                        if os.path.getsize(temp_name) > 10000: # >10KB
                            st.success("✅ Audio Captured Successfully!")
                            audio_path_to_process = temp_name
                except:
                    pass

        # --- TAB 2: UPLOAD (Requirement 4) ---
        with tab2:
            uploaded = st.file_uploader("Upload WAV (Min 6s recommended)", type=['wav'])
            if uploaded:
                temp_name = "upload_rec.wav"
                with open(temp_name, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.audio(uploaded)
                audio_path_to_process = temp_name

    # 4. Prediction Logic
    with col_result:
        st.subheader("2. Analysis Results")
        
        if audio_path_to_process:
            if st.button("🔍 Analyze Accent", type="primary"):
                with st.spinner("🧠 Neural Network is analyzing accent patterns..."):
                    
                    # Load & Process
                    proc_audio = preprocess_audio(audio_path_to_process)
                    
                    # Clean up
                    try: os.remove(audio_path_to_process)
                    except: pass

                    if proc_audio is None:
                        st.error("⚠️ Audio too short or unclear. Please provide at least 1-2 seconds of clear speech.")
                    else:
                        # Extract & Predict
                        feats = extract_features(proc_audio, extractor, hubert)
                        
                        if feats is not None:
                            input_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = model(input_tensor)
                                probs = torch.softmax(logits, dim=1)
                                pred_idx = torch.argmax(probs).item()
                                confidence = probs[0][pred_idx].item()
                            
                            native_lang = classes[pred_idx]
                            food = CUISINE_MAP.get(native_lang.lower(), CUISINE_MAP['unknown'])

                            # --- SEPARATE PREDICTION BOX (Requirement 3) ---
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style="margin:0; color: #4CAF50;">Detected L1: {native_lang.replace('_',' ').upper()}</h2>
                                <p style="color: #AAA;">Confidence: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.success("🍽️ **Cuisine Recommendation:**")
                            st.info(f"**{food}**")
                            
                        else:
                            st.error("⚠️ Could not extract features. Try a different file.")
        else:
            st.info("👈 Waiting for audio input...")

if __name__ == "__main__":
    main()
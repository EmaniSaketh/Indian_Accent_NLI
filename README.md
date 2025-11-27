🗣️ Native Language Identification (NLI) of Indian English Speakers

Project Title: Native Language Identification of Indian English Speakers Using HuBERT

This project develops a Deep Learning system to identify the native language (L1) of an Indian speaker based on their English accent, integrating the predictive model into an interactive web application that provides personalized cuisine recommendations.

🚀 Application Demo

The final application, the Accent-Aware Cuisine Recommender, is a Streamlit web interface that allows users to test the NLI model live via microphone or file upload.

Key Features:

Live Microphone Input: Uses streamlit-webrtc to capture real-time audio from the browser.

Hybrid Feature Extraction: Combines modern HuBERT Embeddings with traditional MFCCs for robust accent classification.

Personalized Output: Predicts the regional L1 (e.g., Tamil, Keralite) and suggests a corresponding regional cuisine (e.g., Appam, Fish Curry).

Robustness: Includes crash-proof logic to handle short, silent, or corrupted audio inputs.

🔬 Technical Scope & Model Architecture

1. Feature Extraction

Feature

Dimension

Source Layer

Purpose

HuBERT Embeddings

768D

Transformer Layer 1

Captures generalized phonetic and prosodic traits (accent patterns).

MFCCs

40D

librosa

Traditional acoustic baseline, capturing spectral shape.

Final Vector

808D (Concatenated)

N/A

Input for the final classifier.

2. Model Architecture

The core prediction is handled by a simple Feed-Forward Neural Network (FNN) trained on the concatenated feature vector. The network uses one hidden layer (808 $\rightarrow$ 256 $\rightarrow$ N classes) and is optimized using the Adam algorithm and Cross-Entropy Loss (CEL).

3. Generalization Insights

Analysis confirmed that the HuBERT Layer 1 features demonstrated superior age generalization (maintaining accuracy when tested on children's voices) compared to MFCCs, validating its ability to isolate stable linguistic accent markers.

🛠️ Setup and Execution

1. Repository Structure

Indian_Accent_NLI/
├── dataset/                    <- Contains L1 subfolders (andhra_pradesh, tamil, etc.)
├── app.py                      <- The final Streamlit application code
├── make_dummy.py               <- Emergency script to quickly create a model file
├── best_nli_model.pth          <- (Generated after training) The saved model weights
└── requirements.txt            <- List of all Python dependencies


2. Installation (Local Setup)

This project must be run from a local terminal (like VS Code or PowerShell).

Clone the Repository:

git clone [https://github.com/EmaniSaketh/Indian_Accent_NLI.git](https://github.com/EmaniSaketh/Indian_Accent_NLI.git)
cd Indian_Accent_NLI


Create and Activate Environment: (Recommended)

python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate # Linux/macOS


Install Dependencies: Install all necessary libraries listed in requirements.txt.

pip install -r requirements.txt


3. Model Training (CRITICAL First Run)

The application requires the model file (best_nli_model.pth) to exist.

Option A: Full Training (Recommended for accuracy, takes ~20 mins on CPU):

python app.py


Wait for the console to print "✅ Model training complete..."

Option B: Quick Bypass (For demonstration only):
If the training is too slow, run the make_dummy.py script to create an instant, but untrained, model file.

python make_dummy.py


4. Run the Streamlit Application

Once the best_nli_model.pth file is created, launch the app:

streamlit run app.py


Access the application in your browser at http://localhost:8501.

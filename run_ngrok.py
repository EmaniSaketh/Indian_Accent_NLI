# ===============================================================================
# FILE: run_ngrok.py (Execution Script)
# INSTRUCTIONS: Run this file ONLY AFTER 'python nli_main.py' finishes training.
# ===============================================================================

import subprocess
import time

STREAMLIT_APP_FILE = "app.py"
STREAMLIT_PORT = 8501
NGROK_TUNNEL_TIMEOUT = 10 # Give Streamlit time to boot

def run_local():
    """Starts the Streamlit app locally."""
    print("Starting Streamlit server locally...")
    
    # Use PowerShell's Start-Process command structure
    # This launches Streamlit in a separate window/process
    try:
        subprocess.run(
            ['start-process', 'streamlit', '-ArgumentList', 'run', STREAMLIT_APP_FILE],
            check=True,
            shell=True # Needed for PowerShell commands to run properly
        )
        print(f"\n✅ Streamlit server launched. Access at http://localhost:{STREAMLIT_PORT}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Failed to launch Streamlit. Ensure 'streamlit' is in your PATH.")
        return

if __name__ == "__main__":
    # 1. Run Streamlit (The app will open in a new window)
    run_local()
    
    # 2. Ngrok/Tunneling requires the command to be run in the terminal separately
    print("\n" + "="*40)
    print("ACTION REQUIRED:")
    print("Open a NEW terminal window and run the tunnel command:")
    print(f"npx localtunnel --port {STREAMLIT_PORT}")
    print("or access locally at http://localhost:8501")
    print("="*40)
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app"""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    print("🚀 Starting Financial RAG Streamlit Application...")
    print("📊 Open your browser and navigate to the URL shown below")
    print("=" * 50)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(app_path), 
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()

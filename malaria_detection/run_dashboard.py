"""
Quick Launcher for Malaria Detection Web App
This script starts the web dashboard automatically
"""

import subprocess
import webbrowser
import time
import sys
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("MALARIA DETECTION SYSTEM - WEB DASHBOARD LAUNCHER")
    print("="*70 + "\n")
    
    # Check if Flask is installed
    try:
        import flask
        print("[OK] Flask is installed")
    except ImportError:
        print("[*] Flask not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask>=2.3.0"])
        print("[OK] Flask installed successfully")
    
    # Check if model exists
    model_path = Path('models/best_model.h5')
    if not model_path.exists():
        print("\n[WARNING] No trained model found!")
        print("   First run: python main.py")
        print("   Then run: python run_dashboard.py")
        input("\nPress Enter to continue anyway...")
    else:
        print("[OK] Model found")
    
    # Start the web app
    print("\n" + "="*70)
    print("Starting web server...")
    print("="*70)
    print("\n[OK] Web Dashboard will open in your browser")
    print("[OK] Dashboard URL: http://127.0.0.1:5000")
    print("[OK] Press CTRL+C to stop the server\n")
    
    try:
        # Run Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n[OK] Web server stopped")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

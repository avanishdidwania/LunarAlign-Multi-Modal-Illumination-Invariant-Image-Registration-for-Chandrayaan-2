import os
import sys
import subprocess

def main():
    print("=== Lunar Image Registration Engine Startup ===")
    
    # Check for models folder or download weights instruction folder
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(models_dir):
        print(f"Verified: Local models folder is present at: {models_dir}")
    else:
        print("Warning: Local 'models' directory not found. Pretrained weights will download and cache dynamically on first execution.")

    # Execute uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "lunar_reg.web.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()

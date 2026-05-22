import os
import zipfile
from pathlib import Path

def create_zip():
    zip_path = "D:/CODE/NCKH/FlappyBird_LeWM_Cloud.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add Dataset
        dataset_dir = Path("D:/stable-wm")
        for file in dataset_dir.glob("*.h5"):
            print(f"Adding {file}")
            zipf.write(file, arcname=f"stable-wm/{file.name}")
            
        # 2. Add Codebase (exclude .venv, .git, lightning_logs, __pycache__)
        code_dir = Path("D:/CODE/NCKH/le-wm")
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d not in ['.venv', '.venv_win', '.git', 'lightning_logs', '__pycache__', 'outputs']]
            for file in files:
                file_path = Path(root) / file
                arcname = f"le-wm/{file_path.relative_to(code_dir)}"
                zipf.write(file_path, arcname=arcname)
                
    print(f"Successfully created {zip_path} (Size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    create_zip()

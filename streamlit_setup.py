import subprocess
from pathlib import Path
import sys

def run_pipeline():
    print("Starting pipeline to generate artifacts...")
    
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    
    scripts = ["simulate_data.py", "preprocess.py", "train_models.py", "cox_model.py"]
    for script in scripts:
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, "-X", "utf8", script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Script {script} failed with exit code {result.returncode}")
        else:
            print(result.stdout)
            
    print("Pipeline completed successfully! All artifacts generated.")

def setup_if_needed():
    artifacts_needed = [
        "artifacts/preprocessor.pkl",
        "artifacts/logistic_regression.pkl",
        "artifacts/xgboost.pkl",
        "artifacts/cox_model.pkl",
        "artifacts/metrics.json",
    ]
    if all(Path(p).exists() for p in artifacts_needed):
        return
    run_pipeline()

if __name__ == "__main__":
    run_pipeline()

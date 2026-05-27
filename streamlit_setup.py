import subprocess
from pathlib import Path

def run_pipeline():
    print("Starting pipeline to generate artifacts...")
    
    # Create folders if they do not exist
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    
    # Run the scripts sequentially
    scripts = ["simulate_data.py", "preprocess.py", "train_models.py", "cox_model.py"]
    for script in scripts:
        print(f"Running {script}...")
        # Running with the current python interpreter and forcing UTF-8 mode to handle unicode characters
        import sys
        result = subprocess.run([sys.executable, "-X", "utf8", script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Script {script} failed with exit code {result.returncode}")
        else:
            print(result.stdout)
            
    print("Pipeline completed successfully! All artifacts generated.")

if __name__ == "__main__":
    run_pipeline()

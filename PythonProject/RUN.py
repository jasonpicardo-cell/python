import subprocess

# Runs sequentially: script1 finishes, then script2 starts
scripts = ["../NSE_SYNC/nse_macro_scanner.py", "../Camarilla/pivot_scanner.py"]

for script in scripts:
    print(f"--- Running {script} ---")
    subprocess.run(["python", script])
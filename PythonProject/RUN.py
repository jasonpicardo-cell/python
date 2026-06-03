import subprocess
import sys
import os


def run_scripts_sequentially():
    scripts = [
        "./NSE_SYNC/nse_macro_scanner.py",
        "./RS.StageAnalysis.VCP.Accum.EMAADX/nse_scanner.py",
        "./Camarilla/pivot_scanner.py",
        "./DSZone/combined_screener.py",
        "./DCF_SCAN/dcf_analysis.py"
    ]

    for script in scripts:
        print(f"[{'-' * 10} Starting: {script} {'-' * 10}]")

        # 1. Convert the relative script path to an absolute path
        abs_script_path = os.path.abspath(script)

        # 2. Extract just the directory where the script lives
        script_dir = os.path.dirname(abs_script_path)

        try:
            # 3. Use cwd=script_dir to run the script "inside" its own folder
            subprocess.run([sys.executable, abs_script_path], check=True, cwd=script_dir)
            print(f"✅ Successfully finished: {script}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred while running {script}")
            print(f"Exit code: {e.returncode}\n")
            # break


if __name__ == "__main__":
    run_scripts_sequentially()
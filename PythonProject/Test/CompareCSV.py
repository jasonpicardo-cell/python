import os
from pathlib import Path

def check_stock_csv_files(txt_filepath, data_folder_path):
    # Convert paths to Path objects for easier cross-platform handling
    txt_file = Path(txt_filepath)
    data_folder = Path(data_folder_path)

    # Check if the text file actually exists before proceeding
    if not txt_file.exists():
        print(f"Error: Could not find the file '{txt_filepath}'")
        return

    # Check if the data folder exists
    if not data_folder.exists():
        print(f"Error: Could not find the directory '{data_folder_path}'. Please check the path.")
        return

    # Read stock names from the text file
    with open(txt_file, 'r', encoding='utf-8') as f:
        # Read lines, remove whitespace/newlines, and ignore empty lines
        stocks = [line.strip() for line in f if line.strip()]

    print(f"Successfully loaded {len(stocks)} stock symbols from {txt_file.name}.")

    found_files = []
    missing_files = []

    # Check for the corresponding .csv file for each stock
    for stock in stocks:
        csv_filename = f"{stock}.csv"
        csv_path = data_folder / csv_filename

        if csv_path.is_file():
            found_files.append(stock)
        else:
            missing_files.append(stock)

    # Print a summary of the results
    print("-" * 30)
    print("RESULTS:")
    print("-" * 30)
    print(f"Total CSVs found:   {len(found_files)}")
    print(f"Total CSVs missing: {len(missing_files)}")

    # Print ALL missing files
    if missing_files:
        print("\nMissing CSV files:")
        for missing in missing_files:
            print(f" - {missing}.csv")

if __name__ == "__main__":
    # Updated to your new directory name
    # Note: Used '../' to represent exactly one directory level up
    STOCKS_LIST_FILE = "nse_all_stocks.txt"
    DATA_DIRECTORY = "../nse_data_cache"

    check_stock_csv_files(STOCKS_LIST_FILE, DATA_DIRECTORY)
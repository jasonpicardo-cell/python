import os
import csv
from datetime import datetime
from collections import deque

#TODO - check nifty 750 stock list
#Store out put with in a file with DATE
#out has to be written to file at end
#Nifty 750 stock name should appear in a highlighed section and other stocks in different sections

def find_outdated_csvs(directory="data_cache"):
    # Generate today's date in YYYY-MM-DD format
    today_str = '2026-05-06' #datetime.today().strftime('%Y-%m-%d')
    outdated_files = []

    # Verify the target directory exists
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, mode='r', encoding='utf-8') as file:
                    # deque(..., maxlen=1) efficiently grabs only the very last row of the CSV
                    last_line_iterator = deque(csv.reader(file), maxlen=1)

                    if last_line_iterator:
                        last_row = last_line_iterator[0]

                        # Check if the row has data and if the first column matches today
                        if not last_row or last_row[0].strip() != today_str:
                            outdated_files.append(filename)
                    else:
                        # Catch completely empty CSV files
                        outdated_files.append(filename)

            except Exception as e:
                print(f"Could not read {filename} due to: {e}")
                outdated_files.append(filename)

    # Output the final list
    if outdated_files:
        print(f"Files missing today's date ({today_str}) in the last row:")
        for file in outdated_files:
            print(f" - {file}")
    else:
        print(f"Success: All CSV files have {today_str} as their last record.")


if __name__ == "__main__":
    find_outdated_csvs()
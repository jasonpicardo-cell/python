import os
import csv
from datetime import datetime


def load_nifty_list(filepath="nifty750.txt"):
    """Reads the Nifty 750 text file and returns a set of stock names."""
    nifty_stocks = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                symbol = line.strip().upper()
                if symbol:
                    nifty_stocks.add(symbol)
    else:
        print(f"Warning: '{filepath}' not found. All files will be treated as 'Other'.")
    return nifty_stocks


def extract_stock_name(filename):
    """
    Extracts the core stock symbol from a filename.
    Example: 'BSE_RELIANCE-X.csv' -> 'RELIANCE'
    """
    # Remove the file extension
    base_name = filename.replace('.csv', '').upper()

    # Strip everything before and including the first '_'
    if '_' in base_name:
        base_name = base_name.split('_', 1)[-1]

    # Strip everything after and including the first '-'
    if '-' in base_name:
        base_name = base_name.split('-')[0]

    return base_name


def find_outdated_csvs(directory="nse_data_cache", nifty_file="nifty750.txt"):
    # Generate today's date
    today_str = '2026-05-07'  # datetime.today().strftime('%Y-%m-%d')

    # Load the Nifty 750 master list
    nifty_750_stocks = load_nifty_list(nifty_file)

    # Two separate lists for the different sections
    nifty_outdated = []
    other_outdated = []

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Iterate through files
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)

            # Extract the clean stock name to check against the Nifty 750 list
            stock_symbol = extract_stock_name(filename)

            try:
                with open(filepath, mode='r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    date_found = False

                    for row in csv_reader:
                        cleaned_row = [cell.strip() for cell in row]
                        if today_str in cleaned_row:
                            date_found = True
                            break

                    # If the date is missing, categorize the file
                    if not date_found:
                        if stock_symbol in nifty_750_stocks:
                            nifty_outdated.append(filename)
                        else:
                            other_outdated.append(filename)

            except Exception as e:
                print(f"Could not read {filename} due to: {e}")
                other_outdated.append(filename)  # Default to 'other' on read failure

    # --- Generate the Output File ---
    if nifty_outdated or other_outdated:
        output_filename = f"missing_data_report_{today_str}.txt"

        with open(output_filename, 'w', encoding='utf-8') as out_file:
            # Section 1: Nifty 750 Stocks
            out_file.write(f"=== NIFTY 750 STOCKS MISSING DATA ({today_str}) ===\n")
            if nifty_outdated:
                for file in sorted(nifty_outdated):
                    out_file.write(f" - {file}\n")
            else:
                out_file.write(" All Nifty 750 stocks are up to date.\n")

            out_file.write("\n")  # Blank line for spacing

            # Section 2: Other Stocks
            out_file.write(f"=== OTHER STOCKS MISSING DATA ({today_str}) ===\n")
            if other_outdated:
                for file in sorted(other_outdated):
                    out_file.write(f" - {file}\n")
            else:
                out_file.write(" All other stocks are up to date.\n")

        print(f"Report successfully generated: {output_filename}")

    else:
        print(f"Success: All CSV files contain the date {today_str}. No report needed.")


if __name__ == "__main__":
    find_outdated_csvs()
import requests
import pandas as pd
import io


def fetch_nse_fno_stocks():
    # The official NSE URL for F&O market lots
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

    # Headers to mimic a real web browser (prevents NSE from blocking the script)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

    print("Connecting to NSE...")
    try:
        # We must use a session to hit the homepage first to establish cookies
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=15)

        # Now fetch the actual CSV file
        print("Downloading F&O data...")
        response = session.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            # Read the CSV data into pandas
            df = pd.read_csv(io.StringIO(response.text))

            # NSE CSV headers often contain trailing spaces, so we strip them
            df.columns = df.columns.str.strip()

            if 'SYMBOL' in df.columns:
                # Extract symbols, clean whitespace, remove empty rows, and make unique
                symbols = df['SYMBOL'].astype(str).str.strip().dropna().unique().tolist()

                # Sort alphabetically for better readability
                symbols.sort()

                # Save to the requested text file
                file_path = "niftyfno.txt"
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write("\n".join(symbols))

                print(f"\nSuccess! Extracted {len(symbols)} F&O symbols.")
                print(f"File saved as: {file_path}")

            else:
                print("Error: 'SYMBOL' column not found in the downloaded data.")
                print("Available columns were:", df.columns.tolist())
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    fetch_nse_fno_stocks()
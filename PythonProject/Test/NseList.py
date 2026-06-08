import requests
import pandas as pd
import io


def generate_nse_stock_file():
    # Official NSE URL for all listed equities
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    # Headers to mimic a standard browser, preventing NSE connection blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

    try:
        session = requests.Session()
        # Initialize a session on the main page to establish cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Request the master CSV
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))

            # Extract the SYMBOL column
            if 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].dropna().astype(str).tolist()

                # Write to text file with line breaks
                with open("nse_all_stocks.txt", "w") as f:
                    f.write("\n".join(symbols))

                print(f"Success! Exported {len(symbols)} NSE stocks to nse_all_stocks.txt")
            else:
                print("Error: 'SYMBOL' column not found in the CSV.")
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_nse_stock_file()
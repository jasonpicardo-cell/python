import ssl

import pandas as pd
from utils import get_property


# Ignore SSL errors on Mac
#ssl._create_default_https_context = ssl._create_unverified_context


def fetch_live_dhan_ids():
    print("Downloading Dhan's master file in chunks (Takes ~10 seconds)...")

    # Using the standard Compact CSV
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    nse_dict = {}

    try:
        # Read the file in chunks of 100,000 rows
        for chunk in pd.read_csv(url, chunksize=100000, low_memory=False):

            # Clean column names (removes hidden spaces like "SEM_SEGMENT ")
            chunk.columns = chunk.columns.str.strip()

            if 'SEM_EXM_EXCH_ID' in chunk.columns:

                # Filter for NSE Equity
                # Segment = E (Equity), Series = EQ (Regular Equities)
                nse_chunk = chunk[
                    (chunk['SEM_EXM_EXCH_ID'] == 'NSE') &
                    (chunk['SEM_SEGMENT'] == 'E') &
                    (chunk['SEM_SERIES'] == 'EQ')
                    ]

                # Add to dictionary
                for _, row in nse_chunk.iterrows():
                    nse_dict[str(row['SEM_TRADING_SYMBOL'])] = str(row['SEM_SMST_SECURITY_ID'])

        print(f"✅ Successfully extracted {len(nse_dict)} NSE stocks!")
        return nse_dict

    except Exception as e:
        print(f"❌ Failed to download master list: {e}")
        return {}
# --- To use it in your run_scanner() function: ---
# Replace your manual dictionary with this single line:
# all_stocks = fetch_live_dhan_ids()

if __name__ == '__main__':
    print(fetch_live_dhan_ids())

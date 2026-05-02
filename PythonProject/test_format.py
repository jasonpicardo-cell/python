import requests

# Bypass the Mac SSL block
requests.packages.urllib3.disable_warnings()
headers = {'User-Agent': 'Mozilla/5.0'}

print("⏳ Fetching raw Fyers CSV row...")
response = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers=headers, verify=False)

# Print the very first line of the raw text
first_line = response.text.split('\n')[0]
print("\n👀 THIS IS WHAT FYERS IS SENDING US:")
print(first_line)




print("⏳ Fetching raw BSE CSV row...")
response = requests.get("https://public.fyers.in/sym_details/BSE_CM.csv", headers=headers, verify=False)

lines = response.text.split('\n')

print("\n👀 THIS IS WHAT FYERS IS SENDING US FOR BSE:")
print(f"Row 1: {lines[0]}")
print(f"Row 500: {lines[500] if len(lines) > 500 else 'N/A'}")
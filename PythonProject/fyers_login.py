import webbrowser
from fyers_apiv3 import fyersModel
from utils import get_property

client_id = get_property("fyers_client_id")
secret_key = get_property("fyers_secret_key")
redirect_uri = get_property("fyers_redirect_uri")

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code"
)

auth_link = session.generate_authcode()
print("\nOpening browser to log into Fyers...")
webbrowser.open(auth_link)

auth_url_pasted = input("\nPaste the ENTIRE URL from the address bar here:\n> ")
auth_code = auth_url_pasted.split("auth_code=")[1].split("&")[0]
session.set_token(auth_code)

response = session.generate_token()
access_token = response['access_token']

with open("fyers_token.txt", "w") as f:
    f.write(access_token)

print("\n✅ Login Successful! Today's token saved to fyers_token.txt.")
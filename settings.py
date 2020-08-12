from os import getenv

from dotenv import load_dotenv

load_dotenv()

api_key = getenv("ALPHA_VANTAGE_API_KEY")
ticker_name = getenv("TICKER_NAME")

assert api_key, "ALPHA_VANTAGE_API_KEY not configured"
assert ticker_name, "TICKER_NAME not configured"

from ast import literal_eval
from os import getenv

from dotenv import load_dotenv

load_dotenv()

ticker_name = getenv("TICKER_NAME")
cyc_len = int(getenv("CYC_LEN", 20))
seed = int(getenv("SEED", 2020))
valid_pct = float(getenv("VALID_PCT", 0.1))
layers = literal_eval(getenv("LAYERS", "[200, 100]"))

assert ticker_name, "TICKER_NAME not configured"

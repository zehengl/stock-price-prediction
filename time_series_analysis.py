# %%
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.metrics import smape
from pmdarima.model_selection import train_test_split
from tqdm import tqdm

from settings import api_key, ticker_name


print(ticker_name)


# %%
df = web.DataReader(ticker_name, "av-daily", end=datetime.now(), api_key=api_key)

df = df.reset_index()
df["index"] = pd.to_datetime(df["index"])

plt.figure(figsize=(12, 6))
ax = sns.lineplot(data=df, x="index", y="close")
ax.set(xlabel="Date", ylabel="Stock Price")


# %%
today = datetime.now()
end_date = today.replace(month=today.month - 1)
train, test = train_test_split(df, test_size=(today - end_date).days)

y_train = train["close"]
y_test = test["close"]

print(f"{train.shape[0]} train samples")
print(f"{test.shape[0]} test samples")


kpss_diffs = ndiffs(y_train, alpha=0.05, test="kpss", max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test="adf", max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")


model = auto_arima(
    y_train,
    d=n_diffs,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    max_p=6,
    trace=2,
)


def forecast_one_step():
    forecast = model.predict(n_periods=1)
    return forecast[0]


forecasts = []
for new_ob in tqdm(y_test):
    forecasts.append(forecast_one_step())
    model.update(new_ob)

print(f"SMAPE: {smape(y_test, forecasts)}")


# %%
plt.figure(figsize=(12, 6))
sns.lineplot(test["index"], forecasts, label="Predicted Price")
sns.lineplot(test["index"], y_test, label="Actual Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")


# %%
print(f"{ticker_name} will probably close at {forecast_one_step():.2f} next day.")


# %%

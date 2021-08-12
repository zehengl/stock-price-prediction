# %%
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from fastai.tabular import *
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.metrics import smape
from pmdarima.model_selection import train_test_split
from tqdm import tqdm

from settings import api_key, ticker_name, cyc_len, seed, valid_pct, layers


# %%
print(f"Try to predict stock price of {ticker_name}.")
print(f"Configuration:")
print(f"cyc_len: {cyc_len}")
print(f"seed: {seed}")
print(f"valid_pct: {valid_pct}")
print(f"layers: {layers}")

df = web.DataReader(ticker_name, "av-daily", end=datetime.now(), api_key=api_key)
df = df.reset_index()
df["index"] = pd.to_datetime(df["index"])


# %%
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
fig.suptitle(f"History of {ticker_name}")
df["high_diff"] = df["high"] - df["close"]
df["low_diff"] = df["low"] - df["close"]
sns.lineplot(data=df, x="index", y="high_diff", color="green", ax=axes[0])
sns.lineplot(data=df, x="index", y="close", ax=axes[1])
sns.lineplot(data=df, x="index", y="low_diff", color="red", ax=axes[2])
fig.savefig(f"history.{ticker_name}.png", bbox_inches="tight", dpi=200)


# %%
today = datetime.now()
end_date = today.replace(month=today.month - 1)
train, test = train_test_split(df, test_size=(today - end_date).days)

print(f"{train.shape[0]} train samples")
print(f"{test.shape[0]} test samples")


# %%
def get_arima_model(train, test, col):
    y_train = train[col]
    y_test = test[col]

    kpss_diffs = ndiffs(y_train, alpha=0.05, test="kpss", max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test="adf", max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)

    print(f"Estimated differencing term for {col}: {n_diffs}")

    model = auto_arima(
        y_train,
        d=n_diffs,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        max_p=6,
        trace=1,
    )

    return model


model_open = get_arima_model(train, test, "open")
model_high = get_arima_model(train, test, "high")
model_close = get_arima_model(train, test, "close")
model_low = get_arima_model(train, test, "low")
model_volume = get_arima_model(train, test, "volume")


# %%
def get_forecasts(test, model, col):
    forecasts = []
    y_test = test[col]
    for new_ob in tqdm(y_test, desc=f"predicting arima {col}"):
        forecast = model.predict(n_periods=1)[0]
        forecasts.append(forecast)
        model.update(new_ob)

    return forecasts


forecasts_open = get_forecasts(test, model_open, "open")
forecasts_high = get_forecasts(test, model_high, "high")
forecasts_close = get_forecasts(test, model_close, "close")
forecasts_low = get_forecasts(test, model_low, "low")
forecasts_volume = get_forecasts(test, model_volume, "volume")


# %%
def get_rmse(forcasts, series):
    pred = tensor(forcasts)
    target = tensor(series.tolist())
    return rmse(pred, target)


print(f"rmse of open [arima]: {get_rmse(forecasts_open, test['open'])}")
print(f"rmse of high [arima]: {get_rmse(forecasts_high, test['high'])}")
print(f"rmse of close [arima]: {get_rmse(forecasts_close, test['close'])}")
print(f"rmse of low [arima]: {get_rmse(forecasts_low, test['low'])}")
print(f"rmse of volume [arima]: {get_rmse(forecasts_volume, test['volume'])}")

print(f"smape of open [arima]: {smape(test['open'], forecasts_open)}")
print(f"smape of high [arima]: {smape(test['high'], forecasts_high)}")
print(f"smape of close [arima]: {smape(test['close'], forecasts_close)}")
print(f"smape of low [arima]: {smape(test['low'], forecasts_low)}")
print(f"smape of volume [arima]: {smape(test['volume'], forecasts_volume)}")


# %%
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(20, 12), sharex=True)
fig.suptitle(f"ARIMA Model of {ticker_name}")

sns.lineplot(test["index"], forecasts_open, label="Arima Open Price", ax=axes[0])
sns.lineplot(test["index"], test["open"], label="Actual Open Price", ax=axes[0])

sns.lineplot(test["index"], forecasts_high, label="Arima High Price", ax=axes[1])
sns.lineplot(test["index"], test["high"], label="Actual High Price", ax=axes[1])

sns.lineplot(test["index"], forecasts_close, label="Arima Close Price", ax=axes[2])
sns.lineplot(test["index"], test["close"], label="Actual Close Price", ax=axes[2])

sns.lineplot(test["index"], forecasts_low, label="Arima Low Price", ax=axes[3])
sns.lineplot(test["index"], test["low"], label="Actual Low Price", ax=axes[3])

sns.lineplot(test["index"], forecasts_volume, label="Arima Volume", ax=axes[4])
sns.lineplot(test["index"], test["volume"], label="Actual Volume", ax=axes[4])


fig.savefig(f"arima.{ticker_name}.png", bbox_inches="tight", dpi=200)


# %%
arima_open = model_open.predict(1)[0]
arima_high = model_high.predict(1)[0]
arima_close = model_close.predict(1)[0]
arima_low = model_low.predict(1)[0]
arima_volume = model_volume.predict(1)[0]
print(f"Based on time series analysis, next day {ticker_name} may have:")
print(f"  a open price at {arima_open:.3f}")
print(f"  a high price at {arima_high:.3f}")
print(f"  a close price at {arima_close:.3f}")
print(f"  a low price at {arima_low:.3f}")
print(f"  a volume at {arima_volume:.0f}")


# %%
dep_var = "close"
cont_names = ["open", "high", "low", "volume"]
procs = [Normalize]

data_test = TabularList.from_df(test, cont_names=cont_names, procs=procs)

data_train = (
    TabularList.from_df(train, path=".", cont_names=cont_names, procs=procs)
    .split_by_rand_pct(valid_pct=valid_pct, seed=seed)
    .label_from_df(cols=dep_var, label_cls=FloatList)
    .add_test(data_test)
    .databunch(num_workers=0)
)


# %%
learn = tabular_learner(data_train, layers=layers, metrics=rmse)
learn.lr_find()
learn.recorder.plot(suggestion=True)


# %%
learn.fit_one_cycle(cyc_len, learn.recorder.min_grad_lr)


# %%
preds, _ = learn.get_preds(DatasetType.Test)
oracle_forecasts = [t[0] for t in preds.tolist()]


# %%
arima_test = pd.DataFrame(
    {
        "open": forecasts_open,
        "high": forecasts_high,
        "low": forecasts_low,
        "volume": forecasts_volume,
    }
)
arima_forecasts = []
for i in tqdm(range(arima_test.shape[0])):
    a = learn.predict(arima_test.iloc[i])[1].item()
    arima_forecasts.append(a)


# %%
fig = plt.figure(figsize=(12, 6))
sns.lineplot(
    test["index"],
    oracle_forecasts,
    label="Predicted Close Price w. Oracle Open/High/Low",
)
sns.lineplot(
    test["index"],
    arima_forecasts,
    label="Predicted Close Price w. ARIMA Open/High/Low",
)
sns.lineplot(test["index"], forecasts_close, label="ARIMA Close Price")
sns.lineplot(test["index"], test["close"], label="Actual Close Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Prediction of {ticker_name}")
fig.savefig(f"prediction.{ticker_name}.png", bbox_inches="tight", dpi=200)


# %%
print(f"rmse of close [arima]: {get_rmse(forecasts_close, test['close'])}")
print(f"rmse of close [dl w. arima]: {get_rmse(arima_forecasts, test['close'])}")
print(f"rmse of close [dl w. oracle]: {get_rmse(oracle_forecasts, test['close'])}")

print(f"smape of close [arima]: {smape(test['close'], forecasts_close)}")
print(f"smape of close [dl w. arima]: {smape(test['close'], arima_forecasts)}")
print(f"smape of close [dl w. oracle]: {smape(test['close'], oracle_forecasts)}")

# %%
learn.predict(
    dict(open=arima_open, high=arima_high, low=arima_low, volume=int(arima_volume))
)

# %%

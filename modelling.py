# %%
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from fastai.tabular.all import *
from pandas_datareader import data as pdr
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.metrics import smape
from pmdarima.model_selection import train_test_split
from tqdm import tqdm

from settings import cyc_len, layers, seed, ticker_name, valid_pct


yf.pdr_override()

# %%
print(f"Try to predict stock price of {ticker_name}.")
print(f"Configuration:")
print(f"cyc_len: {cyc_len}")
print(f"seed: {seed}")
print(f"valid_pct: {valid_pct}")
print(f"layers: {layers}")

df = pdr.get_data_yahoo(ticker_name)
df = df.reset_index()


# %%
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
fig.suptitle(f"History of {ticker_name}")
df["High Diff"] = df["High"] - df["Close"]
df["Low Diff"] = df["Low"] - df["Close"]
sns.lineplot(data=df, x="Date", y="High Diff", color="green", ax=axes[0])
sns.lineplot(data=df, x="Date", y="Close", color="orange", ax=axes[1])
sns.lineplot(data=df, x="Date", y="Low Diff", color="red", ax=axes[2])
fig.savefig(f"history.{ticker_name}.png", bbox_inches="tight", dpi=200)


# %%
today = datetime.now()
end_date = today.replace(month=today.month - 1)
train, test = train_test_split(df, test_size=(today - end_date).days)

print(f"{train.shape[0]} train samples")
print(f"{test.shape[0]} test samples")


# %%
def get_arima_model(train, col):
    y_train = train[col]

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


model_open = get_arima_model(train, "Open")
model_high = get_arima_model(train, "High")
model_close = get_arima_model(train, "Close")
model_low = get_arima_model(train, "Low")
model_volume = get_arima_model(train, "Volume")


# %%
def get_forecasts(test, model, col):
    forecasts = []
    y_test = test[col]
    for new_ob in tqdm(y_test, desc=f"predicting arima {col}"):
        forecast = model.predict(n_periods=1).tolist()[0]
        forecasts.append(forecast)
        model.update(new_ob)

    return forecasts


forecasts_open = get_forecasts(test, model_open, "Open")
forecasts_high = get_forecasts(test, model_high, "High")
forecasts_close = get_forecasts(test, model_close, "Close")
forecasts_low = get_forecasts(test, model_low, "Low")
forecasts_volume = get_forecasts(test, model_volume, "Volume")


# %%
def get_rmse(series, forcasts):
    pred = tensor(forcasts)
    target = tensor(series.tolist())
    return rmse(pred, target).item()


print(f"rmse of open [arima]: {get_rmse(test['Open'], forecasts_open)}")
print(f"rmse of high [arima]: {get_rmse(test['High'], forecasts_high)}")
print(f"rmse of close [arima]: {get_rmse(test['Close'], forecasts_close)}")
print(f"rmse of low [arima]: {get_rmse(test['Low'], forecasts_low)}")
print(f"rmse of volume [arima]: {get_rmse(test['Volume'], forecasts_volume)}")

print(f"smape of open [arima]: {smape(test['Open'], forecasts_open )}")
print(f"smape of high [arima]: {smape(test['High'], forecasts_high )}")
print(f"smape of close [arima]: {smape(test['Close'], forecasts_close )}")
print(f"smape of low [arima]: {smape(test['Low'], forecasts_low )}")
print(f"smape of volume [arima]: {smape(test['Volume'], forecasts_volume )}")


# %%
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(20, 12), sharex=True)
fig.suptitle(f"ARIMA Model of {ticker_name}")

sns.lineplot(
    x=test["Date"],
    y=forecasts_open,
    label="Arima Open Price",
    ax=axes[0],
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["Open"],
    label="Actual Open Price",
    ax=axes[0],
    color="orange",
)

sns.lineplot(
    x=test["Date"],
    y=forecasts_high,
    label="Arima High Price",
    ax=axes[1],
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["High"],
    label="Actual High Price",
    ax=axes[1],
    color="orange",
)

sns.lineplot(
    x=test["Date"],
    y=forecasts_close,
    label="Arima Close Price",
    ax=axes[2],
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["Close"],
    label="Actual Close Price",
    ax=axes[2],
    color="orange",
)

sns.lineplot(
    x=test["Date"],
    y=forecasts_low,
    label="Arima Low Price",
    ax=axes[3],
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["Low"],
    label="Actual Low Price",
    ax=axes[3],
    color="orange",
)

sns.lineplot(
    x=test["Date"],
    y=forecasts_volume,
    label="Arima Volume",
    ax=axes[4],
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["Volume"],
    label="Actual Volume",
    ax=axes[4],
    color="orange",
)


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
y_names = "Close"
cont_names = ["Open", "High", "Low", "Volume"]
procs = [Normalize]
splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(train))

to = TabularPandas(
    train,
    procs=procs,
    cont_names=cont_names,
    y_names=y_names,
    y_block=RegressionBlock(),
    splits=splits,
)
dls = to.dataloaders()


# %%
learn = tabular_learner(dls, layers=layers, metrics=rmse)


# %%
lrs = learn.lr_find(suggest_funcs=(valley))


# %%
learn.fit_one_cycle(cyc_len, lrs.valley)


# %%
dl = learn.dls.test_dl(test)


# %%
preds = learn.get_preds(dl=dl)
oracle_forecasts = preds[0].numpy().T[0]


# %%
arima_test = pd.DataFrame(
    {
        "Open": forecasts_open,
        "High": forecasts_high,
        "Low": forecasts_low,
        "Volume": forecasts_volume,
    }
)
arima_forecasts = []
for i in tqdm(range(arima_test.shape[0])):
    a = learn.predict(arima_test.iloc[i])[1].item()
    arima_forecasts.append(a)


# %%
fig = plt.figure(figsize=(12, 6))
sns.lineplot(
    x=test["Date"],
    y=oracle_forecasts,
    label="Predicted Close Price w. Oracle Open/High/Low",
    color="cyan",
)
sns.lineplot(
    x=test["Date"],
    y=arima_forecasts,
    label="Predicted Close Price w. ARIMA Open/High/Low",
    color="purple",
)
sns.lineplot(
    x=test["Date"],
    y=forecasts_close,
    label="ARIMA Close Price",
    color="lime",
)
sns.lineplot(
    x=test["Date"],
    y=test["Close"],
    label="Actual Close Price",
    color="orange",
)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Prediction of {ticker_name}")
fig.savefig(f"prediction.{ticker_name}.png", bbox_inches="tight", dpi=200)


# %%
print(f"rmse of close [arima]: {get_rmse(test['Close'], forecasts_close)}")
print(f"rmse of close [dl w. arima]: {get_rmse(test['Close'], arima_forecasts)}")
print(f"rmse of close [dl w. oracle]: {get_rmse(test['Close'], oracle_forecasts)}")

print(f"smape of close [arima]: {smape(test['Close'], forecasts_close)}")
print(f"smape of close [dl w. arima]: {smape(test['Close'], arima_forecasts)}")
print(f"smape of close [dl w. oracle]: {smape(test['Close'], oracle_forecasts)}")


# %%
row = pd.Series(
    dict(Open=arima_open, High=arima_high, Low=arima_low, Volume=int(arima_volume))
)
learn.predict(row)[1].item()


# %%

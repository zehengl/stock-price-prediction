<div align="center">
    <img src="https://cdn3.iconfinder.com/data/icons/animal-flat-colors/64/bear-512.png" alt="logo" height="96">
    <img src="https://cdn3.iconfinder.com/data/icons/animal-flat-colors/64/bufalo-512.png" alt="logo" height="96">
</div>

# stock-price-prediction

![coding_style](https://img.shields.io/badge/code%20style-black-000000.svg)

An attempt to predict stock price

**DISCLAIMER**: This repo is for educational purpose. Please don't take the results as financial advices.

## Environment

- Python 3.7
- Windows 10

## Install

Create a virtual environment.

    python -m venv venv
    .\venv\Scripts\activate

Next go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install PyTorch. For example, if you don't have CUDA,

    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Then install dependencies.

    pip install -r .\requirements.txt

> Use `pip install -r requirements-dev.txt` for development.
> It will install `pylint`, `black`, and `jupyter` to enable linting, auto-formatting, and notebook experience.

## Usage

Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) and claim your API key.

Create a `.env` file to store your alpha vantage api key and a ticker.

    # .env
    # required
    ALPHA_VANTAGE_API_KEY="xxx"
    TICKER_NAME="yyy"

    # optional and the defaults
    CYC_LEN=20
    SEED=2020
    VALID_PCT=.1
    LAYERS="[200, 100]"

Run the modelling script

    python modelling.py

## Example Results

### Price Action History

![](examples/history.ABX.TO.png)

### Time Series Analysis

![](examples/arima.ABX.TO.png)

### Predictive Modelling

![](examples/prediction.ABX.TO.png)

<hr>

<sup>

## Credits

- [Bear Logo][1] and [Bull Logo][2] by [pongsakorn tan][3]

</sup>

[1]: https://www.iconfinder.com/icons/4591876/animal_bear_carnivore_cartoon_fauna_head_zoo_icon
[2]: https://www.iconfinder.com/icons/4591900/animal_buffalo_cape_cartoon_fauna_herbivore_zoo_icon
[3]: https://www.iconfinder.com/kerismaker

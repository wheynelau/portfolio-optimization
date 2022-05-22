# Portfolio Optimization in Python with cryptocurrencies

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Support Me](#support-me)

## Overview

This fork is an improvement done on the original portfolio optimization, 
it also condenses everything into a class so it can be ran with a few lines.

## Setup

Right now, the library is not hosted on **PyPi** so you will need to do a local
install on your system if you plan to use it in other scrips you use.

First, clone this repo to your local system. After you clone the repo, make sure
to run the `setup.py` file, so you can install any dependencies you may need. To
run the `setup.py` file, run the following command in your terminal.

```console
pip install -e .
```

This will install all the dependencies listed in the `setup.py` file. Once done
you can use the library wherever you want.

## Usage

Simple example to optimize a portfolio for crypto and stocks

```python
import pandas as pd
from pyopt.client import PyOpt

# Define the symbols: Stocks and/or crypto (Actual input is not case-sensitive)
stocks = ['AAPL', 'MSFT', 'SQ']
crypto = ['BNB','BTC']

# Initialize the client.
pyopt = PyOpt() <--- This can be done before or after defining the symbols

# Add tickers to the optimizer
pyopt.add_stocks(stocks)
pyopt.add_crypto(crypto)

# Run optimizer

pyopt.run()
```

## Support me
Still a student so creating this was just a hobby. Providing feedback or criticisms will greatly help me.
Giving me a star helps too. 

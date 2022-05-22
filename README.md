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

From the terminal, run this command 

```console
pip install git+https://github.com/wheynelau/portfolio-optimization
```

This will install all the dependencies listed in the `setup.py` file. Once done
you can use the library wherever you want.

Potential issues: As this is my first repo if the setup does not work I may not know how to fix,
please do just copy the code if it doesn't work.

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

## Contribute to this project
I added some TODO or ideas for improvements but due to exams I did not manage fix many of them.
If there are issues please post them so I can try to fix them.

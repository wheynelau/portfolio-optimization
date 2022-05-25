# Portfolio Optimization in Python with cryptocurrencies

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Known Issues](#known-issues)
- [TODO](#todo)
- [Support Me](#support-me)
- [Contribute to this project](#contribute-to-this-project)
- [Credits](#credits)

## Overview

This fork is an improvement done on the original portfolio optimization, 
it also condenses everything into a class so it can be ran with a few lines. 
The main algorithms have not been modified much.
It uses yfinance as the provider for historical data and is optimized using
scipy.minimize function and Monte Carlo method. 

## Installation

Right now, the library is not hosted on **PyPi** so you will need to do a local
install on your system if you plan to use it in other scrips you use.

From the terminal, run this command 

```console
pip install git+https://github.com/wheynelau/portfolio-optimization
```

This will install all the dependencies listed in the `setup.py` file. Once done
you can use the library wherever you want.

Potential issues: As this is my first repo if the setup does not work I may not know how to fix,
please do just copy the code if it doesn't work. In the meantime I will also figure out how to improve this.

## Usage

Example to optimize a portfolio for crypto and stocks

_Personal suggestion: Do a quick google for the stock if you can't find it, I didn't know that to look for
VWRA I had to input "VWRA.L"._

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

## Known issues

- [ ] Missing error handling on some inputs

## TODO
 
### Personal

- [ ] Clean up code and make it more readable
- [ ] Improve github skills, in terms of formatting, version control, pull requests etc

### Project
- [ ] Create new exception class for PyOpt
- [ ] PySimpleGUI
- [ ] Minumum weightage purchase. (Useful if low capital and no fractional shares)
- [ ] Print results to html for readability and save it for future reference
- [ ] Input using user provided .csv files
- [x] Max weightage option Update: Not specified securities, but for all
- [ ] Weightage for individual securities . Would be better with gui or menu based
- [ ] Experiment getting input using **kwargs 
- [ ] Provide more settings

## Support me

Still a student so creating this was just a hobby. Providing feedback or criticisms will greatly help me.
Giving me a star helps too. 

## Contribute to this project

I added some TODO or ideas for improvements but due to exams I did not manage fix many of them.
If there are issues please post them so I can try to fix them.

## Credits

Credits to areed1192's github and his youtube for motivating and teaching me the algorithm's for 
Monte Carlo and scipy.minimize. You can check out his infomative youtube channel [here!](https://www.youtube.com/c/SigmaCoding)

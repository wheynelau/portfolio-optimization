from PyOpt import PyOpt

# Initialize

pyopt = PyOpt()

# Order of which you add first doesn't matter

crypto = ['bnb', 'btc', 'xrp', 'eth']

# Example of input without list variable

pyopt.add_stocks(['voo', 'vwra.l', 'qqq', 'gbug', 'spbo'])

pyopt.add_crypto(crypto)

# Set period to test (default:'1y')
pyopt.period = '1y'

# Number of simulations for Monte Carlo (default:5000)
pyopt.simulations = 10000

# Target for scipy minimize (default: 'sharpe')
pyopt.target = 'sharpe'

# Risk free rate (default: 2)
pyopt.RFR = 2
# Setting max weights

# Added function maxweights, set the max weightage

pyopt.maxweights = 20

# Run optimizer

pyopt.run()

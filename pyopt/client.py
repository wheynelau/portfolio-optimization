import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt
from datetime import date

""" 
Improvements to be made: 
1. Clean up if possible and write more pythonic codes ,open to feedback and criticism

TODO List:
1. Create backtest function using given weights

2. Menu based program with error handling

3. Add fractional constraints 
Eg: Stock A costs 50 and with capital of 200, Stock A weightage can only be 0,0.25,0.5,0.75 or 1

4. Print results to a webpage with visualization of capital growth 

5. Use a provided CSV file, useful if the stock/crypto can't be found on yfinance. May be useful for funds

6. Max weightage for specific asset, for this case the runfile gives 100% BNB weights for scipy.
Another situation would be set crypto portfolio to be max 25% of total portfolio
"""


class PyOpt:
    """
    | Class object for portfolio optimization.
    |
    |
    | Notes
    --------

    Not financial advice.

    |
    | Example run
    -------

    >>>Foo = PyOpt()

    >>>stocks = ['aapl','amzn']

    >>>Foo.add_stocks(stocks)

    >>>crypto = ['bnb','btc']

    >>>Foo.add_crypto(crypto)

    >>>Foo.run()

    |
    | For more details please use help(PyOpt)

    """

    def __init__(self):
        """
        Constructs the PyOpt object with the default values
        """

        # Initializes appropriate type
        self._tickers = []
        self._number_of_tickers = 0
        self._symbols_for_yfinance = ""

        # Default values
        self._simulations = 5000
        self._trading_days = 365
        self._period = '1y'
        self._RFR = 2
        self._max_weights = 100

        # Initialize dataframes
        self._fullDf = pd.DataFrame()
        self._log_returns = pd.DataFrame()
        self._simulations_df = pd.DataFrame()

        # For future use
        self._optimize = True

        # Scipy function
        self._minimize = 'sharpe'

    # Returns list of tickers

    @property
    def period(self):
        """
        | From yfinance
        |
        | Data period to download
        | By default this is set to '1y'.
        |
        | Notes:
        --------

        As some assets may be new, the dataset will not be the full length if the age of the asset is lower than the
        period

        |
        | Example:
        --------

        >>> print(Foo.period)
        1y

        :return: Returns the period
        """
        return self._period

    @period.setter
    def period(self, newPeriod: str):
        """
        | From yfinance
        |
        | Data period to download
        | By default this is set to '1y'.
        |
        | Notes:
        --------

        As some assets may be new, the dataset will not be the full length if the age of the asset is lower than the
        period.

        |
        | Example:
        --------

        >>> Foo.period = '2y'

        :param newPeriod: Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

        if newPeriod.lower() not in valid_periods:

            raise TypeError(f"Invalid period: Please choose from one of the following"
                            f"{str(valid_periods)}")

        else:

            self._period = newPeriod

    @property
    def simulations(self):
        """
        | Number of simulations to run for the monte-carlo algorithm.
        |
        | The default value is 5000. It is not recommended to use lower values due to
        insufficient simulations. It is also not recommended to use values that are too
        high.

        | Example:
        --------


        >>> print(Foo.simulations)
        5000


        :return: Returns current simulations value.
        """
        return self._simulations

    @simulations.setter
    def simulations(self, newSimulations: int):
        """
        | Number of simulations to run for the monte-carlo algorithm.
        | The default value is 5000. It is not
        recommended to use lower values due to insufficient simulations. It is also not recommended to use values
        that are too high.

        |
        | Example:
        --------

        >>> Foo.simulations = 10000

        :param newSimulations: Takes in an integer above 0
        """

        if newSimulations <= 1000:
            print(newSimulations, "simulations might be too low. Results may not be accurate.")
            self._simulations = newSimulations

        elif newSimulations > 1000:
            print("Number of simulations set as", newSimulations)
            self._simulations = newSimulations

        else:
            raise TypeError("Please provide a positive integer only. ")

    @property
    def target(self):
        """
        | Target for optimizing function for scipy.optimize
        | Only three acceptable inputs:
        |
        | 'Returns' : Get the maximum returns
        |
        | 'Vol': Get the minumum volatility
        |
        | 'Sharpe' : Gets the maximum sharpe ratio
        |
        | Example:
        --------


        >>> print(Foo.target)
        'sharpe'

        :return: Returns current optimizer target
        """
        return self._minimize

    @target.setter
    def target(self, newTarget: str):
        """
        | Target for optimizing function for scipy.optimize
        | Only three acceptable inputs:
        |
        | 'Returns' : Get the maximum returns
        |
        | 'Vol': Get the minumum volatility
        |
        | 'Sharpe' : Gets the maximum sharpe ratio
        |
        | Example:
        --------

        >>> Foo.target = 'sharpe'

        :param newTarget: string variable. Accepts only 'returns','vol', or 'sharpe'. Non-case senstitive
        """
        # Check if string

        if not isinstance(newTarget, str):
            raise AttributeError("Please input 'sharpe', 'vol' or 'returns' only.")
        newTarget = newTarget.lower()
        # Input validation
        if newTarget not in ['sharpe', 'vol', 'returns']:

            raise ValueError("Please input 'sharpe', 'vol' or 'returns' only.")

        else:
            self._minimize = newTarget

    @property
    def maxweights(self):
        """
        | Get the max weights in percent
        |
        | Default is 100 (%)
        |
        | Example:
        --------

        >>> print(Foo.maxweights)
        100

        :return: Returns current max weights
        """

    @maxweights.setter
    def maxweights(self, newWeights: float):
        """
        | Sets new weights value in percent
        |
        | Must only be run after adding stocks and crypto
        |
        | Example
        --------

        >>> Foo.maxweights = 20.5


        :param newWeights: float between 0-100
        """
        self._tickers = self._symbols_for_yfinance.strip().split(" ")
        self._number_of_tickers = len(self._tickers)
        try:
            float(newWeights)
            if self._symbols_for_yfinance == "":
                print("Please add assets first or run this after, setting default of 100%")
            elif newWeights > 100 or newWeights <= 0:
                print("Please set value between 0-100")
            elif 100 / self._number_of_tickers > newWeights:
                print(f"Please increase weights or reduce assets, "
                      f"can't set max weights lower than {100 / self._number_of_tickers:.2f}%\n"
                      f"Setting default of 100%.")

            else:
                self._max_weights = newWeights
        except ValueError:
            print("Invalid weights set, defaulting to 100%.")

    # For future use, create backtest function
    @property
    def optimize(self):
        return self._optimize

    # For future use, create backtest function
    @optimize.setter
    def optimize(self, newOptimize: str):

        newOptimize = newOptimize.lower()

        if newOptimize not in ['true', 'false']:
            raise TypeError("Please enter True or False")

        else:
            print("Optimization set as", newOptimize.upper())

            self._optimize = True if newOptimize == 'true' else False

    @property
    def RFR(self):
        """
        | Risk-Free rate. The default value is 2 percent.
        | 
        | Example:
        --------
        
        >>>print(Foo.RFR)
        
        2   
        
        :return: Returns RFR in percent.
        """
        return self._RFR

    @RFR.setter
    def RFR(self, newRFR):
        """
        | Risk-Free rate. The default value is 2 percent.
        |
        | Negative values are also accepted but the range is -100 to 100, although unrealistic
        |
        | Example:
        --------

        >>>Foo.RFR = 1.5

        :param newRFR: Input number in percent
        """

        try:
            float(newRFR)
        except ValueError:
            raise ValueError("Please enter a valid Risk Free Rate between -100 and 100")

        if -100 <= newRFR <= 100:
            self._RFR = newRFR

        else:
            raise ValueError("Please enter a valid Risk Free Rate between -100 and 100")

    # Creates a string from given stocks/crypto
    # TODO: Error catching? Although yfinance will throw out an error if invalid input

    def add_stocks(self, stocks: list):
        """
        | Adds stocks to the list to backtest.
        |
        | Example:
        --------

        >>> Foo.add_stocks(['voo','AAPL']


        :param stocks: list of stocks to add. Please note to use [] even if single item. Non-case sensitive
        """
        stocks = ' '.join([str(a).upper() for a in stocks])
        # if stocks added, trading days are 252

        if stocks:
            self._trading_days = 252

        self._symbols_for_yfinance = self._symbols_for_yfinance + " " + stocks

    def add_crypto(self, crypto: list):
        """
        | Adds cryptocurrencies to the list to backtest.
        |
        | Example:
        --------


        >>> Foo.add_crypto(['btc','BNB']

        :param crypto: list of crypto to add. Please note to use [] even if single item. Non-case sensitive
        """

        crypto = ' '.join([str(a).upper() + '-USD' for a in crypto])

        self._symbols_for_yfinance = self._symbols_for_yfinance + " " + crypto

    # Main run function

    def run(self):
        """
        | Main run function
        |
        | Checks if number of symbols are more than 1
        |
        | After that pulls data from yfinance
        |
        | Performs Monte Carlo method followed by scipy.minimize
        |
        | Plots efficient frontier graph and gives weights
        """

        # Remove additional spaces at the end
        # Convert the given string to a list of tickers for dataframe rearrangement
        self._tickers = self._symbols_for_yfinance.strip().split(" ")

        # Number of tickers
        self._number_of_tickers = len(self._tickers)

        # Check if symbols is empty or less than 2
        if self._number_of_tickers <= 1:
            raise ValueError("Needs more than 1 asset to optimize. Use add_stocks or add_crypto")

        # Download data from yfinance
        # Interval of 1 day is fixed

        self._fullDf = yf.download(self._symbols_for_yfinance,
                                   interval='1d',
                                   period=self._period,
                                   actions=False,
                                   group_by='ticker')

        # Prep dataframe
        self._dataframe_prep()

        # Print params
        self._get_params()

        """
        TODO: Test performance using current  weights. Useful if stock/crypto has no fractional shares
        and you're not able to get the exact weights provided by the script
        """
        if self._optimize:
            self._monte_carlo()
            self._summary()
            self._fit()
            plt.show()

        # TODO: Backtest function
        # else:

    # Prep dataframe for optimization

    def _dataframe_prep(self):
        # Drop NaN -----> TODO: Test if not dropping weekends will be better?

        self._fullDf.dropna(inplace=True)

        # Reduce multiindex into one level
        self._fullDf.columns = [' '.join(col) for col in self._fullDf.columns.values]

        # Keep only Adj Close
        self._fullDf = self._fullDf.loc[:, self._fullDf.columns.str.contains('Adj Close', case=True)]

        # Remove Adj Close just to get name of tickers
        self._fullDf.columns = self._fullDf.columns.str.replace(' Adj Close', '')

        # Arrange columns into given tickers for easier readability

        self._fullDf = self._fullDf.reindex(columns=self._tickers)

        # Get pct_change
        self._log_returns = self._fullDf.pct_change()

    def _get_params(self):
        print(f"CURRENT PARAMS\n"
              f"{'=' * 60}\n"
              f"Number of tickers:\t {self._number_of_tickers}\n"
              f"Tickers: \t{str(self._tickers)}\n"
              f"Risk Free Rate: \t{self._RFR} %\t\t"
              f"Period:\t {self._period}\n"
              f"Yearly Trading days (For APY, not affected by period) :\t {self._trading_days} days\n"
              f"Number of simulations:\t {self._simulations}\t\t"
              f"Scipy target: \t{self._minimize.upper()}")

    """Monte Carlo method functions"""

    # Runs monte carlo

    def _monte_carlo(self):

        # Unchanged except variables

        # This code is done by areed1192

        # Initialize the components, to run a Monte Carlo Simulation.

        # Prep an array to store the weights as they are generated, x iterations for each of our 4 symbols.
        all_weights = np.zeros((self._simulations, self._number_of_tickers))

        # Prep an array to store the returns as they are generated, x possible return values.
        ret_arr = np.zeros(self._simulations)

        # Prep an array to store the volatilities as they are generated, x possible volatility values.
        vol_arr = np.zeros(self._simulations)

        # Prep an array to store the sharpe ratios as they are generated, x possible Sharpe Ratios.
        sharpe_arr = np.zeros(self._simulations)

        # Start the simulations.
        for ind in range(self._simulations):
            # First, calculate the weights.
            weights = np.array(np.random.random(self._number_of_tickers))
            weights = weights / np.sum(weights)

            # If loop skips if the value is unchanged or set as the default
            if self._max_weights != 100:

                # Boolean array to check if any value is bigger than the max weightage
                check_max = weights > self._max_weights / 100

                # While there is a value greater than the max weights, the loop continues running
                # If there is a more pythonic way of doing this please do mention it

                while np.any(check_max):
                    weights = np.array(np.random.random(self._number_of_tickers))
                    weights = weights / np.sum(weights)
                    check_max = weights > self._max_weights / 100

            # Add the weights, to the `weights_arrays`.
            all_weights[ind, :] = weights

            # Calculate the expected log returns, and add them to the `returns_array`.
            ret_arr[ind] = np.sum(self._log_returns.mean() * weights * self._trading_days)

            # Calculate the volatility, and add them to the `volatility_array`.
            vol_arr[ind] = np.sqrt(
                np.dot(weights.T, np.dot(self._log_returns.cov() * self._trading_days, weights))
            )

            # Calculate the Sharpe Ratio and Add it to the `sharpe_ratio_array`.
            sharpe_arr[ind] = (ret_arr[ind] - self._RFR / 100) / vol_arr[ind]

        # Let's create our "Master Data Frame", with the weights, the returns, the volatility, and the Sharpe Ratio
        simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]

        # Create a DataFrame from it, then Transpose it so it looks like our original one.
        self._simulations_df = pd.DataFrame(data=simulations_data).T

        # Give the columns the Proper Names.
        self._simulations_df.columns = [
            'Returns',
            'Volatility',
            'Sharpe Ratio',
            'Portfolio Weights'
        ]

        self._simulations_df = self._simulations_df.infer_objects()

    # Gets max sharpe, min vol and plots graph

    def _summary(self):

        # Convert returns to %

        self._simulations_df['Returns'] = self._simulations_df['Returns'] * 100

        # Return the Max Sharpe Ratio from the run.
        self._max_sharpe_ratio = self._simulations_df.loc[self._simulations_df['Sharpe Ratio'].idxmax()]

        # Return the Min Volatility from the run.
        self._min_volatility = self._simulations_df.loc[self._simulations_df['Volatility'].idxmin()]

        # Plot the data on a Scatter plot.
        plt.scatter(
            y=self._simulations_df['Returns'],
            x=self._simulations_df['Volatility'],
            c=self._simulations_df['Sharpe Ratio'],
            cmap='RdYlBu'
        )

        # Give the Plot some labels, and titles.
        plt.title('Portfolio Returns Vs. Risk')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Returns')

        # Plot the Max Sharpe Ratio, using a `Red Star`.
        plt.scatter(
            self._max_sharpe_ratio[1],
            self._max_sharpe_ratio[0],
            marker=(5, 1, 0),
            color='r',
            s=600
        )

        # Plot the Min Volatility, using a `Blue Star`.
        plt.scatter(
            self._min_volatility[1],
            self._min_volatility[0],
            marker=(5, 1, 0),
            color='b',
            s=600
        )

        # Finally, show the plot.
        plt.show(block=False)

        # Print details
        self._print_details()

    # Print details on weights for max sharpe and min vol

    def _print_details(self):

        # Print out max and min volatality

        print('')
        print('=' * 60)
        print('MAX SHARPE RATIO:')
        print('-' * 60)
        details = self._get_weights(self._max_sharpe_ratio)
        print(details)
        print('-' * 60)

        print('')
        print('=' * 60)
        print('MIN VOLATILITY:')
        print('-' * 60)
        details = self._get_weights(self._min_volatility)
        print(details)
        print('-' * 60)

    # Get weights for monte carlo

    def _get_weights(self, resultSeries: pd.Series):

        # Print details for given result series

        print(f"Returns:\t {resultSeries[0]:.3f}%")
        print(f"Volatility:\t {resultSeries[1]:.3f}")
        print(f"Sharpe Ratio:\t {resultSeries[2]:.3f}")
        print("Weights:")
        weights = pd.DataFrame(resultSeries[3])
        result = pd.concat([pd.Series(self._tickers), weights], axis=1)
        result.set_axis(['Tickers', 'Weights'], axis=1, inplace=True)
        result.set_index('Tickers', inplace=True)

        # Convert to xx%
        result['Weights'] = result['Weights'].apply(lambda x: str(round(x * 100, 3)) + "%")

        # Result is a dataframe of tickers/weights

        return result

    """Scipy related functions"""

    # Runs scipy.optimize

    def _fit(self):

        bounds = tuple((0, self._max_weights / 100) for symbol in range(self._number_of_tickers))

        # Define the constraints, here I'm saying that the sum of each weight must not exceed 100%.
        constraints = ({'type': 'eq', 'fun': self._check_sum})

        init_guess = np.full(self._number_of_tickers, 1 / self._number_of_tickers)

        # Perform the operation to minimize the risk.
        optimized_sharpe = sci_opt.minimize(
            fun=self._optimizing_func,
            x0=init_guess,  # minimize this. # Start with these values.
            method='SLSQP',
            bounds=bounds,  # don't exceed these bounds.
            constraints=constraints  # make sure you don't exceed the 100% constraint.
        )

        weights = optimized_sharpe.x
        weights = np.round(weights, 3)
        metrics = self._get_metrics(weights)
        weights_df = pd.DataFrame(weights)
        result = pd.concat([pd.Series(self._tickers), weights_df], axis=1)
        result.set_axis(['Tickers', 'Weights'], axis=1, inplace=True)
        result.set_index('Tickers', inplace=True)

        # Convert to xx%
        result['Weights'] = result['Weights'].apply(lambda x: str(round(x * 100, 3)) + "%")

        # Print the results.
        print('')
        print('=' * 60)
        print('OPTIMIZED SHARPE RATIO USING SCIPY\n')
        print(f'TARGET:\t{self._minimize.upper()}')
        print('-' * 60)
        print(f"Returns:\t {metrics[0] * 100:.3f}%")
        print(f"Volatility:\t {metrics[1]:.3f}")
        print(f"Sharpe Ratio:\t {metrics[2]:.3f}")
        print("Weights:")
        print(result)
        print('-' * 60)

    # Get returns, vol, sharpe from weights for Scipy

    def _get_metrics(self, weights: list) -> np.array:
        """
        ### Overview:
        ----
        With a given set of weights, return the portfolio returns,
        the portfolio volatility, and the portfolio sharpe ratio.

        ### Arguments:
        ----
        weights (list): An array of portfolio weights.

        ### Returns:
        ----
        (np.array): An array containg return value, a volatility value,
            and a sharpe ratio.
        """

        # Convert to a Numpy Array.
        weights = np.array(weights)

        # Calculate the returns, remember to annualize them (252).
        ret = np.sum(self._log_returns.mean() * weights) * self._trading_days

        # Calculate the volatility, remember to annualize them (252).
        vol = np.sqrt(
            np.dot(weights.T, np.dot(self._log_returns.cov() * self._trading_days, weights))
        )

        # Calculate the Sharpe Ratio.
        sr = ret - (self._RFR / 100) / vol

        return np.array([ret, vol, sr])

    # Returns based on the optimizing function

    def _optimizing_func(self, weights: list) -> np.array:

        """The function used to minimize the Sharpe Ratio.

        ### Arguments:
        ----
        weights (list): The weights, we are testing to see
            if it's the minimum.

        ### Returns:
        ----
        Returns depending on the minimizing target
        Returns, vol, sharpe (default)
        (np.array): An numpy array of the portfolio metrics.
        """
        # Return negative sharpe
        if self._minimize.lower() == 'sharpe':
            return -self._get_metrics(weights)[2]

        # Return vol
        elif self._minimize.lower() == 'vol':
            return self._get_metrics(weights)[1]

        # Return negative returns
        else:
            return -self._get_metrics(weights)[0]

    # Constraint

    @staticmethod
    def _check_sum(weights: list) -> float:
        """Ensure the allocations of the "weights", sums to 1 (100%)

        ### Arguments:
        ----
        weights (list): The weights we want to check to see
            if they sum to 1.

        ### Returns:
        ----
        float: The different between 1 and the sum of the weights.
        """
        return 1 - np.sum(weights)

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:26:49 2024

@author: vic
"""
from functools import wraps
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
import pandas as pd
import numpy as np
import subprocess
import traceback
import seaborn as sns
from scipy.stats import shapiro, normaltest, kstest, anderson
import scipy.stats as stats
from datetime import timedelta
import datetime
import os

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        end_time = time.time()
        # Check if the first positional argument is an instance of a class (typically 'self')
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f"**[timer]{class_name}.{func.__name__!r} finished in {end_time - start_time:.4f} secs")
        else:
            print(f"**[timer]{func.__name__!r} finished in {end_time - start_time:.4f} secs")
        return retval
    return wrapper

def plot_pnl_curve(df, returns='strategy_simple_returns_net', benchmark=False):
    ###plot
    #df = df.dropna()
    # Overall PnL curve (already cumulative, no further calculation needed)
    
    if returns == 'strategy_simple_returns_net':
        pnl_curve = df['cstrategy_simple_returns_net']
        
    elif returns == 'strategy_simple_levered_net':
        pnl_curve = df['cstrategy_simple_levered_net']
    
    if benchmark:
        benchmark_curve = df['creturns_simple']    
        #print(benchmark_curve)
    # Filter daily returns for long and short positions
    long_returns = df.loc[df['pred'] == 1, returns]
    short_returns = df.loc[df['pred'] == -1, returns]
        
    # Recalculate cumulative PnL for long and short positions
    pnl_long = (1 + long_returns).cumprod()
    pnl_short = (1 + short_returns).cumprod()
        
    # Plot the PnL curve for the training set
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_curve, label='Overall Strategy', color='blue', linewidth=2)
    if benchmark:
        plt.plot(benchmark_curve, label='benchmark', color='black', linewidth=2)
    plt.plot(pnl_long, label='Long Positions', color='green', linestyle='--')
    plt.plot(pnl_short, label='Short Positions', color='red', linestyle='--')
    plt.title('PnL Curve')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.legend(loc='best')
    plt.grid(axis='y', lw=0.5)
    plt.show()

def plot_scatter(df, y_col, x_col, title=None, xlabel=None, ylabel=None, 
                 color='blue', figsize=(10, 6), regression=False, grid=True, 
                 alpha=0.6, s=30):
    """
    Generate a scatter plot for specified columns in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - x_col (str): Name of column for x-axis
    - y_col (str): Name of column for y-axis
    - title (str): Plot title (default: f'{y_col} vs {x_col}')
    - xlabel/ylabel (str): Axis labels (defaults to column names)
    - color (str): Marker color
    - figsize (tuple): Figure dimensions
    - regression (bool): Whether to plot a regression line
    - grid (bool): Whether to show gridlines
    - alpha (float): Marker transparency (0-1)
    - s (int): Marker size
    """
    # Validate columns
    if x_col not in df.columns or y_col not in df.columns:
        missing = [col for col in [x_col, y_col] if col not in df.columns]
        raise ValueError(f"Columns missing: {missing}")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(x=df[x_col], y=df[y_col], c=color, alpha=alpha, s=s)
    
    # Add regression line
    if regression:
        x = df[x_col]
        y = df[y_col]
        try:
            coeffs = np.polyfit(x, y, 1)
            # Format tiny coefficients to show meaningful precision
            slope_str = f"{coeffs[0]:.2e}".replace("e-0", "e-").replace("e-", " × 10⁻")  # 2.08e-5 → 2.08 × 10⁻5
            intercept_str = f"{coeffs[1]:.2e}".replace("e-0", "e-").replace("e-", " × 10⁻")
            label = f'y = {slope_str} x + {intercept_str}'
        except np.linalg.LinAlgError:
            label = "Regression failed"
        trend_line = np.poly1d(coeffs)
        ax.plot(x, trend_line(x), color='red', linestyle='--', label=label)
        ax.legend()

    # Labels and aesthetics
    ax.set_title(title if title else f'{y_col} vs {x_col}')
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    ax.grid(grid)
    
    plt.tight_layout()
    plt.show()
    return ax

def plot_histogram(df, column, bins=50, kde=True, title=None):
    """
    Plots a histogram of the specified column, dropping missing values.

    Parameters:
    - df: DataFrame containing the data.
    - column: Column name to plot the histogram for.
    - bins: Number of bins in the histogram.
    - kde: Whether to plot a Kernel Density Estimate (KDE).
    - title: Title of the plot.
    """
    column_data = df[column].dropna()  # Drop missing values
    plt.figure(figsize=(10, 6))
    sns.histplot(column_data, bins=bins, kde=kde, color='blue')
    if title is None:
        title = f"Histogram of {column}"
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    
def check_normal_distribution(df, column, alpha=0.05, verbose=True):
    """
    Checks if the data in the specified column follows a normal distribution.
    Uses Shapiro-Wilk, D'Agostino's K^2, Kolmogorov-Smirnov, and Anderson-Darling tests.

    Parameters:
    - df: DataFrame containing the data.
    - column: Column name to check for normality.
    - alpha: Significance level for the normality tests (default: 0.05).
    - verbose: If True, prints the results of the tests.

    Returns:
    - A dictionary with the results of the normality tests.

    Notes:
    - **Shapiro-Wilk Test** (valid for N ≤ 5000):
      - Null Hypothesis: The data is normally distributed.
      - If p-value > alpha, the data is considered normally distributed.
    - **D'Agostino's K^2 Test**:
      - Null Hypothesis: The data is normally distributed.
      - If p-value > alpha, the data is considered normally distributed.
    - **Kolmogorov-Smirnov Test**:
      - Null Hypothesis: The data is normally distributed.
      - If p-value > alpha, the data is considered normally distributed.
    - **Anderson-Darling Test**:
      - Null Hypothesis: The data is normally distributed.
      - If the test statistic < critical value (for the given alpha), the data is considered normally distributed.

    Example:
    >>> df = pd.DataFrame({'returns': np.random.normal(0, 1, 1000)})
    >>> normality_results = check_normal_distribution(df, 'returns')
    """
    column_data = df[column].dropna()  # Drop missing values
    n = len(column_data)

    # Initialize results dictionary
    normality_results = {}

    # Shapiro-Wilk Test
    if n <= 5000:  # Shapiro-Wilk is not recommended for N > 5000
        shapiro_stat, shapiro_p = shapiro(column_data)
        shapiro_result = shapiro_p > alpha
        normality_results['Shapiro-Wilk'] = {
            'statistic': shapiro_stat,
            'p-value': shapiro_p,
            'is_normal': shapiro_result
        }
    else:
        normality_results['Shapiro-Wilk'] = {
            'statistic': None,
            'p-value': None,
            'is_normal': None,
            'message': 'Shapiro-Wilk not recommended for N > 5000'
        }

    # D'Agostino's K^2 Test
    k2_stat, k2_p = normaltest(column_data)
    k2_result = k2_p > alpha
    normality_results['D\'Agostino\'s K^2'] = {
        'statistic': k2_stat,
        'p-value': k2_p,
        'is_normal': k2_result
    }

    # Kolmogorov-Smirnov Test
    ks_stat, ks_p = kstest(column_data, 'norm', args=(column_data.mean(), column_data.std()))
    ks_result = ks_p > alpha
    normality_results['Kolmogorov-Smirnov'] = {
        'statistic': ks_stat,
        'p-value': ks_p,
        'is_normal': ks_result
    }

    # Anderson-Darling Test
    anderson_result = anderson(column_data, dist='norm')
    anderson_stat = anderson_result.statistic
    # Anderson-Darling returns critical values and corresponding significance levels
    # We compare the test statistic to the critical value at the desired alpha level
    critical_values = anderson_result.critical_values
    significance_levels = anderson_result.significance_level
    # Find the critical value corresponding to the desired alpha level
    anderson_critical = None
    for i, level in enumerate(significance_levels):
        if level == alpha * 100:  # Anderson-Darling uses percentages (e.g., 5 for 0.05)
            anderson_critical = critical_values[i]
            break
    if anderson_critical is not None:
        anderson_result_bool = anderson_stat < anderson_critical
    else:
        anderson_result_bool = None
    normality_results['Anderson-Darling'] = {
        'statistic': anderson_stat,
        'critical_value': anderson_critical,
        'is_normal': anderson_result_bool
    }

    # Print results if verbose is True
    if verbose:
        print(f"-----Testing Normal Distribution: {column}-----")

        for test, result in normality_results.items():
            print(f"{test}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        print()

    # Return results
    return normality_results

def plot_qq(df, column, title=None):
    """
    Plots a Q-Q (Quantile-Quantile) plot for the specified column, dropping missing values.

    Parameters:
    - df: DataFrame containing the data.
    - column: Column name to plot the Q-Q plot for.
    - title: Title of the plot (default: "Q-Q Plot").

    Notes:
    - A Q-Q plot is used to compare the distribution of the data to a normal distribution.
    - **Interpretation**:
      - If the points lie approximately on the straight line (45-degree reference line),
        the data is considered to follow a normal distribution.
      - Deviations from the line indicate departures from normality:
        - **Curved Patterns**: Indicate skewness (e.g., right or left skew).
        - **S-Shaped Patterns**: Indicate heavy or light tails (kurtosis).

    Example:
    >>> df = pd.DataFrame({'returns': np.random.normal(0, 1, 1000)})
    >>> plot_qq(df, 'returns', title="Q-Q Plot of Returns")
    """
    column_data = df[column].dropna()  # Drop missing values
    plt.figure(figsize=(10, 6))
    stats.probplot(column_data, dist="norm", plot=plt)
    if title is None:
        title = f"Q-Q plot: {column}"
    plt.title(title)
    
    
def analyze_skewness_kurtosis(df, column, verbose=True):
    """
    Analyzes the skewness and kurtosis of a column in a DataFrame.
    Provides a summary, interpretation, and visualizes the distribution.

    Parameters:
    - df: DataFrame containing the data.
    - column: Column name to analyze.
    - verbose: If True, prints the summary and interpretations.

    Returns:
    - A dictionary with skewness and kurtosis values, along with their interpretations.

    Notes:
    - **Skewness**:
      - Measures the asymmetry of the distribution.
      - **Skewness > 0**: The distribution is right-skewed (tail on the right).
        - Common in situations with frequent small losses and occasional extreme gains (e.g., crypto returns).
      - **Skewness < 0**: The distribution is left-skewed (tail on the left).
        - Common in situations with frequent small gains and occasional extreme losses (e.g., insurance claims).
      - **Skewness = 0**: The distribution is symmetric (e.g., normal distribution).

    - **Kurtosis**:
      - Measures the "tailedness" of the distribution (how much data is in the tails).
      - **Kurtosis > 0**: The distribution has fat tails (leptokurtic).
        - Common in situations with high volatility and extreme events (e.g., financial crashes or spikes).
      - **Kurtosis < 0**: The distribution has thin tails (platykurtic).
        - Rare in financial data but may occur in stable or uniform distributions.
      - **Kurtosis = 0**: The distribution has normal tails (mesokurtic).

    Example:
    >>> df = pd.DataFrame({'returns': np.random.normal(0, 1, 1000)})
    >>> results = analyze_skewness_kurtosis(df, 'returns')
    >>> print(results)
    """
    
    column_data = df[column].dropna()  # Drop missing values

    # Calculate skewness and kurtosis
    skewness = column_data.skew()
    kurtosis = column_data.kurtosis()

    # Interpretation of skewness
    if skewness > 0:
        skew_interpret = "The distribution is right-skewed (tail on the right)."
    elif skewness < 0:
        skew_interpret = "The distribution is left-skewed (tail on the left)."
    else:
        skew_interpret = "The distribution is symmetric."

    # Interpretation of kurtosis
    if kurtosis > 0:
        kurt_interpret = "The distribution has fat tails (leptokurtic)."
    elif kurtosis < 0:
        kurt_interpret = "The distribution has thin tails (platykurtic)."
    else:
        kurt_interpret = "The distribution has normal tails (mesokurtic)."
    
    if verbose:
        # Print summary
        print(f"-----Testing Skewness and Kurtosis: {column}-----")
        print(f"Skewness: {skewness:.2f}")
        print(f"Interpretation: {skew_interpret}")
        print(f"Kurtosis: {kurtosis:.2f}")
        print(f"Interpretation: {kurt_interpret}\n")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(column_data, kde=True, color='blue')
    plt.title(f"Distribution of {column} (Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f})")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

    # Return results
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'skew_interpretation': skew_interpret,
        'kurt_interpretation': kurt_interpret
    }

def calculate_correlation(df, col1, col2, by_year=False, verbose=True):
    """
    Calculates the correlation between two columns in a DataFrame, optionally by year.

    Parameters:
    - df: pandas DataFrame
    - col1: Name of the first column (string)
    - col2: Name of the second column (string)
    - by_year: If True, calculates correlation by year and includes general correlation (bool)
    - verbose: If True, prints the correlation(s) (bool)

    Returns:
    - A dictionary containing correlation metrics:
      - If by_year=False, returns the general correlation.
      - If by_year=True, returns a dictionary with:
        - 'general': General (ungrouped) correlation
        - 'yearly': Correlation for each year
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' not found in the DataFrame.")
    
    # Initialize the results dictionary
    results = {}

    # Calculate general correlation
    general_correlation = df[col1].corr(df[col2])
    results['general'] = general_correlation
    if verbose:
        print(f"General correlation between '{col1}' and '{col2}': {general_correlation:.4f}")

    if by_year:
        # Ensure the DataFrame has a datetime index or a 'date' column
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' not in df.columns:
                raise ValueError("DataFrame must have a datetime index or a 'date' column to calculate correlation by year.")
            df = df.set_index('date')

        # Group by year and calculate correlation for each year
        yearly_correlations = {}
        for year, group in df.groupby(df.index.year):
            correlation = group[col1].corr(group[col2])
            yearly_correlations[year] = correlation
            if verbose:
                print(f"Correlation between '{col1}' and '{col2}' in {year}: {correlation:.4f}")
        
        results['yearly'] = yearly_correlations

    return results

def linear_regression_analysis(df, y, x, by_year=False, print_summary=False):
    """
    Performs a linear regression and returns regression metrics, optionally by year.

    Parameters:
    - df: pandas DataFrame containing the data
    - y: string, name of the dependent variable (target)
    - x: string or list of strings, name(s) of the independent variable(s) (predictors)
    - by_year: If True, performs regression for each year separately and includes general results (bool)
    - print_summary: If True, prints the regression summary (bool)

    Returns:
    - A dictionary containing regression metrics:
      - If by_year=False, returns general regression metrics.
      - If by_year=True, returns a dictionary with:
        - 'general': General (ungrouped) regression metrics
        - 'yearly': Regression metrics for each year
    """
    try:
        if isinstance(x, str):
            x = [x]  # Ensure x is a list for consistency

        # Ensure the DataFrame has a datetime index or a 'date' column
        if by_year:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' not in df.columns:
                    raise ValueError("DataFrame must have a datetime index or a 'date' column to perform regression by year.")
                df = df.set_index('date')

        # Initialize the results dictionary
        results = {}

        # Perform general regression
        general_results = _perform_regression(df, y, x, print_summary)
        if general_results:
            results['general'] = general_results

        # Perform regression by year if requested
        if by_year:
            yearly_results = {}
            for year, group in df.groupby(df.index.year):
                year_result = _perform_regression(group, y, x, print_summary)
                if year_result:
                    yearly_results[year] = year_result
            results['yearly'] = yearly_results

        return results
    except Exception as e:
        print(f"Linear regression analysis failed: {e}")
        print(traceback.format_exc())  # Print the full error traceback
        return None

def _perform_regression(df, y, x, print_summary):
    """
    Helper function to perform linear regression on a given DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data
    - y: string, name of the dependent variable (target)
    - x: list of strings, name(s) of the independent variable(s) (predictors)
    - print_summary: If True, prints the regression summary (bool)

    Returns:
    - A dictionary containing regression metrics:
      - R-squared
      - Adjusted R-squared
      - Coefficients
      - P-values
      - Standard Errors
      - T-statistics
      - Intercept
    """
    try:
        # Prepare the X (independent variables) and y (dependent variable)
        X = df[x]
        X = sm.add_constant(X)  # Add a constant (intercept) to the model
        Y = df[y]

        # Check for NaN or infinite values and drop problematic rows
        combined_data = pd.concat([X, Y], axis=1)  # Combine X and Y for consistent cleaning
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()

        # Reassign cleaned data to X and Y
        X = combined_data[x]
        X = sm.add_constant(X)  # Re-add the constant after cleaning
        Y = combined_data[y]

        # Check if data is empty after cleaning
        if X.empty or Y.empty:
            raise ValueError("Data is empty after cleaning. Cannot perform linear regression.")

        # Fit the regression model
        model = sm.OLS(Y, X).fit()

        # Extract regression metrics
        results = {
            "R-squared": model.rsquared,
            "Adjusted R-squared": model.rsquared_adj,
            "Coefficients": model.params.to_dict(),
            "P-values": model.pvalues.to_dict(),
            "Standard Errors": model.bse.to_dict(),
            "T-statistics": model.tvalues.to_dict(),
            "Intercept": model.params['const'] if 'const' in model.params else None
        }

        if print_summary:
            print("#" * 100)
            print(model.summary())
            print("#" * 100)

        return results
    except Exception as e:
        print(f"Regression failed: {e}")
        print(traceback.format_exc())  # Print the full error traceback
        return None

def describe_with_custom_percentiles(df, column, custom_percentiles=None):
    """
    Computes summary statistics for a column, including custom percentiles.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - column (str): The column name to describe.
    - custom_percentiles (list, optional): List of custom percentiles (e.g., [0.1, 0.2, ..., 1]).
        If None, defaults to 10%, 20%, ..., 100%.

    Returns:
    - pd.Series: Combined statistics including describe() and custom percentiles.
    """
    if custom_percentiles is None:
        # Default to 10%, 20%, ..., 100%
        custom_percentiles = [i / 10 for i in range(1, 11)]

    # Get standard describe() output for the column
    description = df[column].describe()

    # Calculate custom percentiles
    percentiles = df[column].quantile(custom_percentiles)
    percentiles.index = [f"{int(p * 100)}%" for p in percentiles.index]  # Format as percentages

    # Combine describe() output with custom percentiles using pd.concat()
    combined = pd.concat([description, percentiles])

    return combined

def trades_cycle_analysis(df, 
                          simple_returns_column='strategy_simple_returns',
                          simple_returns_net_column='strategy_simple_returns_net',
                          leverage = 1,
                          position='pred'):
    """Enhanced trade analysis with return metrics, MAE, MFE, and their respective row counts."""
    trades = []
    current_position = None
    entry_pos = None
    entry_idx = None
    
    def compute_mae_mfe(trade_df, entry_price, position):
        """Calculate MAE, MFE, and the row count to reach them."""
        if position == 'long':
            mae_values = (entry_price - trade_df['Low']) / entry_price
            mfe_values = (trade_df['High'] - entry_price) / entry_price
        elif position == 'short':
            mae_values = (trade_df['High'] - entry_price) / entry_price
            mfe_values = (entry_price - trade_df['Low']) / entry_price

        # Compute MAE and MFE values, simple multiply leverage for easy calculation, see if need adjust later
        mfe = leverage * mfe_values.max()  # Max favorable excursion
        mae = leverage * mae_values.max()  # Max adverse excursion

        # Find row index where MAE and MFE occur
        mfe_index = mfe_values.idxmax()
        mae_index = mae_values.idxmax()

        # Compute row count from entry to MAE/MFE
        mfe_periods = trade_df.index.get_loc(mfe_index) - trade_df.index.get_loc(trade_df.index[0])
        mae_periods = trade_df.index.get_loc(mae_index) - trade_df.index.get_loc(trade_df.index[0])

        return mfe, mae, mfe_periods, mae_periods
    
    def compute_trade_metrics(trade_df, entry_price, position):
        """Calculate all metrics for a trade window."""
        metrics = {}
        metrics['gross_returns'] = (1 + trade_df[simple_returns_column]).prod() - 1
        metrics['net_returns'] = (1 + trade_df[simple_returns_net_column]).prod() - 1
        metrics['transaction_cost'] = (trade_df[simple_returns_column] - trade_df[simple_returns_net_column]).sum()
        metrics['gross_result'] = 'win' if metrics['gross_returns'] > 0 else ('loss' if metrics['gross_returns'] < 0 else 'breakeven')
        metrics['result'] = 'win' if metrics['net_returns'] > 0 else ('loss' if metrics['net_returns'] < 0 else 'breakeven')
        metrics['mfe'], metrics['mae'], metrics['mfe_periods'], metrics['mae_periods'] = compute_mae_mfe(trade_df, entry_price, position)
        return metrics

    for pos in range(len(df)):
        current_pred = df.iloc[pos][position]
        timestamp = df.index[pos]
        
        if current_position is None:
            if current_pred != 0:
                current_position = current_pred
                entry_pos = pos
                entry_idx = timestamp
        else:
            if current_pred != current_position or current_pred == 0:
                trade_window = df.iloc[entry_pos:pos]
                entry_price = df.loc[entry_idx, 'Open']
                metrics = compute_trade_metrics(trade_window, entry_price, 'long' if current_position == 1 else 'short')
                duration = pos - entry_pos
                
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': df.index[pos - 1],
                    'entry_price': entry_price,
                    'exit_price': df.loc[df.index[pos - 1], 'Close'],
                    'period_highest': max(trade_window.High),
                    'period_lowest': min(trade_window.Low),
                    'duration': duration,
                    'position': 'long' if current_position == 1 else 'short',
                    'status': 'closed',
                    **metrics
                })
                
                if current_pred != 0:
                    current_position = current_pred
                    entry_pos = pos
                    entry_idx = timestamp
                else:
                    current_position = None

    if current_position is not None:
        trade_window = df.iloc[entry_pos:]
        entry_price = df.loc[entry_idx, 'Open']
        metrics = compute_trade_metrics(trade_window, entry_price, 'long' if current_position == 1 else 'short')
        duration = len(df) - entry_pos
        
        trades.append({
            'entry_date': entry_idx,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': df.loc[df.index[-1], 'Close'],
            'period_highest': max(trade_window.High),
            'period_lowest': min(trade_window.Low),
            'duration': duration,
            'position': 'long' if current_position == 1 else 'short',
            'status': 'open',
            **metrics
        })
    
    result_df = pd.DataFrame(trades)
    result_df['gross_returns_pct'] = result_df['gross_returns'] * 100
    result_df['net_returns_pct'] = result_df['net_returns'] * 100
    result_df['transaction_cost_pct'] = result_df['transaction_cost'] * 100
    result_df['mfe_pct'] = result_df['mfe'] * 100
    result_df['mae_pct'] = result_df['mae'] * 100
    result_df['mfe_periods'] = result_df['mfe_periods']
    result_df['mae_periods'] = result_df['mae_periods']
    result_df['cgross_returns_pct'] = (1 + result_df['gross_returns']).cumprod()*100
    result_df['cnet_returns_pct'] = (1 + result_df['net_returns']).cumprod()*100

    cols = [c for c in result_df.columns if c not in ['gross_returns', 'net_returns', 'transaction_cost', 'mfe', 'mae', 'mfe_periods', 'mae_periods']]
    cols += ['mfe_periods', 'mae_periods']
    result_df = result_df.loc[:, cols]
    
    num_trades_cycle_long = len(result_df[result_df['position'] == 'long'])
    num_trades_cycle_short = len(result_df[result_df['position'] == 'short'])
    winning_ratio_all = round(len(result_df[result_df['gross_result'] == 'win']) / len(result_df) * 100, 2) if len(result_df) > 0 else 0.0
    winning_ratio_long = round(len(result_df[(result_df['gross_result'] == 'win') & (result_df['position'] == 'long')]) /
                               num_trades_cycle_long * 100, 2) if num_trades_cycle_long > 0 else 0.0
    winning_ratio_short = round(len(result_df[(result_df['gross_result'] == 'win') & (result_df['position'] == 'short')]) /
                                num_trades_cycle_short * 100, 2) if num_trades_cycle_short > 0 else 0.0
    winning_ratio_all_net = round(len(result_df[result_df['result'] == 'win']) / len(result_df) * 100, 2) if len(result_df) > 0 else 0.0
    winning_ratio_long_net = round(len(result_df[(result_df['result'] == 'win') & (result_df['position'] == 'long')]) /
                                   num_trades_cycle_long * 100, 2) if num_trades_cycle_long > 0 else 0.0
    winning_ratio_short_net = round(len(result_df[(result_df['result'] == 'win') & (result_df['position'] == 'short')]) /
                                    num_trades_cycle_short * 100, 2) if num_trades_cycle_short > 0 else 0.0
    num_trades_cycle_total = len(result_df)
    
    holding_duration_total = result_df.duration.sum()
    holding_duration_long = result_df[result_df['position'] == 'long'].duration.sum() if num_trades_cycle_long > 0 else 0
    holding_duration_short = result_df[result_df['position'] == 'short'].duration.sum() if num_trades_cycle_short > 0 else 0
    
    ratio_l = round(num_trades_cycle_long / num_trades_cycle_total * 100, 2) if num_trades_cycle_total > 0 else 0.0
    ratio_s = round(num_trades_cycle_short / num_trades_cycle_total * 100, 2) if num_trades_cycle_total > 0 else 0.0
    holding_ratio_l = round(holding_duration_long / holding_duration_total * 100, 2) if holding_duration_total > 0 else 0.0
    holding_ratio_s = round(holding_duration_short / holding_duration_total * 100, 2) if holding_duration_total > 0 else 0.0  

    # Calculate time gaps between exit and next entry in periods
    if len(result_df) > 1:
        exit_positions = result_df['exit_date'].apply(df.index.get_loc)
        entry_positions = result_df['entry_date'].apply(df.index.get_loc)
        time_gaps = entry_positions[1:].values - exit_positions[:-1].values
        max_time_gap = time_gaps.max() if len(time_gaps) > 0 else 0
        mean_time_gap = round(time_gaps.mean(), 2) if len(time_gaps) > 0 else 0
        median_time_gap = round(pd.Series(time_gaps).median(), 2) if len(time_gaps) > 0 else 0
    else:
        max_time_gap = 0
        mean_time_gap = 0
        median_time_gap = 0

    metric_mean = {'general': {
                    'duration': round(result_df['duration'].mean(), 2),
                    'gross_returns_pct': round(result_df['gross_returns_pct'].mean(), 2),
                    'net_returns_pct': round(result_df['net_returns_pct'].mean(), 2),
                    'mfe_pct': round(result_df['mfe_pct'].mean(), 2),
                    'mae_pct': round(result_df['mae_pct'].mean(), 2),
                    'mae_periods': round(result_df['mae_periods'].mean(), 2),
                    'mfe_periods': round(result_df['mfe_periods'].mean(), 2),
                    'time_gap_periods': mean_time_gap  # Added mean time gap
                    },
                   'win': {
                    'duration': round(result_df[result_df['result'] == 'win']['duration'].mean(), 2),
                    'gross_returns_pct': round(result_df[result_df['result'] == 'win']['gross_returns_pct'].mean(), 2),
                    'net_returns_pct': round(result_df[result_df['result'] == 'win']['net_returns_pct'].mean(), 2),
                    'mfe_pct': round(result_df[result_df['result'] == 'win']['mfe_pct'].mean(), 2),
                    'mae_pct': round(result_df[result_df['result'] == 'win']['mae_pct'].mean(), 2),
                    'mae_periods': round(result_df[result_df['result'] == 'win']['mae_periods'].mean(), 2),
                    'mfe_periods': round(result_df[result_df['result'] == 'win']['mfe_periods'].mean(), 2),
                    'time_gap_periods': mean_time_gap  # Same mean for all trades (not per win/loss)
                    }, 
                   'loss': {
                    'duration': round(result_df[result_df['result'] == 'loss']['duration'].mean(), 2),
                    'gross_returns_pct': round(result_df[result_df['result'] == 'loss']['gross_returns_pct'].mean(), 2),
                    'net_returns_pct': round(result_df[result_df['result'] == 'loss']['net_returns_pct'].mean(), 2),
                    'mfe_pct': round(result_df[result_df['result'] == 'loss']['mfe_pct'].mean(), 2),
                    'mae_pct': round(result_df[result_df['result'] == 'loss']['mae_pct'].mean(), 2),
                    'mae_periods': round(result_df[result_df['result'] == 'loss']['mae_periods'].mean(), 2),
                    'mfe_periods': round(result_df[result_df['result'] == 'loss']['mfe_periods'].mean(), 2),
                    'time_gap_periods': mean_time_gap  # Same mean for all trades (not per win/loss)
                    }, 
    }

    metric_median = {'general': {
                      'duration': round(result_df['duration'].median(), 2),
                      'gross_returns_pct': round(result_df['gross_returns_pct'].median(), 2),
                      'net_returns_pct': round(result_df['net_returns_pct'].median(), 2),
                      'mfe_pct': round(result_df['mfe_pct'].median(), 2),
                      'mae_pct': round(result_df['mae_pct'].median(), 2),
                      'mae_periods': round(result_df['mae_periods'].median(), 2),
                      'mfe_periods': round(result_df['mfe_periods'].median(), 2),
                      'time_gap_periods': median_time_gap  # Added median time gap
                     },
                     'win': {
                       'duration': round(result_df[result_df['result'] == 'win']['duration'].median(), 2),
                       'gross_returns_pct': round(result_df[result_df['result'] == 'win']['gross_returns_pct'].median(), 2),
                       'net_returns_pct': round(result_df[result_df['result'] == 'win']['net_returns_pct'].median(), 2),
                       'mfe_pct': round(result_df[result_df['result'] == 'win']['mfe_pct'].median(), 2),
                       'mae_pct': round(result_df[result_df['result'] == 'win']['mae_pct'].median(), 2),
                       'mae_periods': round(result_df[result_df['result'] == 'win']['mae_periods'].median(), 2),
                       'mfe_periods': round(result_df[result_df['result'] == 'win']['mfe_periods'].median(), 2),
                       'time_gap_periods': median_time_gap  # Same median for all trades (not per win/loss)
                      },  
                     'loss': {
                       'duration': round(result_df[result_df['result'] == 'loss']['duration'].median(), 2),
                       'gross_returns_pct': round(result_df[result_df['result'] == 'loss']['gross_returns_pct'].median(), 2),
                       'net_returns_pct': round(result_df[result_df['result'] == 'loss']['net_returns_pct'].median(), 2),
                       'mfe_pct': round(result_df[result_df['result'] == 'loss']['mfe_pct'].median(), 2),
                       'mae_pct': round(result_df[result_df['result'] == 'loss']['mae_pct'].median(), 2),
                       'mae_periods': round(result_df[result_df['result'] == 'loss']['mae_periods'].median(), 2),
                       'mfe_periods': round(result_df[result_df['result'] == 'loss']['mfe_periods'].median(), 2),
                       'time_gap_periods': median_time_gap  # Same median for all trades (not per win/loss)
                      },  
    }

    # Optimized calculation of time gap statistics in periods
    if len(result_df) > 1:
        exit_positions = result_df['exit_date'].apply(df.index.get_loc)
        entry_positions = result_df['entry_date'].apply(df.index.get_loc)
        time_gaps = entry_positions[1:].values - exit_positions[:-1].values
        max_time_gap = time_gaps.max() if len(time_gaps) > 0 else 0
        mean_time_gap = round(time_gaps.mean(), 2) if len(time_gaps) > 0 else 0
        median_time_gap = round(pd.Series(time_gaps).median(), 2) if len(time_gaps) > 0 else 0
    else:
        max_time_gap = 0
        mean_time_gap = 0
        median_time_gap = 0

    kpi = {
            'winning_ratio_pct': {
                'gross': {
                    'total': winning_ratio_all,
                    'long': winning_ratio_long,
                    'short': winning_ratio_short,
                },
                'net': {
                    'total': winning_ratio_all_net,
                    'long': winning_ratio_long_net,
                    'short': winning_ratio_short_net,
                }
            },
            'num_trades_cycle': {
                'total': num_trades_cycle_total,
                'long': num_trades_cycle_long,
                'short': num_trades_cycle_short,
                'ratio_l': ratio_l,
                'ratio_s': ratio_s,
            },
            'holding_duration': {
                'total': holding_duration_total,
                'long': holding_duration_long,
                'short': holding_duration_short,
                'ratio_l': holding_ratio_l,
                'ratio_s': holding_ratio_s,
            },
            'metric_mean': metric_mean,
            'metric_median': metric_median,
            'max_time_gap_exit_to_entry_periods': max_time_gap  # Added max time gap
    }
    return {'result': result_df, 'kpi': kpi}

def add_zscore_column(df: pd.DataFrame, metric_col: str, lookback: int = None) -> pd.DataFrame:
    """
    Adds Z-score column to DataFrame using naming convention: {metric_col}_zscore_w{lookback}
    If lookback is None, calculates Z-score using expanding window (no look-ahead bias).

    Parameters:
        df: Input DataFrame with datetime index
        metric_col: Column name to calculate Z-scores for
        lookback: Number of periods for rolling calculation. If None, uses expanding window.

    Returns:
        DataFrame with new Z-score column
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")
    
    if lookback is not None and lookback < 2:
        raise ValueError("Lookback must be ≥ 2 for meaningful standard deviation")

    # Calculate mean and standard deviation
    if lookback is None:
        # Use expanding window to avoid look-ahead bias
        mean = df[metric_col].expanding().mean()
        std = df[metric_col].expanding().std(ddof=0)
    else:
        # Use rolling window
        mean = df[metric_col].rolling(lookback).mean()
        std = df[metric_col].rolling(lookback).std(ddof=0)
    
    # Create new column name
    if lookback is None:
        zscore_col = f"{metric_col}_zscore_expanding"
    else:
        zscore_col = f"{metric_col}_zscore_w{lookback}"
    
    # Calculate and store Z-scores
    df[zscore_col] = (df[metric_col] - mean) / std
    
    return df

def add_robust_zscore_column(df: pd.DataFrame, metric_col: str, lookback: int) -> pd.DataFrame:
    """
    Adds rolling Robust Z-score column to DataFrame using naming convention: {metric_col}_robust_zscore_w{lookback}
    
    Parameters:
        df: Input DataFrame with datetime index.
        metric_col: Column name to calculate Robust Z-scores for.
        lookback: Number of periods for rolling calculation.
    
    Returns:
        DataFrame with new Robust Z-score column.
    
    Raises:
        ValueError: If `metric_col` is not in the DataFrame or `lookback` is invalid.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")
    
    if lookback < 2:
        raise ValueError("Lookback must be ≥ 2 for meaningful MAD calculation")

    # Calculate rolling median and MAD
    median = df[metric_col].rolling(lookback).median()
    mad = (df[metric_col] - median).abs().rolling(lookback).median()
    
    # Create new column name
    robust_zscore_col = f"{metric_col}_robust_zscore_w{lookback}"
    
    # Calculate and store Robust Z-scores
    df[robust_zscore_col] = (df[metric_col] - median) / mad
    
    return df

def calculate_percentile(x):
    """Computes the rolling percentile rank of the last value in a window."""
    return (sum(x[:-1] < x[-1]) / (len(x) - 1)) * 100 if len(x) > 1 else None

def add_percentile_column(df: pd.DataFrame, metric_col: str, rolling_period: int = None, percentiles: int = 10) -> pd.DataFrame:
    """
    Adds either rolling percentile classification & rolling percentile rank columns OR static percentiles.

    Parameters:
        df: Input DataFrame with a datetime index.
        metric_col: Column name to calculate percentiles for.
        rolling_period: Number of periods for rolling calculation (None means compute static percentiles on the entire dataset).
        percentiles: Number of percentiles to divide the metric into (default: 10 for deciles).

    Returns:
        DataFrame with:
          - `{metric_col}_percentile_w{rolling_period}` (if rolling) - Rolling percentile classification
          - `{metric_col}_percentile_v_w{rolling_period}` (if rolling) - Rolling percentile rank
          - `{metric_col}_percentile` (if not rolling) - Static percentiles for the full dataset
          - `{metric_col}_percentile_v` (if not rolling) - Static percentile rank for the entire dataset

    Raises:
        ValueError: If `metric_col` is not in the DataFrame or `rolling_period` is invalid.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")

    df = df.copy()  # Avoid modifying the original DataFrame

    if rolling_period is None:
        # Compute static percentiles over the entire dataset
        df[f"{metric_col}_percentile"] = pd.qcut(df[metric_col], q=percentiles, labels=False, duplicates='drop') + 1

        # Compute static percentile rank using calculate_percentile
        # Define a function to apply per row, treating the full dataset as context
        def static_percentile_rank(value, series):
            if pd.isna(value):
                return np.nan  # Leave NaN as is
            # Mimic calculate_percentile: rank this value against all others
            valid_series = series.dropna()
            if len(valid_series) <= 1:
                return np.nan
            return (sum(valid_series < value) / (len(valid_series) - 1)) * 100

        # Apply to each value in the column
        df[f"{metric_col}_percentile_v"] = df[metric_col].apply(
            lambda x: static_percentile_rank(x, df[metric_col])
        )
    else:
        if rolling_period < 2:
            raise ValueError("Rolling period must be ≥ 2 for meaningful rolling percentiles")

        # Function to compute rolling percentile classification using `pd.qcut`
        def rolling_percentile_classification(series):
            if series.isna().sum() > 0:
                return None
            try:
                return pd.qcut(series, q=percentiles, labels=False, duplicates='drop')[-1] + 1
            except ValueError:
                return None

        # Apply rolling percentile classification
        df[f"{metric_col}_percentile_w{rolling_period}"] = (
            df[metric_col]
            .rolling(rolling_period, min_periods=rolling_period)
            .apply(rolling_percentile_classification, raw=False)
        )

        # Apply rolling percentile rank calculation
        df[f"{metric_col}_percentile_v_w{rolling_period}"] = (
            df[metric_col]
            .rolling(rolling_period, min_periods=rolling_period)
            .apply(calculate_percentile, raw=True)
        )

    return df

def analyze_predictive_power(
    df: pd.DataFrame, 
    returns_col: str, 
    factor_col: str, 
    percentiles: int = 10, 
    rolling_period: int = None,
    statistic="all",
    verbose=True,
    save_chart=False,  
    save_folder="charts",
    file_prefix="factor"
) -> pd.DataFrame:
    """
    Analyzes the predictive power of an indicator on returns by grouping the indicator
    into percentiles and calculating the mean/median of returns for each group.

    Parameters:
        df: Input DataFrame with datetime index.
        returns_col: Column name of the returns (r(t)).
        factor_col: Column name of the indicator (X(t) or X(t-1)).
        percentiles: Number of percentiles to divide the indicator into (default: 10 for deciles).
        rolling_period: Number of periods for rolling percentile calculation (None means no rolling).
        statistic: "all" (default) to plot both mean & median, or specify "mean"/"median".
        verbose: If True, generates visualizations.
        save_chart: If True, saves the chart instead of displaying it.
        save_folder: Directory where the chart should be saved.
        file_prefix: Prefix to add before the factor name in the saved filename.

    Returns:
        Tuple of (DataFrame with results, median marks, mean marks, percentile volatility).

    Raises:
        ValueError: If `factor_col` or `returns_col` is not in the DataFrame.
    """
    if factor_col not in df.columns or returns_col not in df.columns:
        raise ValueError(f"Columns '{factor_col}' or '{returns_col}' not found in DataFrame")

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[factor_col, returns_col])

    # Custom binning function inspired by calculate_percentile
    def assign_percentile_bins(series, n_bins=percentiles):
        if series.isna().any() or len(series) < 2:
            return pd.Series(np.nan, index=series.index)
        # Rank each value against others, then bin into percentiles
        ranks = series.rank(method='dense', na_option='keep')  # Dense ranking to minimize empty bins
        max_rank = ranks.max()
        if pd.isna(max_rank) or max_rank < 2:
            return pd.Series(np.nan, index=series.index)
        # Scale ranks to 0-(n_bins-1), then shift to 1-n_bins
        bins = np.floor((ranks - 1) / max_rank * n_bins).clip(0, n_bins - 1) + 1
        return bins

    if rolling_period:
        if rolling_period < 2:
            raise ValueError("Rolling period must be ≥ 2 for meaningful percentiles")
        
        # Rolling percentile function
        def rolling_percentile(series):
            if len(series) < rolling_period or series.isna().sum() > 0:
                return np.nan
            return assign_percentile_bins(series)[-1]

        df[f"{factor_col}_percentile"] = (
            df[factor_col]
            .rolling(rolling_period, min_periods=rolling_period)
            .apply(rolling_percentile, raw=False)
        )
    else:
        # Static case using custom binning
        df[f"{factor_col}_percentile"] = assign_percentile_bins(df[factor_col])

    df = df.dropna(subset=[f"{factor_col}_percentile"])
    df['percentile_diff'] = df[f"{factor_col}_percentile"].diff()
    percentile_vol = df['percentile_diff'].std()

    result = df.groupby(f"{factor_col}_percentile")[returns_col].agg(['mean', 'median']).reset_index()
    result.columns = ['Percentile', 'Mean_Returns', 'Median_Returns']
    result['Percentile'] = result['Percentile'].astype(int)  # Ensure integer percentiles

    # Scoring
    result["Mean_Score"] = result["Mean_Returns"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    result["Median_Score"] = result["Median_Returns"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    sum_mean_score_1_3 = result[result["Percentile"].between(1, 3)]["Mean_Score"].sum()
    sum_mean_score_8_10 = result[result["Percentile"].between(8, 10)]["Mean_Score"].sum()
    feature_mean_marks = [sum_mean_score_1_3, sum_mean_score_8_10]

    sum_median_score_1_3 = result[result["Percentile"].between(1, 3)]["Median_Score"].sum()
    sum_median_score_8_10 = result[result["Percentile"].between(8, 10)]["Median_Score"].sum()
    feature_median_marks = [sum_median_score_1_3, sum_median_score_8_10]

    if verbose:
        if statistic == 'all':
            visualize_predictive_power(result, percentiles, 'median', factor_col, save_chart, save_folder, file_prefix)
            visualize_predictive_power(result, percentiles, 'mean', factor_col, save_chart, save_folder, file_prefix)
        else:
            visualize_predictive_power(result, percentiles, statistic, factor_col, save_chart, save_folder, file_prefix)

    return result, feature_median_marks, feature_mean_marks, percentile_vol 

def visualize_predictive_power(
    df_returns_factor_p: pd.DataFrame, 
    percentiles: int = 10, 
    statistic: str = "median",
    factor_col: str = "",  
    save_chart: bool = False, 
    save_folder: str = "charts",
    file_prefix: str = "factor"
) -> None:
    """
    Visualizes the predictive power of a factor on returns.

    Parameters:
        df_returns_factor_p: DataFrame from `analyze_predictive_power`.
        percentiles: Number of percentiles (default: 10).
        statistic: "median" or "mean".
        factor_col: Name of the factor for the title.
        save_chart: If True, saves the chart.
        save_folder: Directory for saving.
        file_prefix: Prefix for filenames.

    Raises:
        ValueError: If `statistic` is invalid.
    """
    if statistic not in ["median", "mean"]:
        raise ValueError("Invalid statistic. Choose 'median' or 'mean'.")

    stat_col = "Median_Returns" if statistic == "median" else "Mean_Returns"

    norm = mcolors.Normalize(vmin=df_returns_factor_p[stat_col].min(), vmax=df_returns_factor_p[stat_col].max())
    colors = plt.cm.Oranges(norm(df_returns_factor_p[stat_col]))  

    fig, ax = plt.subplots(figsize=(10, 6))  
    bars = ax.bar(df_returns_factor_p.index, df_returns_factor_p[stat_col], color=colors, edgecolor='black')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=norm)
    sm.set_array([])  
    plt.colorbar(sm, ax=ax, label=f"{statistic.capitalize()} Returns Intensity", orientation='vertical', pad=0.1)

    ax.set_xlabel('Quantiles of Factor')
    ax.set_ylabel(f"{statistic.capitalize()} Returns (t+1)")
    ax.set_title(f"{statistic.capitalize()} Return t+1 by Quantiles of Factor ({factor_col})")

    ax.set_xticks(range(len(df_returns_factor_p['Percentile'])))
    ax.set_xticklabels([f'q{i+1}' for i in range(len(df_returns_factor_p['Percentile']))])  

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    if save_chart:
        os.makedirs(save_folder, exist_ok=True)
        filename = f"{save_folder}/{file_prefix}_{statistic}_returns_by_{factor_col}.png".replace(" ", "_")
        plt.pause(0.1)  # Ensure rendering
        fig.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Chart saved to: {filename}")
    else:
        plt.show()
def add_metric_change(
    df: pd.DataFrame,
    metric_col: str,
    calc_type: str = 'pct_change',
    periods: int = 1,
    suffix: str = None
    ) -> pd.DataFrame:
    """
    Adds calculated change metrics to DataFrame with automatic column naming
    
    Parameters:
        df: Input DataFrame
        metric_col: Column name to calculate from
        calc_type: Type of calculation:
            - 'pct_change': Percentage change
            - 'log_return': Logarithmic return
            - 'diff': Absolute difference
        periods: Number of periods for calculation
        suffix: Custom suffix for column name (automatic if None)
    
    Returns:
        DataFrame with new calculated column
    """
    valid_calcs = ['pct_change', 'log_return', 'diff']
    if calc_type not in valid_calcs:
        raise ValueError(f"Invalid calc_type. Use one of {valid_calcs}")
    
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")

    # Create column suffix mapping
    suffix_map = {
        'pct_change': '_pct_change',
        'log_return': '_log_return',
        'diff': '_diff'
    }
    
    # Calculate transformations
    if calc_type == 'pct_change':
        result = df[metric_col].pct_change(periods=periods)
    elif calc_type == 'log_return':
        result = np.log(df[metric_col] / df[metric_col].shift(periods))
    elif calc_type == 'diff':
        result = df[metric_col].diff(periods=periods)
    
    # Create column name
    suffix = suffix_map[calc_type] if suffix is None else suffix
    new_col = f"{metric_col}{suffix}"
    if periods != 1:
        new_col += f"_{periods}"
    
    df[new_col] = result
    return df

def add_lagged_columns(
    df: pd.DataFrame,
    metric_col: str,
    lags: list = [1, 2, 3],
    fill_na: bool = False
) -> pd.DataFrame:
    """
    Creates lagged versions of a metric column with automatic naming
    
    Parameters:
        df: Input DataFrame
        metric_col: Column name to create lags for
        lags: List of lag periods to create (default [1, 2, 3])
        fill_na: Whether to forward-fill NaN values (default False)
    
    Returns:
        DataFrame with new lagged columns
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")
    
    if not all(isinstance(x, int) and x > 0 for x in lags):
        raise ValueError("All lags must be positive integers")

    # Create lagged columns
    for lag in sorted(lags):
        lag_col = f"{metric_col}_lag{lag}"
        df[lag_col] = df[metric_col].shift(lag)
        
        if fill_na:
            df[lag_col] = df[lag_col].fillna(method='ffill')

    return df

def say_finished(phrase="主席！大功告成"):
    """
    Use macOS built-in TTS (Text-to-Speech) to say '主席！大功告成' in Cantonese.
    """
    try:
        # Specify the Cantonese-compatible voice, e.g., 'Sin-Ji'       
        voice = "Sinji"  # Cantonese voice available on macOS
        subprocess.run(["say", "-v", voice, phrase], check=True)
        print("Finished speaking the completion message in Cantonese.")
    except Exception as e:
        print(f"Error using text-to-speech: {e}")



def N_annual(freq):
    """
    Convert frequency to annual periods.

    :param freq: str - frequency of backtest
    :return: int - number of periods per annum for chosen frequency
    """
    freq_map = {
        '1m': 365 * 60 * 24,
        '5m': 12 * 24 * 365,
        '15m': 4 * 24 * 365,
        '30m': 48 * 365,
        '60m': 24 * 365,
        '1h': 24 * 365,
        '120m': 12 * 365,
        '2h': 12 * 365,
        '240m': 6 * 365,
        '4h': 6 * 365,
        '360m': 4 * 365,
        '6h': 4 * 365,
        '480m': 3 * 365,
        '8h': 3 * 365,
        '720m': 2 * 365,
        '12h': 2 * 365,
        '1d': 365
    }

    if freq not in freq_map:
        raise ValueError('Invalid frequency')
    return freq_map[freq]

def adjust_backtest_start(train_start, outofsample_start, interval, roll_back):
    """
    Adjust train_start and outofsample_start timestamps for backtesting based on the interval and roll_back.

    Parameters:
        train_start (str): Start of the training period (e.g., "2023-10-01 00:00:00").
        outofsample_start (str): Start of the out-of-sample period (e.g., "2023-10-15 00:00:00").
        interval (str): Timeframe interval (e.g., "6h", "1d").
        roll_back (int): Number of intervals to roll back.

    Returns:
        Tuple[str, str]: Adjusted train_start and outofsample_start timestamps.
    """
    # Determine the time format and timedelta unit
    if 'h' in interval:
        time_format = "%Y-%m-%d %H:%M:%S"
        interval_value = int(interval.replace('h', ''))  # Extract the number from '6h', '3h', etc.
        delta_unit = timedelta(hours=interval_value * roll_back)
    elif 'd' in interval:
        time_format = "%Y-%m-%d"
        delta_unit = timedelta(days=roll_back)  # No need to multiply for daily intervals
    else:
        raise ValueError("Invalid interval format. Supported: 'h', 'd'.")

    # Adjust train_start and outofsample_start
    train_start_adjusted = (datetime.datetime.strptime(train_start, time_format) - delta_unit).strftime(time_format)
    outofsample_start_adjusted = (datetime.datetime.strptime(outofsample_start, time_format) - delta_unit).strftime(time_format)

    return train_start_adjusted, outofsample_start_adjusted

def index_is_unique_continuous(df, interval):
    """
    Check if the index of a DataFrame is unique and continuous based on the given interval.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        interval (str): The interval of the index (e.g., '1d', '12h', '6h', '1h').
        symbol (str, optional): Symbol identifier for logging (e.g., 'BTCUSDT').
        nickname (str, optional): Nickname for logging (e.g., 'spot').

    Returns:
        tuple[bool, bool]: A tuple indicating if the index is unique and continuous.
    """
    # Check if the index is unique
    is_unique = df.index.is_unique

    # Calculate the difference between each date in the index and its predecessor
    differences = df.index.to_series().diff().dropna()
    
    # Determine the expected time delta based on the interval
    interval_map = {
        '1d': pd.Timedelta(days=1),
        '12h': pd.Timedelta(hours=12),
        '8h': pd.Timedelta(hours=8),
        '6h': pd.Timedelta(hours=6),
        '4h': pd.Timedelta(hours=4),
        '2h': pd.Timedelta(hours=2),
        '1h': pd.Timedelta(hours=1),
        '30m': pd.Timedelta(minutes=30),
        '15m': pd.Timedelta(minutes=15),
        '5m': pd.Timedelta(minutes=5),
        '3m': pd.Timedelta(minutes=3),
        '1m': pd.Timedelta(minutes=1),
    }
    expected_delta = interval_map.get(interval)

    if expected_delta is None:
        raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {list(interval_map.keys())}")

    # Check for continuity
    is_continuous = all(differences == expected_delta)

    # Identify non-continuous points
    non_continuous_points = differences[differences != expected_delta].index

    # Print the results
    print(f"index is unique: {is_unique}")
    print(f"index is continuous without gaps: {is_continuous}")
    if not is_continuous:
        print("Gaps found at the following indices:")
        gap_df_index = df.loc[non_continuous_points].index
        print(gap_df_index)
    print()
    return is_unique, is_continuous

def limited_ffill(series, n, overwrite=False):
    """
    Forward fills a pandas Series but only for a limited number of subsequent rows.
    Allows optional overwriting of new valid values.

    Parameters:
    series (pd.Series): The pandas Series with missing values to be forward filled.
    n (int): The number of rows to forward fill after each valid value.
    overwrite (bool): If True, overwrites new valid values within N rows. 
                      If False, new valid values remain unchanged.

    Returns:
    pd.Series: The modified Series with limited forward filling applied.
    """
    last_valid = None
    count = 0
    filled_series = series.copy()  # Create a copy to avoid modifying the original in-place

    for i in range(len(series)):
        if not pd.isna(series[i]):  # If value is valid
            if overwrite and last_valid is not None and count < n:
                # Overwrite new valid values within N if enabled
                filled_series[i] = last_valid  
                count += 1
            else:
                last_valid = series[i]  # Update last valid value
                count = 0  # Reset count
        elif last_valid is not None and count < n:  # Fill only up to N rows
            filled_series[i] = last_valid
            count += 1
    
    return filled_series

def count_consecutive_zeros(arr_or_series):
    # Convert to numpy array if it's a Series or list
    is_zero = np.array(arr_or_series) == 0
    diff = np.diff(np.concatenate(([0], is_zero, [0])))
    lengths = np.where(diff == -1)[0] - np.where(diff == 1)[0]
    # Deduplicate with set and sort in descending order
    return sorted(set(lengths.tolist()), reverse=True)
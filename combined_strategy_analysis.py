import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob


from trader_utils import *

def get_xlsx_files(folder_path):
    """
    Retrieve all .xlsx files in the specified folder and sort them in ascending order.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder '{folder_path}' does not exist or is not a directory.")
    xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    xlsx_files.sort()
    return xlsx_files

def left_join_earliest_test_files(test_files):
    """
    Perform a left join on test DataFrames with the earliest index as the base.

    Parameters:
    -----------
    test_files : list
        List of file paths to .xlsx files.
    col : str, default 'strategy_simple_returns_net'
        Base column name to extract and join (both net and levered versions will be included).

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with the earliest index as the base, including both net and levered columns.
    """
    if not test_files:
        raise ValueError("No test files provided.")

    # Initialize list to store DataFrames
    dfs = []

    # Read each file and store DataFrame with its start index
    for file in test_files:
        strategy_name = file.split('_')[-3:]  # Extract strategy name from filename
        strategy_name = "_".join(strategy_name)
        strategy_name = strategy_name.replace('.xlsx', '')
        # Read both net and levered net columns
        df = pd.read_excel(file).loc[:, ['Date', 'strategy_simple_returns_net', 'strategy_simple_levered_net']]
        df.set_index('Date', inplace=True)
        # Rename columns with strategy name
        df = df.rename(columns={
            'strategy_simple_returns_net': f"strategy_simple_returns_net_{strategy_name}",
            'strategy_simple_levered_net': f"strategy_simple_levered_net_{strategy_name}"
        })
        if not df.empty:
            dfs.append((df, df.index.min(), file))

    # Sort by earliest index to find the base DataFrame
    dfs.sort(key=lambda x: x[1])  # Sort by minimum index
    base_df, _, base_file = dfs[0]  # Use the DataFrame with the earliest index
    print(f"Using base DataFrame from {base_file} with earliest index {base_df.index.min()}")

    # Left join remaining DataFrames onto the base
    result_df = base_df.copy()
    for df, _, file in dfs[1:]:
        result_df = result_df.join(df, how='left', rsuffix=f'_from_{os.path.basename(file).replace(".xlsx", "")}')
    
    # Fill NaN values with 0
    result_df = result_df.fillna(0)
    
    return result_df.fillna(0)

def analyze_portfolio_performance(test_files, levered=True, initial_capital=10_000, folder_path="/path/to/your/folder"):
    """
    Analyze portfolio performance from test files, including equity curves, Sharpe ratios,
    maximum drawdowns, and correlation matrix for multiple strategies.

    Parameters:
    -----------
    test_files : list
        List of file paths to test .xlsx files.
    levered : bool, default True
        If True, use 'strategy_simple_levered_net'; if False, use 'strategy_simple_returns_net'.
    initial_capital : float, default 10_000
        Initial capital per strategy (total combined starts at len(strategies) * initial_capital).
    folder_path : str, default "/path/to/your/folder"
        Path to the folder containing the files (for context).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics and values.
    """
    # Step 0: Get merged DataFrame
    if levered:
        df = left_join_earliest_test_files(test_files)
    else:
        df = left_join_earliest_test_files(test_files)
    
    # Step 1: Identify strategy columns dynamically
    net_cols = [col for col in df.columns if col.startswith('strategy_simple_returns_net_')]
    levered_cols = [col.replace('returns_net', 'levered_net') for col in net_cols]
    
    if levered:
        use_cols = levered_cols
    else:
        use_cols = net_cols
    if not net_cols:
        raise ValueError("No strategy return columns found in the DataFrame.")
    
    # Step 2: Calculate portfolio values based on levered flag
    value_cols = {}
    for i, col in enumerate(use_cols):
        base_col = col.replace('strategy_simple_returns_net_', '')
        
        value_cols[f'value_{base_col}'] = (1 + df[col]).cumprod() * (initial_capital / len(use_cols))
    df = df.assign(**value_cols)

    # Combined portfolio value (equal weighting)
    #df['combined_value'] = sum(df[value_cols[f'value_{col.split("_")[-1]}']] for col in use_cols)
    df['combined_value'] = df[list(value_cols.keys())].sum(axis=1, min_count=1)

    # Step 3: Calculate daily returns of the combined portfolio
    df['combined_returns'] = df['combined_value'].pct_change().fillna(0)
    
    
    # adjust for 10000 initial capital for each strategy
    for i, col in enumerate(use_cols):
        base_col = col.replace('strategy_simple_returns_net_', '')
        
        value_cols[f'value_{base_col}'] = (1 + df[col]).cumprod() * (initial_capital)
    df = df.assign(**value_cols)
    

    # Step 4: Calculate Sharpe ratios (annualized, assuming 365 trading days)
    risk_free_rate = 0  # Adjust if needed
    sharpe_ratios = {}
    for col in net_cols:
        base_col = col.replace('strategy_simple_returns_net_', '')
        use_col = levered_cols[net_cols.index(col)] if levered else col
        sharpe_ratios[base_col] = (df[use_col].mean() - risk_free_rate) / df[use_col].std() * np.sqrt(365)

    sharpe_combined = (df['combined_returns'].mean() - risk_free_rate) / df['combined_returns'].std() * np.sqrt(365)

    # Step 5: Define Maximum Drawdown (MDD) function
    def calculate_mdd(portfolio_values):
        running_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - running_max) / running_max
        mdd = drawdowns.min()
        return mdd

    # Calculate MDD for each strategy and combined portfolio
    mdd_values = {}
    for col in value_cols.keys():
        base_col = col.replace('value_', '')
        mdd_values[base_col] = calculate_mdd(df[col])
    mdd_combined = calculate_mdd(df['combined_value'])

    # Step 6: Calculate correlation matrix
    use_cols = [levered_cols[i] if levered else col for i, col in enumerate(use_cols)]
    correlation_matrix = df[use_cols + ['combined_returns']].corr()

    # Step 7: Plot equity curves
    plt.figure(figsize=(10, 6))
    for col in value_cols.keys():
        base_col = col.replace('value_', '')
        plt.plot(df.index, df[col], label=f'Strategy {base_col}', alpha=0.7)
    plt.plot(df.index, df['combined_value'], label='Combined Portfolio', color='green', linewidth=2)
    plt.title('Equity Curves (Starting at ${} per Strategy)'.format(initial_capital))
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    y_max = max(df[value_cols.keys()].max().max(), df['combined_value'].max())
    y_ticks = np.arange(initial_capital, y_max + initial_capital, initial_capital)
    plt.yticks(y_ticks)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Results
    print("Sharpe Ratios:")
    for strategy, sharpe in sharpe_ratios.items():
        print(f"Sharpe Ratio - Strategy {strategy}: {round(sharpe, 4)}")
    print(f"Sharpe Ratio - Combined Portfolio: {round(sharpe_combined, 4)}")
    print("\nMaximum Drawdowns:")
    for strategy, mdd in mdd_values.items():
        print(f"Maximum Drawdown - Strategy {strategy}: {round(mdd, 4)}")
    print(f"Maximum Drawdown - Combined Portfolio: {round(mdd_combined, 4)}")
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    print("-"*100, '\n')
    #print("\nFinal DataFrame (Head):")
    #print(df.head())
    
    return df, correlation_matrix

if __name__ == "__main__":
    # Specify the folder path (replace with your actual path)
    folder_path = "." #current directory
    
    # Get the sorted list of .xlsx files
    files = get_xlsx_files(folder_path)
    
    train_files = []
    test_files = []
    combined_files = []
    # Print the results
    for file in files:
        if 'train' in file:
            train_files.append(file)
        elif 'test' in file:
            test_files.append(file)
        else:
            combined_files.append(file)
        # print(file)
    print("analysis: train set")
    df_train, correlation_matrix_train = analyze_portfolio_performance(train_files, levered=False, initial_capital=10_000)  
    #df_train_levered, correlation_matrix_train_levered = analyze_portfolio_performance(train_files, levered=True, initial_capital=10_000)
    print("analysis: test set")
    df_test, correlation_matrix_test = analyze_portfolio_performance(test_files, levered=False, initial_capital=10_000)
    #df_test_levered, correlation_matrix_test_levered = analyze_portfolio_performance(test_files, levered=True, initial_capital=10_000)
    print("analysis: combined set")
    df_combined, correlation_matrix_combined = analyze_portfolio_performance(combined_files, levered=False, initial_capital=10_000)
    #df_combined_levered, correlation_matrix_combined_levered = analyze_portfolio_performance(combined_files, levered=True, initial_capital=10_000)
    
    # print("time gap: train set")
    # for c in df_train:
    #     if 'returns' in c:
    #         print(f'{c}: {count_consecutive_zeros(df_train[c])}')   
  
    # print("time gap: test set")
    # for c in df_test:
    #     if 'returns' in c:
    #         print(f'{c}: {count_consecutive_zeros(df_test[c])}')
    
    # print("time gap: combined set")
    # for c in df_combined:
    #     if 'returns' in c:
    #         print(f'{c}: {count_consecutive_zeros(df_combined[c])}')
            
            

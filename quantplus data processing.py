import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm


def load_stock_returns(file_path):
    """
    Load stock returns data and preprocess it.

    Parameters:
        file_path (str): Path to the CSV file containing stock returns data.

    Returns:
        pd.DataFrame: Preprocessed stock returns data.
    """
    stock_returns = pd.read_csv(file_path)
    stock_returns['RET'] = stock_returns['RET'].replace(['B', 'C'], np.nan)  # Replace invalid values with NaN
    stock_returns['RET'] = stock_returns['RET'].astype(float)  # Convert return column to float
    return stock_returns


def load_stock_characteristics(file_path):
    """
    Load stock characteristics data.

    Parameters:
        file_path (str): Path to the CSV file containing stock characteristics data.

    Returns:
        pd.DataFrame: Stock characteristics data.
    """
    return pd.read_csv(file_path)


def merge_characteristics_with_returns(stock_characteristics, stock_returns):
    """
    Merge stock characteristics with returns data.

    Parameters:
        stock_characteristics (pd.DataFrame): DataFrame containing stock characteristics.
        stock_returns (pd.DataFrame): DataFrame containing stock returns.

    Returns:
        pd.DataFrame: Merged data.
    """
    merged_data = pd.merge(stock_characteristics, stock_returns, \
                           left_on=['permno', 'DATE'], right_on=['PERMNO', 'date'], how='left')
    merged_data.dropna(subset=['RET'], inplace=True)  # Remove rows where returns are NaN
    merged_data = merged_data.drop(columns=['PERMNO', 'date'])
    return merged_data


def load_macroeconomic_predictors(file_path):
    """
    Load and process macroeconomic predictors data.

    Parameters:
        file_path (str): Path to the CSV file containing macroeconomic predictors.

    Returns:
        pd.DataFrame: Processed macroeconomic predictors.
    """
    macroeconomic_predictors = pd.read_csv(file_path, thousands=",")
    macroeconomic_predictors['dp'] = np.log(macroeconomic_predictors['D12']) - np.log(macroeconomic_predictors['Index'])
    macroeconomic_predictors['EP'] = np.log(macroeconomic_predictors['E12']) - np.log(macroeconomic_predictors['Index'])
    macroeconomic_predictors['tms'] = macroeconomic_predictors['lty'] - macroeconomic_predictors['tbl']
    macroeconomic_predictors['dfy'] = macroeconomic_predictors['BAA'] - macroeconomic_predictors['AAA']
    macroeconomic_predictors = macroeconomic_predictors[['yyyymm', 'dp', 'EP', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']]
    return macroeconomic_predictors


def merge_with_macroeconomic_data(characteristics_with_return, macroeconomic_predictors):
    """
    Merge stock characteristics and returns data with macroeconomic predictors.

    Parameters:
        characteristics_with_return (pd.DataFrame): DataFrame containing stock characteristics and returns.
        macroeconomic_predictors (pd.DataFrame): DataFrame containing macroeconomic predictors.

    Returns:
        pd.DataFrame: Merged data with macroeconomic predictors.
    """
    characteristics_with_return['DATE'] = characteristics_with_return['DATE'].astype(str).str[:6].astype(int)
    merged_data = pd.merge(characteristics_with_return, macroeconomic_predictors, left_on=['DATE'], right_on=['yyyymm'], how='left')
    merged_data['real_RET'] = merged_data['RET'] - merged_data['tbl']  # Calculate excess return
    macroeconomic_columns = ['dp', 'EP', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
    merged_data.drop(columns=['yyyymm', 'sic2', 'RET'] + macroeconomic_columns, inplace=True)
    return merged_data


def clean_and_fill_na(df):
    """
    Filter columns with more than 50% missing values and forward fill NaNs.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaNs filled.
    """
    # Filter out columns with more than 50% NaN values
    nan_summary = df.isna().sum()
    nan_columns = nan_summary[nan_summary > 0]
    nan_info = pd.DataFrame({
        'Column': nan_columns.index,
        'NaN Count': nan_columns.values,
        'NaN Percentage': (nan_columns.values / len(df)) * 100
    })
    columns_to_keep = nan_info[nan_info['NaN Percentage'] <= 50]['Column'].tolist()
    df_filtered = df[columns_to_keep + ['permno', 'DATE', 'real_RET']]
    
    # Sort by 'permno' and 'DATE' and fill NaNs
    df_filtered = df_filtered.sort_values(by=['permno', 'DATE'])
    tqdm.pandas(desc="Processing Groups")
    df_filled = df_filtered.groupby('permno', group_keys=False).progress_apply(lambda group: group.ffill(limit=12))
    return df_filled.reset_index(drop=True)


def normalize_features(df, feature_set):
    """
    Cross-sectionally rank and normalize features.

    Parameters:
        df (pd.DataFrame): DataFrame containing features to be normalized.
        feature_set (list): List of features to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized features.
    """
    def normal(x):
        x = rankdata(x, axis=0, nan_policy='omit')
        x = ((x - 1) / (np.nanmax(x, axis=0) - 1) - 0.5) * 2
        return x
    
    df[feature_set] = df.groupby('DATE')[feature_set].transform(lambda x: normal(x.values))
    return df


def fill_remaining_na(df):
    """
    Fill remaining NaNs with median or zero.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled.
    """
    permno_col = df['permno']
    df = df.groupby('permno').transform(lambda x: x.fillna(x.median()))
    df['permno'] = permno_col
    df = df.fillna(0)  # Fill remaining NaNs with 0
    return df


def save_by_year(df, base_dir):
    """
    Save the DataFrame by year into feather files.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        base_dir (str): Directory to save the yearly feather files.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m')
    for year, group in tqdm(df.groupby(df['DATE'].dt.year), desc="Processing Years"):
        label_file_path = os.path.join(base_dir, f"{year}_real_RET.feather")
        group[['real_RET']].reset_index(drop=True).to_feather(label_file_path)
        print(f"Saved label file: {label_file_path}")
        
        data_file_path = os.path.join(base_dir, f"{year}_df_data.feather")
        group.drop(columns=['real_RET']).reset_index(drop=True).to_feather(data_file_path)
        print(f"Saved data file: {data_file_path}")


def main():
    """
    Main function to execute the data processing workflow.
    """
    # File paths (replace with your actual paths)
    stock_returns_path = 'path/to/stock_returns.csv'
    stock_characteristics_path = 'path/to/stock_characteristics.csv'
    macroeconomic_predictors_path = 'path/to/macroeconomic_predictors.csv'
    output_dir = 'output/directory'

    # Load data
    stock_returns = load_stock_returns(stock_returns_path)
    stock_characteristics = load_stock_characteristics(stock_characteristics_path)
    macroeconomic_predictors = load_macroeconomic_predictors(macroeconomic_predictors_path)

    # Merge data
    merged_data = merge_characteristics_with_returns(stock_characteristics, stock_returns)
    merged_data = merge_with_macroeconomic_data(merged_data, macroeconomic_predictors)

    # Clean and normalize data
    cleaned_data = clean_and_fill_na(merged_data)
    feature_set = cleaned_data.columns[:-3]
    normalized_data = normalize_features(cleaned_data, feature_set)
    final_data = fill_remaining_na(normalized_data)

    # Save processed data by year
    save_by_year(final_data, output_dir)


if __name__ == "__main__":
    main()
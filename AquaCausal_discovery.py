import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from itertools import combinations
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import os
import datetime
import shutil
import logging
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LassoCV
import warnings
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
import multiprocessing
import pickle
import hashlib
from functools import wraps

warnings.filterwarnings('ignore')

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"ATTCN_PCMCI_L1_results_{current_time}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"All output files will be saved to: {output_folder}")

cache_dir = os.path.join(output_folder, "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300


class Config:
    file_name = '干扰后数据集.csv'
    target_col = 'TN_eff'
    seq_length = 10
    test_size = 0.2
    random_state = 42
    tau_min = 0
    tau_max = 2
    alpha_level = 0.01
    max_lags = tau_max
    alpha_lasso = 0.1
    num_channels = [64, 128, 64]
    kernel_size = 3
    num_heads = 4
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    n_iterations = 1000
    node_size = 0.3
    link_width = 1
    significant_threshold = 0.05
    whitelist_features = []
    sink_nodes = ['Flow_in', 'TN_in', 'NH4_in', 'COD_in', 'TP_in', 'SS_in', 'DCS', 'DPRA', 'IR',
                  'ER', 'Was', 'DO_1', 'DO_2', 'DO_3']
    output_dir = output_folder
    outlier_method = 'iqr'
    outlier_threshold = 3
    outlier_handling = 'clip'
    missing_value_method = 'interpolate'
    missing_value_threshold = 0.1
    pfi_percentile = 90
    n_jobs = -1
    use_cache = True
    cache_dir = cache_dir


def cache_result(cache_key_prefix):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not Config.use_cache:
                return func(*args, **kwargs)

            cache_key = f"{cache_key_prefix}_{str(args)}_{str(kwargs)}"
            cache_file = os.path.join(Config.cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    write_log(f"Loaded {cache_key_prefix} from cache", log_file)
                    return result
                except:
                    pass

            result = func(*args, **kwargs)

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass

            return result

        return wrapper

    return decorator


def write_log(message, log_file):
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)


log_file = os.path.join(output_folder, "analysis_log.txt")
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"ATTCN-PCMCI-L1 Analysis Log - Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("-" * 80 + "\n")


def get_output_path(filename):
    return os.path.join(Config.output_dir, filename)


def detect_and_handle_outliers(data, method='iqr', threshold=3, handle_method='clip'):
    cleaned_data = data.copy()
    outlier_info = {}

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers = z_scores > threshold
            outliers = pd.Series(outliers, index=data[col].dropna().index).reindex(data.index, fill_value=False)

        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data[col].values.reshape(-1, 1)) == -1

        outlier_info[col] = {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(data)) * 100,
            'indices': data.index[outliers].tolist()
        }

        if handle_method == 'remove':
            cleaned_data = cleaned_data[~outliers]
        elif handle_method == 'clip':
            if method in ['iqr', 'zscore']:
                cleaned_data.loc[outliers, col] = np.clip(
                    data.loc[outliers, col],
                    lower_bound if 'lower_bound' in locals() else data[col].min(),
                    upper_bound if 'upper_bound' in locals() else data[col].max()
                )
        elif handle_method == 'interpolate':
            cleaned_data.loc[outliers, col] = np.nan
            cleaned_data[col] = cleaned_data[col].interpolate(method='linear')

    return cleaned_data, outlier_info


def handle_missing_values(data, method='interpolate', threshold=0.1):
    cleaned_data = data.copy()
    missing_info = {}

    missing_ratios = data.isnull().sum() / len(data)

    cols_to_drop = missing_ratios[missing_ratios > threshold].index
    if len(cols_to_drop) > 0:
        cleaned_data = cleaned_data.drop(columns=cols_to_drop)
        write_log(f"Dropped columns with >={threshold * 100}% missing values: {list(cols_to_drop)}", log_file)

    for col in cleaned_data.columns:
        missing_count = cleaned_data[col].isnull().sum()
        if missing_count > 0:
            missing_info[col] = {
                'count': missing_count,
                'percentage': (missing_count / len(cleaned_data)) * 100
            }

            if method == 'interpolate':
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill').fillna(method='bfill')
            elif method == 'mean':
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            elif method == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])

    return cleaned_data, missing_info


def check_stationarity(data, significance_level=0.05):
    stationarity_results = {}
    non_stationary_cols = []

    for col in data.select_dtypes(include=[np.number]).columns:
        try:
            adf_result = adfuller(data[col].dropna())

            stationarity_results[col] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < significance_level
            }

            if adf_result[1] >= significance_level:
                non_stationary_cols.append(col)
        except:
            write_log(f"Warning: Could not perform ADF test for {col}", log_file)

    return stationarity_results, non_stationary_cols


def generate_data_quality_report(data, missing_info, outlier_info, stationarity_results, output_dir):
    report_path = os.path.join(output_dir, "data_quality_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==== DATA QUALITY REPORT ====\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. DATA OVERVIEW\n")
        f.write("----------------\n")
        f.write(f"Shape: {data.shape}\n")
        f.write(f"Memory Usage: {data.memory_usage().sum() / 1024 ** 2:.2f} MB\n\n")

        f.write("2. MISSING VALUES\n")
        f.write("-----------------\n")
        for col, info in missing_info.items():
            f.write(f"{col}: {info['count']} ({info['percentage']:.2f}%)\n")
        f.write("\n")

        f.write("3. OUTLIERS\n")
        f.write("-----------\n")
        for col, info in outlier_info.items():
            f.write(f"{col}: {info['count']} ({info['percentage']:.2f}%)\n")
        f.write("\n")

        f.write("4. STATIONARITY TEST\n")
        f.write("--------------------\n")
        for col, res in stationarity_results.items():
            status = "Stationary" if res['is_stationary'] else "Non-stationary"
            f.write(f"{col}: {status} (p-value: {res['p_value']:.4f})\n")

    write_log(f"Data quality report saved to: {report_path}", log_file)


def load_and_normalize_data(config):
    write_log("Loading and normalizing data with quality control...", log_file)

    try:
        data = pd.read_csv(config.file_name)
        write_log(f"Data loaded successfully with shape: {data.shape}", log_file)

        if config.target_col not in data.columns:
            write_log(f"Error: Target column '{config.target_col}' not found in dataset.", log_file)
            raise ValueError(f"Target column '{config.target_col}' not found in dataset.")

        if 'Time' in data.columns:
            time_data = data['Time'].copy()
            data = data.drop('Time', axis=1)
        else:
            time_data = pd.Series(range(len(data)), name='Time')

        data, missing_info = handle_missing_values(
            data,
            method=config.missing_value_method,
            threshold=config.missing_value_threshold
        )
        write_log(f"Missing values handled. Remaining shape: {data.shape}", log_file)

        data, outlier_info = detect_and_handle_outliers(
            data,
            method=config.outlier_method,
            threshold=config.outlier_threshold,
            handle_method=config.outlier_handling
        )
        write_log(f"Outliers handled. Total outliers found: {sum(info['count'] for info in outlier_info.values())}",
                  log_file)

        stationarity_results, non_stationary_cols = check_stationarity(data)
        if non_stationary_cols:
            write_log(f"Non-stationary columns detected: {non_stationary_cols}", log_file)

        generate_data_quality_report(data, missing_info, outlier_info, stationarity_results, config.output_dir)

        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns
        )

        write_log(f"Data normalized. Shape: {normalized_data.shape}", log_file)

        preprocessed_data_path = get_output_path("preprocessed_data.csv")
        data.to_csv(preprocessed_data_path, index=False)

        normalized_data_path = get_output_path("normalized_data.csv")
        normalized_data.to_csv(normalized_data_path, index=False)

        write_log(f"Preprocessed data saved to: {preprocessed_data_path}", log_file)
        write_log(f"Normalized data saved to: {normalized_data_path}", log_file)

        return data, normalized_data, time_data, scaler

    except Exception as e:
        write_log(f"Error in data loading and normalization: {str(e)}", log_file)
        raise


def confirm_target_parameter(data, normalized_data, config):
    write_log(f"Confirming target parameter: {config.target_col}", log_file)

    if config.target_col not in data.columns:
        write_log(f"Error: Target column '{config.target_col}' not found in dataset.", log_file)
        raise ValueError(f"Target column '{config.target_col}' not found in dataset.")

    correlation_with_target = data.corr()[config.target_col].sort_values(ascending=False)

    write_log(f"Top 5 correlated features with {config.target_col}:", log_file)
    for i, (feature, corr) in enumerate(correlation_with_target.iloc[1:6].items()):
        write_log(f"  {i + 1}. {feature}: {corr:.4f}", log_file)

    target = normalized_data[config.target_col]
    features = normalized_data.drop(config.target_col, axis=1)

    return target, features, correlation_with_target


@cache_result('pcmci_analysis')
def perform_pcmci_analysis(normalized_data, config):
    write_log("Running PCMCI causal discovery...", log_file)

    data_array = normalized_data.values
    var_names = normalized_data.columns.tolist()

    dataframe = pp.DataFrame(data_array,
                             datatime=np.arange(len(data_array)),
                             var_names=var_names)

    parcorr = ParCorr()
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=1
    )

    results = pcmci.run_pcmci(
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        alpha_level=config.alpha_level
    )

    write_log("PCMCI analysis completed", log_file)

    pcmci_graph = visualize_pcmci_results(results, var_names, config)

    return results, pcmci, var_names, pcmci_graph


def visualize_pcmci_results(results, var_names, config):
    write_log("Visualizing PCMCI results using tigramite plotting style...", log_file)

    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    n_vars = len(var_names)
    tau_max_plus_one = p_matrix.shape[2]
    tau_max = tau_max_plus_one - 1

    q_matrix = p_matrix < config.alpha_level

    link_width_matrix = np.ones_like(val_matrix) * config.link_width

    graph = np.full(val_matrix.shape, "", dtype='<U3')

    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(1, tau_max + 1):
                if q_matrix[i, j, tau]:
                    graph[i, j, tau] = "-->"

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if q_matrix[i, j, 0]:
                if val_matrix[i, j, 0] >= val_matrix[j, i, 0]:
                    graph[i, j, 0] = "-->"
                    graph[j, i, 0] = "<--"
                else:
                    graph[i, j, 0] = "<--"
                    graph[j, i, 0] = "-->"

    causal_links = []
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(tau_max + 1):
                if graph[i, j, tau] == "-->":
                    causal_links.append({
                        'from': var_names[i],
                        'to': var_names[j],
                        'lag': tau,
                        'effect': val_matrix[i, j, tau],
                        'p_value': p_matrix[i, j, tau]
                    })

    causal_links_df = pd.DataFrame(causal_links)
    if not causal_links_df.empty:
        causal_links_path = get_output_path("pcmci_causal_links.csv")
        causal_links_df.to_csv(causal_links_path, index=False)
        write_log(f"PCMCI causal links saved to: {causal_links_path}", log_file)

    fig = plt.figure(figsize=(14, 12))
    tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_width=link_width_matrix,
        node_size=config.node_size,
        link_colorbar_label='Cross-MCI',
        arrow_linewidth=6,
        figsize=(14, 12),
        node_colorbar_label='Auto-MCI',
        show_autodependency_lags=False,
        label_fontsize=18,
        tick_label_size=16,
        node_label_size=18,
        link_label_fontsize=18,
        curved_radius=0.2,
        alpha=1.0,
        cmap_edges='Blues',
        cmap_nodes='Oranges',
        show_colorbar=True,
        vmin_edges=-1,
        vmax_edges=1.0,
        edge_ticks=0.4,
        arrowhead_size=40,
        save_name=get_output_path("pcmci_causal_graph.png")
    )
    plt.close()

    write_log(f"PCMCI causal graph saved to {get_output_path('pcmci_causal_graph.png')}", log_file)

    G = nx.DiGraph()
    for link in causal_links:
        G.add_edge(link['from'], link['to'],
                   lag=link['lag'],
                   effect=link['effect'],
                   weight=abs(link['effect']))

    return G


def main():
    try:
        config = Config()
        write_log(f"Starting causal analysis pipeline on dataset: {config.file_name}", log_file)
        write_log(f"Parallel computing enabled with {multiprocessing.cpu_count()} cores", log_file)
        write_log(f"Cache enabled: {config.use_cache}", log_file)

        data, normalized_data, time_data, scaler = load_and_normalize_data(config)
        target, features, correlation_with_target = confirm_target_parameter(data, normalized_data, config)
        pcmci_results, pcmci, var_names, pcmci_graph = perform_pcmci_analysis(normalized_data, config)

        normalized_data.to_csv(get_output_path("normalized_data_for_next_step.csv"), index=False)
        with open(get_output_path("pcmci_results.pkl"), 'wb') as f:
            pickle.dump({'pcmci_results': pcmci_results, 'var_names': var_names}, f)

        write_log("\nPCMCI initial causal discovery completed successfully!", log_file)
        write_log(f"All results have been saved to: {config.output_dir}", log_file)
        write_log(f"Proceed to next step with normalized_data_for_next_step.csv and pcmci_results.pkl", log_file)

    except Exception as e:
        write_log(f"ERROR in main process: {str(e)}", log_file)
        import traceback
        write_log(traceback.format_exc(), log_file)


if __name__ == "__main__":
    main()
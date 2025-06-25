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

# 新增导入 - 并行计算、缓存和数据质量控制
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
import multiprocessing
import pickle
import hashlib
from functools import wraps

warnings.filterwarnings('ignore')

# 创建带时间戳的输出文件夹
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"ATTCN_PCMCI_L1_results_{current_time}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"All output files will be saved to: {output_folder}")

# 创建缓存目录
cache_dir = os.path.join(output_folder, "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 设置全局绘图风格 - SCI论文级别
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300


# 配置类 - 可调整参数
class Config:
    # 数据参数
    file_name = '干扰后数据集.csv'
    target_col = 'TN_eff'
    seq_length = 10
    test_size = 0.2
    random_state = 42

    # PCMCI参数
    tau_min = 0
    tau_max = 2
    alpha_level = 0.01

    # Granger因果L1正则化参数
    max_lags = tau_max
    alpha_lasso = 0.1

    # TCN模型参数
    num_channels = [64, 128, 64]
    kernel_size = 3
    num_heads = 4
    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # 稳健性测试参数
    n_iterations = 1000

    # 可视化参数
    node_size = 0.3
    link_width = 1
    significant_threshold = 0.05

    # 设置白名单参数 - 这些特征将被强制保留
    whitelist_features = []

    # 设置汇节点集合 - 指向这些节点的边将被删除
    sink_nodes = ['Flow_in', 'TN_in', 'NH4_in', 'COD_in', 'TP_in', 'SS_in', 'DCS', 'DPRA', 'IR',
                  'ER', 'Was', 'DO_1', 'DO_2', 'DO_3']

    # 输出参数
    output_dir = output_folder

    # 数据质量控制参数
    outlier_method = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold = 3
    outlier_handling = 'clip'  # 'remove', 'clip', 'interpolate'
    missing_value_method = 'interpolate'  # 'interpolate', 'forward_fill', 'mean', 'drop'
    missing_value_threshold = 0.1

    # PFI参数 - 使用百分位数方法
    pfi_percentile = 90  # 保留90%的特征

    # 并行计算参数
    n_jobs = -1  # 使用所有CPU核心
    use_cache = True
    cache_dir = cache_dir


# 缓存装饰器
def cache_result(cache_key_prefix):
    """
    缓存函数结果的装饰器
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not Config.use_cache:
                return func(*args, **kwargs)

            # 生成缓存键
            cache_key = f"{cache_key_prefix}_{str(args)}_{str(kwargs)}"
            cache_file = os.path.join(Config.cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")

            # 尝试从缓存加载
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    write_log(f"Loaded {cache_key_prefix} from cache", log_file)
                    return result
                except:
                    pass

            # 计算结果
            result = func(*args, **kwargs)

            # 保存到缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass

            return result

        return wrapper

    return decorator


# 记录日志函数
def write_log(message, log_file):
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)


# 创建日志文件
log_file = os.path.join(output_folder, "analysis_log.txt")
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"ATTCN-PCMCI-L1 Analysis Log - Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("-" * 80 + "\n")


# 创建带时间戳的输出文件名
def get_output_path(filename):
    return os.path.join(Config.output_dir, filename)


# ------------------------------ 数据质量控制函数 ------------------------------
def detect_and_handle_outliers(data, method='iqr', threshold=3, handle_method='clip'):
    """
    检测并处理异常值

    参数:
    - data: DataFrame, 输入数据
    - method: str, 检测方法 ('iqr', 'zscore', 'isolation_forest')
    - threshold: float, 异常值阈值
    - handle_method: str, 处理方法 ('remove', 'clip', 'interpolate')

    返回:
    - cleaned_data: DataFrame, 处理后的数据
    - outlier_info: dict, 异常值信息
    """
    cleaned_data = data.copy()
    outlier_info = {}

    # 获取数值列
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

        # 记录异常值信息
        outlier_info[col] = {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(data)) * 100,
            'indices': data.index[outliers].tolist()
        }

        # 处理异常值
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
    """
    处理缺失值

    参数:
    - data: DataFrame, 输入数据
    - method: str, 处理方法 ('interpolate', 'forward_fill', 'mean', 'drop')
    - threshold: float, 缺失值比例阈值，超过则删除该列

    返回:
    - cleaned_data: DataFrame, 处理后的数据
    - missing_info: dict, 缺失值信息
    """
    cleaned_data = data.copy()
    missing_info = {}

    # 计算每列缺失值比例
    missing_ratios = data.isnull().sum() / len(data)

    # 删除缺失值过多的列
    cols_to_drop = missing_ratios[missing_ratios > threshold].index
    if len(cols_to_drop) > 0:
        cleaned_data = cleaned_data.drop(columns=cols_to_drop)
        write_log(f"Dropped columns with >={threshold * 100}% missing values: {list(cols_to_drop)}", log_file)

    # 处理剩余缺失值
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
    """
    检验时间序列平稳性

    参数:
    - data: DataFrame, 时间序列数据
    - significance_level: float, 显著性水平

    返回:
    - stationarity_results: dict, 平稳性检验结果
    - non_stationary_cols: list, 非平稳列
    """
    stationarity_results = {}
    non_stationary_cols = []

    for col in data.select_dtypes(include=[np.number]).columns:
        # ADF检验
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
    """
    生成详细的数据质量报告
    """
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


# ------------------------------ A. 改进的数据加载与预处理 ------------------------------
def load_and_normalize_data(config):
    write_log("Loading and normalizing data with quality control...", log_file)

    try:
        # 加载数据
        data = pd.read_csv(config.file_name)
        write_log(f"Data loaded successfully with shape: {data.shape}", log_file)

        # 检查目标列是否存在
        if config.target_col not in data.columns:
            write_log(f"Error: Target column '{config.target_col}' not found in dataset.", log_file)
            raise ValueError(f"Target column '{config.target_col}' not found in dataset.")

        # 将时间列设为索引（如果存在）
        if 'Time' in data.columns:
            time_data = data['Time'].copy()
            data = data.drop('Time', axis=1)
        else:
            time_data = pd.Series(range(len(data)), name='Time')

        # 1. 处理缺失值
        data, missing_info = handle_missing_values(
            data,
            method=config.missing_value_method,
            threshold=config.missing_value_threshold
        )
        write_log(f"Missing values handled. Remaining shape: {data.shape}", log_file)

        # 2. 检测和处理异常值
        data, outlier_info = detect_and_handle_outliers(
            data,
            method=config.outlier_method,
            threshold=config.outlier_threshold,
            handle_method=config.outlier_handling
        )
        write_log(f"Outliers handled. Total outliers found: {sum(info['count'] for info in outlier_info.values())}",
                  log_file)

        # 3. 检验平稳性
        stationarity_results, non_stationary_cols = check_stationarity(data)
        if non_stationary_cols:
            write_log(f"Non-stationary columns detected: {non_stationary_cols}", log_file)

        # 4. 生成数据质量报告
        generate_data_quality_report(data, missing_info, outlier_info, stationarity_results, config.output_dir)

        # 5. 标准化数据
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns
        )

        write_log(f"Data normalized. Shape: {normalized_data.shape}", log_file)

        # 保存预处理后的数据
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


# ------------------------------ C. 目标参数确认 ------------------------------
def confirm_target_parameter(data, normalized_data, config):
    write_log(f"Confirming target parameter: {config.target_col}", log_file)

    # 检查目标列是否存在
    if config.target_col not in data.columns:
        write_log(f"Error: Target column '{config.target_col}' not found in dataset.", log_file)
        raise ValueError(f"Target column '{config.target_col}' not found in dataset.")

    # 计算目标变量与其他变量的相关性
    correlation_with_target = data.corr()[config.target_col].sort_values(ascending=False)

    write_log(f"Top 5 correlated features with {config.target_col}:", log_file)
    for i, (feature, corr) in enumerate(correlation_with_target.iloc[1:6].items()):
        write_log(f"  {i + 1}. {feature}: {corr:.4f}", log_file)

    # 提取目标变量
    target = normalized_data[config.target_col]
    features = normalized_data.drop(config.target_col, axis=1)

    return target, features, correlation_with_target


# ------------------------------ D. PCMCI因果发现（带缓存） ------------------------------
@cache_result('pcmci_analysis')
def perform_pcmci_analysis(normalized_data, config):
    write_log("Running PCMCI causal discovery...", log_file)

    # 准备PCMCI数据格式
    data_array = normalized_data.values
    var_names = normalized_data.columns.tolist()

    # 创建TimeSeries对象
    dataframe = pp.DataFrame(data_array,
                             datatime=np.arange(len(data_array)),
                             var_names=var_names)

    # 初始化PCMCI
    parcorr = ParCorr()
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=1
    )

    # 运行PCMCI算法
    results = pcmci.run_pcmci(
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        alpha_level=config.alpha_level
    )

    write_log("PCMCI analysis completed", log_file)

    # 保存PCMCI结果
    pcmci_graph = visualize_pcmci_results(results, var_names, config)

    return results, pcmci, var_names, pcmci_graph


def visualize_pcmci_results(results, var_names, config):
    """使用tigramite的绘图风格可视化PCMCI结果 - 兼容原始PCMCI7参数"""
    write_log("Visualizing PCMCI results using tigramite plotting style...", log_file)

    # 获取p值矩阵（显著性）
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    n_vars = len(var_names)
    tau_max_plus_one = p_matrix.shape[2]
    tau_max = tau_max_plus_one - 1

    # 创建显著性矩阵
    q_matrix = p_matrix < config.alpha_level

    # 创建与val_matrix相同形状的link_width数组
    link_width_matrix = np.ones_like(val_matrix) * config.link_width

    # 创建图结构
    graph = np.full(val_matrix.shape, "", dtype='<U3')

    # 处理时间滞后(tau > 0)的连接
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(1, tau_max + 1):
                if q_matrix[i, j, tau]:
                    graph[i, j, tau] = "-->"

    # 处理同时刻(tau = 0)的连接
    for i in range(n_vars):
        for j in range(i + 1, n_vars):  # 只处理上三角矩阵
            if q_matrix[i, j, 0]:
                if val_matrix[i, j, 0] >= val_matrix[j, i, 0]:
                    graph[i, j, 0] = "-->"
                    graph[j, i, 0] = "<--"
                else:
                    graph[i, j, 0] = "<--"
                    graph[j, i, 0] = "-->"

    # 创建因果链接列表
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

    # 保存因果链接为CSV
    causal_links_df = pd.DataFrame(causal_links)
    if not causal_links_df.empty:
        causal_links_path = get_output_path("pcmci_causal_links.csv")
        causal_links_df.to_csv(causal_links_path, index=False)
        write_log(f"PCMCI causal links saved to: {causal_links_path}", log_file)

    # 使用与PCMCI7.py完全相同的参数调用plot_graph函数
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
        label_fontsize=18,  # 控制标签和colorbar标签的字体大小
        tick_label_size=16,  # 控制刻度标签的字体大小
        node_label_size=18,  # 控制节点标签的字体大小
        link_label_fontsize=18,  # 控制连接线上的标签字体大小
        curved_radius=0.2,
        alpha=1.0,  # 透明度
        cmap_edges='Blues',  # 连接线的colormap
        cmap_nodes='Oranges',  # 节点的colormap
        show_colorbar=True,  # 只显示边的colorbar，不显示节点的
        vmin_edges=-1,  # 连接线colorbar的最小值
        vmax_edges=1.0,  # 连接线colorbar的最大值
        edge_ticks=0.4,  # 连接线colorbar的刻度间隔
        arrowhead_size=40,  # 箭头大小
        save_name=get_output_path("pcmci_causal_graph.png")  # 保存文件名
    )
    plt.close()

    write_log(f"PCMCI causal graph saved to {get_output_path('pcmci_causal_graph.png')}", log_file)

    # 为了与其他函数保持一致性，返回一个NetworkX图对象
    G = nx.DiGraph()
    for link in causal_links:
        G.add_edge(link['from'], link['to'],
                   lag=link['lag'],
                   effect=link['effect'],
                   weight=abs(link['effect']))

    return G


# ------------------------------ E. 因果稀疏性增强 - Granger因果 + L1正则化（并行化） ------------------------------
def granger_causality_with_l1(data, pcmci_results, var_names, config):
    write_log("Performing Granger causality with L1 regularization (parallel)...", log_file)

    # 从PCMCI结果提取显著因果关系
    p_matrix = pcmci_results['p_matrix']
    val_matrix = pcmci_results['val_matrix']
    n_vars = len(var_names)
    tau_max = p_matrix.shape[2] - 1

    # 准备并行任务
    tasks = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:  # 避免自因果
                # 检查PCMCI是否发现了因果关系
                is_pcmci_causal = False
                for tau in range(tau_max + 1):
                    if p_matrix[i, j, tau] < config.alpha_level:
                        is_pcmci_causal = True
                        break

                if is_pcmci_causal:
                    tasks.append((i, j, var_names, data, val_matrix, tau_max, config))

    # 并行执行Granger因果检验
    n_cores = multiprocessing.cpu_count() if config.n_jobs == -1 else config.n_jobs
    results = Parallel(n_jobs=n_cores)(
        delayed(test_granger_causality_single)(task) for task in tasks
    )

    # 收集有效的结果
    enhanced_links = []
    for result in results:
        if result is not None:
            enhanced_links.extend(result)

    # 将增强的因果关系保存为DataFrame
    enhanced_df = pd.DataFrame(enhanced_links)
    if not enhanced_df.empty:
        enhanced_path = get_output_path("enhanced_causal_links.csv")
        enhanced_df.to_csv(enhanced_path, index=False)
        write_log(f"Enhanced causal links saved to: {enhanced_path}", log_file)
        write_log(f"Found {len(enhanced_links)} enhanced causal relationships after L1 regularization", log_file)
    else:
        write_log("No significant causal relationships found after L1 regularization", log_file)

    # 可视化增强后的因果图
    enhanced_graph = visualize_enhanced_causal_graph(enhanced_links, var_names, config)

    return enhanced_links, enhanced_graph


def test_granger_causality_single(args):
    """
    单个Granger因果检验（用于并行化）
    """
    i, j, var_names, data, val_matrix, tau_max, config = args

    write_log(f"Testing Granger causality from {var_names[i]} to {var_names[j]}...", log_file)

    # 提取因变量和自变量
    y = data[var_names[j]].values
    x = data[var_names[i]].values

    # 创建滞后特征矩阵
    lag_features = []
    for lag in range(1, config.max_lags + 1):
        # 为因变量创建滞后特征
        y_lag = np.roll(y, lag)
        y_lag[:lag] = y_lag[lag]  # 处理边界
        lag_features.append(y_lag)

        # 为自变量创建滞后特征
        x_lag = np.roll(x, lag)
        x_lag[:lag] = x_lag[lag]  # 处理边界
        lag_features.append(x_lag)

    # 将滞后特征组合为特征矩阵
    X = np.column_stack(lag_features)

    # 使用LassoCV进行L1正则化的回归
    lasso = LassoCV(cv=5, random_state=config.random_state)
    lasso.fit(X[config.max_lags:], y[config.max_lags:])

    # 检查x的滞后项的系数
    x_lag_coeffs = lasso.coef_[1::2]  # 提取x的滞后系数

    # 如果x的任何滞后系数显著非零，保留因果关系
    enhanced_links = []
    if np.any(np.abs(x_lag_coeffs) > 1e-5):
        granger_strength = np.sum(np.abs(x_lag_coeffs))
        significant_lags = np.where(np.abs(x_lag_coeffs) > 1e-5)[0] + 1

        write_log(
            f"  Granger causality confirmed with L1, strength: {granger_strength:.4f}, significant lags: {significant_lags}",
            log_file)

        for lag in significant_lags:
            enhanced_links.append({
                'from': var_names[i],
                'to': var_names[j],
                'lag': int(lag),
                'granger_strength': float(np.abs(x_lag_coeffs[lag - 1])),
                'original_effect': float(val_matrix[i, j, min(lag, tau_max)]),
                'coefficient': float(x_lag_coeffs[lag - 1])
            })
    else:
        write_log(f"  Granger causality NOT confirmed with L1 regularization.", log_file)

    return enhanced_links if enhanced_links else None


def visualize_enhanced_causal_graph(enhanced_links, var_names, config):
    """使用与PCMCI7兼容的参数可视化增强因果图"""
    write_log("Visualizing enhanced causal graph with tigramite-inspired style...", log_file)

    # 如果没有增强链接，返回空图
    if not enhanced_links:
        write_log("No enhanced links to visualize.", log_file)
        return nx.DiGraph()

    # 准备tigramite绘图所需的矩阵
    n_vars = len(var_names)

    # 保证有链接存在时才计算tau_max
    if enhanced_links:
        tau_max = max([link['lag'] for link in enhanced_links])
    else:
        tau_max = 0

    # 创建初始矩阵
    val_matrix = np.zeros((n_vars, n_vars, tau_max + 1))
    graph = np.full((n_vars, n_vars, tau_max + 1), "", dtype='<U3')
    link_width = np.ones((n_vars, n_vars, tau_max + 1)) * config.link_width

    # 为每个因果链接填充矩阵
    for link in enhanced_links:
        i = var_names.index(link['from'])
        j = var_names.index(link['to'])
        tau = link['lag']
        val_matrix[i, j, tau] = link['coefficient']
        graph[i, j, tau] = "-->"

    # 使用与PCMCI7.py完全相同的参数调用plot_graph函数
    fig = plt.figure(figsize=(14, 12))
    tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_width=link_width,
        node_size=config.node_size,
        link_colorbar_label='Granger Strength',
        arrow_linewidth=6,
        figsize=(14, 12),
        show_autodependency_lags=False,
        label_fontsize=18,  # 控制标签和colorbar标签的字体大小
        tick_label_size=16,  # 控制刻度标签的字体大小
        node_label_size=18,  # 控制节点标签的字体大小
        link_label_fontsize=18,  # 控制连接线上的标签字体大小
        curved_radius=0.2,
        alpha=1.0,  # 透明度
        cmap_edges='Blues',  # 连接线的colormap
        cmap_nodes='Oranges',  # 节点的colormap
        show_colorbar=True,  # 只显示边的colorbar，不显示节点的
        vmin_edges=-1,  # 连接线colorbar的最小值
        vmax_edges=1.0,  # 连接线colorbar的最大值
        edge_ticks=0.4,  # 连接线colorbar的刻度间隔
        arrowhead_size=40,  # 箭头大小
        save_name=get_output_path("enhanced_causal_graph.png")
    )
    plt.close()

    write_log(f"Enhanced causal graph saved to {get_output_path('enhanced_causal_graph.png')}", log_file)

    # 为了与其他函数保持一致性，返回一个NetworkX图对象
    G = nx.DiGraph()
    for link in enhanced_links:
        G.add_edge(link['from'], link['to'],
                   lag=link['lag'],
                   weight=link['granger_strength'],
                   coefficient=link['coefficient'])

    return G


# ------------------------------ H. ACE/RACE估计 ------------------------------
def calculate_causal_effects(data, enhanced_links, config):
    write_log("Calculating causal effects (ACE/RACE)...", log_file)

    causal_effects = []

    # 为每个增强的因果链接计算因果效应
    for link in enhanced_links:
        cause = link['from']
        effect = link['to']
        lag = link['lag']

        # 计算ACE
        ace = calculate_ace(data, cause, effect, lag)

        # 计算RACE
        race = calculate_race(data, cause, effect, lag)

        causal_effects.append({
            'cause': cause,
            'effect': effect,
            'lag': lag,
            'ACE': ace,
            'RACE': race,
            'granger_strength': link['granger_strength']
        })

    # 创建DataFrame并保存
    effects_df = pd.DataFrame(causal_effects)
    if not effects_df.empty:
        effects_path = get_output_path("causal_effects.csv")
        effects_df.to_csv(effects_path, index=False)
        write_log(f"Causal effects saved to: {effects_path}", log_file)
    else:
        write_log("No causal effects to calculate.", log_file)

    # 可视化因果效应
    visualize_causal_effects(effects_df, config)

    return causal_effects


def calculate_ace(data, cause, effect, lag=0):
    """计算平均因果效应 (Average Causal Effect)"""
    # 获取原因和结果变量
    cause_data = data[cause].values

    # 创建滞后的效应变量
    if lag > 0 and lag < len(cause_data):
        effect_data = data[effect].values[lag:]
        cause_data = cause_data[:-lag]
    else:
        effect_data = data[effect].values

    # 根据原因变量的中位数分割数据
    threshold = np.median(cause_data)
    high_cause = effect_data[cause_data > threshold]
    low_cause = effect_data[cause_data <= threshold]

    # 计算平均因果效应
    ace = np.mean(high_cause) - np.mean(low_cause)
    return ace


def calculate_race(data, cause, effect, lag=0):
    """计算相对平均因果效应 (Relative Average Causal Effect)"""
    ace = calculate_ace(data, cause, effect, lag)

    # 获取效应变量的标准差
    if lag > 0 and lag < len(data):
        effect_std = np.std(data[effect].values[lag:])
    else:
        effect_std = np.std(data[effect].values)

    # 计算相对平均因果效应
    race = ace / effect_std if effect_std != 0 else 0
    return race


def visualize_causal_effects(effects_df, config):
    if effects_df.empty:
        write_log("No causal effects to visualize.", log_file)
        return

    # 创建因果效应对比图 - 增加图形宽度
    plt.figure(figsize=(20, 10))  # 增加宽度

    # 排序数据以便更好的可视化
    sorted_df = effects_df.sort_values('ACE', ascending=False)

    # 创建因果对标签
    labels = [f"{row['cause']} → {row['effect']} (τ={row['lag']})" for _, row in sorted_df.iterrows()]

    # 绘制ACE和RACE的对比条形图
    x = np.arange(len(labels))
    width = 0.35

    ax = plt.gca()
    rects1 = ax.bar(x - width / 2, sorted_df['ACE'], width, label='ACE', color='royalblue', edgecolor='black',
                    linewidth=1)
    rects2 = ax.bar(x + width / 2, sorted_df['RACE'], width, label='RACE', color='lightcoral', edgecolor='black',
                    linewidth=1)

    # 添加标签和标题 - 增大字体
    plt.xlabel('Causal Relationships', fontsize=20, fontweight='bold')
    plt.ylabel('Effect Magnitude', fontsize=20, fontweight='bold')
    plt.title('ACE and RACE for Causal Relationships', fontsize=24, fontweight='bold')

    # 调整x轴标签 - 增大字体并倾斜45度
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=12)  # 修改rotation为45度，ha为'right'
    plt.yticks(fontsize=14)  # 添加y轴刻度字体大小

    # 调整布局以确保倾斜的标签不会被裁剪
    plt.tight_layout(pad=3)  # 增加更多填充空间
    plt.legend(fontsize=18)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 保存图像
    effects_viz_path = get_output_path("causal_effects_comparison.png")
    plt.savefig(effects_viz_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Causal effects visualization saved to: {effects_viz_path}", log_file)

    # 创建热力图展示因果效应
    pivot_data = effects_df.pivot_table(
        index='cause',
        columns='effect',
        values='ACE',
        aggfunc='mean'
    ).fillna(0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0,
                linewidths=0.5, fmt='.3f', cbar_kws={'label': 'ACE'},
                annot_kws={"size": 12})  # 增大热力图中的注释字体大小

    # 增大热力图的标题和轴标签字体
    plt.title('Average Causal Effects Heatmap', fontsize=24, fontweight='bold')
    plt.xlabel('Effect Variables', fontsize=18, fontweight='bold')
    plt.ylabel('Cause Variables', fontsize=18, fontweight='bold')

    # 增大坐标轴刻度标签
    plt.xticks(fontsize=14, rotation=45, ha='right')  # 增大x轴刻度标签并旋转
    plt.yticks(fontsize=14)  # 增大y轴刻度标签

    # 调整色条标签大小
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('ACE', fontsize=16, fontweight='bold')
    cbar.tick_params(labelsize=14)

    plt.tight_layout()

    # 保存热力图
    heatmap_path = get_output_path("ace_heatmap.png")
    plt.savefig(heatmap_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"ACE heatmap saved to: {heatmap_path}", log_file)


# ------------------------------ J. TCN-PFI重要性分析 ------------------------------
# TCN模型定义
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    batch_first=True)

    def forward(self, x):
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        return attn_output, attn_weights


class ATTTCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, num_heads=4):
        super(ATTTCN, self).__init__()
        self.tcn_blocks = nn.ModuleList()
        num_layers = len(num_channels)
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            self.tcn_blocks.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))

        self.attention = MultiHeadAttentionLayer(embed_dim=num_channels[-1], num_heads=num_heads)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        attention_output, attention_weights = self.attention(x)
        x = attention_output.mean(dim=1)
        x = self.fc(x)
        return x, attention_weights


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X, y, seq_length=10):
    """创建时间序列训练序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_evaluate_tcn(features, target, config, enhanced_links):
    write_log("Training TCN model with attention and performing PFI analysis...", log_file)

    # 准备数据
    X = features.values
    y = target.values.reshape(-1, 1)

    # 创建时间序列序列
    X_seq, y_seq = create_sequences(X, y, seq_length=config.seq_length)

    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=config.test_size, random_state=config.random_state
    )

    write_log(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}", log_file)

    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Using device: {device}", log_file)

    # 初始化模型
    input_dim = X_train.shape[2]
    output_dim = 1

    model = ATTTCN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_channels=config.num_channels,
        kernel_size=config.kernel_size,
        num_heads=config.num_heads
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练模型
    write_log("Starting model training...", log_file)
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 每10个epoch评估一次模型
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0.0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(outputs.cpu().numpy())

            # 计算评估指标
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            write_log(f"Epoch {epoch + 1}/{config.epochs}, "
                      f"Train Loss: {train_loss / len(train_loader):.4f}, "
                      f"Test Loss: {test_loss / len(test_loader):.4f}, "
                      f"R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}", log_file)

    # 保存模型
    model_path = get_output_path("attcn_model.pth")
    torch.save(model.state_dict(), model_path)
    write_log(f"Model saved to: {model_path}", log_file)

    # 计算最终评估指标
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, _ = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    final_r2 = r2_score(y_true, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    final_mae = mean_absolute_error(y_true, y_pred)

    write_log("\nFinal Model Performance:", log_file)
    write_log(f"R2: {final_r2:.4f}", log_file)
    write_log(f"RMSE: {final_rmse:.4f}", log_file)
    write_log(f"MAE: {final_mae:.4f}", log_file)

    # 创建散点图比较预测值和真实值
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, edgecolors='w')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=16, fontweight='bold')
    plt.title('True vs Predicted Values', fontsize=20, fontweight='bold')

    # 添加性能指标文本框
    textstr = f'R² = {final_r2:.4f}\nRMSE = {final_rmse:.4f}\nMAE = {final_mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    pred_plot_path = get_output_path("true_vs_predicted.png")
    plt.savefig(pred_plot_path, dpi=800, bbox_inches='tight')
    plt.close()

    # 执行PFI分析
    write_log("\nPerforming Permutation Feature Importance (PFI) analysis...", log_file)
    pfi_scores = calculate_pfi(model, X_test_tensor=torch.tensor(X_test, dtype=torch.float32).to(device),
                               y_test_tensor=torch.tensor(y_test, dtype=torch.float32).to(device),
                               feature_names=features.columns, device=device)

    # 基于PFI筛选重要特征 - 使用百分位数方法
    selected_features, pfi_threshold = select_features_by_pfi_percentile(pfi_scores, features.columns, config)

    # 可视化PFI结果和选择的特征
    visualize_pfi_results(pfi_scores, features.columns, config, enhanced_links, selected_features, pfi_threshold)

    return model, pfi_scores, selected_features


def calculate_pfi(model, X_test_tensor, y_test_tensor, feature_names, device):
    """
    使用与ATTCN-PC16相同的方法计算PFI比率
    """
    model.eval()

    # 计算基线性能
    with torch.no_grad():
        original_pred, _ = model(X_test_tensor)
        original_loss = mean_squared_error(y_test_tensor.cpu().numpy(), original_pred.cpu().numpy())

    # 对每个特征计算PFI
    pfi_results = []

    for i in range(X_test_tensor.shape[2]):
        # 创建特征排列
        X_permuted = X_test_tensor.clone()
        permuted_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, :, i] = X_test_tensor[permuted_idx, :, i]

        # 使用排列后的特征进行预测
        with torch.no_grad():
            perm_pred, _ = model(X_permuted)
            perm_loss = mean_squared_error(y_test_tensor.cpu().numpy(), perm_pred.cpu().numpy())

        # 按ATTCN-PC16中的方法计算PFI比率: 原始损失/排列后损失
        pfi_ratio = original_loss / perm_loss

        pfi_results.append({
            'feature': feature_names[i],
            'pfi_score': pfi_ratio
        })

    # 排序结果
    pfi_df = pd.DataFrame(pfi_results)

    # 保存PFI结果
    pfi_path = get_output_path("pfi_scores.csv")
    pfi_df.to_csv(pfi_path, index=False)
    write_log(f"PFI scores saved to: {pfi_path}", log_file)

    return pfi_df


def select_features_by_pfi_percentile(pfi_df, feature_names, config):
    """
    根据PFI分数选择重要特征 - 使用百分位数方法
    保留PFI分数最低的90%特征（因为低分表示重要）
    """
    pfi_scores = pfi_df['pfi_score'].values

    # 计算阈值：保留90%的特征
    pfi_threshold = np.percentile(pfi_scores, config.pfi_percentile)
    write_log(f"PFI threshold for feature selection (percentile {config.pfi_percentile}): {pfi_threshold:.4f}",
              log_file)

    # 选择PFI分数小于阈值的特征
    selected_features = pfi_df[pfi_df['pfi_score'] < pfi_threshold]['feature'].tolist()

    # 处理白名单特征
    for f in config.whitelist_features:
        if f in feature_names and f not in selected_features:
            selected_features.append(f)
            write_log(f"Added whitelist feature '{f}' to selected features", log_file)

    # 确保选中合理数量的特征
    if len(selected_features) < 5:
        # 如果选中的特征太少，使用top-k方法
        selected_features = pfi_df.nsmallest(10, 'pfi_score')['feature'].tolist()
        write_log(f"Too few features selected, using top-10 features", log_file)

    # 确保目标变量被添加到选定特征中
    if config.target_col not in selected_features and config.target_col not in feature_names:
        selected_features.append(config.target_col)
        write_log(f"Added target variable '{config.target_col}' to selected features", log_file)

    write_log(f"\nSelected {len(selected_features)}/{len(feature_names)} features using percentile method:", log_file)
    for feat in selected_features:
        if feat in pfi_df['feature'].values:
            pfi_score = pfi_df[pfi_df['feature'] == feat]['pfi_score'].values[0]
            write_log(f"  {feat}: {pfi_score:.4f}", log_file)
        else:
            write_log(f"  {feat}: (target variable or not analyzed)", log_file)

    return selected_features, pfi_threshold


def visualize_pfi_results(pfi_df, feature_names, config, enhanced_links, selected_features, pfi_threshold):
    """可视化PFI结果, 包括强调选定的特征"""
    # 准备绘图数据
    sorted_pfi = pfi_df.sort_values('pfi_score', ascending=True).copy()

    # 创建颜色映射 - 选中的特征用绿色，未选中的用灰色
    colors = ['green' if feat in selected_features else 'gray'
              for feat in sorted_pfi['feature']]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(sorted_pfi['feature'], sorted_pfi['pfi_score'],
                    color=colors, edgecolor='black', linewidth=1)

    # 添加阈值线
    plt.axvline(x=pfi_threshold, color='crimson', linestyle='--', linewidth=2,
                label=f'Threshold ({pfi_threshold:.4f})')

    # 添加标题和标签 - 增大字体
    plt.title('Permutation Feature Importance', fontsize=24, fontweight='bold')  # 增大字体
    plt.xlabel('PFI Ratio (original loss / permuted loss)', fontsize=20, fontweight='bold')  # 增大字体
    plt.ylabel('Feature', fontsize=20, fontweight='bold')  # 增大字体

    # 为每个条形添加数值标签 - 增大字体
    for i, v in enumerate(sorted_pfi['pfi_score']):
        plt.text(v + 0.02, i, f"{v:.4f}", va='center', fontsize=16)  # 增大字体从12到14

    # 增大坐标轴刻度标签
    plt.xticks(fontsize=16)  # 增大x轴刻度标签
    plt.yticks(fontsize=16)  # 增大y轴刻度标签

    # 修改后的图例代码 - 增大字体
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='crimson', linestyle='--', linewidth=2,
               label=f'Threshold ({pfi_threshold:.4f})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=16,  # 增大标记大小
               label='Selected features'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=16,  # 增大标记大小
               label='Excluded features')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=16)  # 增大图例字体从12到16

    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    pfi_plot_path = get_output_path("pfi_importance.png")
    plt.savefig(pfi_plot_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"PFI visualization saved to: {pfi_plot_path}", log_file)

    # 创建PFI分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(pfi_df['pfi_score'], bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(pfi_df['pfi_score']), color='red', linestyle='--',
                label=f'Mean ({np.mean(pfi_df["pfi_score"]):.4f})')
    plt.axvline(x=np.median(pfi_df['pfi_score']), color='blue', linestyle='-.',
                label=f'Median ({np.median(pfi_df["pfi_score"]):.4f})')
    plt.axvline(x=pfi_threshold, color='crimson', linestyle='-',
                label=f'Threshold ({pfi_threshold:.4f})')

    # 增大直方图的标题和标签字体
    plt.title('Distribution of PFI Scores', fontsize=24, fontweight='bold')  # 增大字体
    plt.xlabel('PFI Ratio', fontsize=20, fontweight='bold')  # 增大字体
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')  # 增大字体
    plt.legend(fontsize=16)  # 增大图例字体从12到16

    # 增大坐标轴刻度标签
    plt.xticks(fontsize=14)  # 增大x轴刻度标签
    plt.yticks(fontsize=14)  # 增大y轴刻度标签

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存直方图
    pfi_hist_path = get_output_path("pfi_distribution.png")
    plt.savefig(pfi_hist_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"PFI distribution saved to: {pfi_hist_path}", log_file)


# ------------------------------ L. 稳健性检验（并行化） ------------------------------
def perform_robustness_tests(data, enhanced_links, config):
    write_log("Performing robustness tests: CSR, DDA, RCC, Placebo Test (parallel)...", log_file)

    # 准备并行任务
    tasks = [(data, link, config) for link in enhanced_links]

    # 并行执行稳健性测试
    n_cores = multiprocessing.cpu_count() if config.n_jobs == -1 else config.n_jobs
    results = Parallel(n_jobs=n_cores)(
        delayed(test_robustness_single)(task) for task in tasks
    )

    # 创建稳健性结果DataFrame
    robustness_df = pd.DataFrame(results)

    if not robustness_df.empty:
        # 保存稳健性结果
        robustness_path = get_output_path("robustness_results.csv")
        robustness_df.to_csv(robustness_path, index=False)
        write_log(f"Robustness test results saved to: {robustness_path}", log_file)

        # 可视化稳健性结果
        visualize_robustness_results(robustness_df, config)
    else:
        write_log("No robustness results to save.", log_file)

    return robustness_df


def test_robustness_single(args):
    """
    单个链接的稳健性测试（用于并行化）
    """
    data, link, config = args

    cause = link['from']
    effect = link['to']
    lag = link['lag']

    # 创建滞后数据
    if lag > 0:
        lagged_data = data.copy()
        lagged_data[f"{cause}_lag{lag}"] = lagged_data[cause].shift(lag)
        lagged_data = lagged_data.dropna()
        test_cause = f"{cause}_lag{lag}"
    else:
        lagged_data = data.copy()
        test_cause = cause

    # 计算CSR
    csr = calculate_csr(lagged_data, test_cause, effect, config.n_iterations)

    # 计算DDA
    dda = calculate_dda(lagged_data, test_cause, effect)

    # 执行RCC测试
    rcc_score = test_robustness_rcc(lagged_data, test_cause, effect, config.n_iterations)

    # 执行安慰剂测试
    pt_p_value = test_robustness_pt(lagged_data, test_cause, effect, config.n_iterations)

    return {
        'cause': cause,
        'effect': effect,
        'lag': lag,
        'CSR': csr,
        'DDA': dda,
        'RCC_score': rcc_score,
        'PT_p_value': pt_p_value
    }


def generate_random_common_cause(n_samples):
    """生成随机共同原因变量"""
    return np.random.normal(0, 1, n_samples)


def create_placebo_treatment(original_treatment):
    """创建安慰剂处理 - 通过打乱原始处理"""
    return np.random.permutation(original_treatment)


def calculate_csr(data, cause, effect, num_bootstrap=1000):
    """计算因果稳定比率 (Causal Stability Ratio)"""
    n_samples = len(data)
    positive_count = 0

    for _ in range(num_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_data = data.iloc[bootstrap_idx]
        ace = calculate_ace(bootstrap_data, cause, effect)
        if ace > 0:
            positive_count += 1

    csr = positive_count / num_bootstrap
    return csr


def calculate_dda(data, var1, var2):
    """计算方向依赖分析 (Directional Dependence Analysis)"""
    ace_forward = calculate_ace(data, var1, var2)
    ace_backward = calculate_ace(data, var2, var1)

    denominator = abs(ace_forward) + abs(ace_backward)
    if denominator == 0:
        return 0
    dda = (ace_forward - ace_backward) / denominator
    return dda


def test_robustness_rcc(data, cause, effect, n_iterations=1000):
    """使用随机共同原因测试稳健性"""
    original_ace = calculate_ace(data, cause, effect)
    n_samples = len(data)
    ace_with_rcc = []

    for _ in range(n_iterations):
        rcc = generate_random_common_cause(n_samples)
        data_with_rcc = data.copy()
        data_with_rcc['RCC'] = rcc
        new_ace = calculate_ace(data_with_rcc, cause, effect)
        ace_with_rcc.append(new_ace)

    ace_std = np.std(ace_with_rcc)
    robustness_score = 1 - (ace_std / abs(original_ace)) if original_ace != 0 else 0
    return robustness_score


def test_robustness_pt(data, cause, effect, n_iterations=1000):
    """使用安慰剂处理测试稳健性"""
    original_ace = calculate_ace(data, cause, effect)
    ace_with_pt = []

    for _ in range(n_iterations):
        placebo = create_placebo_treatment(data[cause].values)
        data_with_pt = data.copy()
        data_with_pt[cause] = placebo
        new_ace = calculate_ace(data_with_pt, cause, effect)
        ace_with_pt.append(new_ace)

    placebo_mean = np.mean(ace_with_pt)
    placebo_std = np.std(ace_with_pt)
    z_score = (original_ace - placebo_mean) / (placebo_std + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return p_value


def visualize_robustness_results(robustness_df, config):
    """可视化稳健性测试结果 - 包含所有四个指标"""
    if robustness_df.empty:
        write_log("No robustness results to visualize.", log_file)
        return

    # 创建2x2布局的图
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # 为整个图形添加标题
    fig.suptitle('Robustness Test Results - All Metrics', fontsize=28, fontweight='bold', y=0.98)  # 增大字体

    # 1. CSR vs |DDA|散点图 (左上)
    ax1 = axs[0, 0]
    scatter1 = ax1.scatter(robustness_df['CSR'], robustness_df['DDA'].abs(),
                           s=120, alpha=0.7, c=robustness_df['RCC_score'],
                           cmap='viridis', edgecolors='black', linewidth=1.5)

    # 添加参考线
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='|DDA|=0.5')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='CSR=0.5')
    ax1.axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='CSR=0.6 (threshold)')
    ax1.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='CSR=0.4 (threshold)')

    # 增大标签字体
    ax1.set_xlabel('CSR (Causal Stability Ratio)', fontsize=18, fontweight='bold')  # 增大字体
    ax1.set_ylabel('|DDA| (Directional Dependence)', fontsize=18, fontweight='bold')  # 增大字体
    ax1.set_title('Causal Stability vs Directional Dependence', fontsize=20, fontweight='bold')  # 增大字体
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # 增大坐标轴刻度标签
    ax1.tick_params(axis='both', labelsize=18)  # 增大刻度标签

    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('RCC Score', fontsize=18, fontweight='bold')  # 增大字体
    cbar1.ax.tick_params(labelsize=18)  # 增大色条刻度字体

    # 2. RCC vs PT p-value散点图 (右上)
    ax2 = axs[0, 1]
    scatter2 = ax2.scatter(robustness_df['RCC_score'],
                           -np.log10(robustness_df['PT_p_value'].clip(1e-10, 1)),
                           s=120, alpha=0.7, c=robustness_df['CSR'],
                           cmap='coolwarm', edgecolors='black', linewidth=1.5)

    # 添加参考线
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='RCC=0.5 (threshold)')

    # 增大标签字体
    ax2.set_xlabel('RCC Score (Robustness)', fontsize=18, fontweight='bold')  # 增大字体
    ax2.set_ylabel('-log10(PT p-value)', fontsize=18, fontweight='bold')  # 增大字体
    ax2.set_title('RCC Robustness vs Placebo Test Significance', fontsize=20, fontweight='bold')  # 增大字体
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(fontsize=18)  # 增大图例字体

    # 增大坐标轴刻度标签
    ax2.tick_params(axis='both', labelsize=18)  # 增大刻度标签

    # 添加颜色条
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('CSR', fontsize=18, fontweight='bold')  # 增大字体
    cbar2.ax.tick_params(labelsize=18)  # 增大色条刻度字体

    # 3. 所有指标的箱线图 (左下)
    ax3 = axs[1, 0]

    # 准备数据
    box_data = [
        robustness_df['CSR'].values,
        robustness_df['DDA'].abs().values,
        robustness_df['RCC_score'].values,
        1 - robustness_df['PT_p_value'].values  # 转换为1-p以便更好地展示
    ]

    box = ax3.boxplot(box_data, labels=['CSR', '|DDA|', 'RCC', '1-PT_p'],
                      patch_artist=True, showmeans=True, meanline=True)

    # 设置箱线图颜色
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # 添加阈值线
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='0.95 (p=0.05)')

    # 增大标签字体
    ax3.set_ylabel('Score Value', fontsize=18, fontweight='bold')  # 增大字体
    ax3.set_title('Distribution of All Robustness Metrics', fontsize=20, fontweight='bold')  # 增大字体
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(fontsize=18)  # 增大图例字体

    # 增大坐标轴刻度标签
    ax3.tick_params(axis='both', labelsize=18)  # 增大刻度标签

    # 4. 相关性热力图 (右下)
    ax4 = axs[1, 1]

    # 计算相关性矩阵
    metrics_df = robustness_df[['CSR', 'DDA', 'RCC_score', 'PT_p_value']].copy()
    metrics_df['|DDA|'] = metrics_df['DDA'].abs()
    metrics_df['-log10(PT_p)'] = -np.log10(metrics_df['PT_p_value'].clip(1e-10, 1))
    metrics_df = metrics_df[['CSR', '|DDA|', 'RCC_score', '-log10(PT_p)']]

    corr_matrix = metrics_df.corr()

    # 绘制热力图
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 设置刻度
    ax4.set_xticks(np.arange(len(corr_matrix.columns)))
    ax4.set_yticks(np.arange(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, fontsize=18)  # 增大刻度标签字体
    ax4.set_yticklabels(corr_matrix.columns, fontsize=18)  # 增大刻度标签字体

    # 旋转x轴标签
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标注 - 增大字体
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=18)  # 增大字体

    # 增大标题字体
    ax4.set_title('Correlation Matrix of Robustness Metrics', fontsize=20, fontweight='bold')  # 增大字体

    # 添加颜色条
    cbar4 = plt.colorbar(im, ax=ax4)
    cbar4.set_label('Correlation Coefficient', fontsize=16, fontweight='bold')  # 增大字体
    cbar4.ax.tick_params(labelsize=18)  # 增大色条刻度字体

    # 添加整体统计信息 - 增大字体
    textstr = f'Total Relationships: {len(robustness_df)}\n'
    textstr += f'Robust (all criteria): {len(robustness_df[(robustness_df["RCC_score"] > 0.5) & (robustness_df["PT_p_value"] < 0.05) & ((robustness_df["CSR"] > 0.6) | (robustness_df["CSR"] < 0.4))])}'

    fig.text(0.02, 0.02, textstr, transform=fig.transFigure, fontsize=18,  # 增大字体
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    robustness_viz_path = get_output_path("robustness_tests_visualization.png")
    plt.savefig(robustness_viz_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Robustness visualization saved to: {robustness_viz_path}", log_file)

    # 额外：创建一个综合评分雷达图
    create_robustness_radar_chart(robustness_df, config)


def create_robustness_radar_chart(robustness_df, config):
    """创建稳健性指标的雷达图"""
    if robustness_df.empty:
        return

    # 选择前10个最强的因果关系（如果有的话）
    if len(robustness_df) > 10:
        # 计算综合得分并排序
        robustness_df['composite_score'] = (
                robustness_df['RCC_score'] *
                (1 - robustness_df['PT_p_value']) *
                robustness_df['CSR'].apply(lambda x: max(x, 1 - x))  # 考虑CSR的方向性
        )
        top_relationships = robustness_df.nlargest(10, 'composite_score')
    else:
        top_relationships = robustness_df

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    # 设置指标
    metrics = ['CSR', '|DDA|', 'RCC', '1-PT_p']
    num_vars = len(metrics)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 绘制每个因果关系
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_relationships)))

    for idx, (_, row) in enumerate(top_relationships.iterrows()):
        values = [
            row['CSR'],
            abs(row['DDA']),
            row['RCC_score'],
            1 - row['PT_p_value']
        ]
        values += values[:1]  # 闭合图形

        label = f"{row['cause']}→{row['effect']} (lag={row['lag']})"
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=label, alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # 设置刻度和标签
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

    # 添加参考线
    for value in [0.5, 0.95]:
        ax.plot(angles, [value] * len(angles), 'k--', linewidth=1, alpha=0.3)

    # 设置标题和图例
    ax.set_title('Robustness Metrics Radar Chart\n(Top 10 Causal Relationships)',
                 fontsize=18, fontweight='bold', pad=20)

    # 调整图例位置
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()

    # 保存雷达图
    radar_path = get_output_path("robustness_radar_chart.png")
    plt.savefig(radar_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Robustness radar chart saved to: {radar_path}", log_file)


# ------------------------------ M. 因果图重构 ------------------------------
def reconstruct_causal_graph(robustness_df, pfi_scores, selected_features, config):
    write_log("Reconstructing final causal graph...", log_file)

    # 如果稳健性结果为空，则返回
    if robustness_df.empty:
        write_log("No robustness results for causal graph reconstruction.", log_file)
        return None

    # 筛选稳健的因果关系
    robust_causal = robustness_df[
        (robustness_df['RCC_score'] > 0.5) &  # RCC稳健性分数大于0.5
        (robustness_df['PT_p_value'] < 0.05) &  # 安慰剂测试p值小于0.05
        ((robustness_df['CSR'] > 0.6) | (robustness_df['CSR'] < 0.4))  # CSR指示明确方向
        ].copy()

    if robust_causal.empty:
        write_log("No robust causal relationships found after filtering.", log_file)
        return None

    # 筛选选定特征之间的因果关系
    if selected_features:
        # 确保目标变量在特征列表中
        if config.target_col not in selected_features:
            selected_features.append(config.target_col)
            write_log(f"Added target variable '{config.target_col}' to selected features for causal filtering",
                      log_file)

        original_robust_causal = robust_causal.copy()  # 保存原始筛选结果

        robust_causal = robust_causal[
            (robust_causal['cause'].isin(selected_features)) &
            (robust_causal['effect'].isin(selected_features))
            ]

        write_log(f"Filtered causal relationships to only include selected features. "
                  f"Remaining relationships: {len(robust_causal)}", log_file)

        # 检查目标变量是否在筛选后的因果关系中
        has_target = (config.target_col in robust_causal['cause'].values or
                      config.target_col in robust_causal['effect'].values)

        if not has_target and not robust_causal.empty:
            # 找出与目标变量相关的关系
            target_relations = original_robust_causal[
                (original_robust_causal['cause'] == config.target_col) |
                (original_robust_causal['effect'] == config.target_col)
                ]
            if not target_relations.empty:
                # 添加最强的目标变量关系
                strongest_relation = target_relations.sort_values('RCC_score', ascending=False).iloc[0:1]
                robust_causal = pd.concat([robust_causal, strongest_relation], ignore_index=True)
                write_log(f"Added strongest relationship involving target variable to ensure its inclusion", log_file)

        if robust_causal.empty:
            write_log("No robust causal relationships among selected features.", log_file)

            # 如果没有关系，尝试放宽条件找出与目标变量相关的关系
            target_relations = robustness_df[
                ((robustness_df['cause'] == config.target_col) | (robustness_df['effect'] == config.target_col)) &
                (robustness_df['RCC_score'] > 0.3)  # 放宽RCC标准
                ].sort_values('RCC_score', ascending=False)

            if not target_relations.empty:
                robust_causal = target_relations.head(2).copy()  # 取最强的两个关系
                write_log(f"Using relaxed criteria, found {len(robust_causal)} relationships involving target variable",
                          log_file)
            else:
                return None

    # 合并PFI重要性分数
    if not pfi_scores.empty:
        pfi_dict = dict(zip(pfi_scores['feature'], pfi_scores['pfi_score']))

        # 添加原因和结果的PFI分数
        robust_causal['cause_pfi'] = robust_causal['cause'].map(lambda x: pfi_dict.get(x, 0))
        robust_causal['effect_pfi'] = robust_causal['effect'].map(lambda x: pfi_dict.get(x, 0))

        # 综合考虑PFI和稳健性，计算最终因果强度
        robust_causal['causal_strength'] = (
                robust_causal['RCC_score'] *
                (1 - robust_causal['PT_p_value']) *
                robust_causal['cause_pfi']
        )
    else:
        # 如果没有PFI分数，只使用稳健性指标
        robust_causal['causal_strength'] = robust_causal['RCC_score'] * (1 - robust_causal['PT_p_value'])

    # 移除指向汇节点的边
    if hasattr(config, 'sink_nodes') and config.sink_nodes:
        orig_len = len(robust_causal)
        robust_causal = robust_causal[~robust_causal['effect'].isin(config.sink_nodes)]
        filtered_count = orig_len - len(robust_causal)
        if filtered_count > 0:
            write_log(f"Removed {filtered_count} causal relationships pointing to sink nodes.", log_file)

    # a. 创建最终因果图
    final_graph = visualize_final_causal_graph(robust_causal, config)

    # b. 创建最强因果关系表
    top_causal = robust_causal.sort_values('causal_strength', ascending=False)
    top_causal_path = get_output_path("top_causal_relationships.csv")
    top_causal.to_csv(top_causal_path, index=False)
    write_log(f"Top causal relationships saved to: {top_causal_path}", log_file)

    # c. 梳理各项因果性质
    write_log("\nFinal Causal Relationships:", log_file)
    write_log("-" * 80, log_file)

    for idx, row in top_causal.iterrows():
        write_log(f"Relationship {idx + 1}: {row['cause']} → {row['effect']} (lag={row['lag']})", log_file)
        write_log(f"  Causal Strength: {row['causal_strength']:.4f}", log_file)
        write_log(f"  RCC Score: {row['RCC_score']:.4f}", log_file)
        write_log(f"  Placebo Test p-value: {row['PT_p_value']:.4f}", log_file)
        write_log(f"  CSR: {row['CSR']:.4f}", log_file)
        write_log(f"  DDA: {row['DDA']:.4f}", log_file)
        if 'cause_pfi' in row:
            write_log(f"  Cause PFI: {row['cause_pfi']:.4f}", log_file)
        if 'effect_pfi' in row:
            write_log(f"  Effect PFI: {row['effect_pfi']:.4f}", log_file)
        write_log("-" * 40, log_file)

    return robust_causal


def visualize_final_causal_graph(robust_causal, config):
    """使用与PCMCI7兼容的参数可视化最终因果图"""
    if robust_causal.empty:
        write_log("No robust causal relationships to visualize.", log_file)
        return None

    # 获取所有唯一的节点
    all_nodes = list(set(robust_causal['cause'].unique()) | set(robust_causal['effect'].unique()))
    n_vars = len(all_nodes)
    var_names = all_nodes

    # 确定最大时滞
    if len(robust_causal) > 0:
        tau_max = int(robust_causal['lag'].max())
    else:
        tau_max = 0

    # 创建矩阵
    val_matrix = np.zeros((n_vars, n_vars, tau_max + 1))
    graph = np.full((n_vars, n_vars, tau_max + 1), "", dtype='<U3')
    link_width = np.ones((n_vars, n_vars, tau_max + 1)) * config.link_width

    # 填充矩阵
    for _, row in robust_causal.iterrows():
        i = var_names.index(row['cause'])
        j = var_names.index(row['effect'])
        tau = int(row['lag'])
        val_matrix[i, j, tau] = row['causal_strength']
        graph[i, j, tau] = "-->"

    # 使用与PCMCI7.py完全相同的参数调用plot_graph函数
    fig = plt.figure(figsize=(14, 12))
    tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_width=link_width,
        node_size=config.node_size,
        link_colorbar_label='Causal Strength',
        arrow_linewidth=6,
        figsize=(14, 12),
        show_autodependency_lags=False,
        label_fontsize=18,  # 控制标签和colorbar标签的字体大小
        tick_label_size=16,  # 控制刻度标签的字体大小
        node_label_size=18,  # 控制节点标签的字体大小
        link_label_fontsize=18,  # 控制连接线上的标签字体大小
        curved_radius=0.2,
        alpha=1.0,  # 透明度
        cmap_edges='Blues',  # 连接线的colormap
        cmap_nodes='Oranges',  # 节点的colormap
        show_colorbar=True,  # 只显示边的colorbar，不显示节点的
        vmin_edges=-1,  # 连接线colorbar的最小值
        vmax_edges=1.0,  # 连接线colorbar的最大值
        edge_ticks=0.4,  # 连接线colorbar的刻度间隔
        arrowhead_size=40,  # 箭头大小
        save_name=get_output_path("final_causal_graph.png")
    )
    plt.close()

    write_log(f"Final causal graph saved to {get_output_path('final_causal_graph.png')}", log_file)

    # 为了与其他函数保持一致性，返回一个NetworkX图对象
    G = nx.DiGraph()
    for _, row in robust_causal.iterrows():
        G.add_edge(row['cause'], row['effect'],
                   lag=row['lag'],
                   weight=row['causal_strength'])

    return G


# ------------------------------ N. 多维度评估报告 ------------------------------
def generate_comprehensive_report(data, normalized_data, target, features, pcmci_results, enhanced_links,
                                  causal_effects, pfi_scores, selected_features, robustness_df, robust_causal, config):
    write_log("Generating comprehensive evaluation report...", log_file)

    # 创建评估报告
    report_path = get_output_path("comprehensive_evaluation_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===============================================================\n")
        f.write("              COMPREHENSIVE CAUSAL ANALYSIS REPORT              \n")
        f.write("===============================================================\n\n")

        # 1. 数据概述
        f.write("1. DATA OVERVIEW\n")
        f.write("----------------\n")
        f.write(f"Dataset: {config.file_name}\n")
        f.write(f"Number of samples: {data.shape[0]}\n")
        f.write(f"Number of features: {data.shape[1]}\n")
        f.write(f"Target variable: {config.target_col}\n\n")

        # 2. PCMCI初步因果发现
        f.write("2. PCMCI INITIAL CAUSAL DISCOVERY\n")
        f.write("--------------------------------\n")

        if pcmci_results is not None:
            p_matrix = pcmci_results['p_matrix']
            val_matrix = pcmci_results['val_matrix']
            n_vars = p_matrix.shape[0]
            n_significant = np.sum(p_matrix < config.alpha_level)

            f.write(f"Number of variables tested: {n_vars}\n")
            f.write(f"Maximum time lag: {config.tau_max}\n")
            f.write(f"Alpha level: {config.alpha_level}\n")
            f.write(f"Number of significant links found: {n_significant}\n\n")
        else:
            f.write("PCMCI analysis not performed or failed.\n\n")

        # 3. 因果稀疏性增强
        f.write("3. CAUSALITY ENHANCEMENT WITH L1 REGULARIZATION\n")
        f.write("---------------------------------------------\n")

        if enhanced_links:
            f.write(f"Number of enhanced causal links: {len(enhanced_links)}\n")
            f.write("Top 5 strongest Granger causal relationships:\n")

            # 按Granger强度排序
            sorted_links = sorted(enhanced_links, key=lambda x: x['granger_strength'], reverse=True)[:5]
            for i, link in enumerate(sorted_links):
                f.write(
                    f"  {i + 1}. {link['from']} → {link['to']} (lag={link['lag']}, strength={link['granger_strength']:.4f})\n")
            f.write("\n")
        else:
            f.write("No enhanced causal links found.\n\n")

        # 4. 因果效应分析
        f.write("4. CAUSAL EFFECTS ANALYSIS\n")
        f.write("-------------------------\n")

        if causal_effects:
            # 按ACE绝对值排序
            sorted_effects = sorted(causal_effects, key=lambda x: abs(x['ACE']), reverse=True)[:5]
            f.write("Top 5 strongest causal effects (by |ACE|):\n")
            for i, effect in enumerate(sorted_effects):
                f.write(f"  {i + 1}. {effect['cause']} → {effect['effect']} (lag={effect['lag']})\n")
                f.write(f"     ACE: {effect['ACE']:.4f}, RACE: {effect['RACE']:.4f}\n")
            f.write("\n")
        else:
            f.write("No causal effects analyzed.\n\n")

        # 5. TCN-PFI重要性分析
        f.write("5. TCN MODEL WITH PFI IMPORTANCE ANALYSIS\n")
        f.write("---------------------------------------\n")

        if not pfi_scores.empty:
            f.write("Top 10 most important features (by PFI score):\n")
            top_pfi = pfi_scores.sort_values('pfi_score').head(10)  # 按PFI升序排列
            for i, (_, row) in enumerate(top_pfi.iterrows()):
                f.write(f"  {i + 1}. {row['feature']}: {row['pfi_score']:.4f}\n")

            f.write(f"\nFeature Selection Method: Percentile ({config.pfi_percentile}%)\n")
            f.write(f"Selected Features ({len(selected_features)} features):\n")
            for i, feature in enumerate(selected_features):
                # 安全地检查特征是否在PFI分数中
                matched_rows = pfi_scores[pfi_scores['feature'] == feature]
                if not matched_rows.empty:
                    pfi_score = matched_rows['pfi_score'].values[0]
                    f.write(f"  {i + 1}. {feature}: {pfi_score:.4f}\n")
                else:
                    # 特殊处理目标变量或不在PFI分析中的特征
                    if feature == config.target_col:
                        f.write(f"  {i + 1}. {feature}: (target variable)\n")
                    else:
                        f.write(f"  {i + 1}. {feature}: (not in PFI analysis)\n")
            f.write("\n")
        else:
            f.write("PFI analysis not performed or no results available.\n\n")

        # 6. 稳健性测试
        f.write("6. ROBUSTNESS TESTS\n")
        f.write("------------------\n")

        if not robustness_df.empty:
            f.write(f"Number of causal relationships tested: {len(robustness_df)}\n")

            # 稳健性统计
            f.write("\nRobustness Statistics:\n")
            f.write(f"  Average RCC Score: {robustness_df['RCC_score'].mean():.4f}\n")
            f.write(f"  Average CSR: {robustness_df['CSR'].mean():.4f}\n")
            f.write(f"  Average |DDA|: {robustness_df['DDA'].abs().mean():.4f}\n")

            # 显著通过安慰剂测试的关系数量
            significant_pt = sum(robustness_df['PT_p_value'] < 0.05)
            f.write(f"  Relationships passing Placebo Test (p<0.05): {significant_pt} "
                    f"({100 * significant_pt / len(robustness_df):.1f}%)\n\n")
        else:
            f.write("Robustness tests not performed or no results available.\n\n")

        # 7. 最终因果结构
        f.write("7. FINAL CAUSAL STRUCTURE\n")
        f.write("------------------------\n")

        if robust_causal is not None and not robust_causal.empty:
            f.write(f"Number of robust causal relationships: {len(robust_causal)}\n\n")

            # 按因果强度排序
            top_robust = robust_causal.sort_values('causal_strength', ascending=False)

            f.write("Final Causal Relationships (by causal strength):\n")
            for i, (_, row) in enumerate(top_robust.iterrows()):
                f.write(f"  {i + 1}. {row['cause']} → {row['effect']} (lag={row['lag']})\n")
                f.write(f"     Causal Strength: {row['causal_strength']:.4f}\n")
                f.write(f"     RCC Score: {row['RCC_score']:.4f}, PT p-value: {row['PT_p_value']:.4f}\n")
                f.write(f"     CSR: {row['CSR']:.4f}, DDA: {row['DDA']:.4f}\n")

            # 分析目标变量的主要影响因素
            target_effects = robust_causal[robust_causal['effect'] == config.target_col]

            if not target_effects.empty:
                f.write(f"\nKey Influencers of {config.target_col}:\n")
                target_influencers = target_effects.sort_values('causal_strength', ascending=False)
                for i, (_, row) in enumerate(target_influencers.iterrows()):
                    f.write(f"  {i + 1}. {row['cause']} (lag={row['lag']}, strength={row['causal_strength']:.4f})\n")
            else:
                f.write(f"\nNo direct causal influences found for {config.target_col}.\n")

            f.write("\n")
        else:
            f.write("No robust causal relationships identified in the final analysis.\n\n")

        # 8. 结论和建议
        f.write("8. CONCLUSIONS AND RECOMMENDATIONS\n")
        f.write("---------------------------------\n")

        if robust_causal is not None and not robust_causal.empty:
            # 分析目标变量关系
            target_causes = robust_causal[robust_causal['effect'] == config.target_col]
            target_effects = robust_causal[robust_causal['cause'] == config.target_col]

            if not target_causes.empty:
                f.write(f"Primary causal factors affecting {config.target_col}:\n")
                for i, (_, row) in enumerate(target_causes.sort_values('causal_strength', ascending=False).iterrows()):
                    f.write(f"  {i + 1}. {row['cause']} (strength={row['causal_strength']:.4f}, lag={row['lag']})\n")
                f.write("\n")

            if not target_effects.empty:
                f.write(f"Variables influenced by {config.target_col}:\n")
                for i, (_, row) in enumerate(target_effects.sort_values('causal_strength', ascending=False).iterrows()):
                    f.write(f"  {i + 1}. {row['effect']} (strength={row['causal_strength']:.4f}, lag={row['lag']})\n")
                f.write("\n")

            # 识别关键因果路径
            f.write("Key causal pathways identified:\n")
            G = nx.DiGraph()
            for _, row in robust_causal.iterrows():
                G.add_edge(row['cause'], row['effect'], weight=row['causal_strength'])

            # 尝试找到从各种节点到目标的路径
            if config.target_col in G.nodes():
                paths_to_target = []
                for node in G.nodes():
                    if node != config.target_col:
                        try:
                            for path in nx.all_simple_paths(G, node, config.target_col, cutoff=3):
                                paths_to_target.append(path)
                        except:
                            pass  # 忽略没有路径的情况

                # 输出发现的路径
                if paths_to_target:
                    for i, path in enumerate(paths_to_target[:5]):  # 最多显示5条路径
                        path_str = " → ".join(path)
                        f.write(f"  {i + 1}. {path_str}\n")
                else:
                    f.write("  No clear causal pathways to target identified.\n")
            else:
                f.write("  Target variable not found in causal graph.\n")
        else:
            f.write("Insufficient robust causal relationships to form conclusions.\n")

        f.write("\n")

        # 9. 分析局限性
        f.write("9. LIMITATIONS OF ANALYSIS\n")
        f.write("-------------------------\n")
        f.write("- Time series length may limit the ability to detect longer lag relationships.\n")
        f.write("- Observational data cannot fully confirm causality without experimental validation.\n")
        f.write("- Hidden confounders may still exist despite robustness tests.\n")
        f.write("- Linear assumptions in some tests might not capture complex nonlinear relationships.\n")
        f.write("- Results should be interpreted in context with domain knowledge.\n")

    write_log(f"Comprehensive evaluation report saved to: {report_path}", log_file)

    return report_path


# ------------------------------ 主流程 ------------------------------
def main():
    try:
        config = Config()
        write_log(f"Starting causal analysis pipeline on dataset: {config.file_name}", log_file)
        write_log(f"Parallel computing enabled with {multiprocessing.cpu_count()} cores", log_file)
        write_log(f"Cache enabled: {config.use_cache}", log_file)

        # 1. 数据加载和归一化处理（包含数据质量控制）
        data, normalized_data, time_data, scaler = load_and_normalize_data(config)

        # 2. 确认目标参数
        target, features, correlation_with_target = confirm_target_parameter(data, normalized_data, config)

        # 3. PCMCI因果发现（带缓存）
        pcmci_results, pcmci, var_names, pcmci_graph = perform_pcmci_analysis(normalized_data, config)

        # 4. 因果稀疏性增强 - Granger因果 + L1正则化（并行化）
        enhanced_links, enhanced_graph = granger_causality_with_l1(normalized_data, pcmci_results, var_names, config)

        # 5. ACE/RACE估计
        causal_effects = calculate_causal_effects(normalized_data, enhanced_links, config)

        # 6. TCN-PFI重要性分析 - 使用百分位数方法
        model, pfi_scores, selected_features = train_evaluate_tcn(features, target, config, enhanced_links)

        # 7. 稳健性检验（并行化）
        robustness_df = perform_robustness_tests(normalized_data, enhanced_links, config)

        # 8. 因果图重构
        robust_causal = reconstruct_causal_graph(robustness_df, pfi_scores, selected_features, config)

        # 9. 多维度评估报告
        report_path = generate_comprehensive_report(
            data, normalized_data, target, features, pcmci_results, enhanced_links,
            causal_effects, pfi_scores, selected_features, robustness_df, robust_causal, config
        )

        # 10. 复制当前脚本到输出目录
        try:
            script_path = os.path.realpath(__file__)
            script_name = os.path.basename(script_path)
            script_copy_path = get_output_path(f"{script_name}")
            shutil.copy2(script_path, script_copy_path)
            write_log(f"Script copied to: {script_copy_path}", log_file)
        except Exception as e:
            write_log(f"Warning: Could not copy script file: {e}", log_file)

        write_log("\nCausal analysis pipeline completed successfully!", log_file)
        write_log(f"All results have been saved to: {config.output_dir}", log_file)

    except Exception as e:
        write_log(f"ERROR in main process: {str(e)}", log_file)
        import traceback
        write_log(traceback.format_exc(), log_file)


if __name__ == "__main__":
    main()
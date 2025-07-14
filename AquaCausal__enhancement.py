import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
import os
import datetime
import logging
import pickle
from sklearn.linear_model import LassoCV
import warnings
from joblib import Parallel, delayed
import multiprocessing
import hashlib
from functools import wraps

warnings.filterwarnings('ignore')

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
    pfi_percentile = 90
    n_jobs = -1
    use_cache = True


def find_output_folder():
    base_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("ATTCN_PCMCI_L1_results_")]
    if not base_dirs:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"ATTCN_PCMCI_L1_results_{current_time}"
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "cache"))
        print(f"Created new output directory: {output_dir}")
    else:
        output_dir = sorted(base_dirs, reverse=True)[0]
        print(f"Using existing output directory: {output_dir}")

        if not os.path.exists(os.path.join(output_dir, "cache")):
            os.makedirs(os.path.join(output_dir, "cache"))

    return output_dir


def cache_result(cache_key_prefix):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not Config.use_cache:
                return func(*args, **kwargs)

            cache_key = f"{cache_key_prefix}_{str(args)}_{str(kwargs)}"
            cache_file = os.path.join(config.cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")

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


def get_output_path(filename):
    return os.path.join(config.output_dir, filename)


def granger_causality_with_l1(data, pcmci_results, var_names, config):
    write_log("Performing Granger causality with L1 regularization (parallel)...", log_file)

    p_matrix = pcmci_results['p_matrix']
    val_matrix = pcmci_results['val_matrix']
    n_vars = len(var_names)
    tau_max = p_matrix.shape[2] - 1

    tasks = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                is_pcmci_causal = False
                for tau in range(tau_max + 1):
                    if p_matrix[i, j, tau] < config.alpha_level:
                        is_pcmci_causal = True
                        break

                if is_pcmci_causal:
                    tasks.append((i, j, var_names, data, val_matrix, tau_max, config))

    n_cores = multiprocessing.cpu_count() if config.n_jobs == -1 else config.n_jobs
    results = Parallel(n_jobs=n_cores)(
        delayed(test_granger_causality_single)(task) for task in tasks
    )

    enhanced_links = []
    for result in results:
        if result is not None:
            enhanced_links.extend(result)

    enhanced_df = pd.DataFrame(enhanced_links)
    if not enhanced_df.empty:
        enhanced_path = get_output_path("enhanced_causal_links.csv")
        enhanced_df.to_csv(enhanced_path, index=False)
        write_log(f"Enhanced causal links saved to: {enhanced_path}", log_file)
        write_log(f"Found {len(enhanced_links)} enhanced causal relationships after L1 regularization", log_file)
    else:
        write_log("No significant causal relationships found after L1 regularization", log_file)

    enhanced_graph = visualize_enhanced_causal_graph(enhanced_links, var_names, config)

    return enhanced_links, enhanced_graph


def test_granger_causality_single(args):
    i, j, var_names, data, val_matrix, tau_max, config = args

    write_log(f"Testing Granger causality from {var_names[i]} to {var_names[j]}...", log_file)

    y = data[var_names[j]].values
    x = data[var_names[i]].values

    lag_features = []
    for lag in range(1, config.max_lags + 1):
        y_lag = np.roll(y, lag)
        y_lag[:lag] = y_lag[lag]
        lag_features.append(y_lag)

        x_lag = np.roll(x, lag)
        x_lag[:lag] = x_lag[lag]
        lag_features.append(x_lag)

    X = np.column_stack(lag_features)

    lasso = LassoCV(cv=5, random_state=config.random_state)
    lasso.fit(X[config.max_lags:], y[config.max_lags:])

    x_lag_coeffs = lasso.coef_[1::2]

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
    write_log("Visualizing enhanced causal graph with tigramite-inspired style...", log_file)

    if not enhanced_links:
        write_log("No enhanced links to visualize.", log_file)
        return nx.DiGraph()

    n_vars = len(var_names)

    if enhanced_links:
        tau_max = max([link['lag'] for link in enhanced_links])
    else:
        tau_max = 0

    G = nx.DiGraph()
    for link in enhanced_links:
        G.add_edge(link['from'], link['to'],
                   lag=link['lag'],
                   weight=link['granger_strength'],
                   coefficient=link['coefficient'])

    enhanced_links_path = get_output_path("enhanced_links.pkl")
    with open(enhanced_links_path, 'wb') as f:
        pickle.dump(enhanced_links, f)
    write_log(f"Enhanced links saved to: {enhanced_links_path}", log_file)

    return G


def calculate_ace(data, cause, effect, lag=0):
    cause_data = data[cause].values

    if lag > 0 and lag < len(cause_data):
        effect_data = data[effect].values[lag:]
        cause_data = cause_data[:-lag]
    else:
        effect_data = data[effect].values

    threshold = np.median(cause_data)
    high_cause = effect_data[cause_data > threshold]
    low_cause = effect_data[cause_data <= threshold]

    ace = np.mean(high_cause) - np.mean(low_cause)
    return ace


def calculate_race(data, cause, effect, lag=0):
    ace = calculate_ace(data, cause, effect, lag)

    if lag > 0 and lag < len(data):
        effect_std = np.std(data[effect].values[lag:])
    else:
        effect_std = np.std(data[effect].values)

    race = ace / effect_std if effect_std != 0 else 0
    return race


def calculate_causal_effects(data, enhanced_links, config):
    write_log("Calculating causal effects (ACE/RACE)...", log_file)

    causal_effects = []

    for link in enhanced_links:
        cause = link['from']
        effect = link['to']
        lag = link['lag']

        ace = calculate_ace(data, cause, effect, lag)
        race = calculate_race(data, cause, effect, lag)

        causal_effects.append({
            'cause': cause,
            'effect': effect,
            'lag': lag,
            'ACE': ace,
            'RACE': race,
            'granger_strength': link['granger_strength']
        })

    effects_df = pd.DataFrame(causal_effects)
    if not effects_df.empty:
        effects_path = get_output_path("causal_effects.csv")
        effects_df.to_csv(effects_path, index=False)
        write_log(f"Causal effects saved to: {effects_path}", log_file)
    else:
        write_log("No causal effects to calculate.", log_file)

    return causal_effects


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
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_evaluate_tcn(features, target, config, enhanced_links):
    write_log("Training TCN model with attention and performing PFI analysis...", log_file)

    X = features.values
    y = target.values.reshape(-1, 1)

    X_seq, y_seq = create_sequences(X, y, seq_length=config.seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=config.test_size, random_state=config.random_state
    )

    write_log(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}", log_file)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Using device: {device}", log_file)

    input_dim = X_train.shape[2]
    output_dim = 1

    model = ATTTCN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_channels=config.num_channels,
        kernel_size=config.kernel_size,
        num_heads=config.num_heads
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

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

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            write_log(f"Epoch {epoch + 1}/{config.epochs}, "
                      f"Train Loss: {train_loss / len(train_loader):.4f}, "
                      f"Test Loss: {test_loss / len(test_loader):.4f}, "
                      f"R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}", log_file)

    model_path = get_output_path("attcn_model.pth")
    torch.save(model.state_dict(), model_path)
    write_log(f"Model saved to: {model_path}", log_file)

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

    pfi_scores = calculate_pfi(model, X_test_tensor=torch.tensor(X_test, dtype=torch.float32).to(device),
                               y_test_tensor=torch.tensor(y_test, dtype=torch.float32).to(device),
                               feature_names=features.columns, device=device)

    selected_features, pfi_threshold = select_features_by_pfi_percentile(pfi_scores, features.columns, config)

    visualize_pfi_results(pfi_scores, features.columns, config, enhanced_links, selected_features, pfi_threshold)

    return model, pfi_scores, selected_features


def calculate_pfi(model, X_test_tensor, y_test_tensor, feature_names, device):
    model.eval()

    with torch.no_grad():
        original_pred, _ = model(X_test_tensor)
        original_loss = mean_squared_error(y_test_tensor.cpu().numpy(), original_pred.cpu().numpy())

    pfi_results = []

    for i in range(X_test_tensor.shape[2]):
        X_permuted = X_test_tensor.clone()
        permuted_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, :, i] = X_test_tensor[permuted_idx, :, i]

        with torch.no_grad():
            perm_pred, _ = model(X_permuted)
            perm_loss = mean_squared_error(y_test_tensor.cpu().numpy(), perm_pred.cpu().numpy())

        pfi_ratio = original_loss / perm_loss

        pfi_results.append({
            'feature': feature_names[i],
            'pfi_score': pfi_ratio
        })

    pfi_df = pd.DataFrame(pfi_results)

    pfi_path = get_output_path("pfi_scores.csv")
    pfi_df.to_csv(pfi_path, index=False)
    write_log(f"PFI scores saved to: {pfi_path}", log_file)

    return pfi_df


def select_features_by_pfi_percentile(pfi_df, feature_names, config):
    pfi_scores = pfi_df['pfi_score'].values

    pfi_threshold = np.percentile(pfi_scores, config.pfi_percentile)
    write_log(f"PFI threshold for feature selection (percentile {config.pfi_percentile}): {pfi_threshold:.4f}",
              log_file)

    selected_features = pfi_df[pfi_df['pfi_score'] < pfi_threshold]['feature'].tolist()

    for f in config.whitelist_features:
        if f in feature_names and f not in selected_features:
            selected_features.append(f)
            write_log(f"Added whitelist feature '{f}' to selected features", log_file)

    if len(selected_features) < 5:
        selected_features = pfi_df.nsmallest(10, 'pfi_score')['feature'].tolist()
        write_log(f"Too few features selected, using top-10 features", log_file)

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

    selected_features_path = get_output_path("selected_features.pkl")
    with open(selected_features_path, 'wb') as f:
        pickle.dump({'selected_features': selected_features, 'pfi_threshold': pfi_threshold}, f)
    write_log(f"Selected features saved to: {selected_features_path}", log_file)

    return selected_features, pfi_threshold


def visualize_pfi_results(pfi_df, feature_names, config, enhanced_links, selected_features, pfi_threshold):
    sorted_pfi = pfi_df.sort_values('pfi_score', ascending=True).copy()

    colors = ['green' if feat in selected_features else 'gray'
              for feat in sorted_pfi['feature']]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(sorted_pfi['feature'], sorted_pfi['pfi_score'],
                    color=colors, edgecolor='black', linewidth=1)

    plt.axvline(x=pfi_threshold, color='crimson', linestyle='--', linewidth=2,
                label=f'Threshold ({pfi_threshold:.4f})')

    plt.title('Permutation Feature Importance', fontsize=24, fontweight='bold')
    plt.xlabel('PFI Ratio (original loss / permuted loss)', fontsize=20, fontweight='bold')
    plt.ylabel('Feature', fontsize=20, fontweight='bold')

    for i, v in enumerate(sorted_pfi['pfi_score']):
        plt.text(v + 0.02, i, f"{v:.4f}", va='center', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='crimson', linestyle='--', linewidth=2,
               label=f'Threshold ({pfi_threshold:.4f})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=16,
               label='Selected features'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=16,
               label='Excluded features')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=16)

    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    pfi_plot_path = get_output_path("pfi_importance.png")
    plt.savefig(pfi_plot_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"PFI visualization saved to: {pfi_plot_path}", log_file)

    plt.figure(figsize=(10, 6))
    plt.hist(pfi_df['pfi_score'], bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(pfi_df['pfi_score']), color='red', linestyle='--',
                label=f'Mean ({np.mean(pfi_df["pfi_score"]):.4f})')
    plt.axvline(x=np.median(pfi_df['pfi_score']), color='blue', linestyle='-.',
                label=f'Median ({np.median(pfi_df["pfi_score"]):.4f})')
    plt.axvline(x=pfi_threshold, color='crimson', linestyle='-',
                label=f'Threshold ({pfi_threshold:.4f})')

    plt.title('Distribution of PFI Scores', fontsize=24, fontweight='bold')
    plt.xlabel('PFI Ratio', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    pfi_hist_path = get_output_path("pfi_distribution.png")
    plt.savefig(pfi_hist_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"PFI distribution saved to: {pfi_hist_path}", log_file)


def main():
    try:
        global config, log_file
        output_dir = find_output_folder()
        log_file = os.path.join(output_dir, "analysis_log.txt")

        config = Config()
        config.output_dir = output_dir
        config.cache_dir = os.path.join(output_dir, "cache")

        write_log("\n" + "=" * 50, log_file)
        write_log("Starting Phase 2: L1 Granger causality and PFI feature selection", log_file)
        write_log("=" * 50 + "\n", log_file)

        normalized_data = pd.read_csv(os.path.join(output_dir, "normalized_data_for_next_step.csv"))

        with open(os.path.join(output_dir, "pcmci_results.pkl"), 'rb') as f:
            pcmci_data = pickle.load(f)
            pcmci_results = pcmci_data['pcmci_results']
            var_names = pcmci_data['var_names']

        write_log(f"Loaded normalized data with shape: {normalized_data.shape}", log_file)
        write_log(f"Loaded PCMCI results with {len(var_names)} variables", log_file)

        target = normalized_data[config.target_col]
        features = normalized_data.drop(config.target_col, axis=1)

        enhanced_links, enhanced_graph = granger_causality_with_l1(normalized_data, pcmci_results, var_names, config)

        causal_effects = calculate_causal_effects(normalized_data, enhanced_links, config)

        model, pfi_scores, selected_features = train_evaluate_tcn(features, target, config, enhanced_links)

        write_log("\nPhase 2 completed successfully!", log_file)
        write_log(f"All results have been saved to: {config.output_dir}", log_file)
        write_log(f"Proceed to final phase for robustness tests and final causal graph", log_file)

    except Exception as e:
        write_log(f"ERROR in main process: {str(e)}", log_file)
        import traceback
        write_log(traceback.format_exc(), log_file)


if __name__ == "__main__":
    main()
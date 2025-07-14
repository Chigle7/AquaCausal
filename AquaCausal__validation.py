import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from scipy import stats
import os
import datetime
import pickle
import warnings
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns

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
    n_iterations = 1000
    node_size = 0.3
    link_width = 1
    significant_threshold = 0.05
    whitelist_features = []
    sink_nodes = ['Flow_in', 'TN_in', 'NH4_in', 'COD_in', 'TP_in', 'SS_in', 'DCS', 'DPRA', 'IR',
                  'ER', 'Was', 'DO_1', 'DO_2', 'DO_3']
    n_jobs = -1


def find_output_folder():
    base_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("ATTCN_PCMCI_L1_results_")]
    if not base_dirs:
        raise ValueError("No output directory found from previous phases")
    output_dir = sorted(base_dirs, reverse=True)[0]
    print(f"Using existing output directory: {output_dir}")
    return output_dir


def write_log(message, log_file):
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)


def get_output_path(filename):
    return os.path.join(config.output_dir, filename)


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


def perform_robustness_tests(data, enhanced_links, config):
    write_log("Performing robustness tests: CSR, DDA, RCC, Placebo Test (parallel)...", log_file)

    tasks = [(data, link, config) for link in enhanced_links]

    n_cores = multiprocessing.cpu_count() if config.n_jobs == -1 else config.n_jobs
    results = Parallel(n_jobs=n_cores)(
        delayed(test_robustness_single)(task) for task in tasks
    )

    robustness_df = pd.DataFrame(results)

    if not robustness_df.empty:
        robustness_path = get_output_path("robustness_results.csv")
        robustness_df.to_csv(robustness_path, index=False)
        write_log(f"Robustness test results saved to: {robustness_path}", log_file)

        visualize_robustness_results(robustness_df, config)
    else:
        write_log("No robustness results to save.", log_file)

    return robustness_df


def test_robustness_single(args):
    data, link, config = args

    cause = link['from']
    effect = link['to']
    lag = link['lag']

    if lag > 0:
        lagged_data = data.copy()
        lagged_data[f"{cause}_lag{lag}"] = lagged_data[cause].shift(lag)
        lagged_data = lagged_data.dropna()
        test_cause = f"{cause}_lag{lag}"
    else:
        lagged_data = data.copy()
        test_cause = cause

    csr = calculate_csr(lagged_data, test_cause, effect, config.n_iterations)
    dda = calculate_dda(lagged_data, test_cause, effect)
    rcc_score = test_robustness_rcc(lagged_data, test_cause, effect, config.n_iterations)
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
    return np.random.normal(0, 1, n_samples)


def create_placebo_treatment(original_treatment):
    return np.random.permutation(original_treatment)


def calculate_csr(data, cause, effect, num_bootstrap=1000):
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
    ace_forward = calculate_ace(data, var1, var2)
    ace_backward = calculate_ace(data, var2, var1)

    denominator = abs(ace_forward) + abs(ace_backward)
    if denominator == 0:
        return 0
    dda = (ace_forward - ace_backward) / denominator
    return dda


def test_robustness_rcc(data, cause, effect, n_iterations=1000):
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
    if robustness_df.empty:
        write_log("No robustness results to visualize.", log_file)
        return

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    fig.suptitle('Robustness Test Results - All Metrics', fontsize=28, fontweight='bold', y=0.98)

    ax1 = axs[0, 0]
    scatter1 = ax1.scatter(robustness_df['CSR'], robustness_df['DDA'].abs(),
                           s=120, alpha=0.7, c=robustness_df['RCC_score'],
                           cmap='viridis', edgecolors='black', linewidth=1.5)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='|DDA|=0.5')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='CSR=0.5')
    ax1.axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='CSR=0.6 (threshold)')
    ax1.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='CSR=0.4 (threshold)')

    ax1.set_xlabel('CSR (Causal Stability Ratio)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('|DDA| (Directional Dependence)', fontsize=18, fontweight='bold')
    ax1.set_title('Causal Stability vs Directional Dependence', fontsize=20, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    ax1.tick_params(axis='both', labelsize=18)

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('RCC Score', fontsize=18, fontweight='bold')
    cbar1.ax.tick_params(labelsize=18)

    ax2 = axs[0, 1]
    scatter2 = ax2.scatter(robustness_df['RCC_score'],
                           -np.log10(robustness_df['PT_p_value'].clip(1e-10, 1)),
                           s=120, alpha=0.7, c=robustness_df['CSR'],
                           cmap='coolwarm', edgecolors='black', linewidth=1.5)

    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='RCC=0.5 (threshold)')

    ax2.set_xlabel('RCC Score (Robustness)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('-log10(PT p-value)', fontsize=18, fontweight='bold')
    ax2.set_title('RCC Robustness vs Placebo Test Significance', fontsize=20, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(fontsize=18)

    ax2.tick_params(axis='both', labelsize=18)

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('CSR', fontsize=18, fontweight='bold')
    cbar2.ax.tick_params(labelsize=18)

    ax3 = axs[1, 0]

    box_data = [
        robustness_df['CSR'].values,
        robustness_df['DDA'].abs().values,
        robustness_df['RCC_score'].values,
        1 - robustness_df['PT_p_value'].values
    ]

    box = ax3.boxplot(box_data, labels=['CSR', '|DDA|', 'RCC', '1-PT_p'],
                      patch_artist=True, showmeans=True, meanline=True)

    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='0.95 (p=0.05)')

    ax3.set_ylabel('Score Value', fontsize=18, fontweight='bold')
    ax3.set_title('Distribution of All Robustness Metrics', fontsize=20, fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(fontsize=18)

    ax3.tick_params(axis='both', labelsize=18)

    ax4 = axs[1, 1]

    metrics_df = robustness_df[['CSR', 'DDA', 'RCC_score', 'PT_p_value']].copy()
    metrics_df['|DDA|'] = metrics_df['DDA'].abs()
    metrics_df['-log10(PT_p)'] = -np.log10(metrics_df['PT_p_value'].clip(1e-10, 1))
    metrics_df = metrics_df[['CSR', '|DDA|', 'RCC_score', '-log10(PT_p)']]

    corr_matrix = metrics_df.corr()

    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax4.set_xticks(np.arange(len(corr_matrix.columns)))
    ax4.set_yticks(np.arange(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, fontsize=18)
    ax4.set_yticklabels(corr_matrix.columns, fontsize=18)

    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=18)

    ax4.set_title('Correlation Matrix of Robustness Metrics', fontsize=20, fontweight='bold')

    cbar4 = plt.colorbar(im, ax=ax4)
    cbar4.set_label('Correlation Coefficient', fontsize=16, fontweight='bold')
    cbar4.ax.tick_params(labelsize=18)

    textstr = f'Total Relationships: {len(robustness_df)}\n'
    textstr += f'Robust (all criteria): {len(robustness_df[(robustness_df["RCC_score"] > 0.5) & (robustness_df["PT_p_value"] < 0.05) & ((robustness_df["CSR"] > 0.6) | (robustness_df["CSR"] < 0.4))])}'

    fig.text(0.02, 0.02, textstr, transform=fig.transFigure, fontsize=18,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    robustness_viz_path = get_output_path("robustness_tests_visualization.png")
    plt.savefig(robustness_viz_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Robustness visualization saved to: {robustness_viz_path}", log_file)

    create_robustness_radar_chart(robustness_df, config)


def create_robustness_radar_chart(robustness_df, config):
    if robustness_df.empty:
        return

    if len(robustness_df) > 10:
        robustness_df['composite_score'] = (
                robustness_df['RCC_score'] *
                (1 - robustness_df['PT_p_value']) *
                robustness_df['CSR'].apply(lambda x: max(x, 1 - x))
        )
        top_relationships = robustness_df.nlargest(10, 'composite_score')
    else:
        top_relationships = robustness_df

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    metrics = ['CSR', '|DDA|', 'RCC', '1-PT_p']
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set3(np.linspace(0, 1, len(top_relationships)))

    for idx, (_, row) in enumerate(top_relationships.iterrows()):
        values = [
            row['CSR'],
            abs(row['DDA']),
            row['RCC_score'],
            1 - row['PT_p_value']
        ]
        values += values[:1]

        label = f"{row['cause']}→{row['effect']} (lag={row['lag']})"
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=label, alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

    for value in [0.5, 0.95]:
        ax.plot(angles, [value] * len(angles), 'k--', linewidth=1, alpha=0.3)

    ax.set_title('Robustness Metrics Radar Chart\n(Top 10 Causal Relationships)',
                 fontsize=18, fontweight='bold', pad=20)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()

    radar_path = get_output_path("robustness_radar_chart.png")
    plt.savefig(radar_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Robustness radar chart saved to: {radar_path}", log_file)


def reconstruct_causal_graph(robustness_df, pfi_scores, selected_features, config):
    write_log("Reconstructing final causal graph...", log_file)

    if robustness_df.empty:
        write_log("No robustness results for causal graph reconstruction.", log_file)
        return None

    robust_causal = robustness_df[
        (robustness_df['RCC_score'] > 0.5) &
        (robustness_df['PT_p_value'] < 0.05) &
        ((robustness_df['CSR'] > 0.6) | (robustness_df['CSR'] < 0.4))
        ].copy()

    if robust_causal.empty:
        write_log("No robust causal relationships found after filtering.", log_file)
        return None

    if selected_features:
        if config.target_col not in selected_features:
            selected_features.append(config.target_col)
            write_log(f"Added target variable '{config.target_col}' to selected features for causal filtering",
                      log_file)

        original_robust_causal = robust_causal.copy()

        robust_causal = robust_causal[
            (robust_causal['cause'].isin(selected_features)) &
            (robust_causal['effect'].isin(selected_features))
            ]

        write_log(f"Filtered causal relationships to only include selected features. "
                  f"Remaining relationships: {len(robust_causal)}", log_file)

        has_target = (config.target_col in robust_causal['cause'].values or
                      config.target_col in robust_causal['effect'].values)

        if not has_target and not robust_causal.empty:
            target_relations = original_robust_causal[
                (original_robust_causal['cause'] == config.target_col) |
                (original_robust_causal['effect'] == config.target_col)
                ]
            if not target_relations.empty:
                strongest_relation = target_relations.sort_values('RCC_score', ascending=False).iloc[0:1]
                robust_causal = pd.concat([robust_causal, strongest_relation], ignore_index=True)
                write_log(f"Added strongest relationship involving target variable to ensure its inclusion", log_file)

        if robust_causal.empty:
            write_log("No robust causal relationships among selected features.", log_file)

            target_relations = robustness_df[
                ((robustness_df['cause'] == config.target_col) | (robustness_df['effect'] == config.target_col)) &
                (robustness_df['RCC_score'] > 0.3)
                ].sort_values('RCC_score', ascending=False)

            if not target_relations.empty:
                robust_causal = target_relations.head(2).copy()
                write_log(f"Using relaxed criteria, found {len(robust_causal)} relationships involving target variable",
                          log_file)
            else:
                return None

    if not pfi_scores.empty:
        pfi_dict = dict(zip(pfi_scores['feature'], pfi_scores['pfi_score']))

        robust_causal['cause_pfi'] = robust_causal['cause'].map(lambda x: pfi_dict.get(x, 0))
        robust_causal['effect_pfi'] = robust_causal['effect'].map(lambda x: pfi_dict.get(x, 0))

        robust_causal['causal_strength'] = (
                robust_causal['RCC_score'] *
                (1 - robust_causal['PT_p_value']) *
                robust_causal['cause_pfi']
        )
    else:
        robust_causal['causal_strength'] = robust_causal['RCC_score'] * (1 - robust_causal['PT_p_value'])

    if hasattr(config, 'sink_nodes') and config.sink_nodes:
        orig_len = len(robust_causal)
        robust_causal = robust_causal[~robust_causal['effect'].isin(config.sink_nodes)]
        filtered_count = orig_len - len(robust_causal)
        if filtered_count > 0:
            write_log(f"Removed {filtered_count} causal relationships pointing to sink nodes.", log_file)

    final_graph = visualize_final_causal_graph(robust_causal, config)

    top_causal = robust_causal.sort_values('causal_strength', ascending=False)
    top_causal_path = get_output_path("top_causal_relationships.csv")
    top_causal.to_csv(top_causal_path, index=False)
    write_log(f"Top causal relationships saved to: {top_causal_path}", log_file)

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
    if robust_causal.empty:
        write_log("No robust causal relationships to visualize.", log_file)
        return None

    all_nodes = list(set(robust_causal['cause'].unique()) | set(robust_causal['effect'].unique()))

    G = nx.DiGraph()
    for _, row in robust_causal.iterrows():
        G.add_edge(row['cause'], row['effect'],
                   lag=row['lag'],
                   weight=row['causal_strength'])

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)

    edges = G.edges(data=True)
    weights = [d['weight'] * 3 for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='navy',
                           arrowsize=20, arrowstyle='-|>', connectionstyle='arc3,rad=0.1')

    edge_labels = {(u, v): f"τ={d['lag']}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.axis('off')
    plt.title('Final Causal Graph', fontsize=20, fontweight='bold')
    plt.tight_layout()

    final_graph_path = get_output_path("final_causal_graph.png")
    plt.savefig(final_graph_path, dpi=800, bbox_inches='tight')
    plt.close()

    write_log(f"Final causal graph saved to {final_graph_path}", log_file)

    return G


def generate_comprehensive_report(data, target, features, enhanced_links, causal_effects,
                                  pfi_scores, selected_features, robustness_df, robust_causal, config):
    write_log("Generating comprehensive evaluation report...", log_file)

    report_path = get_output_path("comprehensive_evaluation_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===============================================================\n")
        f.write("              COMPREHENSIVE CAUSAL ANALYSIS REPORT              \n")
        f.write("===============================================================\n\n")

        f.write("1. DATA OVERVIEW\n")
        f.write("----------------\n")
        f.write(f"Number of samples: {data.shape[0]}\n")
        f.write(f"Number of features: {data.shape[1]}\n")
        f.write(f"Target variable: {config.target_col}\n\n")

        f.write("2. CAUSALITY ENHANCEMENT WITH L1 REGULARIZATION\n")
        f.write("---------------------------------------------\n")

        if enhanced_links:
            f.write(f"Number of enhanced causal links: {len(enhanced_links)}\n")
            f.write("Top 5 strongest Granger causal relationships:\n")

            sorted_links = sorted(enhanced_links, key=lambda x: x['granger_strength'], reverse=True)[:5]
            for i, link in enumerate(sorted_links):
                f.write(
                    f"  {i + 1}. {link['from']} → {link['to']} (lag={link['lag']}, strength={link['granger_strength']:.4f})\n")
            f.write("\n")
        else:
            f.write("No enhanced causal links found.\n\n")

        f.write("3. CAUSAL EFFECTS ANALYSIS\n")
        f.write("-------------------------\n")

        if causal_effects:
            sorted_effects = sorted(causal_effects, key=lambda x: abs(x['ACE']), reverse=True)[:5]
            f.write("Top 5 strongest causal effects (by |ACE|):\n")
            for i, effect in enumerate(sorted_effects):
                f.write(f"  {i + 1}. {effect['cause']} → {effect['effect']} (lag={effect['lag']})\n")
                f.write(f"     ACE: {effect['ACE']:.4f}, RACE: {effect['RACE']:.4f}\n")
            f.write("\n")
        else:
            f.write("No causal effects analyzed.\n\n")

        f.write("4. PFI IMPORTANCE ANALYSIS\n")
        f.write("-------------------------\n")

        if not pfi_scores.empty:
            f.write("Top 10 most important features (by PFI score):\n")
            top_pfi = pfi_scores.sort_values('pfi_score').head(10)
            for i, (_, row) in enumerate(top_pfi.iterrows()):
                f.write(f"  {i + 1}. {row['feature']}: {row['pfi_score']:.4f}\n")

            f.write(f"\nSelected Features ({len(selected_features)} features):\n")
            for i, feature in enumerate(selected_features):
                matched_rows = pfi_scores[pfi_scores['feature'] == feature]
                if not matched_rows.empty:
                    pfi_score = matched_rows['pfi_score'].values[0]
                    f.write(f"  {i + 1}. {feature}: {pfi_score:.4f}\n")
                else:
                    if feature == config.target_col:
                        f.write(f"  {i + 1}. {feature}: (target variable)\n")
                    else:
                        f.write(f"  {i + 1}. {feature}: (not in PFI analysis)\n")
            f.write("\n")
        else:
            f.write("PFI analysis not performed or no results available.\n\n")

        f.write("5. ROBUSTNESS TESTS\n")
        f.write("------------------\n")

        if not robustness_df.empty:
            f.write(f"Number of causal relationships tested: {len(robustness_df)}\n")

            f.write("\nRobustness Statistics:\n")
            f.write(f"  Average RCC Score: {robustness_df['RCC_score'].mean():.4f}\n")
            f.write(f"  Average CSR: {robustness_df['CSR'].mean():.4f}\n")
            f.write(f"  Average |DDA|: {robustness_df['DDA'].abs().mean():.4f}\n")

            significant_pt = sum(robustness_df['PT_p_value'] < 0.05)
            f.write(f"  Relationships passing Placebo Test (p<0.05): {significant_pt} "
                    f"({100 * significant_pt / len(robustness_df):.1f}%)\n\n")
        else:
            f.write("Robustness tests not performed or no results available.\n\n")

        f.write("6. FINAL CAUSAL STRUCTURE\n")
        f.write("------------------------\n")

        if robust_causal is not None and not robust_causal.empty:
            f.write(f"Number of robust causal relationships: {len(robust_causal)}\n\n")

            top_robust = robust_causal.sort_values('causal_strength', ascending=False)

            f.write("Final Causal Relationships (by causal strength):\n")
            for i, (_, row) in enumerate(top_robust.iterrows()):
                f.write(f"  {i + 1}. {row['cause']} → {row['effect']} (lag={row['lag']})\n")
                f.write(f"     Causal Strength: {row['causal_strength']:.4f}\n")
                f.write(f"     RCC Score: {row['RCC_score']:.4f}, PT p-value: {row['PT_p_value']:.4f}\n")
                f.write(f"     CSR: {row['CSR']:.4f}, DDA: {row['DDA']:.4f}\n")

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

        f.write("7. CONCLUSIONS AND RECOMMENDATIONS\n")
        f.write("---------------------------------\n")

        if robust_causal is not None and not robust_causal.empty:
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

            f.write("Key causal pathways identified:\n")
            G = nx.DiGraph()
            for _, row in robust_causal.iterrows():
                G.add_edge(row['cause'], row['effect'], weight=row['causal_strength'])

            if config.target_col in G.nodes():
                paths_to_target = []
                for node in G.nodes():
                    if node != config.target_col:
                        try:
                            for path in nx.all_simple_paths(G, node, config.target_col, cutoff=3):
                                paths_to_target.append(path)
                        except:
                            pass

                if paths_to_target:
                    for i, path in enumerate(paths_to_target[:5]):
                        path_str = " → ".join(path)
                        f.write(f"  {i + 1}. {path_str}\n")
                else:
                    f.write("  No clear causal pathways to target identified.\n")
            else:
                f.write("  Target variable not found in causal graph.\n")
        else:
            f.write("Insufficient robust causal relationships to form conclusions.\n")

        f.write("\n")

        f.write("8. LIMITATIONS OF ANALYSIS\n")
        f.write("-------------------------\n")
        f.write("- Time series length may limit the ability to detect longer lag relationships.\n")
        f.write("- Observational data cannot fully confirm causality without experimental validation.\n")
        f.write("- Hidden confounders may still exist despite robustness tests.\n")
        f.write("- Linear assumptions in some tests might not capture complex nonlinear relationships.\n")
        f.write("- Results should be interpreted in context with domain knowledge.\n")

    write_log(f"Comprehensive evaluation report saved to: {report_path}", log_file)

    return report_path


def main():
    try:
        global config, log_file
        output_dir = find_output_folder()
        log_file = os.path.join(output_dir, "analysis_log.txt")

        config = Config()
        config.output_dir = output_dir

        write_log("\n" + "=" * 50, log_file)
        write_log("Starting Phase 3: Robustness tests and final causal graph", log_file)
        write_log("=" * 50 + "\n", log_file)

        normalized_data = pd.read_csv(os.path.join(output_dir, "normalized_data_for_next_step.csv"))

        with open(os.path.join(output_dir, "enhanced_links.pkl"), 'rb') as f:
            enhanced_links = pickle.load(f)

        with open(os.path.join(output_dir, "selected_features.pkl"), 'rb') as f:
            features_data = pickle.load(f)
            selected_features = features_data['selected_features']
            pfi_threshold = features_data['pfi_threshold']

        pfi_scores = pd.read_csv(os.path.join(output_dir, "pfi_scores.csv"))

        write_log(f"Loaded normalized data with shape: {normalized_data.shape}", log_file)
        write_log(f"Loaded {len(enhanced_links)} enhanced links", log_file)
        write_log(f"Loaded {len(selected_features)} selected features", log_file)

        target = normalized_data[config.target_col]
        features = normalized_data.drop(config.target_col, axis=1)

        try:
            causal_effects_df = pd.read_csv(os.path.join(output_dir, "causal_effects.csv"))
            causal_effects = causal_effects_df.to_dict('records')
            write_log(f"Loaded {len(causal_effects)} causal effects", log_file)
        except:
            write_log("Could not load causal effects, using empty list", log_file)
            causal_effects = []

        robustness_df = perform_robustness_tests(normalized_data, enhanced_links, config)

        robust_causal = reconstruct_causal_graph(robustness_df, pfi_scores, selected_features, config)

        report_path = generate_comprehensive_report(
            normalized_data, target, features, enhanced_links,
            causal_effects, pfi_scores, selected_features, robustness_df, robust_causal, config
        )

        write_log("\nFinal phase completed successfully!", log_file)
        write_log(f"All results have been saved to: {config.output_dir}", log_file)
        write_log(f"The comprehensive report is available at: {report_path}", log_file)

    except Exception as e:
        write_log(f"ERROR in main process: {str(e)}", log_file)
        import traceback
        write_log(traceback.format_exc(), log_file)


if __name__ == "__main__":
    main()
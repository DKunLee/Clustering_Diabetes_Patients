import pandas as pd
import numpy as np
import math
import ast

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from scipy.spatial.distance import euclidean, cosine
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

from sklearn.metrics import silhouette_score


def read_csv(file_path:str):
    """
    Load a CSV file and convert 'info_mat' and 'all_medications' columns from string to list types.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_csv(file_path)
    df["info_mat"] = df["info_mat"].apply(ast.literal_eval)
    df["all_medications"] = df["all_medications"].apply(ast.literal_eval)
    return df


def load_med_oh(file_path:str):
    """
    Load a medication one-hot encoding dictionary from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        Dict[str, List[int]]: Mapping of medication names to one-hot vectors.
    """
    df = pd.read_csv(file_path, index_col=0)
    return df.T.to_dict(orient='list')


def get_map(df:pd.DataFrame, start_year:int=2019, end_year:int=2023, bins_method:str="half_year") -> pd.DataFrame:
    """
    Generate a mapping DataFrame for patient medication data across time bins.

    Parameters:
        df (pd.DataFrame): Input patient DataFrame.
        start_year (int): Start year for bins.
        end_year (int): End year for bins.
        bins_method (str): One of 'year', 'month', 'half_year', or 'quarter'.

    Returns:
        pd.DataFrame: DataFrame with patient_id and columns for each time bin.
    """
    if bins_method == "year":
        bin_labels = [f"{year}" for year in range(start_year, end_year + 1)]
    elif bins_method == "month":
        bin_labels = [f"{year}-{month:02d}" for year in range(start_year, end_year + 1) for month in range(1, 13)]
    elif bins_method == "half_year":
        bin_labels = [f"{year}-H{half}" for year in range(start_year, end_year + 1) for half in range(1, 3)]
    elif bins_method == "quarter":
        bin_labels = [f"{year}-Q{quarter}" for year in range(start_year, end_year + 1) for quarter in range(1, 5)]
    else:
        raise ValueError("Invalid bins_method. Choose from 'year', 'month', 'half_year', or 'quarter'.")

    generate_map_df = pd.DataFrame(columns=["patient_id"] + bin_labels)
    generate_map_df["patient_id"] = df["patient_id"]

    for index, row in df.iterrows():
        info_mat = row["info_mat"]
        if not isinstance(info_mat, list):
            info_mat = [info_mat] * len(bin_labels)
        for iter, column in enumerate(bin_labels):
            generate_map_df.at[index, column] = row["info_mat"][iter]

    return generate_map_df


def binarize_data(df:pd.DataFrame):
    """
    Convert medication frequency vectors to binary indicators.

    Parameters:
        df (pd.DataFrame): DataFrame with list-valued columns.

    Returns:
        pd.DataFrame: DataFrame with binary lists.
    """
    time_bins = list(df.columns[1:])
    binarized_df = df.copy()

    for col in time_bins:
        binarized_df[col] = binarized_df[col].apply(lambda x: [1 if v > 0 else 0 for v in x])

    return binarized_df


def cluster_patients(df: pd.DataFrame, k: int = None, metric: str = 'euclidean'):
    """
    Perform hierarchical clustering on patient medication vectors.

    Parameters:
        df (pd.DataFrame): DataFrame with patient vectors in list-valued columns.
        k (int, optional): Number of clusters for fcluster. If None, no labels added.
        metric (str): Distance metric: 'euclidean' or 'cosine'.

    Returns:
        pd.DataFrame: Reordered DataFrame, optionally with 'cluster' column.
    """
    time_bins = df.columns[1:-1]

    flattened_matrix = np.array([np.concatenate(row[time_bins].values) for _, row in df.iterrows()])

    linkage_matrix = None

    if metric == 'euclidean':
        linkage_matrix = linkage(flattened_matrix, method='ward', metric='euclidean')
    elif metric == 'cosine':
        linkage_matrix = linkage(flattened_matrix, method='average', metric='cosine')
    else:
        raise ValueError("Invalid metric. Choose 'euclidean' or 'cosine'.")

    ordered_indices = leaves_list(linkage_matrix)
    ordered_df = df.iloc[ordered_indices].reset_index(drop=True)

    # Test this
    if k is not None:
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        cluster_labels = cluster_labels[ordered_indices]
        ordered_df['cluster'] = cluster_labels
    
    return ordered_df


def plot_maps(df: pd.DataFrame, clustered_df: pd.DataFrame, class_shorts: dict):
    """
    Plot original and clustered medication usage color maps side by side.
    """
    # Hard Coding with Expertise
    class_colors = {
        "Insulin": "red",
        "MET": "blue",
        "SGLT2": "green",
        "GLP_1": "yellow",
        "DPP_4": "brown",
        "SUL": "pink",
        "TZD": "orange",
        "Other": "black"
    }

    def get_color_vector(vec: list):
        total = sum(vec)
        if total == 0:
            return (1.0, 1.0, 1.0)

        w_color = np.zeros(3)
        for med, val in zip(class_shorts.keys(), vec):
            if val > 0:
                color_rgb = np.array(mcolors.to_rgb(class_colors[med]))
                w_color += color_rgb * (val / total)

        return tuple(w_color)
    
    def create_color_matrix(df: pd.DataFrame):
        bins = list(df.columns[1:-1])
        color_matrix = np.array([
            [get_color_vector(row[col]) for col in bins] for _, row in df.iterrows()
        ])
        return color_matrix, bins
    
    color_matrix_df, bins_df = create_color_matrix(df)
    color_matrix_clustered, bins_clustered = create_color_matrix(clustered_df)

    patient_ids_df = df["patient_id"].tolist()
    patient_ids_clustered = clustered_df["patient_id"].tolist()

    fig_width = 7
    fig_height = min(max(10, len(patient_ids_df) * 0.3, len(patient_ids_clustered) * 0.3), 10)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)

    for ax, color_matrix, bins, patient_ids, title in zip(
        axes,
        [color_matrix_df, color_matrix_clustered],
        [bins_df, bins_clustered],
        [patient_ids_df, patient_ids_clustered],
        ["Original Data", "Clustered Data"]
    ):
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bins, rotation=45, ha="right")
        ax.set_title(f"{title} Patient X Binary bins", fontsize=10)

        for i in range(len(patient_ids)):
            for j in range(len(bins)):
                rect = plt.Rectangle((j, i), 1, 1, color=color_matrix[i, j], edgecolor='black')
                ax.add_patch(rect)

        ax.set_xlim(0, len(bins))
        ax.set_ylim(0, len(patient_ids))
        ax.invert_yaxis()

    cluster_counts = clustered_df['cluster'].value_counts().sort_index()
    cluster_boundaries = cluster_counts.cumsum()[:-1]

    x_min, x_max = ax.get_xlim()
    margin = (x_max-x_min) * 0.05
    ax.set_xlim(x_min-margin, x_max+margin)
    for boundary in cluster_boundaries:
        axes[1].axhline(boundary, color='black', linewidth=1)

    legend_patches = [
        mpatches.Patch(color=color, label=med) for med, color in class_colors.items()
    ]
    axes[1].legend(
        handles=legend_patches,
        title="Color Representation",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    plt.show()


def plot_clustered_map(clustered_df: pd.DataFrame, class_shorts: dict, title:str):
    """
    Plot a single clustered medication usage map.
    """
    class_colors = {
        "Insulin": "red",
        "MET": "blue",
        "SGLT2": "green",
        "GLP_1": "yellow",
        "DPP_4": "brown",
        "SUL": "pink",
        "TZD": "orange",
        "Other": "black"
    }

    def get_color_vector(vec: list):
        total = sum(vec)
        if total == 0:
            return (1.0, 1.0, 1.0)

        w_color = np.zeros(3)
        for med, val in zip(class_shorts.keys(), vec):
            if val > 0:
                color_rgb = np.array(mcolors.to_rgb(class_colors[med]))
                w_color += color_rgb * (val / total)

        return tuple(w_color)
    
    def create_color_matrix(df):
        bins = list(df.columns[1:-1])
        color_matrix = np.array([
            [get_color_vector(row[col]) for col in bins] for _, row in df.iterrows()
        ])
        return color_matrix, bins

    color_matrix_clustered, bins_clustered = create_color_matrix(clustered_df)
    patient_ids_clustered = clustered_df["patient_id"].tolist()

    fig_width = 7
    fig_height = min(max(10, len(patient_ids_clustered) * 0.3), 10)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.set_xticks(np.arange(len(bins_clustered)))
    ax.set_xticklabels(bins_clustered, rotation=45, ha="right")
    ax.set_title(title, fontsize=10)

    for i in range(len(patient_ids_clustered)):
        for j in range(len(bins_clustered)):
            rect = plt.Rectangle((j, i), 1, 1, color=color_matrix_clustered[i, j], edgecolor='black')
            ax.add_patch(rect)

    ax.set_xlim(0, len(bins_clustered))
    ax.set_ylim(0, len(patient_ids_clustered))
    ax.invert_yaxis()

    cluster_counts = clustered_df['cluster'].value_counts().sort_index()
    cluster_boundaries = cluster_counts.cumsum()[:-1]

    x_min, x_max = ax.get_xlim()
    margin = (x_max-x_min) * 0.05
    ax.set_xlim(x_min-margin, x_max + margin)
    for boundary in cluster_boundaries:
        ax.axhline(boundary, color='black', linewidth=1)

    legend_patches = [
        mpatches.Patch(color=color, label=med) for med, color in class_colors.items()
    ]
    ax.legend(
        handles=legend_patches,
        title="Color Representation",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    plt.show()


def plot_clusters_line_graphs(clustered_df: pd.DataFrame):
    """
    Plot line graphs of medication usage percentages over time for each cluster.
    """
    class_colors = {
        "Insulin": "red",
        "MET": "blue",
        "SGLT2": "green",
        "GLP_1": "brown",
        "DPP_4": "yellow",
        "SUL": "pink",
        "TZD": "orange",
        "Other": "black"
    }

    df = clustered_df.copy()
    vec_cols = df.columns[1:-1]
    
    for col in vec_cols:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=float))
    
    df['info_vec'] = df.apply(lambda row: np.stack([row[col] for col in vec_cols], axis=0), axis=1)
    
    def percentage_2d_array(series):
        stacked = np.stack(series, axis=0)
        sum_matrix = np.sum(stacked, axis=0)
        percentage = (sum_matrix / len(series)) * 100
        return percentage

    perc_info = df.groupby('cluster')['info_vec'].agg(percentage_2d_array).reset_index()
    
    n_clusters = len(perc_info)
    ncols = 5
    nrows = math.ceil(n_clusters / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.ravel()

    medication_labels = ["Insulin", "MET", "SGLT2", "GLP_1", "DPP_4", "SUL", "TZD", "Other"]
    medication_colors = [class_colors[label] for label in medication_labels]
    
    for ax, (_, row) in zip(axes, perc_info.iterrows()):
        matrix = row['info_vec']
        n_time_bins, n_meds = matrix.shape
        t = np.arange(n_time_bins)
        
        for j in range(n_meds):
            # Use the dictionary values; fallback to gray if index exceeds defined meds.
            color = medication_colors[j] if j < len(medication_colors) else "gray"
            label = medication_labels[j] if j < len(medication_labels) else ""
            ax.plot(t, matrix[:, j], label=label, color=color)
        
        ax.set_title(f"Cluster {row['cluster']}")
        ax.set_xlabel("Time Bins")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
    
    for extra_ax in axes[len(perc_info):]:
        extra_ax.set_visible(False)
    
    handles = []
    for label, color in zip(medication_labels, medication_colors):
        if label=="GLP_1":
            color = "yellow"
        elif label=="DPP_4":
            color = "brown"
        handles.append(plt.Line2D([0], [0], color=color, label=label))
    fig.legend(handles=handles, loc='upper center', ncol=len(medication_labels), fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def k_evaluation(df, method='ward', metric='euclidean', k_min=2, k_max=40, ax=None):
    """
    Evaluate clustering for a range of k using elbow and silhouette curves.
    """
    time_cols = df.columns[1:]
    
    df = df.copy()
    
    for col in time_cols:
        df[col] = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
    
    df['vector'] = df[time_cols].apply(lambda row: np.concatenate(row.values), axis=1)
    
    data_matrix = np.stack(df['vector'].values)

    k_values  = range(k_min, k_max + 1)
    wcss_vals = []
    sil_vals  = []

    Z_full = linkage(data_matrix, method=method, metric=metric)

    for k in k_values:
        labels = fcluster(Z_full, t=k, criterion='maxclust')

        wcss_k = 0.0
        for cid in np.unique(labels):
            pts = data_matrix[labels == cid]
            centroid = pts.mean(axis=0)
            wcss_k += ((pts - centroid) ** 2).sum()
        wcss_vals.append(wcss_k)

        sil_vals.append(silhouette_score(data_matrix, labels))

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax_wcss, ax_sil = ax

    ax_wcss.plot(k_values, wcss_vals, marker='o')
    ax_wcss.set_title("Elbow curve (WCSS)")
    ax_wcss.set_xlabel("Number of clusters (k)")
    ax_wcss.set_ylabel("WCSS")
    ax_wcss.grid(True)

    ax_sil.plot(k_values, sil_vals, marker='o')
    ax_sil.set_title("Silhouette score")
    ax_sil.set_xlabel("Number of clusters (k)")
    ax_sil.set_ylabel("Silhouette")
    ax_sil.grid(True)

    plt.tight_layout()
    plt.show()


def combine_two_cluster(df: pd.DataFrame, _to: int, _from: int):
    """
    Merge one cluster into another and reassign cluster labels sequentially.
    """
    df['cluster'] = df['cluster'].replace(_from, _to)

    unique_clusters = sorted(df['cluster'].unique())
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters, start=1)}
    df['cluster'] = df['cluster'].map(cluster_mapping)

    clusters = df["cluster"].unique()
    clustered_dfs = [cluster_patients(df[df["cluster"]==cluster].copy()) for cluster in clusters]
    clustered_df = pd.concat(clustered_dfs, axis=0, ignore_index=True)

    return clustered_df


# Extra-Experiment
def combined_cluster_patients(df:pd.DataFrame, threshold:float, k:int, kk:int):
    """
    Perform a combined clustering approach with threshold-based reassignments and hierarchical merging.
    """
    time_bins = df.columns[1:]
    flattened_matrix = np.array([np.concatenate(row[time_bins].values) for _, row in df.iterrows()])

    linkage_matrix = linkage(flattened_matrix, method='ward', metric='euclidean')

    clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
    df['cluster'] = clusters

    time_bins = df.columns[1:-1]
    flat_mat = np.array([np.concatenate(row) for row in df[time_bins].values])
    df['flat_mat'] = list(flat_mat)

    cluster_centroid_dict = df.groupby('cluster')['flat_mat'].apply(lambda x: np.mean(np.vstack(x), axis=0)).to_dict()
    df['centroid'] = df['cluster'].map(cluster_centroid_dict)

    centroid = flat_mat.mean(axis=0)
    df['distance_to_centroid'] = [euclidean(row, centroid) for row in flat_mat]

    copied_df = df.copy()
    cluster_centroids = (df[['cluster', 'centroid']].drop_duplicates(subset=['cluster']).set_index('cluster')['centroid'].to_dict())

    for idx, row in copied_df.iterrows():
        current_distance = row['distance_to_centroid']
        current_cluster = row['cluster']
        
        if current_distance > threshold:
            patient_vec = row['flat_mat']

            original_cosine_dist = cosine(patient_vec, cluster_centroids[current_cluster])

            best_cluster = current_cluster
            best_dist = original_cosine_dist

            for c, centroid_vec in cluster_centroids.items():
                dist = cosine(patient_vec, centroid_vec)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c

            if best_cluster != current_cluster:
                copied_df.at[idx, 'cluster'] = best_cluster
                copied_df.at[idx, 'distance_to_centroid'] = best_dist

    new_centroids = {}
    for c in copied_df['cluster'].unique():
        cluster_vectors = copied_df.loc[copied_df['cluster'] == c, 'flat_mat']
        new_centroids[c] = np.mean(np.vstack(cluster_vectors), axis=0)

    cluster_centroids = new_centroids

    df_cluster_centroids = (copied_df[['cluster', 'centroid']].drop_duplicates(subset=['cluster']).reset_index(drop=True))

    X_centroids = np.vstack(df_cluster_centroids['centroid'].values)

    Z = linkage(X_centroids, method='average', metric='cosine')

    labels = fcluster(Z, t=kk, criterion='maxclust')
    df_cluster_centroids['hier_cluster'] = labels

    df_merged = copied_df.merge(df_cluster_centroids[['cluster', 'hier_cluster']], on='cluster',how='left')

    df_merged['cluster'] = df_merged['hier_cluster']
    df_merged.drop(columns='hier_cluster', inplace=True)

    df_merged.sort_values(by='cluster', ascending=True, inplace=True)

    df_merged = df_merged.drop(['flat_mat','centroid', 'distance_to_centroid'], axis=1)

    clusters = df_merged["cluster"].unique()
    clustered_dfs = [cluster_patients(df_merged[df_merged["cluster"]==cluster].copy()) for cluster in clusters]
    clustered_df = pd.concat(clustered_dfs, axis=0, ignore_index=True)

    return clustered_df
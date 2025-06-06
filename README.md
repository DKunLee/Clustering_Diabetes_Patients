----------------------------------------------------------
⚠️ **Important Notice on Data Privacy**
----------------------------------------------------------
All `patient_id` displayed or analyzed in this project have been securely anonymized using the `anonymous_head` function. This function applies a **non-reversible, cryptographic hash**, ensuring that no original patient identities can be retrieved or traced. This project adheres strictly to data privacy standards and ethical data handling practices.

# Clustering_Diabetes_Patients

This repository contains the code and notebooks for preprocessing and analyzing electronic health record (EHR) data to identify optimal clustering methods for diabetes patients based on their medication patterns. This is a sub-project of our research team’s larger diabetes study; Code in this repo was written by the author (DK Lee), and the ideas were discussed with the team.

> **Note:** The `data/` directory (containing raw patient data) is omitted from version control due to privacy restrictions. Before running workflows, ensure you have a local `data/extracted/` folder populated with preprocessed CSVs.

## Repository Structure

```text
├── data/                        # (Excluded) Raw/intermediate data files
│   └── extracted/               # Outputs from preprocessing (CSV files)
├── Extraction.ipynb             # Data extraction and info-matrix creation notebook
├── Hierarchy_Clustering_Analysis.ipynb  # Clustering experiments and visualizations notebook
├── analysis_tools/              # Python modules for preprocessing and analysis
│   ├── preprocess.py            # Functions for loading, cleaning, anonymizing, encoding data
│   └── analysis.py              # Functions for mapping, clustering, plotting, evaluation
└── README.md                    # This file
```

## Features

* **Data Loading & Conversion**: Read RData/RDS files, anonymize patient IDs, and convert medications into time‑binned matrices.
* **One‑Hot Encoding**: Generate and export one‑hot vectors for medication types.
* **Info Matrix Construction**: Aggregate medication usage across user‑defined time bins (year, half‑year, quarter, month).
* **Clustering Tools**: Hierarchical clustering (Ward’s method and cosine distance), combined threshold-based reassignments, cluster merging, and k‑evaluation (elbow and silhouette).
* **Visualization**: Color‑mapped patient×time heatmaps, clustered maps, and cluster-wise line‑graph summaries of medication usage.

## Usage

### 1. Preprocessing

Open **`Extraction.ipynb`** and run through:

1. Load raw EHR data (`.RData`/`.rds`) using `analysis_tools.preprocess.data_load()`.
2. Inspect and anonymize patient IDs with `analysis_tools.preprocess.anonymous_head()`.
3. Build the medication time‑bin info matrix via `analysis_tools.preprocess.get_info_matrix()`.
4. One‑hot encode medications with `analysis_tools.preprocess.medication_one_hot_encoding()` and save CSVs.
5. Sample random patients using `analysis_tools.preprocess.get_random_patients()`.

Processed output CSVs (e.g., `info_matrix.csv`, `med_oh.csv`) should be placed in `data/extracted/`.

### 2. Analysis & Clustering

Open **`Hierarchy_Clustering_Analysis.ipynb`** and follow:

1. Load preprocessed CSVs with `analysis_tools.analysis.read_csv()` and `analysis_tools.analysis.load_med_oh()`.
2. Reconstruct patient×time maps (`analysis_tools.analysis.get_map()`) and optionally binarize (`analysis_tools.analysis.binarize_data()`).
3. Run clustering with `analysis_tools.analysis.cluster_patients()`, `analysis_tools.analysis.combined_cluster_patients()`, or merge clusters via `analysis_tools.analysis.combine_two_cluster()`.
4. Evaluate cluster counts using `analysis_tools.analysis.k_evaluation()` (elbow & silhouette plots).
5. Visualize results through `analysis_tools.analysis.plot_maps()`, `analysis_tools.analysis.plot_clustered_map()`, and `analysis_tools.analysis.plot_clusters_line_graphs()`.

## Function Summary

All modules reside in the **`analysis_tools/`** directory.

### preprocess.py

* **`anonymous_head(df: DataFrame, n: int=5) -> DataFrame`**
  Return the first `n` rows with MD5‑anonymized `patient_id`.
* **`data_load(file_path: str) -> DataFrame`**
  Load `.RData`/`.rds` files via `pyreadr`.
* **`get_info_matrix(df, medication_oh, start_year, end_year, bins_method) -> DataFrame`**
  Build medication usage matrix over time bins.
* **`convert_medications(df, column, old, new) -> DataFrame`**
  Rename medication labels in-place.
* **`medication_one_hot_encoding(df) -> dict`**
  Create one‑hot vectors for each medication category.
* **`save_columns_as_csv(df, columns, file_name) -> None`**
  Export selected DataFrame columns to a CSV.
* **`save_one_hot_encoded_medications(med_oh, file_name) -> None`**
  Export one‑hot dictionary to CSV.
* **`get_random_patients(df, num_patients, random_state) -> DataFrame`**
  Randomly sample patients from the DataFrame.

### analysis.py

* **`read_csv(file_path: str) -> DataFrame`**
  Load `info_mat` and `all_medications` from CSV into list objects.
* **`load_med_oh(file_path: str) -> dict`**
  Load medication one‑hot dictionary from CSV.
* **`get_map(df, start_year, end_year, bins_method) -> DataFrame`**
  Expand `info_mat` into separate time‑bin columns.
* **`binarize_data(df) -> DataFrame`**
  Convert count lists to binary presence/absence.
* **`cluster_patients(df, k, metric) -> DataFrame`**
  Perform hierarchical clustering (Ward or cosine).
* **`combined_cluster_patients(df, threshold, k, kk) -> DataFrame`**
  Apply threshold‑based reassignments and hierarchical merging.
* **`combine_two_cluster(df, to_cluster, from_cluster) -> DataFrame`**
  Merge one cluster into another and reindex labels.
* **`k_evaluation(df, method, metric, k_min, k_max) -> None`**
  Plot elbow (WCSS) and silhouette curves for a range of `k`.
* **`plot_maps(df, clustered_df, class_shorts) -> None`**
  Display side‑by‑side heatmaps of original vs. clustered data.
* **`plot_clustered_map(clustered_df, class_shorts, title) -> None`**
  Display a heatmap of clustered usage patterns.
* **`plot_clusters_line_graphs(clustered_df) -> None`**
  Show medication usage trends over time for each cluster.

## Research Paper

A full manuscript detailing the clustering methodology and results will be available during summer 2025.

$Link to paper \(TBD\)$

## Author

DK Lee ([https://github.com/DKunLee](https://github.com/DKunLee))

## License

This project is released under the MIT License. See `LICENSE` for details.

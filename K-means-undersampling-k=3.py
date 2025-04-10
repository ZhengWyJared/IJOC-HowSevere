import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import itertools
import os

# Set the grid
quantiles = {
    (1, 0): [0.20, 0.26, 0.32, 0.38],  # Label 1, Cluster 0
    (1, 1): [0.20, 0.26, 0.32, 0.38],  # Label 1, Cluster 1
    (1, 2): [0.20, 0.26, 0.32, 0.38],  # Label 1, Cluster 2
    (2, 0): [0.40, 0.46, 0.52, 0.58],  # Label 2, Cluster 0
    (2, 1): [0.40, 0.46, 0.52, 0.58],  # Label 2, Cluster 1
    (2, 2): [0.40, 0.46, 0.52, 0.58]  # Label 2, Cluster 2
}
quantile_combinations = list(itertools.product(
    quantiles[(1, 0)], quantiles[(1, 1)], quantiles[(1, 2)],
    quantiles[(2, 0)], quantiles[(2, 1)], quantiles[(2, 2)]
))
results_dir = '..../K=3-Kmeans-undersampling-gridsearch'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
train_file_path = '.../mydatafinal_train.xlsx'
data_full = pd.read_excel(train_file_path)
original_features_full = data_full.iloc[:, 1:-1].values
labels_full = data_full.iloc[:, 0].values
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(original_features_full)
data_full['Cluster_Label'] = np.nan         
data_full['Distance_to_Own'] = np.nan
random_seed = 0
# Perform KMeans clustering on samples with label=1
mask_label1 = (data_full.iloc[:, 0] == 1)
features_label1_norm = features_normalized[mask_label1]
kmeans_label1 = KMeans(n_clusters=3, n_init=1000, random_state=random_seed, init='k-means++')
kmeans_label1.fit(features_label1_norm)
cluster_labels_1 = kmeans_label1.labels_
distances_1 = np.linalg.norm(features_label1_norm - kmeans_label1.cluster_centers_[cluster_labels_1], axis=1)
data_full.loc[mask_label1, 'Cluster_Label'] = cluster_labels_1
data_full.loc[mask_label1, 'Distance_to_Own'] = distances_1
# Perform KMeans clustering on samples with label=2
mask_label2 = (data_full.iloc[:, 0] == 2)
features_label2_norm = features_normalized[mask_label2]
kmeans_label2 = KMeans(n_clusters=3, n_init=1000, random_state=random_seed, init='k-means++')
kmeans_label2.fit(features_label2_norm)
cluster_labels_2 = kmeans_label2.labels_
distances_2 = np.linalg.norm(features_label2_norm - kmeans_label2.cluster_centers_[cluster_labels_2], axis=1)
data_full.loc[mask_label2, 'Cluster_Label'] = cluster_labels_2
data_full.loc[mask_label2, 'Distance_to_Own'] = distances_2


def train_with_quantile_combination(quantile_combination):
    quantiles_dict = {
        (1, 0): quantile_combination[0],
        (1, 1): quantile_combination[1],
        (1, 2): quantile_combination[2],
        (2, 0): quantile_combination[3],
        (2, 1): quantile_combination[4],
        (2, 2): quantile_combination[5]
    }
    data = data_full.copy()    
# Collect removed points to be used as an additional test set later
    removed_df_list = []
    for label in [1, 2]:
        for cluster in [0, 1]:
            mask = (data.iloc[:, 0] == label) & (data['Cluster_Label'] == cluster)
            cluster_data = data[mask]
            if len(cluster_data) == 0:
                continue
            q = quantiles_dict[(label, cluster)]
            thr_distance = np.percentile(cluster_data['Distance_to_Own'], (1 - q) * 100)
# Remove points with distances ≤ threshold (those closest to the center)
            remove_data = cluster_data[cluster_data['Distance_to_Own'] <= thr_distance]
            removed_df_list.append(remove_data.copy())
            data = data.drop(remove_data.index)
    removed_data_all = pd.concat(removed_df_list, axis=0) if removed_df_list else pd.DataFrame()

    quantile_str = ''.join(str(int(q * 100)) for q in quantile_combination)
    output_path = os.path.join(results_dir, f"mydata2k{quantile_str}.xlsx")
    data.to_excel(output_path, index=False)
    removed_file_path = os.path.join(results_dir, f"mydata2k{quantile_str}_removed.xlsx")
    if not removed_data_all.empty:
        removed_data_all.to_excel(removed_file_path, index=False)
    else:
        pd.DataFrame().to_excel(removed_file_path, index=False)
    label_counts = data.iloc[:, 0].value_counts()
    label_counts_str = ' '.join([f'{lb}:{ct}' for lb, ct in label_counts.items()])

# Use XGBOOST for prediction
    labels_for_train = data.iloc[:, 0].values
    features_for_train = data.iloc[:, 1:-3].values  
    unique_labels = sorted(np.unique(labels_for_train))
    labels_mapped = labels_for_train - 1
    test_file_orig = '.../mydatafinal_test.xlsx'
    test_data_orig = pd.read_excel(test_file_orig)
    X_test_orig = test_data_orig.iloc[:, 1:-1].values
    y_test_orig = test_data_orig.iloc[:, 0].values
    y_test_mapped = y_test_orig - 1
    param_grid = {
        'max_depth': [1, 3, 5, 7, 9, 11],
        'learning_rate': [0.01],
        'n_estimators': [100, 300, 500, 700, 900, 1100],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(unique_labels),
        n_jobs=1
    )
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='recall_macro', 
        cv=5,
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(features_for_train, labels_mapped)
    best_model = grid_search.best_estimator_
# Predict on the original test set
    pred_labels_orig = best_model.predict(X_test_orig)
    accuracy_orig = accuracy_score(y_test_mapped, pred_labels_orig)
    precision_orig = precision_score(y_test_mapped, pred_labels_orig, average=None, labels=[0,1,2])
    recall_orig = recall_score(y_test_mapped, pred_labels_orig, average=None, labels=[0,1,2])

# Predict on the removed points
    if not removed_data_all.empty:
        removed_indices = removed_data_all.index
        X_removed = original_features_full[removed_indices]
        y_removed = removed_data_all.iloc[:, 0].values
        y_removed_mapped = y_removed - 1

        pred_removed = best_model.predict(X_removed)
        accuracy_removed = accuracy_score(y_removed_mapped, pred_removed)

        precision_removed_all = precision_score(y_removed_mapped, pred_removed, average=None, labels=[0,1,2])
        recall_removed_all = recall_score(y_removed_mapped, pred_removed, average=None, labels=[0,1,2])

        precision_removed_all[2] = 0.0
        recall_removed_all[2] = 0.0
    else:
        accuracy_removed = 0
        precision_removed_all = [0.0, 0.0, 0.0]
        recall_removed_all = [0.0, 0.0, 0.0]
    result = {
        'Quantile_Combination': quantile_combination,
        'Frequency': label_counts_str,

        'Accuracy': accuracy_orig,
        'Label_1_Precision': precision_orig[0],
        'Label_1_Recall': recall_orig[0],
        'Label_2_Precision': precision_orig[1],
        'Label_2_Recall': recall_orig[1],
        'Label_3_Precision': precision_orig[2],
        'Label_3_Recall': recall_orig[2],

        'RemovedPoints_Accuracy': accuracy_removed,
        'RemovedPoints_Label_1_Precision': precision_removed_all[0],
        'RemovedPoints_Label_1_Recall': recall_removed_all[0],
        'RemovedPoints_Label_2_Precision': precision_removed_all[1],
        'RemovedPoints_Label_2_Recall': recall_removed_all[1],
        'RemovedPoints_Label_3_Precision': precision_removed_all[2],
        'RemovedPoints_Label_3_Recall': recall_removed_all[2],
    }
    return result

final_results = Parallel(n_jobs=-1)(
    delayed(train_with_quantile_combination)(quantile_set)
    for quantile_set in quantile_combinations
)

# Save results
results_df = pd.DataFrame(final_results)
final_results_path = os.path.join(results_dir, 'final_results.xlsx')
results_df.to_excel(final_results_path, index=False)

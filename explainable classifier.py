import torch
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import time
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from collections import Counter

def precompute_piecewise_vectors(features, num_intervals_list, feature_min_max, device):
    num_samples, num_features = features.shape
    all_piecewise_vectors = []

    for i in range(num_features):
        min_val, max_val = feature_min_max[i]
        intervals = np.linspace(min_val, max_val, num_intervals_list[i])

        feature_vectors = []
        for value in features[:, i]:
            if value < min_val:
                value = min_val
            vector = torch.zeros(num_intervals_list[i] - 1, device=device) 
            for j in range(1, num_intervals_list[i]):
                if value < intervals[j]:
                    vector[j - 1] = (value - intervals[j - 1]) / (intervals[j] - intervals[j - 1])
                    break
                elif j < num_intervals_list[i] - 1:
                    vector[j - 1] = 1
                    if value < intervals[j + 1]:
                        vector[j] = (value - intervals[j]) / (intervals[j + 1] - intervals[j])
                        break
                else:
                    vector[j - 1] = 1  
            feature_vectors.append(vector)
        all_piecewise_vectors.append(torch.stack(feature_vectors))
    return all_piecewise_vectors


def scoring_function(precomputed_vectors, num_intervals_list, k, lambda_, delta_u, v, device):
    num_features = len(num_intervals_list)
    aggregated_terms = []
    for i in range(num_features):
        marginal_scores = torch.mv(precomputed_vectors[i], delta_u[i]) + 20  
        aggregated_terms.append(marginal_scores.unsqueeze(0)) 
    aggregated_tensor = torch.cat(aggregated_terms, dim=0)  
    if lambda_ == 0:
        score_sum = torch.exp(torch.mean(torch.log(aggregated_tensor), dim=0))
    elif lambda_ > 0:
        max_value = aggregated_tensor.max(dim=0, keepdim=True)[0]
        scaled_tensor = aggregated_tensor / max_value
        mean_of_powers = torch.mean(scaled_tensor ** lambda_, dim=0)
        score_sum = (mean_of_powers ** (1 / lambda_)) * max_value.squeeze(0)
    else:
        min_value = aggregated_tensor.min(dim=0, keepdim=True)[0]
        scaled_tensor = aggregated_tensor / min_value
        mean_of_powers = torch.mean(scaled_tensor ** lambda_, dim=0)
        score_sum = (mean_of_powers ** (1 / lambda_)) * min_value.squeeze(0)
    interaction_terms = torch.zeros(marginal_scores.shape[0], device=device) 
    for i in range(num_features):
        for j in range(i + 1, num_features):
            for m in range(num_intervals_list[i] - 1):
                for n in range(num_intervals_list[j] - 1):
                    interaction_term = (
                        v[i][m] @ v[j][n]
                    ) * precomputed_vectors[i][:, m] * precomputed_vectors[j][:, n]
                    interaction_terms = interaction_terms + interaction_term
    score_sum = score_sum + interaction_terms
    return score_sum


def loss_function(variables, precomputed_vectors, labels, num_intervals_list, k, C1, C2, C3, tau, device):
    lambda_, delta_u, v = unflatten_variables(variables, len(precomputed_vectors), num_intervals_list, k, device)
    scores = scoring_function(precomputed_vectors, num_intervals_list, k, lambda_, delta_u, v, device)
    num_samples = scores.shape[0]
    indices = torch.triu_indices(num_samples, num_samples, offset=1)
    scores_i = scores[indices[0]]
    scores_j = scores[indices[1]]
    labels_i = labels[indices[0]]
    labels_j = labels[indices[1]]    
    epsilon=1e-6    
    labels_i = labels_i.long()
    labels_j = labels_j.long() 
    w_ij = torch.tensor(weight_matrix, device=device)[labels_i, labels_j]
    yi_j = torch.where(labels_i > labels_j, 1, torch.where(labels_i < labels_j, -1, 0))
    diff = yi_j * (scores_i - scores_j)
    loss = w_ij * torch.where(diff < tau, 0.5 * (tau - diff) ** 2, 0)  
    total_loss = torch.sum(loss)      
    reg_term_1 = C1 * (
        sum((delta_u[i][j + 1] - delta_u[i][j])**2 for i in range(len(delta_u)) for j in range(len(delta_u[i]) - 1))
        + sum(delta_u[i][0]**2 for i in range(len(delta_u)))
    )
    reg_term_2 = C2 * sum(v[i][s][f]**2 for i in range(len(v)) for s in range(len(v[i])) for f in range(len(v[i][s])))
    reg_term_3 = C3 * (lambda_ - 1)**2
    total_loss = total_loss + reg_term_1 + reg_term_2  + reg_term_3
    return total_loss

def unflatten_variables(variables, num_features, num_intervals_list, k, device):
    index = 0
    lambda_ = variables[index] 
    index += 1
    delta_u = []
    for i in range(num_features):
        delta_u.append(variables[index:index + num_intervals_list[i] - 1])  # [num_features][num_intervals_list[i]-1]
        index += num_intervals_list[i] - 1
    v = []
    for i in range(num_features):
        v.append(variables[index:index + (num_intervals_list[i] - 1) * k].view(num_intervals_list[i] - 1, k))  # [num_features][num_intervals_list[i]-1][k]
        index += (num_intervals_list[i] - 1) * k

    return lambda_, delta_u, v
def precompute_weights(labels):
    freq_dict = Counter()
    num_samples = len(labels)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):  
            label_pair = tuple(sorted((labels[i], labels[j])))  
            if label_pair[0] != label_pair[1]:  
                freq_dict[label_pair] += 1
    inverse_freq_dict = {pair: 1 / freq for pair, freq in freq_dict.items()}
    total_inverse_freq = sum(inverse_freq_dict.values())
    w_dict = {pair: weight / total_inverse_freq for pair, weight in inverse_freq_dict.items()}
    symmetric_w_dict = {}
    for pair, weight in w_dict.items():
        symmetric_w_dict[pair] = weight
        symmetric_w_dict[(pair[1], pair[0])] = weight  
    return symmetric_w_dict

def build_weight_matrix(w_dict, num_labels):
    weight_matrix = np.zeros((num_labels, num_labels))
    for (label_i, label_j), weight in w_dict.items():
        weight_matrix[label_i, label_j] = weight
        weight_matrix[label_j, label_i] = weight

    return weight_matrix
num_labels = 3

############################################################
import pandas as pd
import numpy as np
import torch
from itertools import combinations
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import compute_sample_weight
import optuna
import os
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "multiprocessing"
import pickle
from tqdm import tqdm 
from itertools import product
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.samplers import TPESampler, RandomSampler

train_file = '.../train32.xlsx'       # training set
test_file_orig = '.../mydatafinal_test.xlsx'                  # original test set
test_file_add = '.../addtest32.xlsx' # added test set (removed points by the undersampling)

train_data = pd.read_excel(train_file)
print("Training data sample:\n", train_data.head())
device = torch.device("cpu")
print("Using device:", device)
labels = train_data.iloc[:, 0].to_numpy()  
features_df = train_data.iloc[:, 1:-2]
print("Extracted training features:\n", features_df.head())

test_data_orig = pd.read_excel(test_file_orig)
features_test_orig = torch.tensor(test_data_orig.iloc[:, 1:-1].to_numpy(), device=device, dtype=torch.float32)
labels_test_orig = torch.tensor(test_data_orig.iloc[:, 0].to_numpy(), device=device, dtype=torch.float32)
features_test_df = test_data_orig.iloc[:, 1:-1]
combined_features = pd.concat([features_df, features_test_df], axis=0)
feature_min_max = [(combined_features[col].min(), combined_features[col].max()) for col in combined_features.columns]
w_dict = precompute_weights(labels.tolist())

category_counts = [
    features_df[col].nunique() if features_df[col].nunique() <= 5 else None
    for col in features_df.columns
]
features = torch.tensor(features_df.to_numpy(), device=device, dtype=torch.float32)
labels = torch.tensor(labels, device=device, dtype=torch.float32)

test_data_add = pd.read_excel(test_file_add)
features_test_add = torch.tensor(test_data_add.iloc[:, 1:-2].to_numpy(), device=device, dtype=torch.float32)
labels_test_add = torch.tensor(test_data_add.iloc[:, 0].to_numpy(), device=device, dtype=torch.float32)

interval_candidates = [6, 7, 8, 9, 10]


precomputed_cache = {}

def cache_all_precomputed_vectors(features, interval_candidates, feature_min_max, device, category_counts):
    for num_intervals in interval_candidates:
        num_intervals_list = [
            count if count is not None else num_intervals
            for count in category_counts
        ]
        key = tuple(num_intervals_list)
        if key not in precomputed_cache:
            precomputed_vectors = precompute_piecewise_vectors(features, num_intervals_list, feature_min_max, device)
            precomputed_cache[key] = precomputed_vectors
    print("Precomputed vectors cached for all interval candidates.")


def get_precomputed_vectors(num_intervals_list, indices=None):
    key = tuple(num_intervals_list)
    precomputed_vectors = precomputed_cache[key]
    
    if indices is not None:
        return [precomputed_vectors[i][indices] for i in range(len(precomputed_vectors))]    
    return precomputed_vectors


cache_all_precomputed_vectors(features, interval_candidates, feature_min_max, device, category_counts)
num_repeats = 1
tau = 5 
num_iterations = 200 


all_loss_histories = []
optimization_times = []
optimal_parameters_list = []
def get_global_indices(features, features_train):
    indices = []
    for train_feature in features_train:
        index = torch.where((features == train_feature).all(dim=1))[0]
        if len(index) > 0:
            indices.append(index.item()) 
    return indices

def flatten_variables(lambda_, delta_u, v):
    device = lambda_.device
    return torch.cat(
        [torch.tensor([lambda_], device=device)] +
        [du.flatten().to(device) for du in delta_u] + 
        [vi.flatten().to(device) for vi in v]
    )



def initialize_variables(features, num_intervals_list, k, device):
    num_features = features.shape[1]    
    torch.manual_seed(0)
    lambda_ = torch.tensor(1.0, device=device, requires_grad=True)
    delta_u = [torch.full((intervals - 1,), 1 / len(num_intervals_list), device=device) for intervals in num_intervals_list]
    
    v = [torch.empty(intervals - 1, k, device=device).uniform_(-1, 1) for intervals in num_intervals_list]
    flattened_variables = flatten_variables(lambda_, delta_u, v)

    return flattened_variables[1:], lambda_ 
iteration_counter = 0
print_every = 100 
def optimize_with_pytorch(features_train, labels_train, precomputed_vectors_train, 
                          num_intervals_list, k, C1, C2, C3, tau, num_iterations, device):

    initial_variables, lambda_ = initialize_variables(features_train, num_intervals_list, k, device)
    initial_variables.requires_grad_(True)
    lambda_.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [{'params': initial_variables, 'lr': 0.01}, {'params': [lambda_], 'lr': 0.01}]
    )

    dataset = TensorDataset(torch.arange(len(features_train)))  
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    loss_history = []
    min_loss = float('inf')
    min_loss_variables = None
    increase_count = 0
    early_stopping_threshold_increase = 15 

    with tqdm(total=num_iterations, desc="Training Progress") as pbar:
        for iteration in range(num_iterations):
            for batch_indices_tensor in loader:
                batch_indices = batch_indices_tensor[0]  
                
                batch_features, batch_labels = features_train[batch_indices], labels_train[batch_indices]
                batch_piecewise_vectors = [precomputed_vectors_train[i][batch_indices] for i in range(len(precomputed_vectors_train))]
                
                optimizer.zero_grad()
                loss = loss_function(
                    torch.cat([lambda_.view(1), initial_variables], dim=0),
                    batch_piecewise_vectors, batch_labels, 
                    num_intervals_list, k, C1, C2, C3, tau, device
                )
                loss.backward()
                torch.nn.utils.clip_grad_value_([initial_variables, lambda_], clip_value=50)
                optimizer.step()
            current_loss = loss_function(
                torch.cat([lambda_.view(1), initial_variables], dim=0),
                precomputed_vectors_train, 
                labels_train, num_intervals_list, k, C1, C2, C3, tau, device
            )
            loss_history.append(current_loss.item())

            pbar.update(1)
            pbar.set_postfix(loss=current_loss.item(), lambda_=lambda_.item())
            if current_loss < min_loss:
                if current_loss < min_loss * 0.999:
                    increase_count = 0
                else:
                    increase_count += 1
                min_loss = current_loss
                min_loss_variables = torch.cat([lambda_.detach().clone().view(1), initial_variables.detach().clone()])
            else:
                increase_count += 1            
            if increase_count > early_stopping_threshold_increase:
                print("Early stopping triggered.")
                break
    return min_loss_variables, loss_history

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [tensor_to_numpy(t) for t in tensor]
    else:
        return tensor
all_accuracies = []
all_mean_absolute_errors = []
all_macro_precisions = []
all_macro_recalls = []
all_weighted_maes = []
all_label_precisions = []
all_label_recalls = []
CV_optimization_times = []
best_params_list = []  
all_test_results = []
all_train_results = []
all_confusion_matrices = []
all_train_accuracies = []

save_folder = '...'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for i in tqdm(range(num_repeats), desc="Overall Progress (Repeats)", position=0):
    print(f"Starting C-V run {i+1}")
    def single_fold_training(
        fold_number, local_train_idx, local_val_idx,
        precomputed_vectors_full, features, labels,
        num_intervals_list, k, C1, C2, C3, tau, device
    ):
        fold_id = f"Fold {fold_number}"
        precomputed_vectors_train = [
            precomputed_vectors_full[f_idx][local_train_idx] 
            for f_idx in range(len(precomputed_vectors_full))
        ]
        precomputed_vectors_val = [
            precomputed_vectors_full[f_idx][local_val_idx]
            for f_idx in range(len(precomputed_vectors_full))
        ]

        features_train_fold = features[local_train_idx]
        labels_train_fold = labels[local_train_idx]
        features_val_fold = features[local_val_idx]
        labels_val_fold = labels[local_val_idx]

        print(f"{fold_id}: Starting training")
        final_variables, _ = optimize_with_pytorch(
            features_train_fold,
            labels_train_fold,
            precomputed_vectors_train, 
            num_intervals_list, k, C1, C2, C3, tau,
            num_iterations, device
        )
        lambda_opt, delta_u_opt, v_opt = unflatten_variables(
            final_variables,
            features_train_fold.shape[1],
            num_intervals_list,
            k,
            device
        )
        train_scores_fold = scoring_function(
            precomputed_vectors_train, num_intervals_list,
            k, lambda_opt, delta_u_opt, v_opt, device
        )
        val_scores_fold = scoring_function(
            precomputed_vectors_val, num_intervals_list,
            k, lambda_opt, delta_u_opt, v_opt, device
        )
        train_labels = labels_train_fold.cpu().numpy()
        train_reliability = np.zeros_like(train_labels, dtype=float)
        for i, (score_i, label_i) in enumerate(zip(train_scores_fold, train_labels)):
            num, denom = 0.0, 0.0
            for j, (score_j, label_j) in enumerate(zip(train_scores_fold, train_labels)):
                if i == j:
                    continue 
                if label_j != label_i:
                    denom += 1.0
                    if ((score_j > score_i and label_j > label_i) or 
                        (score_j < score_i and label_j < label_i)):
                        num += 1.0
            train_reliability[i] = num / denom if denom != 0 else 0.0
        unique_labels_fold = np.unique(labels_train_fold.cpu().numpy())
        predicted_labels_val = []
        for val_score in val_scores_fold:
            confidences = []
            for h in unique_labels_fold:
                num = 0
                denom = 0
                for train_score, train_label, rel in zip(train_scores_fold, labels_train_fold.cpu().numpy(), train_reliability):
                    if train_label != h:
                        denom += 1
                        if (train_score > val_score and train_label > h) or (train_score < val_score and train_label < h):
                            num += rel
                confidences.append(num / denom if denom != 0 else 0)
            predicted_label_val = unique_labels_fold[np.argmax(confidences)]
            predicted_labels_val.append(predicted_label_val)

        predicted_labels_val = np.array(predicted_labels_val)
        true_labels_val = labels_val_fold.cpu().numpy()
        f1_macro = recall_score(true_labels_val, predicted_labels_val, average='macro', zero_division=0)
        print(f"{fold_id}: Finished training with Macro F1 Score: {f1_macro:.4f}")
        return f1_macro
    param_grid = {
        "common_num_intervals": range(6, 10),
        "k": range(6, 10),
        "C1": [0, 0.1],
        "C2": [0.1, 1],
        "C3": [0, 0.05]
    }
    param_combinations = list(product(
        param_grid["common_num_intervals"],
        param_grid["k"],
        param_grid["C1"],
        param_grid["C2"],
        param_grid["C3"]
    ))
    def evaluate_params(params, features, labels, tau, device):
        common_num_intervals, k, C1, C2, C3 = params
        num_intervals_list = [
            count if count is not None else common_num_intervals
            for count in category_counts
        ]
        precomputed_vectors_full = precompute_piecewise_vectors(
            features,
            num_intervals_list,
            feature_min_max,
            device
        )
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        f1_scores = []
        for fold_number, (local_train_idx, local_val_idx) in enumerate(skf.split(features, labels)):
            f1_score_fold = single_fold_training(
                fold_number + 1,
                local_train_idx,
                local_val_idx,
                precomputed_vectors_full,
                features,
                labels,
                num_intervals_list, k, C1, C2, C3, tau,
                device
            )
            f1_scores.append(f1_score_fold)

        avg_f1_score = np.mean(f1_scores)
        return params, avg_f1_score, f1_scores
    cv_start_time = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params, features, labels, tau, device)
        for params in param_combinations
    )
    best_result = max(results, key=lambda x: x[1]) 
    best_params, best_f1_score, best_fold_f1_scores = best_result
    cv_end_time = time.time()
    cv_optimization_time = cv_end_time - cv_start_time

    CV_optimization_times.append(cv_optimization_time)
    best_params_list.append(best_params)
    print(f"Best CV Params: {best_params}")
    print(f"Best cross-validation f1score: {best_f1_score}")
    print(f"CV Optimization Time: {cv_optimization_time} seconds")
    common_num_intervals = best_params[0]
    num_intervals_list = [
        count if count is not None else common_num_intervals
        for count in category_counts
    ]
    k = best_params[1]
    C1 = best_params[2]
    C2 = best_params[3]
    C3 = best_params[4]
    precomputed_vectors_train_full = precompute_piecewise_vectors(
        features,
        num_intervals_list,
        feature_min_max,
        device
    )
    start_time = time.time()
    final_variables, loss_history = optimize_with_pytorch(
        features,
        labels,
        precomputed_vectors_train_full,
        num_intervals_list,
        k, C1, C2, C3,
        tau,
        num_iterations,
        device
    )
    end_time = time.time()
    optimization_time = end_time - start_time
    optimization_times.append(optimization_time)
    all_loss_histories.append(loss_history.copy())
    num_features = features.shape[1]
    lambda_opt, delta_u_opt, v_opt = unflatten_variables(
        final_variables,
        num_features,
        num_intervals_list,
        k,
        device
    )
    train_scores_full = scoring_function(
        precomputed_vectors_train_full,
        num_intervals_list,
        k,
        lambda_opt, delta_u_opt, v_opt,
        device
    )
    train_labels = labels.cpu().numpy()
    train_reliability = np.zeros_like(train_labels, dtype=float)
    for i, (score_i, label_i) in enumerate(zip(train_scores_full, train_labels)):
        num, denom = 0.0, 0.0
        for j, (score_j, label_j) in enumerate(zip(train_scores_full, train_labels)):
            if i == j:
                continue 
            if label_j != label_i:
                denom += 1.0
                if ((score_j > score_i and label_j > label_i) or 
                    (score_j < score_i and label_j < label_i)):
                    num += 1.0
        train_reliability[i] = num / denom if denom != 0 else 0.0

    unique_labels_train = np.unique(labels.cpu().numpy())
    train_predicted_labels = []
    for tr_score in train_scores_full:
        confs = []
        for h in unique_labels_train:
            num, denom = 0, 0
            for other_tr_score, other_tr_label, rel in zip(train_scores_full, labels.cpu().numpy(), train_reliability):
                if other_tr_label != h:
                    denom += 1
                    if ((other_tr_score > tr_score) and (other_tr_label > h)) or \
                       ((other_tr_score < tr_score) and (other_tr_label < h)):
                        num += rel
            confs.append(num / denom if denom != 0 else 0)
        predicted_train_label = unique_labels_train[np.argmax(confs)]
        train_predicted_labels.append(predicted_train_label)
    train_predicted_labels = np.array(train_predicted_labels)
    train_accuracy = np.mean(train_predicted_labels == labels.cpu().numpy())


    # predict on the original test set

    precomputed_vectors_test_orig = precompute_piecewise_vectors(
        features_test_orig,
        num_intervals_list,
        feature_min_max,
        device
    )
    test_scores_orig = scoring_function(
        precomputed_vectors_test_orig,
        num_intervals_list,
        k,
        lambda_opt, delta_u_opt, v_opt,
        device
    )
    unique_labels_all_train = np.unique(labels.cpu().numpy()) 
    predicted_labels_orig = []
    confidences_list_orig = []
    for sc in test_scores_orig:
        confs = []
        for h in unique_labels_all_train:
            num, denom = 0, 0
            for tr_score, tr_label, rel in zip(train_scores_full, labels.cpu().numpy(), train_reliability):
                if tr_label != h:
                    denom += 1
                    if ((tr_score > sc) and (tr_label > h)) or ((tr_score < sc) and (tr_label < h)):
                        num += rel
            confs.append(num / denom if denom != 0 else 0)
        pred_lbl = unique_labels_all_train[np.argmax(confs)]
        predicted_labels_orig.append(pred_lbl)
        confidences_list_orig.append(confs)

    predicted_labels_orig = np.array(predicted_labels_orig)
    true_labels_orig = labels_test_orig.cpu().numpy()
    accuracy_orig = np.mean(predicted_labels_orig == true_labels_orig)
    mean_absolute_error_orig = np.mean(np.abs(predicted_labels_orig - true_labels_orig))

    label_precisions_orig = precision_score(true_labels_orig, predicted_labels_orig, average=None, zero_division=0)
    label_recalls_orig = recall_score(true_labels_orig, predicted_labels_orig, average=None, zero_division=0)
    macro_precision_orig = precision_score(true_labels_orig, predicted_labels_orig, average='macro', zero_division=0)
    macro_recall_orig = recall_score(true_labels_orig, predicted_labels_orig, average='macro', zero_division=0)

    sample_weights_orig = compute_sample_weight(class_weight='balanced', y=true_labels_orig)
    errors_orig = np.abs(predicted_labels_orig - true_labels_orig)
    weighted_mae_orig = np.average(errors_orig, weights=sample_weights_orig)
    conf_matrix_orig = confusion_matrix(true_labels_orig, predicted_labels_orig, labels=unique_labels_all_train)
    all_accuracies.append(accuracy_orig)
    all_mean_absolute_errors.append(mean_absolute_error_orig)
    all_macro_precisions.append(macro_precision_orig)
    all_macro_recalls.append(macro_recall_orig)
    all_weighted_maes.append(weighted_mae_orig)
    all_label_precisions.append(label_precisions_orig)
    all_label_recalls.append(label_recalls_orig)
    all_confusion_matrices.append(conf_matrix_orig)
    all_train_accuracies.append(train_accuracy)
    test_results_orig = {
        'sample_ids': np.arange(len(true_labels_orig)),
        'scores': test_scores_orig,
        'piecewise_vectors': tensor_to_numpy(precomputed_vectors_test_orig),
        'predicted_labels': predicted_labels_orig,
        'confidences': confidences_list_orig
    }    
    precomputed_vectors_test_add = precompute_piecewise_vectors(
        features_test_add,
        num_intervals_list,
        feature_min_max,
        device
    )
    test_scores_add = scoring_function(
        precomputed_vectors_test_add,
        num_intervals_list,
        k,
        lambda_opt, delta_u_opt, v_opt,
        device
    )
    predicted_labels_add = []
    confidences_list_add = []
    for sc in test_scores_add:
        confs = []
        for h in unique_labels_all_train:
            num, denom = 0, 0
            for tr_score, tr_label, rel in zip(train_scores_full, labels.cpu().numpy(), train_reliability):
                if tr_label != h:
                    denom += 1
                    if ((tr_score > sc) and (tr_label > h)) or ((tr_score < sc) and (tr_label < h)):
                        num += rel
            confs.append(num / denom if denom != 0 else 0)
        pred_lbl_add = unique_labels_all_train[np.argmax(confs)]
        predicted_labels_add.append(pred_lbl_add)
        confidences_list_add.append(confs)

    predicted_labels_add = np.array(predicted_labels_add)
    true_labels_add = labels_test_add.cpu().numpy()
    accuracy_add = np.mean(predicted_labels_add == true_labels_add)
    mean_absolute_error_add = np.mean(np.abs(predicted_labels_add - true_labels_add))

    label_precisions_add = precision_score(true_labels_add, predicted_labels_add, average=None, zero_division=0)
    label_recalls_add = recall_score(true_labels_add, predicted_labels_add, average=None, zero_division=0)
    macro_precision_add = precision_score(true_labels_add, predicted_labels_add, average='macro', zero_division=0)
    macro_recall_add = recall_score(true_labels_add, predicted_labels_add, average='macro', zero_division=0)

    sample_weights_add = compute_sample_weight(class_weight='balanced', y=true_labels_add)
    errors_add = np.abs(predicted_labels_add - true_labels_add)
    weighted_mae_add = np.average(errors_add, weights=sample_weights_add)
    conf_matrix_add = confusion_matrix(true_labels_add, predicted_labels_add, labels=unique_labels_all_train)



     # predict on the added test set
    test_results_add = {
        'sample_ids': np.arange(len(true_labels_add)),
        'scores': test_scores_add,
        'piecewise_vectors': tensor_to_numpy(precomputed_vectors_test_add),
        'predicted_labels': predicted_labels_add,
        'confidences': confidences_list_add
    }

    optimal_parameters_list.append({
        'lambda': tensor_to_numpy(lambda_opt),
        'delta_u': tensor_to_numpy(delta_u_opt),
        'v': tensor_to_numpy(v_opt)
    })

    train_results = {
        'train_scores': train_scores_full,
        'labels_train': labels.cpu().numpy()
    }
    all_train_results.append(train_results)
    iteration_results_dict = {
        'optimization_time': optimization_time,
        'macro_precision': macro_precision_ori
        'macro_recall': macro_recall_orig,
        'label_precisions': label_precisions_orig,
        'label_recalls': label_recalls_orig,
        'weighted_mae': weighted_mae_orig,
        'accuracy': accuracy_orig,
        'mean_absolute_error': mean_absolute_error_orig,
        'confusion_matrix': conf_matrix_orig,
        'train_accuracy': train_accuracy,
        'train_results': train_results,
        'test_results': test_results_orig,
        'test_results_add': test_results_add,
        'accuracy_add': accuracy_add,
        'mean_absolute_error_add': mean_absolute_error_add,
        'macro_precision_add': macro_precision_add,
        'macro_recall_add': macro_recall_add,
        'confusion_matrix_add': conf_matrix_add,
        'weighted_mae_add': weighted_mae_add,
        'label_precisions_add': label_precisions_add,
        'label_recalls_add': label_recalls_add,
        'best_params': best_params,
        'cv_optimization_time': cv_optimization_time,
        'best_fold_f1scores': best_fold_f1_scores, 
        'best_average_f1score': best_f1_score,
        'optimal_parameters_list': optimal_parameters_list,
        'loss_history': loss_history.copy()
    }
    all_test_results.append(test_results_orig)
    interim_save_path = os.path.join(save_folder, f'learning_results-CV8020weights-single-final-0.05.pkl')
    with open(interim_save_path, 'wb') as interim_file:
        pickle.dump(iteration_results_dict, interim_file)
    print(f"Interim results saved to {interim_save_path}")

# plot the loss function
for i, loss_hist in enumerate(all_loss_histories):
    plt.plot(loss_hist, label=f'Run {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function Descent')
plt.legend()
plt.show()
print("Average Optimization Time:", np.mean(optimization_times))
print("Average cv Optimization Time:", np.mean(CV_optimization_times))




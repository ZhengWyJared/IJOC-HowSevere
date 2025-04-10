import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns

interim_save_path = '..../learning_results.pkl'


if os.path.exists(interim_save_path):
    with open(interim_save_path, 'rb') as f:
        iteration_results_dict = pickle.load(f)
    optimal_parameters_list = iteration_results_dict['optimal_parameters_list']

file_path = '.../train32.xlsx'
data = pd.read_excel(file_path)
labels = data.iloc[:, 0].to_numpy()  
features_df = data.iloc[:, 1:-2] 
print("Extracted features:")
print(features_df.head())  
feature_min_max = [(features_df[col].min(), features_df[col].max()) for col in features_df.columns]
feature_names = [
    "Age",
    "Heart failure 0-1",
    "Interstitial lung disease 0-1",
    "Pneumonia 0-1",
    "Altered consciousness 0-1",
    "Diastolic blood pressure",
    "Heart rate",
    "Hemoglobin",
    "pH",
    r"PaCO$_2$",
    "Serum albumin",
    "Blood urea nitrogen"
]
feature_names1 = [
    "Age",
    "HF",
    "ILD",
    "PN",
    "AC",
    "DBP",
    "HR",
    "Hb",
    "pH",
    r"PaCO$_2$",
    "Alb",
    "BUN"
]


def plot_segmented_functions(delta_u, feature_ranges):
    all_arrays = [arr for sublist in delta_u for arr in sublist]    
    range_differences = []  
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 14))    
    for index, deltas in enumerate(all_arrays):
        min_val, max_val = feature_ranges[index]
        x = np.linspace(min_val, max_val, len(deltas) + 1)
        y = np.cumsum([0] + list(deltas))        
        range_difference = y.max() - y.min()
        range_differences.append(range_difference)
        row = index // 4
        col = index % 4
        ax = axes[row, col]        
        ax.plot(x, y, marker='o', linestyle='-')
        ax.set_title(f'Range: {range_difference:.5f}', fontsize=17)
        ax.set_xlabel(feature_names[index], fontsize=18.5)
        if col == 0:
            ax.set_ylabel('Score', fontsize=19)
        else:
            ax.set_ylabel('')  
        ax.grid(True)
        ax.set_xticks(x)  
        ax.tick_params(axis='both', which='major', labelsize=13)  
        if feature_names[index] == "pH":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))

        
    plt.tight_layout()
    plt.show()
    return range_differences

# Extract optimized parameters
delta_us = [params['delta_u'] for params in optimal_parameters_list]
v =[params['v'] for params in optimal_parameters_list]
print(f"Total number of sublists in v: {len(v)}")
for i, sublist in enumerate(v):
    print(f"Sublist {i+1}: Length = {len(sublist)}")
    for j, element in enumerate(sublist):
        print(f"  Element {j+1}: Shape = {np.array(element).shape}")

# Plot piecewise linear marginal functions for each feature
range_differences = plot_segmented_functions(delta_us, feature_min_max)
v = v[0]
num_features = len(v)  
D = v[0].shape[1]    
pairs_values = {}
for k in range(num_features):
    for l in range(k+1, num_features):
        r_kl = np.sum(np.dot(v[k], v[l].T))
        pairs_values[(k, l)] = r_kl
print(pairs_values)
pairs_values_adjusted = {}
for (k, l), value in pairs_values.items():
    pairs_values_adjusted[(k + 1, l + 1)] = value
p = 4.3866496
single_values = np.abs(np.array(range_differences)) 
pairwise_values = np.abs(np.array(list(pairs_values_adjusted.values())))  
def power_mean(values, p):
    return (np.mean(values**p))**(1/p)
single_power_mean = power_mean(single_values, p)
sum_pairwise = np.sum(pairwise_values)
total = single_power_mean + sum_pairwise
w_Mp = single_power_mean / total
w_pairs = sum_pairwise / total
w_single_in = (single_values**p) / np.sum(single_values**p)
single_weights = w_Mp * w_single_in
w_pairs_in = pairwise_values / sum_pairwise
pairwise_weights = w_pairs * w_pairs_in
weights = []
weights.extend([(f"{feature_names[i]}", single_weights[i]) for i in range(len(feature_names))])
weights.extend([
    (f"{feature_names[k - 1]} & {feature_names[l - 1]}", pairwise_weights[i])
    for i, (k, l) in enumerate(pairs_values_adjusted.keys())
])

# Sort weights in descending order
weights_sorted = sorted(weights, key=lambda x: x[1], reverse=True)
labels, values = zip(*weights_sorted)
plt.figure(figsize=(10, 40))  
y_positions = np.arange(len(labels)) 
plt.barh(y_positions, values, align='center', height=0.9)
plt.yticks(y_positions, labels, fontsize=14)
plt.xticks(fontsize=16)
plt.xlabel("Equivalent Weights", fontsize=16)
plt.ylim(-0.5, len(labels) - 0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

    
    
# Initialize Shapley values
shapley_values = np.zeros(len(feature_names))
# Add single feature weights
shapley_values += single_weights
for (k, l), weight in zip(pairs_values_adjusted.keys(), pairwise_weights):
    shapley_values[k - 1] += weight / 2 
    shapley_values[l - 1] += weight / 2  
shapley_results = [(feature_names[i], shapley_values[i]) for i in range(len(feature_names))]
shapley_results_sorted = sorted(shapley_results, key=lambda x: x[1], reverse=True)
labels, values = zip(*shapley_results_sorted)
plt.figure(figsize=(10, 8))
plt.barh(labels, values, align='center')
plt.xlabel("Shapley Values", fontsize=20) 
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()   
    



# Construct heatmap of pairwise interactions
heatmap_matrix = np.zeros((len(feature_names), len(feature_names)))
# Fill symmetric matrix using absolute interaction values
for (k, l), value in pairs_values_adjusted.items():
    heatmap_matrix[k - 1, l - 1] = value 
    heatmap_matrix[l - 1, k - 1] = value
mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))
plt.figure(figsize=(3, 8))
heatmap = sns.heatmap(
    heatmap_matrix, 
    mask=mask, 
    annot=True,  
    fmt=".2f", 
    cmap="coolwarm",
    xticklabels=feature_names1, 
    yticklabels=feature_names1,
    annot_kws={"size": 23} 
)
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=23) 
colorbar.ax.set_title("Interaction Effect", fontsize=23, pad=10)
plt.xticks(fontsize=24) 
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()



# Plot detailed segmented heatmaps for feature pairs
feature_ranges = feature_min_max
top_5_pairs = sorted(
    pairs_values_adjusted.items(),  
    key=lambda x: abs(x[1]),       
    reverse=True                  
)[:5]  # taking top 5 as examples
for (k, l), _ in top_5_pairs:
    k_index = k - 1  
    l_index = l - 1
    k_min, k_max = feature_ranges[k_index]
    l_min, l_max = feature_ranges[l_index]

    k_segments = v[k_index].shape[0]
    l_segments = v[l_index].shape[0]
    k_values = np.linspace(k_min, k_max, k_segments + 1)
    l_values = np.linspace(l_min, l_max, l_segments + 1)
    M = np.dot(v[k_index], v[l_index].T) 
    M_cumulative = np.cumsum(np.cumsum(M, axis=0), axis=1)
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        M_cumulative,
        annot=True, 
        fmt=".2f",  
        cmap="coolwarm",
        xticklabels=False,  
        yticklabels=False, 
        annot_kws={"size": 16}
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16) 
    colorbar.ax.set_xlabel("Cumulative Interaction", fontsize=17, labelpad=10) 
    colorbar.ax.xaxis.set_label_position('top') 
    plt.xticks(
        ticks=np.arange(0, l_segments + 1),  
        labels=[f"{val:.2f}" for val in l_values],
        rotation=30,
        ha='right',
        fontsize=15.5 
    )
    plt.yticks(
        ticks=np.arange(0, k_segments + 1),  
        labels=[f"{val:.2f}" for val in k_values],
        rotation=0,
        fontsize=15.5 
    )
    plt.xlabel(f"{feature_names[l_index]}", fontsize=18)
    plt.ylabel(f"{feature_names[k_index]}", fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
# Plot the radar chart for visualizing the counterfactual explanations of the No. 699 patient   
    original_values = [
        69.0000015258789, 1.0, 0.0, 1.0, 0.0,
        47.99999912381172, 89.00000056028367, 58.00000047683716,
        7.551999955821037, 73.69999748885633, 34.00000065326691, 9.279999469757081
    ]
    optimal_values = [
        69.0, 1.0, 0.0, 1.0, 0.0,
        65.32523921728134, 88.99999909400941, 106.0,
        7.551999922811985, 73.70000036358834, 41.16053869724274, 9.279999469757081
    ]
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)
    original_values_normalized = [
        normalize(value, feature_min_max[i][0], feature_min_max[i][1])
        for i, value in enumerate(original_values)
    ]
    optimal_values_normalized = [
        normalize(value, feature_min_max[i][0], feature_min_max[i][1])
        for i, value in enumerate(optimal_values)
    ]
    original_values_normalized += original_values_normalized[:1]
    optimal_values_normalized += optimal_values_normalized[:1]
    angles = np.linspace(0, 2 * np.pi, len(feature_names1), endpoint=False).tolist()
    angles += angles[:1]  
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, original_values_normalized, label='Original Values', color='blue', linestyle='--', linewidth=2)  # 原始值用虚线


    ax.fill(angles, original_values_normalized, color='blue', alpha=0.25)
    ax.plot(angles, optimal_values_normalized, label='Optimal Values', color='orange', linestyle='-', linewidth=2)  # 最优值用实线
    ax.fill(angles, optimal_values_normalized, color='orange', alpha=0.25)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names1, fontsize=13, wrap=True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=15)
    plt.title("Counterfactual intervention", fontsize=17, pad=20)
    plt.tight_layout()
    plt.show()

    
    


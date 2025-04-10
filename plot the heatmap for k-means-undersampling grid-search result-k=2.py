import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast


file_path = '..../K=2-Kmeans-undersampling-gridsearch/final_results_weightrecall.xlsx'
df = pd.read_excel(file_path)

dimension1 = []
dimension2 = []
color_values = []

for index, row in df.iterrows():
    quantile_str = row['Quantile_Combination']
    quantiles = ast.literal_eval(quantile_str)
    first_two = quantiles[:2]
    last_two = quantiles[2:4]    
    dim1_value = sum(first_two) / len(first_two)
    dim2_value = sum(last_two) / len(last_two)
    color_value = row.iloc[-1] 
    dimension1.append(dim1_value)
    dimension2.append(dim2_value)
    color_values.append(color_value)
heatmap_data = pd.DataFrame({
    'Dimension1': dimension1,
    'Dimension2': dimension2,
    'ColorValue': color_values
})
heatmap_data['Dimension1'] = heatmap_data['Dimension1'].round(5)
heatmap_data['Dimension2'] = heatmap_data['Dimension2'].round(5)
heatmap_matrix = heatmap_data.pivot_table(
    index='Dimension1',
    columns='Dimension2',
    values='ColorValue',
    aggfunc='max' 
)
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_matrix, annot=False, cmap='YlGnBu', cbar_kws={'label': ''})
cbar = plt.gca().collections[0].colorbar  
cbar.ax.set_xlabel('Macro Recall', fontsize=22, labelpad=10) 
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=17)
plt.xlabel('Average of the two quantiles-label 2', fontsize=22)
plt.ylabel('Average of the two quantiles-label 1', fontsize=22)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()
